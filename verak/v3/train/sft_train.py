"""QLoRA warm start with inference-exact, action-only token loss; no RFT."""
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

from ..common import file_sha, read_json, write_json
from .sft_data import PHASE, REVISION, config_for, load_samples


def target_positions(labels):
    """Prediction positions for supervised next tokens; batch size one only."""
    import torch
    if labels.ndim != 2 or labels.shape[0] != 1 or labels[0, 0] != -100:
        raise ValueError('Sparse action loss requires one sample and a masked first token')
    positions = torch.nonzero(labels[0, 1:] != -100, as_tuple=False).flatten()
    if not len(positions):
        raise ValueError('No action targets in sample')
    return positions


def action_loss(model, inputs, num_items_in_batch=None):
    """Masked next-token CE without projecting unsupervised context positions."""
    import torch.nn.functional as F
    labels = inputs['labels']
    positions = target_positions(labels)
    outputs = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'],
                    logits_to_keep=positions, use_cache=False)
    targets = labels[0, positions + 1]
    loss = F.cross_entropy(outputs.logits[0].float(), targets, reduction='sum')
    denominator = num_items_in_batch if num_items_in_batch is not None else len(targets)
    return loss / denominator, outputs


def train(role, *, resume=False):
    config = config_for()
    root = config['paths'][PHASE + '_output']
    os.environ.update(CUDA_VISIBLE_DEVICES='0', HF_HUB_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
                      WANDB_DISABLED='true', PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    sys.path.insert(0, str(root / 'python_deps'))
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainerCallback, set_seed
    from trl import SFTConfig, SFTTrainer
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Training must see physical GPU0 only')
    manifest_path = root / 'data/manifest.json'
    manifest = read_json(manifest_path)
    output = root / 'adapters' / role
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'complete.json').exists():
        raise ValueError('This role is already trained; do not retrain a completed run')
    if role == 'korean' and not (root / 'adapters/global/complete.json').exists():
        raise ValueError('GLOBAL must finish before KOREAN')
    samples = {}
    for part in ('train', 'validation'):
        meta = manifest['roles'][role][part]
        if file_sha(Path(meta['path'])) != meta['sha256']:
            raise ValueError('Export changed')
        samples[part] = load_samples(meta['path'])
    recipe = {'role': role, 'base_revision': REVISION, 'export_sha256': file_sha(manifest_path),
        'r': 16, 'alpha': 32, 'dropout': .05, 'target_modules': 'all-linear', 'learning_rate': 1e-4,
        'epochs': 2, 'context': 8192, 'gradient_checkpointing': True, 'micro_batch': 1,
        'gradient_accumulation': 8, 'effective_batch': 8, 'packing': False, 'bf16': True,
        'quantization': 'NF4, double quantization, BF16 compute', 'optimizer': 'paged_adamw_8bit',
        'scheduler': 'cosine', 'warmup_ratio': .03, 'weight_decay': 0., 'max_grad_norm': 1., 'seed': 71,
        'loss': 'sum current-action next-token CE / target tokens in accumulation window',
        'validation_loss': 'token-weighted action-only NLL over all held-out turns',
        'code_sha256': file_sha(Path(__file__)), 'gpu': torch.cuda.get_device_name(0),
        'versions': {name: importlib.metadata.version(name) for name in
                     ('torch', 'transformers', 'peft', 'accelerate', 'bitsandbytes', 'datasets', 'trl')}}
    recipe_path = output / 'recipe.json'
    if recipe_path.exists():
        if read_json(recipe_path) != recipe:
            raise ValueError('Frozen training recipe changed')
        if not resume:
            raise ValueError('Existing unfinished training requires --resume')
    else:
        write_json(recipe_path, recipe)
    set_seed(71)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_num_threads(8)
    tokenizer = AutoTokenizer.from_pretrained(str(config['paths']['policy_base']), local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(str(config['paths']['policy_base']), local_files_only=True,
        device_map={'': 0}, torch_dtype=torch.bfloat16, attn_implementation='sdpa',
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16))
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
                                           gradient_checkpointing_kwargs={'use_reentrant': False})
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=.05,
        target_modules='all-linear', bias='none', task_type='CAUSAL_LM'))
    trainable = [(name, p.numel()) for name, p in model.named_parameters() if p.requires_grad]
    if not trainable or any('lora_' not in name for name, _ in trainable):
        raise ValueError('Only policy LoRA parameters may train')
    write_json(output / 'trainable.json', {'parameters': sum(n for _, n in trainable), 'tensors': trainable})
    def collate(rows):
        if len(rows) != 1:
            raise ValueError('Do not silently change micro-batch or pad supervised tokens')
        row = rows[0]
        return {'input_ids': torch.tensor([row['input_ids']], dtype=torch.long),
                'labels': torch.tensor([row['labels']], dtype=torch.long),
                'attention_mask': torch.ones((1, len(row['input_ids'])), dtype=torch.long)}
    class ActionTrainer(SFTTrainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            # Project only target prediction positions; mathematically the same
            # masked CE, avoiding a [8192,128259] dense output allocation.
            with torch.autocast('cuda', dtype=torch.bfloat16):
                loss, outputs = action_loss(model, inputs, num_items_in_batch)
            return (loss, outputs) if return_outputs else loss

        def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix='eval'):
            # Every action token has equal weight, regardless of action length.
            was_training = self.model.training
            self.model.eval()
            loss_sum, count = 0., 0
            started_eval = time.monotonic()
            with torch.no_grad():
                for sample in samples['validation']:
                    batch = self._prepare_inputs(collate([sample]))
                    n = int((batch['labels'] != -100).sum())
                    with self.compute_loss_context_manager():
                        loss = self.compute_loss(self.model, batch)
                    loss_sum += float(loss) * n
                    count += n
            if was_training:
                self.model.train()
            metrics = {metric_key_prefix + '_loss': loss_sum / count,
                       metric_key_prefix + '_target_tokens': count,
                       metric_key_prefix + '_runtime': time.monotonic() - started_eval}
            self.log(metrics)
            self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
            return metrics
    class EpochRecord(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs:
                print(json.dumps({'role': role, 'step': state.global_step, 'epoch': state.epoch, **logs}), flush=True)
        def on_save(self, args, state, control, **kwargs):
            epoch = round(state.epoch)
            checkpoint = output / f'checkpoint-{state.global_step}'
            adapter = output / f'epoch_{epoch}'
            adapter.mkdir(exist_ok=True)
            for name in ('adapter_config.json', 'adapter_model.safetensors'):
                shutil.copy2(checkpoint / name, adapter / name)
            tokenizer.save_pretrained(adapter)
            write_json(adapter / 'provenance.json', {'role': role, 'epoch': epoch,
                'global_step': state.global_step, 'checkpoint': str(checkpoint),
                'recipe_sha256': file_sha(recipe_path), 'base_revision': REVISION,
                'adapter_sha256': file_sha(adapter / 'adapter_model.safetensors'),
                'log_history': state.log_history})
            print(json.dumps({'checkpoint_saved': str(adapter), 'epoch': epoch}), flush=True)
    args = SFTConfig(output_dir=str(output), num_train_epochs=2, learning_rate=1e-4,
        per_device_train_batch_size=1, per_device_eval_batch_size=1, gradient_accumulation_steps=8,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={'use_reentrant': False},
        bf16=True, fp16=False, max_length=8192, packing=False, dataset_kwargs={'skip_prepare_dataset': True},
        optim='paged_adamw_8bit', lr_scheduler_type='cosine', warmup_ratio=.03, weight_decay=0.,
        max_grad_norm=1., eval_strategy='epoch', save_strategy='epoch', logging_steps=10,
        save_total_limit=None, seed=71, data_seed=71, report_to='none', disable_tqdm=True,
        remove_unused_columns=False, dataloader_num_workers=0, prediction_loss_only=True,
        dataloader_pin_memory=True, average_tokens_across_devices=False)
    trainer = ActionTrainer(model=model, args=args, processing_class=tokenizer, data_collator=collate,
        train_dataset=Dataset.from_list(samples['train']), eval_dataset=Dataset.from_list(samples['validation']),
        callbacks=[EpochRecord()])
    trainer.model_accepts_loss_kwargs = True
    started = time.monotonic()
    if not resume:
        write_json(output / 'initial_validation.json', trainer.evaluate())
    checkpoints = sorted(output.glob('checkpoint-*'), key=lambda p: int(p.name.split('-')[1]))
    trained = trainer.train(resume_from_checkpoint=str(checkpoints[-1]) if resume and checkpoints else None)
    for epoch in (1, 2):
        if not (output / f'epoch_{epoch}/adapter_model.safetensors').exists():
            raise RuntimeError('An epoch checkpoint is missing')
    write_json(output / 'complete.json', {'role': role, 'completed': True, 'metrics': trained.metrics,
        'log_history': trainer.state.log_history, 'elapsed_s': time.monotonic() - started,
        'max_cuda_allocated_bytes': torch.cuda.max_memory_allocated(),
        'max_cuda_reserved_bytes': torch.cuda.max_memory_reserved(), 'recipe_sha256': file_sha(recipe_path)})
    print(json.dumps({'role': role, 'completed': True, 'elapsed_s': time.monotonic() - started}), flush=True)
