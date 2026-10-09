"""Two-device QLoRA continuation, with globally token-normalized action loss."""
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys
import time

from ..common import file_sha, read_json, write_json
from ..train.sft_data import REVISION, load_samples
from ..train.sft_train import action_loss
from .config import PHASE


def distributed_action_loss(model, inputs, num_items_in_batch=None, *, world_size=1):
    loss, outputs = action_loss(model, inputs, num_items_in_batch)
    # Trainer gathers target counts across ranks AND the accumulation window.
    # DDP averages rank gradients; undo that average, but only for that denominator.
    if num_items_in_batch is not None:
        loss = loss * world_size
    return loss, outputs


def train(config, role, *, resume=False, oneshot=False):
    phase = 'oneshot_baseline' if oneshot else PHASE
    root = config['paths']['repo'] / 'verak/v3/outputs' / phase
    os.environ.update(HF_HUB_OFFLINE='1', TOKENIZERS_PARALLELISM='false', WANDB_DISABLED='true',
                      PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    sys.path.insert(0, str(config['paths']['phase7_sft_output'] / 'python_deps'))
    import torch
    import torch.distributed as dist
    from datasets import Dataset
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainerCallback, set_seed
    from trl import SFTConfig, SFTTrainer

    rank, local_rank = int(os.environ['RANK']), int(os.environ['LOCAL_RANK'])
    if int(os.environ['WORLD_SIZE']) != 2 or torch.cuda.device_count() != 2:
        raise RuntimeError('This authorized training requires exactly two data-parallel GPUs')
    torch.cuda.set_device(local_rank)
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = True
    manifest_path = root / 'data/manifest.json'
    manifest = read_json(manifest_path)
    output = root / 'adapters' / role
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'complete.json').exists():
        raise ValueError('Completed training must not be repeated')
    if not oneshot and role == 'korean' and not (root / 'adapters/global/complete.json').exists():
        raise ValueError('Finish GLOBAL before KOREAN')
    samples = {}
    for part in ('train', 'validation'):
        metadata = manifest['roles'][role][part]
        if file_sha(metadata['path']) != metadata['sha256']:
            raise ValueError('Training export changed')
        samples[part] = load_samples(metadata['path'])
    if not samples['train']:
        raise ValueError('Empty training set')
    initial = config['paths']['phase7_sft_output'] / 'adapters' / role / 'epoch_2'
    learning_rate = 1e-4 if oneshot else 5e-5
    recipe = {'role': role, 'phase': phase, 'base_revision': REVISION,
        'export_sha256': file_sha(manifest_path), 'r': 16, 'alpha': 32, 'dropout': .05,
        'target_modules': 'all-linear', 'learning_rate': learning_rate, 'epochs': 1,
        'initial_adapter': None if oneshot else str(initial),
        'initial_adapter_sha256': None if oneshot else file_sha(initial / 'adapter_model.safetensors'),
        'context': 8192, 'gradient_checkpointing': True, 'world_size': 2, 'micro_batch': 1,
        'gradient_accumulation_per_rank': 4, 'effective_batch': 8, 'packing': False,
        'bf16': True, 'quantization': 'NF4, double quantization, BF16 compute',
        'optimizer': 'paged_adamw_8bit', 'scheduler': 'cosine', 'warmup_ratio': .03,
        'weight_decay': 0., 'max_grad_norm': 1., 'seed': 83 if not oneshot else 71,
        'loss': 'sum target CE / target tokens in both ranks and accumulation window; DDP factor compensated',
        'validation_loss': 'token-weighted held-out NLL; each example evaluated once across both ranks',
        'optimizer_state': 'new round optimizer, not SFT optimizer continuation',
        'code_sha256': file_sha(Path(__file__)), 'gpu': torch.cuda.get_device_name(local_rank),
        'versions': {name: importlib.metadata.version(name) for name in
            ('torch', 'transformers', 'peft', 'accelerate', 'bitsandbytes', 'datasets', 'trl')}}
    recipe_path = output / 'recipe.json'
    if rank == 0:
        if recipe_path.exists():
            if read_json(recipe_path) != recipe:
                raise ValueError('Frozen training recipe changed')
            if not resume:
                raise ValueError('Unfinished training requires --resume')
        else:
            write_json(recipe_path, recipe)
    set_seed(recipe['seed'])
    tokenizer = AutoTokenizer.from_pretrained(str(config['paths']['policy_base']), local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(str(config['paths']['policy_base']), local_files_only=True,
        device_map={'': local_rank}, torch_dtype=torch.bfloat16, attn_implementation='sdpa',
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16))
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
                                            gradient_checkpointing_kwargs={'use_reentrant': False})
    if oneshot:
        model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=.05,
            target_modules='all-linear', bias='none', task_type='CAUSAL_LM'))
    else:
        model = PeftModel.from_pretrained(model, str(initial), is_trainable=True)
        peft_config = model.peft_config['default']
        if (peft_config.r, peft_config.lora_alpha, peft_config.lora_dropout) != (16, 32, .05):
            raise ValueError('Continuation adapter recipe differs from accepted SFT')
    trainable = [(name, p.numel()) for name, p in model.named_parameters() if p.requires_grad]
    if not trainable or any('lora_' not in name for name, _ in trainable):
        raise ValueError('Only the LoRA may train')
    if rank == 0:
        write_json(output / 'trainable.json', {'parameters': sum(n for _, n in trainable), 'tensors': trainable})

    def collate(rows):
        if len(rows) != 1:
            raise ValueError('Sparse target projection requires micro-batch one')
        row = rows[0]
        return {'input_ids': torch.tensor([row['input_ids']], dtype=torch.long),
                'labels': torch.tensor([row['labels']], dtype=torch.long),
                'attention_mask': torch.ones((1, len(row['input_ids'])), dtype=torch.long)}

    class ActionTrainer(SFTTrainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            with torch.autocast('cuda', dtype=torch.bfloat16):
                loss, outputs = distributed_action_loss(model, inputs, num_items_in_batch,
                                                        world_size=self.accelerator.num_processes)
            return (loss, outputs) if return_outputs else loss

        def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix='eval'):
            was_training = self.model.training
            self.model.eval()
            sums = torch.zeros(2, dtype=torch.float64, device=self.args.device)
            started_eval = time.monotonic()
            # self.model is the unwrapped model: uneven shard lengths cannot deadlock DDP.
            with torch.no_grad():
                for sample in samples['validation'][rank::2]:
                    batch = self._prepare_inputs(collate([sample]))
                    n = int((batch['labels'] != -100).sum())
                    with self.compute_loss_context_manager():
                        loss = self.compute_loss(self.model, batch)
                    sums[0] += loss.detach().double() * n
                    sums[1] += n
            dist.all_reduce(sums, op=dist.ReduceOp.SUM)
            if was_training:
                self.model.train()
            metrics = {metric_key_prefix + '_loss': float(sums[0] / sums[1]) if sums[1] else None,
                metric_key_prefix + '_target_tokens': int(sums[1]),
                metric_key_prefix + '_runtime': time.monotonic() - started_eval}
            self.log(metrics)
            self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
            return metrics

    class EpochRecord(TrainerCallback):
        def on_epoch_end(self, args, state, control, **kwargs):
            control.should_save = True

        def on_log(self, args, state, control, logs=None, **kwargs):
            if rank == 0 and logs:
                print(json.dumps({'role': role, 'step': state.global_step, 'epoch': state.epoch, **logs}), flush=True)

        def on_save(self, args, state, control, **kwargs):
            if rank != 0 or state.epoch < 1 - 1e-8:
                return
            checkpoint = output / f'checkpoint-{state.global_step}'
            adapter = output / 'epoch_1'
            adapter.mkdir(exist_ok=True)
            for name in ('adapter_config.json', 'adapter_model.safetensors'):
                shutil.copy2(checkpoint / name, adapter / name)
            tokenizer.save_pretrained(adapter)
            write_json(adapter / 'provenance.json', {'role': role, 'epoch': 1,
                'global_step': state.global_step, 'checkpoint': str(checkpoint),
                'recipe_sha256': file_sha(recipe_path), 'base_revision': REVISION,
                'adapter_sha256': file_sha(adapter / 'adapter_model.safetensors'),
                'log_history': state.log_history, 'initial_adapter_sha256': recipe['initial_adapter_sha256']})

    args = SFTConfig(output_dir=str(output), num_train_epochs=1, learning_rate=learning_rate,
        per_device_train_batch_size=1, per_device_eval_batch_size=1, gradient_accumulation_steps=4,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={'use_reentrant': False},
        bf16=True, fp16=False, max_length=8192, packing=False, dataset_kwargs={'skip_prepare_dataset': True},
        optim='paged_adamw_8bit', lr_scheduler_type='cosine', warmup_ratio=.03, weight_decay=0.,
        max_grad_norm=1., eval_strategy='epoch', save_strategy='steps', save_steps=100, logging_steps=10,
        save_total_limit=3, seed=recipe['seed'], data_seed=recipe['seed'], report_to='none', disable_tqdm=True,
        remove_unused_columns=False, dataloader_num_workers=0, prediction_loss_only=True,
        dataloader_pin_memory=True, average_tokens_across_devices=True,
        ddp_find_unused_parameters=False, ddp_timeout=86400)
    trainer = ActionTrainer(model=model, args=args, processing_class=tokenizer, data_collator=collate,
        train_dataset=Dataset.from_list(samples['train']), eval_dataset=Dataset.from_list(samples['validation']),
        callbacks=[EpochRecord()])
    trainer.model_accepts_loss_kwargs = True
    if trainer.accelerator.num_processes != 2 or not dist.is_initialized():
        raise RuntimeError('Data parallelism was not initialized')
    dist.barrier()
    if read_json(recipe_path) != recipe:
        raise ValueError('Ranks disagree on the frozen training recipe')
    started = time.monotonic()
    if not (output / 'initial_validation.json').exists():
        initial_metrics = trainer.evaluate()
        if rank == 0:
            write_json(output / 'initial_validation.json', initial_metrics)
        dist.barrier()
    checkpoints = sorted((p for p in output.glob('checkpoint-*') if all((p / name).exists()
        for name in ('trainer_state.json', 'adapter_model.safetensors', 'optimizer.pt', 'scheduler.pt'))),
        key=lambda p: int(p.name.split('-')[1]))
    trained = trainer.train(resume_from_checkpoint=str(checkpoints[-1]) if resume and checkpoints else None)
    dist.barrier()
    if not (output / 'epoch_1/adapter_model.safetensors').exists():
        raise RuntimeError('Final epoch checkpoint is missing')
    memory = torch.tensor([torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()],
                          dtype=torch.int64, device=trainer.args.device)
    dist.all_reduce(memory, op=dist.ReduceOp.MAX)
    if rank == 0:
        write_json(output / 'complete.json', {'role': role, 'completed': True, 'metrics': trained.metrics,
            'log_history': trainer.state.log_history, 'elapsed_s': time.monotonic() - started,
            'max_cuda_allocated_bytes': int(memory[0]), 'max_cuda_reserved_bytes': int(memory[1]),
            'world_size': 2, 'recipe_sha256': file_sha(recipe_path)})
        print(json.dumps({'role': role, 'completed': True, 'elapsed_s': time.monotonic() - started}), flush=True)
    dist.barrier()
    dist.destroy_process_group()
