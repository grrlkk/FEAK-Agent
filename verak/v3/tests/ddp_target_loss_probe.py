"""CPU-only two-rank check of the installed Trainer's accumulation semantics."""
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.nn.functional as F
from datasets import Dataset
from transformers import Trainer, TrainingArguments

from verak.v3.rft1.train import distributed_action_loss


def main():
    torch.set_num_threads(1)
    torch.manual_seed(17)
    rank = int(os.environ['RANK'])

    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.logits = torch.nn.Parameter(torch.randn(6, 7))

        def forward(self, input_ids, attention_mask, logits_to_keep=None, use_cache=False):
            ids = input_ids if logits_to_keep is None else input_ids[:, logits_to_keep]
            return SimpleNamespace(logits=self.logits[ids])

    model = Toy()
    initial = model.logits.detach().clone()
    labels = [[-100, -100, -100, 2, 3, 4], [-100, 1, -100, -100, -100, -100],
              [-100, -100, 2, 3, -100, -100], [-100, 2, 3, 4, 5, 6],
              [-100, 1, 2, 3, 4, 5], [-100, -100, -100, -100, -100, 2],
              [-100, -100, -100, -100, 1, 2], [-100, -100, 3, -100, 4, -100]]
    rows = [{'input_ids': list(range(6)), 'labels': row} for row in labels]
    total = sum(sum(v != -100 for v in row) for row in labels)
    target = initial.clone().requires_grad_(True)
    expected_loss = sum(F.cross_entropy(target[:-1], torch.tensor(row[1:]),
        ignore_index=-100, reduction='sum') for row in labels) / total
    expected_loss.backward()
    expected = initial - .1 * target.grad

    def collate(values):
        assert len(values) == 1
        return {'input_ids': torch.tensor([values[0]['input_ids']]),
                'labels': torch.tensor([values[0]['labels']]),
                'attention_mask': torch.ones((1, 6), dtype=torch.long)}

    class TargetTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            loss, outputs = distributed_action_loss(model, inputs, num_items_in_batch,
                                                     world_size=self.accelerator.num_processes)
            return (loss, outputs) if return_outputs else loss

    output = Path(os.environ['VERAK_DDP_PROBE_OUTPUT'])
    args = TrainingArguments(output_dir=str(output / 'trainer'), use_cpu=True, ddp_backend='gloo',
        per_device_train_batch_size=1, gradient_accumulation_steps=4, max_steps=1,
        learning_rate=.1, optim='sgd', lr_scheduler_type='constant', weight_decay=0, max_grad_norm=0,
        average_tokens_across_devices=True, ddp_find_unused_parameters=False,
        save_strategy='no', eval_strategy='no', logging_strategy='no',
        remove_unused_columns=False, report_to='none', disable_tqdm=True, dataloader_pin_memory=False)
    trainer = TargetTrainer(model=model, args=args, train_dataset=Dataset.from_list(rows), data_collator=collate)
    trainer.model_accepts_loss_kwargs = True
    assert trainer.accelerator.num_processes == 2 and not torch.cuda.is_available()
    trainer.train()
    maximum_error = float((trainer.model.logits.detach() - expected).abs().max())
    assert maximum_error < 2e-6, maximum_error
    if rank == 0:
        output.mkdir(exist_ok=True, parents=True)
        (output / 'result.json').write_text(json.dumps({'ranks': 2, 'target_tokens': total,
            'maximum_parameter_error': maximum_error, 'cuda_used': False,
            'same_update_as_global_token_mean': True}) + '\n')
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
