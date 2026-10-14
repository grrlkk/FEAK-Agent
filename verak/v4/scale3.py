"""Explicit v4.3 B2/B3 commands; no scheduler, scorer launcher, or training."""
import argparse
import json
from pathlib import Path

from .scale3_common import ROOT, low_priority_cpu, saved_cost_estimate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('cost-estimate', 'content-freeze', 'content-run',
        'content-report', 'corruption-freeze', 'corruption-run', 'corruption-report', 'corruption-gpu-finalize'))
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    component = 'B3' if args.command.startswith('corruption-') else 'B2'
    low_priority_cpu(component)
    if args.command == 'cost-estimate':
        value = saved_cost_estimate()
    elif args.command.startswith('content-'):
        from . import scale3_content as module
        operation = args.command.removeprefix('content-')
        value = module.freeze_sample(args.root) if operation == 'freeze' else module.run(args.root, limit=args.limit) if operation == 'run' else module.report(args.root)
    else:
        from . import scale3_corruption as module
        operation = args.command.removeprefix('corruption-')
        if operation == 'freeze':
            value, _ = module.freeze_sample(args.root)
        elif operation == 'run':
            value = module.run(args.root, limit=args.limit)
        elif operation == 'report':
            value = module.report(args.root)
        else:
            # Tokenizer loading is local CPU only. The function consumes saved
            # reference files; it cannot start a GPU job or contact a scorer.
            from transformers import AutoTokenizer
            from .common import load_config
            tokenizer = AutoTokenizer.from_pretrained(str(load_config()['paths']['policy_base']), local_files_only=True)
            value = module.gpu_finalize(args.root, tokenizer=tokenizer)
    print(json.dumps({k: v for k, v in value.items() if k not in ('source_metadata', 'episode_ids', 'selected',
        'source_ids', 'measured', 'score_proofs', 'errors', 'prior_cohort_sha256', 'test_question_hashes')}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
