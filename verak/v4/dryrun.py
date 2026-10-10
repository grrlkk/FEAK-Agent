"""Explicit entrypoints for the authorized small v4.3 technical dry run."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import time

from .dryrun_common import (ROOT, ROLES, BASE, REVISION, constrain_cpu, verify_manifest,
    read_json, write_json, file_sha, atomic_new, reference)


def preflight():
    from .policy_prompts_v43 import verify_frozen, FROZEN
    from .dryrun_data import export
    from .dryrun_eval_data import prepare
    constrain_cpu()
    if (ROOT/'preflight.json').exists():
        saved = read_json(ROOT/'preflight.json')
        verify_manifest()
        if any(file_sha(Path(__file__).with_name(name)) != digest for name, digest in saved['code_sha256'].items()):
            raise ValueError('Saved preflight refers to another dry-run implementation')
        return saved
    manifest = export()
    evaluation = prepare()
    protected = verify_frozen()
    audits = manifest['action_audit']
    result = {'status': 'ready_for_root_review', 'at': time.time(),
        'data_manifest': reference(ROOT/'data/manifest.json'),
        'source_manifest': reference(ROOT/'data/source_manifest.json'),
        'evaluation_manifest': reference(ROOT/'evaluation/manifest.json'),
        'base_path': str(BASE.resolve()), 'base_revision': REVISION,
        'frozen_v43': reference(FROZEN), 'environment_hashes': protected['environment_sha256'],
        'prompt_tokens': {r: protected['roles'][r]['policy_tokens'] for r in ROLES},
        'role_source_counts': {r: {p: m['sources'] for p, m in parts.items()} for r, parts in manifest['roles'].items()},
        'exported_action_targets': len(audits), 'failed_export_contract_checks': 0,
        'maximum_prefix_tokens': max(a['prefix_tokens'] for a in audits),
        'maximum_total_tokens': max(a['total_tokens'] for a in audits),
        'exact_prefix_and_system_every_call': True, 'all_observations_tools_notices_masked': True,
        'canonical_equals_executed_first_teacher_action': True, 'end_turn_has_loss': True,
        'packing': False, 'truncation': False, 'evaluation_cohorts': {k: len(v['cases']) for k, v in evaluation['cohorts'].items()},
        'strict_student_parser': True, 'teacher_trimming_at_inference': False,
        'quality_scores_or_judges': False, 'new_paid_calls': 0, 'B3_training_included': False,
        'recipe': {'physical_gpu': 1, 'roles_sequential': list(ROLES), 'fresh_base_each_role': True,
            'r': 16, 'alpha': 32, 'lr': 1e-4, 'epochs': 1, 'context': 8192,
            'gradient_checkpointing': True, 'NF4': True, 'BF16': True, 'microbatch': 1, 'gradient_accumulation': 8},
        'code_sha256': {p.name: file_sha(p) for p in sorted(Path(__file__).parent.glob('dryrun*.py'))},
        'packages': {name: importlib.metadata.version(name) for name in
                     ('torch', 'transformers', 'peft', 'accelerate', 'bitsandbytes', 'datasets')}}
    atomic_new(ROOT/'preflight.json', result)
    return result


def report():
    from .dryrun_common import verify_manifest
    manifest = verify_manifest()
    value = {'technical_dry_run_only': True, 'quality_results': False, 'paid_calls': 0,
        'data_manifest': reference(ROOT/'data/manifest.json'), 'roles': manifest['roles'],
        'export_contract': manifest['contract'], 'training': {}, 'evaluation': None}
    for role in ROLES:
        output = ROOT/'adapters'/role
        value['training'][role] = {'status': 'not_started'}
        for name in ('failure', 'complete'):
            if (output/(name+'.json')).exists():
                value['training'][role] = {'status': name, 'artifact': reference(output/(name+'.json'))}
    if (ROOT/'evaluation/complete.json').exists():
        value['evaluation'] = read_json(ROOT/'evaluation/complete.json')
    for name in ('training_failure', 'inference_failure'):
        if (ROOT/(name+'.json')).exists():
            value[name] = read_json(ROOT/(name+'.json'))
    write_json(ROOT/'component_metrics.json', value)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['export', 'prepare-eval', 'preflight', 'train-only', 'run',
        'train-role', 'infer', 'evaluate-local', 'report'])
    parser.add_argument('role', nargs='?', choices=ROLES)
    args = parser.parse_args()
    if args.command in {'export', 'prepare-eval', 'preflight', 'evaluate-local', 'report'}:
        constrain_cpu()
    if args.command == 'export':
        from .dryrun_data import export
        result = export()
    elif args.command == 'prepare-eval':
        from .dryrun_eval_data import prepare
        result = prepare()
    elif args.command == 'preflight':
        result = preflight()
    elif args.command == 'train-role':
        from .dryrun_train import train
        if args.role is None:
            parser.error('train-role requires a role')
        result = train(args.role)
    elif args.command == 'train-only':
        from .dryrun_service import train_only
        result = train_only()
    elif args.command == 'run':
        from .dryrun_service import train_only, serve_and_evaluate
        try:
            train_only()
            serve_and_evaluate()
        finally:
            # File-only reporting also preserves a failed stage; no retries.
            report()
        result = None
    elif args.command == 'infer':
        from .dryrun_service import serve_and_evaluate
        result = serve_and_evaluate()
    elif args.command == 'evaluate-local':
        from .dryrun_student import evaluate
        result = evaluate()
    else:
        result = report()
    print(json.dumps({'command': args.command, 'completed': True, 'output_root': str(ROOT)}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
