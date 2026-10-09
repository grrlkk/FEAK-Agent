"""CPU/API-only GLOBAL teachers; approved cumulative $40 cap, no training."""
import argparse
import json

from ..global_boost.config import config_for, constrain_cpu


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('stage', choices=['prepare', 'qc', 'teacher', 'measure', 'report',
        'continue', 'launch', 'restart', 'prefetch', 'prefetch-loop', 'launch-prefetch', 'restart-prefetch',
        'expand', 'launch-expansion', 'restart-expansion', 'expansion-plan', 'prepare-expansion',
        'audit-variants', 'cpu-ready', 'gpu-finalize', 'finalize', 'launch-finalizer', 'restart-finalizer',
        'v4-stop-legacy', 'v4-compare', 'v4-prepare', 'v4-run', 'v4-launch'])
    parser.add_argument('--max-api-calls', type=int, default=20000)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    constrain_cpu()
    config = config_for()
    if args.stage in {'v4-stop-legacy', 'v4-compare'}:
        from ..global_boost.v4 import stop_legacy, compare_saved
        result = {'v4-stop-legacy': stop_legacy, 'v4-compare': compare_saved}[args.stage](config)
        if args.stage == 'v4-compare':
            result = {k:v for k,v in result.items() if k != 'input_sha256'}
    elif args.stage == 'v4-prepare':
        from ..global_boost.v4_data import prepare
        value = prepare(config)
        result = {k:v for k,v in value.items() if k not in {'source_policy','source_inventory'}}
    elif args.stage in {'v4-run','v4-launch'}:
        from ..global_boost.v4_service import run, launch
        result = (run if args.stage == 'v4-run' else launch)(config)
    elif args.stage == 'prepare':
        from ..global_boost.prepare import prepare
        result = prepare(config)
        result = {k: result[k] for k in ('counts', 'shortfall', 'gpu_used')}
    elif args.stage == 'qc':
        from ..global_boost.qc import run
        result = run(config, args.max_api_calls)
    elif args.stage == 'teacher':
        from ..global_boost.teacher import run
        result = run(config, max_api_calls=args.max_api_calls, limit=args.limit)
    elif args.stage == 'measure':
        from ..global_boost.measure import run
        result = run(config, limit=args.limit)
    elif args.stage == 'report':
        from ..global_boost.report import report
        result = report(config)
    elif args.stage in {'prefetch', 'prefetch-loop'}:
        from ..global_boost.prefetch import enqueue, watch
        result = (enqueue if args.stage == 'prefetch' else watch)(config)
    elif args.stage in {'expand', 'launch-expansion', 'restart-expansion', 'expansion-plan', 'prepare-expansion', 'audit-variants'}:
        from ..global_boost.expansion import run, launch, restart, projected_batch, prepare_batch, audit_variants
        if args.stage == 'prepare-expansion':
            value = prepare_batch(config, projected_batch(config))
            result = {'batch': str(value['paths']['data_boost_global_output']) if value else None}
        elif args.stage == 'audit-variants':
            result = {k: v for k, v in audit_variants(config).items() if k != 'per_source'}
        else:
            result = {'expand': run, 'launch-expansion': launch, 'restart-expansion': restart, 'expansion-plan': projected_batch}[args.stage](config)
    elif args.stage in {'finalize', 'launch-finalizer', 'restart-finalizer', 'cpu-ready', 'gpu-finalize'}:
        from ..global_boost.aggregate import finalize, launch, cpu_ready, gpu_finalize
        result = (launch(config, restart=True) if args.stage == 'restart-finalizer' else
                  {'finalize': finalize, 'launch-finalizer': launch,
                   'cpu-ready': cpu_ready, 'gpu-finalize': gpu_finalize}[args.stage](config))
        if args.stage == 'gpu-finalize':
            result = {k: v for k, v in result.items() if k != 'selected'}
    else:
        from ..global_boost.service import launch, launch_prefetch, restart, run
        result = (launch_prefetch(config, restart=True) if args.stage == 'restart-prefetch' else
            {'launch': launch, 'restart': restart, 'continue': run, 'launch-prefetch': launch_prefetch}[args.stage](config))
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
