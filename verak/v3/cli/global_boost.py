"""CPU/API-only weak-GLOBAL teacher expansion, separate $12 cap, no training."""
import argparse
import json

from ..global_boost.config import config_for, constrain_cpu


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('stage', choices=['prepare', 'qc', 'teacher', 'measure', 'report', 'continue', 'launch'])
    parser.add_argument('--max-api-calls', type=int, default=20000)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    constrain_cpu()
    config = config_for()
    if args.stage == 'prepare':
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
    else:
        from ..global_boost.service import launch, run
        result = (launch if args.stage == 'launch' else run)(config)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
