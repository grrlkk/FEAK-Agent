"""Phase 6 decisions/baselines only; this CLI never trains a policy."""
import argparse

from ..common import load_config
from ..eval.api import Phase6API
from ..eval.filter import run_filter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['filter', 'prepare', 'run', 'summarize', 'validate', 'report'], required=True)
    parser.add_argument('--max-api-calls', type=int, required=True)
    parser.add_argument('--conditions', nargs='+', choices=['two_stage', 'single', 'check_once',
        'real', 'rewrite_sol', 'rewrite_kanana', 'local_link'])
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    if args.limit is not None and not 1 <= args.limit <= 100:
        parser.error('--limit must be between 1 and 100')
    config = load_config()
    api = Phase6API(config, args.max_api_calls)
    try:
        if args.stage == 'filter':
            if args.max_api_calls > 260:
                raise ValueError('Filter requires --max-api-calls <= 260')
            run_filter(config, api)
        elif args.stage == 'prepare':
            from ..eval.design import prepare
            design, _, _ = prepare(config)
            print({k: len(design[k]) for k in ('paired_ids', 'check_ids', 'real_ids', 'local_source_ids')})
        elif args.stage == 'run':
            from ..eval.experiments import execute
            execute(config, api, conditions=args.conditions, limit=args.limit)
        elif args.stage == 'summarize':
            from ..eval.summary import summarize_phase6
            summarize_phase6(config, api)
        elif args.stage == 'validate':
            from ..eval.validate import validate
            result = validate(config, api)
            print({'passed': result['passed'], 'checks': result['checks']})
        else:
            from ..eval.report import render_report
            print(render_report(config))
    finally:
        api.close()


if __name__ == '__main__':
    main()
