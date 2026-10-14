"""Phase 7 diagnostics and bounded Luna comparison; no bulk or training commands."""
import argparse

from ..common import load_config
from ..eval.api import Phase6API
from ..train.teacher_comparison import PHASE, prepare, execute


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare', 'diagnose', 'run', 'evaluate', 'report'])
    parser.add_argument('--max-api-calls', type=int, required=True)
    args = parser.parse_args()
    config = load_config()
    if args.stage == 'prepare':
        prepare(config)
    elif args.stage == 'diagnose':
        from ..train.teacher_diagnostics import diagnose
        diagnose(config)
    elif args.stage == 'evaluate':
        from ..train.teacher_evaluation import score_completed_global
        score_completed_global(config)
    elif args.stage == 'report':
        from ..train.teacher_report import report
        report(config)
    else:
        api = Phase6API(config, args.max_api_calls, phase=PHASE)
        try:
            execute(config, api)
        finally:
            api.close()


if __name__ == '__main__':
    main()
