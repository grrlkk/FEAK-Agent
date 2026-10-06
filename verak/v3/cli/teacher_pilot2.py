"""Bounded second teacher pilot; no bulk generation or training entry point."""
import argparse

from ..common import load_config
from ..eval.api import Phase6API
from ..train import pilot2_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['build', 'qc', 'finalize', 'prepare', 'run', 'report', 'diagnostic'])
    parser.add_argument('--max-api-calls', type=int, required=True)
    args = parser.parse_args()
    config = load_config()
    if args.stage == 'build':
        pilot2_data.build(config)
    elif args.stage == 'finalize':
        pilot2_data.finalize(config)
    elif args.stage in {'qc', 'run'}:
        api = Phase6API(config, args.max_api_calls, phase='phase7_pilot2')
        try:
            if args.stage == 'qc':
                pilot2_data.judge(config, api)
            else:
                from ..train.pilot2 import execute
                execute(config, api)
        finally:
            api.close()
    else:
        from ..train import pilot2
        getattr(pilot2, args.stage)(config)


if __name__ == '__main__':
    main()
