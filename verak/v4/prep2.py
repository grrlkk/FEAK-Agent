"""Versioned CPU/API-only PREP2 entry point. No import may initialize CUDA."""
import argparse
from .common import constrain_cpu


def main():
    constrain_cpu()
    p = argparse.ArgumentParser()
    p.add_argument('command',choices=('sources','maps','content','scorer-labels','report'))
    args = p.parse_args()
    if args.command == 'sources':
        from .prep2_data import freeze,materialize
        freeze()
        materialize('content100')
    elif args.command == 'maps':
        from .prep2_maps import run
        run()
    elif args.command == 'content':
        from .prep2_content import run
        run()
    elif args.command == 'scorer-labels':
        from .prep2_scorer_audit import run
        run()
    else:
        from .prep2_report import report
        report()


if __name__ == '__main__':
    main()
