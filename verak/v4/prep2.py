"""Versioned CPU/API-only PREP2 entry point. No import may initialize CUDA."""
import argparse
from .common import constrain_cpu


def main():
    constrain_cpu()
    p = argparse.ArgumentParser()
    p.add_argument('command',choices=('sources','maps','content','scorer-labels','report'))
    p.add_argument('--watch',action='store_true',help='File-only report refresh until C and D stop')
    args = p.parse_args()
    if args.watch and args.command != 'report':
        p.error('--watch is only valid for the file-only report command')
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
        import time
        from .prep2_report import report
        from .prep2_common import ROOT
        while True:
            report()
            if not args.watch or all((ROOT/c/'complete.json').exists() for c in ('C','D')):
                break
            time.sleep(30)


if __name__ == '__main__':
    main()
