"""Isolated PREP3 CPU/API tasks. Does not load a scorer or training code."""
import argparse
from .common import constrain_cpu


def main():
    constrain_cpu()
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=('maps','revalidate','sources','rejudge','content','report'))
    args=parser.parse_args()
    if args.command=='maps':
        from .prep3_maps import run
        run()
    elif args.command=='revalidate':
        from .prep3_maps import revalidate
        revalidate()
    elif args.command=='sources':
        from .prep3_data import freeze_content,materialize
        freeze_content(); materialize('content')
    elif args.command=='rejudge':
        from .prep3_rejudge import run
        run()
    elif args.command=='content':
        from .prep3_content import run
        run()
    else:
        from .prep3_report import report
        report()


if __name__=='__main__':
    main()
