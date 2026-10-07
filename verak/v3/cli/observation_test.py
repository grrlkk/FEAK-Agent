"""Run only the explicitly authorized observation comparison, under one $8 cap."""
import argparse
import socket

from ..eval.api import Phase6API
from ..observation.experiment import config_for, prepare, graphs, run_agents, spot_check


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['prepare','graphs','agents','spot','markers','report'])
    parser.add_argument('--max-api-calls',type=int,required=True)
    parser.add_argument('--limit',type=int)
    args=parser.parse_args()
    config=config_for('gpt-6.1-sol' if args.stage in {'spot','markers'} else 'gpt-6-luna')
    if args.stage=='prepare':
        design,*_=prepare(config)
        print({k:len(design[k]) for k in ('corrupted_ids','real_ids','quality_corrupted_ids','stability_ids','spot_ids')})
        return
    if args.stage=='report':
        from ..observation.report import report
        report(config)
        return
    api=Phase6API(config,args.max_api_calls,phase='observation_test')
    try:
        api.client()
        socket.getaddrinfo('api.openai.com',443)
        if args.stage=='graphs':
            graphs(config,api,limit=args.limit)
        elif args.stage=='agents':
            run_agents(config,api,limit=args.limit)
        elif args.stage=='spot':
            spot_check(config,api)
        else:
            from ..observation.markers import run_checks
            run_checks(config,api)
    finally:
        api.close()


if __name__=='__main__':
    main()
