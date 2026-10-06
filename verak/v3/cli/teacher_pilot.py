"""Prepare/run/report only the approved 100-essay Phase 7 teacher pilot."""
import argparse

from ..common import load_config


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',required=True,choices=['prepare','run','summarize'])
    parser.add_argument('--max-api-calls',required=True,type=int)
    args=parser.parse_args()
    config=load_config()
    from ..train.pilot import prepare,execute
    if args.stage=='prepare':
        design,_,_=prepare(config)
        print({'total':design['count'],'pilot':design['pilot_counts'],'seed':design['seed']})
        return
    from ..eval.api import Phase6API
    api=Phase6API(config,args.max_api_calls,phase='phase7_pilot')
    try:
        if args.stage=='run':execute(config,api)
        else:
            from ..train.pilot_report import summarize
            print(summarize(config,api))
    finally:api.close()


if __name__=='__main__':main()
