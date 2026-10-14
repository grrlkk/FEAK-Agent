"""Collect the authorized bulk teacher attempts; no SFT or RFT command."""
import argparse
import json

from ..common import write_json
from ..eval.resources import Resources
from ..view_data import load_episode_examples
from ..train.teacher_bulk import (PHASE, BulkAPI, collection_lock, config_for,
                                 execute, load_environment, preflight, prepare)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare', 'preflight', 'run', 'report'])
    parser.add_argument('--max-api-calls', required=True, type=int)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--independent-unseeded', action='store_true',
        help='Acknowledge that seeds71/72 order essays; provider requests are independently sampled without seed support')
    parser.add_argument('--awaiting-seed-clarification', action='store_true',
        help='Write a readiness-only report with zero new requests while sampling clarification is pending')
    args = parser.parse_args()
    config = config_for()
    root = config['paths'][PHASE + '_output']
    if args.stage == 'prepare':
        design, _ = prepare(config)
        print(json.dumps({k: design[k] for k in ('essays', 'requested_slots', 'new_slots', 'sampling_note')}, ensure_ascii=False))
        return
    if args.stage == 'report':
        from ..train.teacher_bulk_report import report
        report(config, awaiting_seed_clarification=args.awaiting_seed_clarification)
        return
    load_environment(config)
    with collection_lock(root):
        if args.stage == 'preflight':
            _, corpus = prepare(config)
            examples = {e.id: e for e in load_episode_examples(config, 'agent_train')}
            resources = Resources(config, examples, output_key=PHASE + '_output')
            try:
                print(json.dumps(preflight(config, resources, corpus), ensure_ascii=False))
            finally:
                resources.close()
            return
        if not args.independent_unseeded:
            parser.error('Responses does not support sampling seed; run requires --independent-unseeded acknowledgement')
        api = BulkAPI(config, args.max_api_calls, phase=PHASE)
        try:
            execute(config, api, limit=args.limit)
        finally:
            api.close()
            write_json(root / 'last_budget.json', api.accounting())


if __name__ == '__main__':
    main()
