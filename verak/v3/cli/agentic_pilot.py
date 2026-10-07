"""Run the authorized v2 pilot; no bulk generation or training entry point."""
import argparse
import json
import socket
import os
from dotenv import dotenv_values

from feak_tc.runtime.openai import load_api_environment
from ..eval.api import Phase6API
from ..common import write_json
from ..agentic.data import PHASE, config_for, question_audit, prepare


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['audit', 'graphs', 'corrupted', 'real', 'markers', 'relevance', 'report', 'prompts'])
    parser.add_argument('--max-api-calls', type=int, required=True)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    config = config_for('gpt-6.1-sol' if args.stage in {'markers', 'relevance'} else 'gpt-6-luna')
    root = config['paths'][PHASE + '_output']
    if args.stage == 'audit':
        print(json.dumps(question_audit(config), ensure_ascii=False))
        prepare(config)
        return
    if args.stage == 'prompts':
        from transformers import AutoTokenizer
        from ..agentic.experiment import prompt_check
        tokenizer = AutoTokenizer.from_pretrained(str(config['paths']['policy_base']), local_files_only=True)
        print(prompt_check(config, tokenizer))
        return
    if args.stage == 'report':
        from ..agentic.report import report
        report(config)
        return
    load_api_environment()
    # Match the established local launchers: pass keys into the process
    # environment, never into config, logs, prompts or command-line arguments.
    values = dotenv_values(config['paths']['scorer_package'] / '.env')
    for name in ('OPENAI_API_KEY', 'BAREUN_API_KEY'):
        if not os.environ.get(name) and values.get(name):
            os.environ[name] = values[name]
    if args.stage in {'graphs', 'corrupted', 'real'}:
        if not os.environ.get('BAREUN_API_KEY'):
            raise RuntimeError('BAREUN_API_KEY unavailable; refuse paid dispatch')
        from ..env.analysis import ParagraphAnalyzer
        analysis = ParagraphAnalyzer(config, cache_dir=root / 'bareun_preflight')
        analysis.profile('글의 내용을 확인한다.')
    api = Phase6API(config, args.max_api_calls, phase=PHASE)
    try:
        api.client()  # Load environment before Bareun initialization as well.
        socket.getaddrinfo('api.openai.com', 443)
        if args.stage == 'graphs':
            from ..agentic.experiment import graphs
            graphs(config, api, args.limit)
        elif args.stage in {'corrupted', 'real'}:
            from ..agentic.experiment import pilot
            pilot(config, api, args.stage, args.limit)
        else:
            from ..agentic.evaluation import markers, relevance
            (markers if args.stage == 'markers' else relevance)(config, api)
    finally:
        api.close()
        write_json(root / 'last_budget.json', api.accounting())


if __name__ == '__main__':
    main()
