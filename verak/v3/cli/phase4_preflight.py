"""Read-only model availability check under the shared Phase 4 API ledger."""
import argparse
import json
import os
import re
import time
from pathlib import Path

from verak.v3.api import PhaseBudget
from verak.v3.common import load_config, read_json, write_json


def select_luna(ids):
    snapshots = [m for m in ids if re.fullmatch(r'gpt-6-luna-\d{4}-\d{2}-\d{2}', m)]
    return (max(snapshots) if snapshots else ('gpt-6-luna' if 'gpt-6-luna' in ids else None)),snapshots


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-api-calls', type=int, required=True)
    args = parser.parse_args()
    config = load_config()
    out = config['paths']['repo'] / 'verak/v3/outputs/phase4'
    out.mkdir(parents=True, exist_ok=True)
    result_path = out / 'models.json'
    if result_path.exists():
        result = read_json(result_path)
    else:
        from openai import OpenAI
        client = OpenAI(api_key=os.environ['OPENAI_API_KEY'],
                        base_url='https://api.openai.com/v1', timeout=600, max_retries=0)
        budget = PhaseBudget(out / 'api_budget.json', args.max_api_calls,
                             authorized_ceiling=650, phase='v3_phase4')
        number = budget.reserve()
        def log(row):
            with (out / 'calls.jsonl').open('a', encoding='utf-8') as f:
                f.write(json.dumps({'phase_call': number, 'stage': 'list_models', **row}) + '\n')
                f.flush()
                os.fsync(f.fileno())
        log({'event': 'reserved_before_request'})
        try:
            models = client.models.list().data
            ids = sorted(m.id for m in models)
            exact,snapshots = select_luna(ids)
            result = {'checked_at': time.time(), 'ids': ids, 'luna_model': exact,
                      'dated_snapshots': snapshots, 'generation_calls': 0}
            write_json(result_path, result)
            log({'status': 'completed', 'usage': None, 'confirmed_usd': 0})
        except Exception as error:
            log({'status': 'error', 'error_type': type(error).__name__, 'usage': None})
            raise
        finally:
            client.close()
    print(json.dumps({'luna_model': result['luna_model'],
                      'dated_snapshots': result['dated_snapshots'],
                      'flagships': [m for m in result['ids'] if m.startswith('gpt-6')]}, ensure_ascii=False))
    if not result['luna_model']:
        raise SystemExit('GPT-6 Luna unavailable; stop Phase 4')
    from verak.v3.common import DEFAULT_CONFIG
    current=DEFAULT_CONFIG.read_text()
    updated,count=re.subn(r'(cheap_model:\n  model:)[^\n]*',
                         r'\1 '+result['luna_model'],current)
    if count!=1:
        raise ValueError('Cannot locate unique cheap_model config')
    DEFAULT_CONFIG.write_text(updated)


if __name__ == '__main__':
    main()
