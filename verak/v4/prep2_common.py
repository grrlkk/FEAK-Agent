"""PREP2 isolation, authorization and durable budgets. No scorer/model loading."""
from .common import (REPO, ROOT as PREP1, GENRES, read_json, write_json, file_sha,
    sha_text, safe_id, atomic_new, collection_lock, load_config, constrain_cpu)
from .paid import PrepAPI
from verak.v3.v2_ops.local import load_environment

ROOT = REPO / 'verak/v4/outputs/prep2'
CAPS = {'C': 15., 'D': 10.}
FACT = 'assistant rubric feedback and 1–9 scores are strong-LLM authored; only grader_1/grader_2 (1–5) are human annotations'


def contract():
    path = ROOT/'contract.json'
    value = {'version': 'v4_prep2', 'feedback_provenance': FACT,
        'method_sha256': file_sha(REPO/'imple/FEAK_AGENT_METHOD.md'),
        'prior_design_sha256': file_sha(PREP1/'design.json'),
        'prior_map_final_sha256': file_sha(PREP1/'C/final_complete.json'),
        'prior_content_design_sha256': file_sha(PREP1/'D/design.json'),
        'caps_usd': CAPS, 'gpu': False, 'training': False, 'scorer_calls': 0,
        'rft_and_extra_global_collection_untouched': True,
        'map_gate': 'both re-extractions valid for at least 39 of the fixed 43 invalid pilot sources',
        'map_rule': 'main/key endpoints on cross-paragraph sentence edges only if paragraph count > 6',
        'off_topic': 'advisory only; never sufficient for deletion',
        'D_attempts': 2, 'D_new_sources': 'exclude all prior B train300, including prior D100',
        'D_max_assigned_items': 4, 'D_delegation_items': [1, 2], 'D_delegation_steps': 6,
        'D_insert_cap': 'at most 2 successful INSERTs per attempt across delegations; UNDO does not replenish',
        'D_korean_steps': 14, 'D_korean_task': 'own located items plus whole-essay typo/spacing pass',
        'D_gate': 'no invention, meaning preserved, better, >=half assigned items fully addressed; no added repetition; natural',
        'provider_sampling_seed': None, 'independent_attempt_requests': True}
    if path.exists():
        if read_json(path) != value:
            raise ValueError('PREP2 contract or protected inputs changed')
    else:
        atomic_new(path, value)
    return value


def api_for(component, kind='luna'):
    if component not in CAPS or kind not in {'sol', 'luna'}:
        raise ValueError('Unauthorized PREP2 component/model')
    contract()
    config = load_config()
    load_environment(config)
    luna = read_json(config['paths']['phase4_output']/'models.json')['luna_model']
    phase = 'v4_prep2_'+component.lower()
    config[phase] = {'model': 'gpt-6.1-sol' if kind == 'sol' else luna,
        'max_cost_usd': CAPS[component], 'max_concurrent_requests': 4, 'phase_api_ceiling': 20000}
    config['paths'][phase+'_output'] = ROOT/component
    api = PrepAPI(config, 20000, phase=phase)
    api.allowed_models = {luna, 'gpt-6.1-sol'}
    return api


def accounting(component):
    """Read-only progress: never construct or settle a live API client."""
    import sqlite3
    path = ROOT/component/'api/ledger.sqlite'
    if not path.exists():
        return {'calls': 0, 'confirmed_usd': 0., 'reserved_usd': 0., 'pending': 0}
    with sqlite3.connect(f'file:{path}?mode=ro', uri=True, timeout=60) as db:
        rows = db.execute('SELECT status,reserved,confirmed FROM calls').fetchall()
    return {'calls': sum(r[0] != 'blocked_before_send' for r in rows),
        'confirmed_usd': sum(r[2] for r in rows), 'reserved_usd': sum(r[1] for r in rows),
        'pending': sum(r[0] == 'pending' for r in rows)}
