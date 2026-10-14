"""Evaluate a completed GLOBAL stage even when the later teacher call failed."""
from pathlib import Path

from ..common import read_json, write_json
from ..view_data import load_episode_examples
from ..corrupt.document import Document
from ..eval.resources import Resources
from ..reward.total import rewards
from .teacher_comparison import PHASE, prepare


def score_completed_global(config):
    design, corpus = prepare(config)
    root = config['paths'][PHASE+'_output']
    paths = [p for e in design['efforts'] for p in (root/('luna_'+e)/'episodes').glob('*.json')]
    rows = [read_json(p) for p in paths]
    targets = [r for r in rows if not r['completed'] and r.get('stage1_layout') and 'global' in r['termination']]
    # Establish equality with saved full-episode rewards before evaluating failures.
    references = [r for r in rows if r['completed']][:5]
    examples = {e.id: e for e in load_episode_examples(config, 'agent_train')}
    resources = Resources(config, examples, output_key=PHASE+'_output')
    results, checks = {}, []
    try:
        for row in references+targets:
            key = row['comparison_condition']+':'+row['corpus_episode_id']
            path = root/'completed_global_rewards'/(key.replace(':', '_')+'.json')
            if path.exists():
                result = read_json(path)
            else:
                data = corpus[row['corpus_episode_id']]
                state = resources.worker()
                source = resources.source(data['source_id'])
                corrupted = resources.corrupted(data)
                stage1 = Document.restore(row['stage1_layout'], state.bank)
                for doc in (corrupted, stage1):
                    state.analysis.refresh(doc, {p.pid for p in doc.paragraphs})
                q = resources.score(data['question'], stage1.text)
                reward = rewards(source, corrupted, stage1, data['records'], genre=data['genre'],
                    q_corrupted=data['q_corrupted'], q_final=q['mean'], q_stage1=q['mean'],
                    config=config['reward'], mode='two_stage', stage1=stage1,
                    stage1_actions=row['actions_by_role']['global'], stage2_actions=[],
                    similarity=resources.similarity, tau=config['similarity']['tau'],
                    preexisting_spell_spans=data.get('preexisting_spell_spans', ()))['global']
                result = {'key': key, 'global_only_reward': reward, 'stage1_score': q,
                    'reason': 'completed GLOBAL stage; no replacement generation', 'gpt_calls': 0}
                write_json(path, result)
            if row['completed']:
                differences = {k: result['global_only_reward'][k]-row['reward']['global'][k]
                    for k in ('R', 'R_rec', 'R_q', 'R_over', 'R_step')}
                checks.append({'key': key, 'differences': differences})
                if any(abs(v) > 1e-7 for v in differences.values()):
                    raise ValueError('GLOBAL post-hoc evaluation differs from the frozen runtime reward')
            else:
                results[key] = result
    finally:
        resources.close()
    write_json(root/'completed_global_evaluation.json', {'gpt_calls': 0, 'completed_stages': len(results),
        'reference_checks': checks, 'results': results})
    print({'completed_global_stages_evaluated': len(results), 'reference_checks': len(checks), 'gpt_calls': 0}, flush=True)
