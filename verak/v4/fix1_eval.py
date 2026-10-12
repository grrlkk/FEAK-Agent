"""Local GPU diagnostic execution only; private records are used after generation."""
from copy import deepcopy
import json
import time

from .common import REPO, atomic_new, load_config, read_json, write_json
from .dryrun_common import tokenizer, local_analysis_environment
from .environment_v43 import BoundaryParagraphs
from .fix1_taxonomy import ROOT
from .full_data import ref, verified
from .fix1_infer import CONDITIONS, SETTINGS, LocalBackend, run_case, summarize
from . import orchestrator_v44 as orch


def evaluate():
    from .fix1_run import verify_preflight
    preflight = verify_preflight()
    orch.verify(); local_analysis_environment()
    config = load_config(); tok = tokenizer()
    config['scorer']['gpu'] = 0  # only physical GPU1 is exposed
    config['paths']['output'] = ROOT/'scorer'
    scorer = None
    def score(question, text):
        nonlocal scorer
        if scorer is None:
            from verak.v3.score.kanana import KananaScorer
            scorer = KananaScorer(config, cache_enabled=False)
            prior = read_json(REPO/'verak/v4/outputs/train44/smoke/scorer_identity.json')
            if scorer.fingerprint != prior['fingerprint']:
                raise ValueError('GPU reference scorer fingerprint changed')
            write_json(ROOT/'scorer_identity.json', {'physical_gpu': 1, 'fingerprint': scorer.fingerprint,
                'settings': config['scorer'], 'identical_to_V4_TRAIN': True, 'CPU_Q_scores_used': False})
        return scorer.score(question, text, use_cache=False).to_dict()
    manifest = verified(preflight['evaluation_manifest'])
    analysis = BoundaryParagraphs(config, cache_dir=ROOT/'bareun')
    collected = []
    try:
        for setting in SETTINGS:
            clients = {condition: LocalBackend(condition, setting) for condition in CONDITIONS}
            try:
                for name, cohort in manifest['cohorts'].items():
                    attempts = {condition: [] for condition in CONDITIONS}
                    for index, case in enumerate(cohort['cases']):
                        saved = verified(case['row'])
                        # Strict allowlist: no Sol plan, feedback or corruption
                        # record is loaded for policy generation.
                        row = {k: saved[k] for k in ('source_id', 'genre', 'question', 'text', 'profile',
                            'layout', 'normalization', 'paragraphs', 'environment_version') if k in saved}
                        order = ['old', 'new'] if (index+setting['run']) % 2 else ['new', 'old']
                        for condition in order:
                            output = ROOT/'evaluation'/setting['name']/f"run{setting['run']}"/condition/name
                            result = run_case(row, case_id=case['case_id'], output=output,
                                backend=clients[condition], tok=tok, analysis=analysis, score=score)
                            attempts[condition].append(result)
                            write_json(ROOT/'progress.json', {'setting': setting, 'condition': condition,
                                'cohort': name, 'case_id': case['case_id'], 'completed_this_cell': len(attempts[condition]),
                                'at': time.time()})
                    for condition in CONDITIONS:
                        output = ROOT/'evaluation'/setting['name']/f"run{setting['run']}"/condition/name
                        metrics = summarize(attempts[condition], len(cohort['cases']))
                        write_json(output/'metrics.json', metrics)
                        collected.append({'setting': deepcopy(setting), 'condition': condition,
                            'cohort': name, 'metrics': metrics, 'directory': str(output)})
            finally:
                for client in clients.values():
                    client.close()
        atomic_new(ROOT/'evaluation/complete.json', {'cells': collected, 'episodes': 240,
            'test_cohort_untouched': True, 'paid_API_calls': 0, 'prompt_versions_unchanged': True,
            'old_and_new_environment': 'v4.6', 'private_plans_or_records_at_inference': False,
            'settings_reported_separately': True})
    finally:
        if scorer is not None:
            scorer.close()


if __name__ == '__main__':
    evaluate()
