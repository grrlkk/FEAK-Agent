"""Post-hoc source-sentence change on real essays; no policy or scorer calls."""
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
import json
import os

import numpy as np

from ..common import file_sha, read_json, sha_text, write_json
from ..corrupt.document import Document, source_document
from ..env.analysis import ParagraphAnalyzer
from ..phase2 import restore_profile
from ..reward.overedit import overedit
from ..train.pilot import safe_id
from .data import PHASE, prepare


class _LayoutOnlyTokens:
    """ParagraphAnalyzer.refresh fills all token fields before any measurement."""
    def tokens(self, text):
        return []

    def seed(self, document):
        pass


def _paragraph_texts(layout):
    return [''.join(u['leading'] + u['text'] for u in p['units']) for p in layout['paragraphs'] if p['units']]


class CachedParagraphs(ParagraphAnalyzer):
    """Reuse exact saved analyzer inputs; neighbors are cache keys, not API inputs."""
    def __init__(self, config, roots, texts, *, allow_local_bareun=False):
        output = config['paths'][PHASE + '_output'] / 'real_overedit_cache/bareun_paragraphs'
        super().__init__(config, cache_dir=output)
        self.allow_local_bareun = allow_local_bareun
        self.by_text, self.used = defaultdict(list), []
        self.sources = [output] + list(dict.fromkeys(Path(p) for p in roots))
        for root in self.sources:
            for path in sorted(root.glob('*.json')):
                try:
                    value = read_json(path)
                except json.JSONDecodeError:
                    # A concurrently running editor may still be publishing a cache file.
                    continue
                if value['text'] in texts:
                    self.by_text[value['text']].append(path)

    def _load(self, path, text, kind):
        value = read_json(path)
        if value['text'] != text:
            raise ValueError('Saved paragraph input mismatch')
        self.used.append({'path': str(path), 'sha256': file_sha(path), 'text_sha256': sha_text(text),
                          'reuse': kind, 'analyzer_version': value['profile']['analyzer_version']})
        self.hits += 1
        return restore_profile(value['profile'])

    def profile(self, text, neighbors=()):
        key = sha_text(json.dumps([text, sha_text(json.dumps(neighbors, ensure_ascii=False))], ensure_ascii=False))
        for root in self.sources:
            path = root / (key + '.json')
            if path.exists():
                return self._load(path, text, 'exact_text_and_neighbor_cache_key')
        candidates = self.by_text.get(text, [])
        if candidates:
            # ParagraphAnalyzer passes only text to Bareun; neighbors do not enter
            # the analyzer call. Refuse ambiguous cached tokenizations nonetheless.
            tokens = set()
            for path in candidates:
                profile = read_json(path)['profile']
                tokens.add(json.dumps([profile['analyzer_version'],
                    [t for s in profile['sentences'] for t in s['tokens']]], sort_keys=True, ensure_ascii=False))
            if len(tokens) != 1:
                raise ValueError('Conflicting cached tokenizations of the same paragraph text')
            return self._load(candidates[0], text, 'identical_analyzer_text_different_neighbor_cache_key')
        if not self.allow_local_bareun:
            raise ValueError('Missing saved paragraph tokens; local Bareun fallback not enabled')
        if self.config['bareun'].get('host', '127.0.0.1') not in {'127.0.0.1', 'localhost', '::1'}:
            raise ValueError('Post-hoc preservation permits only local Bareun fallback')
        key_env = self.config['bareun'].get('api_key_env', 'BAREUN_API_KEY')
        if not os.environ.get(key_env):
            from dotenv import dotenv_values
            value = dotenv_values(self.config['paths']['scorer_package'] / '.env').get(key_env)
            if value:
                os.environ[key_env] = value
        profile = super().profile(text, neighbors)
        path = self.cache_dir / (key + '.json')
        self.by_text[text].append(path)
        self.used.append({'path': str(path), 'sha256': file_sha(path), 'text_sha256': sha_text(text),
                          'reuse': 'local_bareun_fallback', 'analyzer_version': profile.analyzer_version})
        return profile


def restore_final(layout, text, analysis):
    document = Document.restore(layout, _LayoutOnlyTokens())
    if document.text != text:
        raise ValueError('Final text does not exactly round-trip from its saved layout')
    if len({u.sid for u in document.units}) != len(document.units):
        raise ValueError('Duplicate stable sentence IDs in final layout')
    analysis.refresh(document, {p.pid for p in document.paragraphs})
    if document.snapshot() != layout:
        raise ValueError('Paragraph refresh changed saved text, layout, IDs, or analyzer version')
    return document


def paired_change(rows, *, completed_only=True):
    by_setting = {setting: {r['episode_id']: r for r in rows if r['setting'] == setting and
        r['status'] == 'available' and (r['completed'] or not completed_only)} for setting in ('agentic', 'baseline')}
    ids = sorted(by_setting['agentic'].keys() & by_setting['baseline'].keys())
    if not ids:
        return {'n': 0, 'ids': [], 'new': None, 'baseline': None, 'delta': None, 'ci95': None}
    left = [by_setting['agentic'][i]['R_over'] for i in ids]
    right = [by_setting['baseline'][i]['R_over'] for i in ids]
    differences = np.asarray(left) - np.asarray(right)
    boot = np.random.default_rng(89).choice(differences, size=(10000, len(ids)), replace=True).mean(axis=1)
    return {'n': len(ids), 'ids': ids, 'new': mean(left), 'baseline': mean(right), 'delta': float(differences.mean()),
            'ci95': np.quantile(boot, [.025, .975]).tolist(),
            'method': 'paired episode bootstrap, seed 89, 10000 resamples; no question clustering correction'}


def real_overedit(config, *, allow_local_bareun=False, reference_count=3):
    """Persist comparable real-essay distances; never manufacture recovery rewards."""
    if config[PHASE].get('version', 2) != 3:
        raise ValueError('This post-hoc real-essay diagnostic belongs to agentic pilot v3')
    design, train, _, examples = prepare(config)
    root = config['paths'][PHASE + '_output']
    tasks, references, texts = [], [], set()
    for eid in design['real_ids']:
        for setting, path in [('baseline', Path(design['baseline_files'][eid]['path'])),
                              ('agentic', root / 'episodes' / (safe_id(eid) + '.json'))]:
            row = read_json(path) if path.exists() else None
            tasks.append((eid, setting, path, row))
            if row and row.get('final_layout'):
                texts.update(_paragraph_texts(row['final_layout']))
    # Compare reconstruction against existing corruption rewards where a gold
    # reward is actually available. This validation never scores or regenerates.
    for setting in ('baseline', 'agentic'):
        candidates = []
        for eid in design['corrupted_ids']:
            path = Path(design['baseline_files'][eid]['path']) if setting == 'baseline' else root / 'episodes' / (safe_id(eid) + '.json')
            row = read_json(path) if path.exists() else None
            if row and row.get('completed') and row.get('reward'):
                candidates.append((eid, setting, path, row))
        candidates.sort(key=lambda item: (-item[3]['reward']['combined']['R_over'], item[0]))
        for item in candidates[:reference_count]:
            references.append(item)
            texts.update(_paragraph_texts(item[3]['final_layout']))
    roots = [root / 'bareun_paragraphs', config['paths']['repo'] / 'verak/v3/outputs/observation_test/bareun_paragraphs',
             config['paths']['phase7_teacher_output'] / 'bareun_paragraphs']
    analysis = CachedParagraphs(config, roots, texts, allow_local_bareun=allow_local_bareun)
    method = {'definition': 'overedit(source, source, final, records=[])', 'formula': '0.5*morpheme_distance + 0.5*order_distance',
              'meaning': 'amount of change on all source sentences; no claim that changes were unnecessary or incorrect',
              'eligibility': 'all original source sentence IDs; no corruption exclusions and no spelling-span exclusions',
              'insertions': 'the inherited definition does not directly charge newly inserted sentence IDs',
              'restoration': 'frozen Phase-2 full-source profile; saved final IDs/layout; existing ParagraphAnalyzer.refresh for every final paragraph',
              'helper_sha256': file_sha(__file__),
              'overedit_sha256': file_sha(Path(__file__).parents[1] / 'reward/overedit.py'),
              'paragraph_analysis_sha256': file_sha(Path(__file__).parents[1] / 'env/analysis.py')}
    results, checks = [], []
    for eid, setting, path, row in references + tasks:
        reference = eid in train
        result = {'episode_id': eid, 'setting': setting, 'completed': bool(row and row.get('completed')), 'status': 'unavailable'}
        try:
            if row is None or not row.get('final_layout'):
                raise ValueError('Saved episode/final layout unavailable')
            source_id = train[eid]['source_id'] if reference else eid
            source = source_document(config, examples[source_id], _LayoutOnlyTokens())
            source_path = config['paths']['phase2_output'] / 'bareun_profiles' / f'{examples[source_id].source_line}.json'
            provenance = {'episode_path': str(path), 'episode_sha256': file_sha(path),
                          'source_profile_path': str(source_path), 'source_profile_sha256': file_sha(source_path),
                          'source_text_sha256': sha_text(source.text), 'final_text_sha256': sha_text(row['final_text'])}
            if setting == 'baseline' and file_sha(path) != design['baseline_files'][eid]['sha256']:
                raise ValueError('Saved two-stage baseline changed')
            if not reference:
                graph_path = root / 'graphs' / (safe_id(eid) + '.json')
                graph = read_json(graph_path)
                baseline_path = Path(design['baseline_files'][eid]['path'])
                baseline = read_json(baseline_path)
                if source.snapshot() != graph['input_layout'] or source.snapshot() != baseline['initial_layout']:
                    raise ValueError('Source, graph input and two-stage initial layouts differ')
                if source.text != examples[eid].text or sha_text(source.text) != graph['input_hash']:
                    raise ValueError('Real source text does not match frozen graph input')
                provenance.update(graph_path=str(graph_path), graph_sha256=file_sha(graph_path),
                                  baseline_path=str(baseline_path), baseline_sha256=file_sha(baseline_path),
                                  identical_source_graph_baseline_layout=True)
            first_profile = len(analysis.used)
            final = restore_final(row['final_layout'], row['final_text'], analysis)
            records = train[eid]['records'] if reference else []
            spans = train[eid].get('preexisting_spell_spans', ()) if reference else ()
            value = overedit(source, source, final, records, preexisting_spell_spans=spans)
            result.update(status='available', R_over=value['value'], morpheme=value['morpheme'], order=value['order'],
                          detail=value, provenance=provenance, paragraph_profiles=analysis.used[first_profile:])
            if reference:
                expected = row['reward']['combined']['R_over']
                result.update(saved_R_over=expected, difference=value['value'] - expected,
                              passed=abs(value['value'] - expected) < 1e-7)
        except (ValueError, KeyError, OSError, RuntimeError) as exc:
            result['error'] = {'type': type(exc).__name__, 'message': str(exc)}
            if reference:
                result['passed'] = False
        (checks if reference else results).append(result)
        write_json(root / 'real_overedit_cases' / (('reference_' if reference else '') + setting + '_' + safe_id(eid) + '.json'), result)
    summaries = {}
    for setting in ('agentic', 'baseline'):
        selected = [r for r in results if r['setting'] == setting]
        available = [r for r in selected if r['status'] == 'available']
        completed = [r for r in available if r['completed']]
        summaries[setting] = {'intended': 30, 'available': len(available), 'completed_available': len(completed),
                              'unavailable': [r['episode_id'] for r in selected if r['status'] != 'available'],
                              'mean_all_saved_finals': mean(r['R_over'] for r in available) if available else None,
                              'mean_completed': mean(r['R_over'] for r in completed) if completed else None,
                              'morpheme_completed': mean(r['morpheme'] for r in completed) if completed else None,
                              'order_completed': mean(r['order'] for r in completed) if completed else None}
    result = {'method': method, 'summary': summaries, 'paired_completed': paired_change(results),
              'paired_all_saved_finals': paired_change(results, completed_only=False), 'essays': results,
              'reconstruction_references': checks, 'reference_checks_passed': bool(checks) and all(r['passed'] for r in checks),
              'model_api_calls': 0, 'kanana_calls': 0, 'local_bareun_profile_calls': len(analysis.calls),
              'paragraph_cache_hits': analysis.hits, 'paragraph_reuse': dict(Counter(r['reuse'] for r in analysis.used)),
              'combined_real_reward': None}
    write_json(root / 'real_overedit.json', result)
    if any(r['status'] == 'available' and not r['passed'] for r in checks):
        raise ValueError('Post-hoc reconstruction differs from a saved corruption R_over; inspect real_overedit.json')
    return result
