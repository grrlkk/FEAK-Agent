"""Rebuild and audit GLOBAL rewards from saved GPU scores and Bareun profiles only."""
from collections import Counter
from dataclasses import replace
import copy
import json
import math
from pathlib import Path
from types import SimpleNamespace

from ..common import file_sha, pair_key, read_json, sha_text, write_json
from ..corrupt.document import Document, source_document
from ..env.analysis import ParagraphAnalyzer
from ..phase2 import restore_profile
from ..reward.total import rewards
from ..train.teacher_bulk import atomic_new
from ..view_data import load_episode_examples
from .config import PHASE
from .expansion import root_for
from .resources import score_gpu_reference

ABS_TOLERANCE = 1e-12


def differences(expected, actual, path='reward'):
    """Compare every term/detail; integers, booleans, strings and shapes stay exact."""
    if isinstance(expected, dict) and isinstance(actual, dict):
        if expected.keys() != actual.keys():
            return [{'path': path, 'reason': 'keys_differ'}]
        return [d for key in expected for d in differences(expected[key], actual[key], path+'.'+str(key))]
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            return [{'path': path, 'reason': 'length_differs'}]
        return [d for i, (a, b) in enumerate(zip(expected, actual)) for d in differences(a, b, f'{path}[{i}]')]
    if type(expected) is float and type(actual) in (float, int):
        if math.isfinite(expected) and math.isfinite(actual) and abs(expected-actual) <= ABS_TOLERANCE:
            return []
    elif type(expected) is type(actual) and expected == actual:
        return []
    return [{'path': path, 'expected': expected, 'actual': actual}]


class SavedParagraphs(ParagraphAnalyzer):
    """Neighbors only identify the cache: Bareun.profile received text alone.

    Exact-context cache hits are preferred. A text-only reuse is permitted only
    when every saved analysis of that exact text in its cache tier agrees.
    No analyzer can be constructed or contacted by this class.
    """
    def __init__(self, config, directories):
        self.config, self.directories = config, tuple(directories)
        self.hits, self.calls = 0, []
        self.indexes, self.parsed, self.proofs = {}, {}, {}
        self.uses = Counter()

    def _index(self, directory):
        if directory not in self.indexes:
            index = {}
            for path in sorted(directory.glob('*.json')):
                value = read_json(path)
                profile = value['profile']
                signature = sha_text(json.dumps({'analyzer_version': profile.get('analyzer_version'),
                    'sentences': profile['sentences']}, sort_keys=True, ensure_ascii=False))
                index.setdefault(value['text'], []).append((path, signature))
            self.indexes[directory] = index
        return self.indexes[directory]

    def profile(self, text, neighbors=()):
        key = sha_text(json.dumps([text, sha_text(json.dumps(neighbors, ensure_ascii=False))], ensure_ascii=False))
        path = next((d/(key+'.json') for d in self.directories if (d/(key+'.json')).exists()), None)
        mode = 'exact_context'
        if path is None:
            mode = 'identical_analyzer_text'
            for directory in self.directories:
                choices = self._index(directory).get(text, [])
                if choices:
                    if len({signature for _, signature in choices}) != 1:
                        raise ValueError('Ambiguous saved Bareun profiles for exact paragraph text: '+sha_text(text))
                    path = choices[0][0]
                    break
        if path is None:
            raise FileNotFoundError('No saved Bareun paragraph profile: '+sha_text(text))
        name = str(path)
        if name not in self.parsed:
            value = read_json(path)
            if value['text'] != text:
                raise ValueError('Saved Bareun paragraph text differs')
            self.parsed[name] = restore_profile(value['profile'])
            self.proofs[name] = {'path': name, 'sha256': file_sha(path), 'text_sha256': sha_text(text)}
        self.uses[mode] += 1
        self.hits += 1
        return copy.deepcopy(self.parsed[name])


class SavedResources:
    def __init__(self, config):
        self.config = config
        root = root_for(config)
        outputs = config['paths']['repo']/'verak/v3/outputs'
        self.analysis = SavedParagraphs(config, [root/'bareun_paragraphs'] + [outputs/n/'bareun_paragraphs'
            for n in ('phase5','phase6','phase7_teacher','teacher_bulk_two_stage','phase7_sft','v2_ops','v2_ops_retry')])
        self.examples = {e.id: e for e in load_episode_examples(config, 'agent_train')}
        self.sources, self.documents, self.source_proofs = {}, {}, {}

    def source(self, row):
        sid = row['source_id']
        if sid not in self.sources:
            example = self.examples[sid]
            self.sources[sid] = source_document(self.config, example, SimpleNamespace(seed=lambda _: None))
            path = self.config['paths']['phase2_output']/'bareun_profiles'/f'{example.source_line}.json'
            self.source_proofs[str(path)] = {'path': str(path), 'sha256': file_sha(path), 'source_id': sid}
        source = self.sources[sid]
        if source.text != row['source_text']:
            raise ValueError('Frozen source text differs from its saved source profile')
        return source

    def restore(self, layout):
        key = sha_text(json.dumps(layout, sort_keys=True, ensure_ascii=False))
        if key not in self.documents:
            # These temporary empty token lists are replaced from the exact
            # paragraph analysis before any recovery/over-edit function runs.
            doc = Document.restore(layout, SimpleNamespace(tokens=lambda _: []))
            self.analysis.refresh(doc, {p.pid for p in doc.paragraphs})
            if doc.analyzer_version != layout['analyzer_version']:
                raise ValueError('Saved Bareun analyzer version differs from the teacher state')
            self.documents[key] = doc
        return self.documents[key]


def rebuild(config, candidate, raw, resources, fingerprint, complete_sha):
    result, scores = dict(raw), {}
    if raw.get('stage1_layout') is not None:
        source = resources.source(candidate)
        corrupted = resources.restore(candidate['corrupted_layout'])
        stage1 = resources.restore(raw['stage1_layout'])
        for name, doc in [('corrupted', corrupted), ('stage1', stage1)]:
            scores[name] = score_gpu_reference(config, candidate['question'], doc.text)
            if scores[name]['scorer_fingerprint'] != fingerprint:
                raise ValueError('Saved GPU score has a different reference fingerprint')
        result['global_only_reward'] = rewards(source, corrupted, stage1, candidate['records'],
            genre=candidate['genre'], q_corrupted=scores['corrupted']['mean'], q_stage1=scores['stage1']['mean'],
            q_final=scores['stage1']['mean'], config=config['reward'], mode='two_stage', stage1=stage1,
            stage1_actions=raw['actions_by_role']['global'], stage2_actions=[], similarity=None,
            tau=config['similarity']['tau'], preexisting_spell_spans=candidate['preexisting_spell_spans'])['global']
    result.update(reward_status='global_only_gpu_reference', cpu_scores=scores, quality_scores=scores,
        score_source='gpu_reference', unmeasured_roles=['korean','combined'], gpu_used=False,
        canonical_for_selection=True, scorer_approval_sha256=complete_sha)
    return result


def verify_all(config):
    """Repair absent files; preserve existing files and stop on any reward mismatch."""
    root = root_for(config)
    ready = read_json(root/'cpu_ready.json')
    manifest = read_json(ready['manifest_path'])
    if file_sha(ready['manifest_path']) != ready['manifest_sha256']:
        raise ValueError('Frozen teacher manifest changed')
    complete_path = root.parent/'gpu_rescore/global_complete.json'
    complete, complete_sha = read_json(complete_path), file_sha(complete_path)
    if complete['manifest_sha256'] != ready['manifest_sha256'] or complete['fingerprint'] != manifest['reference_gpu_fingerprint']:
        raise ValueError('Saved GPU pass does not match the frozen teachers')
    destination = root/'gpu_reward_recovery_validation.json'
    if destination.exists():
        old = read_json(destination)
        if old.get('status') == 'complete' and old['manifest_sha256'] == ready['manifest_sha256'] and old['gpu_complete_sha256'] == complete_sha:
            for proof in old['artifacts']:
                if file_sha(proof['path']) != proof['sha256']:
                    raise ValueError('Validated reward or saved input changed: '+proof['path'])
            return old
    if complete.get('errors'):
        raise ValueError('Saved GPU scoring has terminal errors; no CPU fallback is allowed')
    inventory_path = root/'gpu_reward_recovery_inventory.json'
    if not inventory_path.exists():
        inventory = []
        for item in manifest['episodes']:
            target = Path(item['batch'])/f"attempt_{item['attempt']}"/('gpu_reference_'+complete['fingerprint'][:12])/Path(item['raw_path']).name
            error_path = target.parent.parent/'measurement_errors'/target.name
            error = read_json(error_path) if not target.exists() and error_path.exists() else None
            inventory.append({'path':str(target),'existed':target.exists(),
                'sha256':file_sha(target) if target.exists() else None,'attempt':item['attempt'],
                'missing_error':error,'missing_error_path':str(error_path) if error else None})
        atomic_new(inventory_path,{'manifest_sha256':ready['manifest_sha256'],'gpu_complete_sha256':complete_sha,'files':inventory})
    inventory = read_json(inventory_path)
    if inventory['manifest_sha256'] != ready['manifest_sha256'] or inventory['gpu_complete_sha256'] != complete_sha:
        raise ValueError('Reward recovery inventory belongs to a different frozen run')
    originals = {i['path']:i for i in inventory['files']}
    artifacts, score_proofs = [], []
    for request in manifest['requests']:
        key = pair_key(request['question'], request['text'])
        path = Path(complete['responses_root'])/(key+'.json')
        value = read_json(path)
        if (key != request['key'] or value.get('execution_device') != 'gpu_reference' or value.get('error')
                or value['fingerprint'] != complete['fingerprint'] or value['result']['cache_key'] != key
                or not math.isfinite(value['result']['mean'])):
            raise ValueError('Invalid saved GPU request/result pair: '+key)
        score_proofs.append({'path': str(path), 'sha256': file_sha(path), 'key': key, 'mean': value['result']['mean']})
    if len(score_proofs) != complete['request_count']:
        raise ValueError('Not every GPU request has a validated saved result')
    resources = SavedResources(config)
    items, failures, missing_by_attempt, causes = [], [], Counter(), Counter()
    old_count = new_count = exact = 0
    for index, item in enumerate(manifest['episodes'], 1):
        target = Path(item['batch'])/f"attempt_{item['attempt']}"/('gpu_reference_'+complete['fingerprint'][:12])/Path(item['raw_path']).name
        existed = target.exists()
        original = originals[str(target)]
        old_count += original['existed']
        new_count += not original['existed']
        if not original['existed']:
            missing_by_attempt[str(item['attempt'])] += 1
            if original['missing_error']:
                error = original['missing_error']
                causes[error['type']+': '+error['message']] += 1
        try:
            if original['existed'] and file_sha(target) != original['sha256']:
                raise ValueError('Original GPU reward file changed during recovery')
            for name in ('raw','candidate'):
                if file_sha(item[name+'_path']) != item[name+'_sha256']:
                    raise ValueError('Frozen teacher/candidate changed')
            raw, candidate = read_json(item['raw_path']), read_json(item['candidate_path'])
            result = rebuild(config, candidate, raw, resources, complete['fingerprint'], complete_sha)
            result['raw_generation_sha256'] = item['raw_sha256']
            for state, score in result['quality_scores'].items():
                if not any(q['state'] == state and q['key'] == score['cache_key'] for q in item['quality_inputs']):
                    raise ValueError('Rebuilt scorer input differs from the frozen GPU manifest')
            if existed:
                previous = read_json(target)
                diff = differences(result.get('global_only_reward'), previous.get('global_only_reward'))
                diff += differences(result['quality_scores'], previous.get('quality_scores'), 'quality_scores')
                if previous.get('raw_generation_sha256') != item['raw_sha256'] or previous.get('score_source') != 'gpu_reference':
                    diff.append({'path':'provenance','reason':'raw hash or score source differs'})
                if diff:
                    raise ValueError('Existing saved reward differs: '+json.dumps(diff[:10],ensure_ascii=False))
                exact += original['existed'] and result.get('global_only_reward') == previous.get('global_only_reward')
            else:
                atomic_new(target, result)
            proof = {'path': str(target), 'sha256': file_sha(target)}
            artifacts.extend([proof, {'path':item['raw_path'],'sha256':item['raw_sha256']},
                              {'path':item['candidate_path'],'sha256':item['candidate_sha256']}])
            items.append({'episode_id':item['episode_id'],'attempt':item['attempt'],'previously_present':original['existed'],
                'reward_available':bool(result.get('global_only_reward')),'reward_path':str(target),'reward_sha256':proof['sha256']})
        except Exception as exc:
            failures.append({'episode_id':item['episode_id'],'attempt':item['attempt'],'path':str(target),
                'type':type(exc).__name__,'message':str(exc)})
        if index % 50 == 0:
            write_json(root/'gpu_reward_recovery_progress.json',{'checked':index,'total':len(manifest['episodes']),
                'existing':old_count,'missing':new_count,'failures':failures,'paid_calls':0,'gpu_calls':0,'bareun_calls':0})
            print(f'Saved-GPU reward verification {index}/{len(manifest["episodes"])}; failures={len(failures)}',flush=True)
    artifacts += score_proofs + list(resources.analysis.proofs.values()) + list(resources.source_proofs.values())
    artifacts.append({'path':str(inventory_path),'sha256':file_sha(inventory_path)})
    report = {'status':'complete' if not failures else 'blocked','manifest_sha256':ready['manifest_sha256'],
        'gpu_complete_sha256':complete_sha,'fingerprint':complete['fingerprint'],'episodes':len(manifest['episodes']),
        'previously_present_rewards':old_count,'missing_rewards_before':new_count,'repaired_rewards':sum(not x['previously_present'] for x in items),
        'missing_by_attempt':dict(missing_by_attempt),'missing_causes':dict(causes),'verified_reward_files':len(items),
        'existing_rewards_exactly_equal':exact,'numeric_absolute_tolerance':ABS_TOLERANCE,'numeric_relative_tolerance':0,
        'reward_differences':failures,'saved_GPU_request_count':len(score_proofs),'saved_GPU_scores_validated':len(score_proofs),
        'paragraph_profile_uses':dict(resources.analysis.uses),'paragraph_profiles':list(resources.analysis.proofs.values()),
        'source_profiles':list(resources.source_proofs.values()),'artifacts':artifacts,'episodes_verified':items,
        'recovery_implementation_sha256':file_sha(Path(__file__)),'v1_reward_sha256':{str(p):file_sha(p) for p in
            [Path(__file__).parents[1]/'reward'/n for n in ('total.py','recovery.py','overedit.py')]},
        'paragraph_cache_reuse':'Prefer exact context key. Neighbors enter the cache key only; saved identical-text profiles must agree. No analyzer was constructed.',
        'paid_calls':0,'scorer_calls':0,'gpu_calls':0,'bareun_calls':0,'CPU_audit_required_for_final_report':True,
        'CPU_audit_required_for_GPU_selection':False,'training':False}
    write_json(destination, report)
    if failures:
        raise RuntimeError(f'{len(failures)} saved-GPU reward rebuild/validation failures; see '+str(destination))
    return report
