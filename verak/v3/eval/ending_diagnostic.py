"""Offline Phase 6 tense/aspect and pronoun-NP transition audit, no profile changes."""
from collections import Counter
from dataclasses import replace
from difflib import SequenceMatcher
import re

from ..common import read_json, sha_text, write_json
from ..phase2 import read_jsonl, write_jsonl, restore_profile


def observed_sentences(observation):
    if '[글]\n' not in observation:
        return None
    body = observation.split('[글]\n', 1)[1].split('\n[한국어 관찰:', 1)[0]
    body = body.split('\n[Korean document profile:', 1)[0]
    return dict(re.findall(r'^([SN]\d+[a-z]*) \| (.*)$', body, flags=re.M))


class OfflineSentenceTokens:
    def __init__(self, root):
        self.tokens = {}
        # Paragraph analyses and individual EDIT analyses are both cached here.
        for path in sorted((root/'bareun_paragraphs').glob('*.json')):
            saved = read_json(path)
            for sentence in restore_profile(saved['profile']).sentences:
                tokens = [replace(t, start=t.start-sentence.start, end=t.end-sentence.start) for t in sentence.tokens]
                self.tokens.setdefault(sentence.text, tokens)

    def get(self, text):
        if text not in self.tokens:
            raise ValueError('Missing saved Bareun sentence: '+sha_text(text))
        return self.tokens[text]


def tense_signature(tokens):
    """All observed tense/aspect markers, with local surfaces as audit evidence.

    Past/future EP plus productive EC+VX aspect constructions. Pure -고 -> -며
    changes with the same auxiliary do not count. This is a marker diagnostic,
    not an assertion that every past-EP removal is a meaning error.
    """
    result = []
    for i, t in enumerate(tokens):
        if t.tag == 'EP' and t.form in {'았', '었', '였', '겠', '더'}:
            result.append(('PAST' if t.form in {'았', '었', '였'} else 'PROSPECTIVE' if t.form == '겠' else 'RETROSPECTIVE', t.form, t.start, t.end))
        if t.tag == 'EC' and i+1 < len(tokens):
            nxt = tokens[i+1]
            if nxt.tag == 'VX':
                cls = None
                if t.form == '고' and nxt.form in {'있', '계시'}:
                    cls = 'PROGRESSIVE'
                elif t.form in {'아', '어', '여'} and nxt.form in {'오', '가'}:
                    cls = 'CONTINUATIVE_'+nxt.form
                elif t.form in {'아', '어', '여'} and nxt.form in {'있', '계시'}:
                    cls = 'RESULTATIVE'
                elif t.form in {'아', '어', '여', '고'} and nxt.form in {'버리', '말'}:
                    cls = 'COMPLETIVE_'+nxt.form
                if cls:
                    result.append((cls, t.form+' '+nxt.form, t.start, nxt.end))
    return result


def changed_tense_markers(before, after, old, new):
    """Ignore tagger differences on an unchanged surface and modal -겠- alone."""
    diffs = [v for v in SequenceMatcher(None, before, after, autojunk=False).get_opcodes() if v[0] != 'equal']
    def touched(markers, side):
        result = []
        for item in markers:
            if item[0] == 'PROSPECTIVE':
                continue  # -겠- may be politeness/conjecture rather than tense.
            a, b = item[2:]
            if any(a < v[side+1] and b > v[side] for v in diffs):
                result.append(item)
        return result
    left, right = touched(old, 1), touched(new, 3)
    return [v[0] for v in left] != [v[0] for v in right], left, right


def pronoun_replacements(before, after, old_tokens, new_tokens):
    """NP(subject) substitutions at an aligned site, not arbitrary noun additions."""
    old_candidates = []
    for match in re.finditer(r'(?<![가-힣])(?:그들|그|이)(?:은|는|이|가)(?=\s|[,，])', before):
        if any(t.tag in {'JKS', 'JX'} and t.start < match.end() and t.end > match.start() for t in old_tokens):
            old_candidates.append(match)
    replacements = []
    opcodes = SequenceMatcher(None, before, after, autojunk=False).get_opcodes()
    for match in old_candidates:
        # Anchor right at the end of the subject's shared particle/space. At least
        # one diff must intersect its lexical pronoun rather than another clause.
        overlaps = [(tag, a, b, c, d) for tag, a, b, c, d in opcodes
                    if tag != 'equal' and (a < match.end()-1 and b > match.start() or a == b == match.start())]
        if not overlaps:
            continue
        # Locate the corresponding marker via equal suffix mapping or the replacement.
        candidates = []
        for tag, a, b, c, d in opcodes:
            if tag == 'equal' and a <= match.end()-1 < b:
                candidates.append(c + match.end()-1-a)
        for marker in new_tokens:
            if marker.tag not in {'JKS', 'JX'} or marker.form not in {'은', '는', '이', '가'}:
                continue
            if candidates and not any(marker.start <= p < marker.end for p in candidates):
                continue
            if not candidates and not any(c <= marker.start <= d+2 for _, a, b, c, d in overlaps):
                continue
            index = new_tokens.index(marker)
            start = marker.start
            for t in reversed(new_tokens[:index]):
                if t.tag in {'NNG', 'NNP', 'NNB', 'NP', 'XSN', 'MM', 'NR', 'SN'}:
                    start = t.start
                else:
                    break
            surface = after[start:marker.end]
            if surface and surface != match.group() and any(t.tag in {'NNG', 'NNP'} and start <= t.start < marker.start for t in new_tokens):
                replacements.append({'before': match.group(), 'after': surface,
                                     'before_span': list(match.span()), 'after_span': [start, marker.end]})
    return replacements


def run(config):
    root = config['paths']['phase6_output']
    output = config['paths']['phase7_pilot2_output']
    bank = OfflineSentenceTokens(root)
    transitions, missing, accepted_edits = [], [], 0
    for path in sorted((root/'real/episodes').glob('*.json')):
        episode = read_json(path)
        event_path = root/'real/events'/path.with_suffix('.jsonl').name
        events = read_jsonl(event_path)
        current = None
        for event in events:
            observation = event.get('observation')
            after = observed_sentences(observation) if observation else None
            if event['event'] == 'reset':
                current = after
                continue
            if event['event'] != 'action':
                continue
            action = event['action']
            if action['valid'] and action['action'] == 'EDIT':
                accepted_edits += 1
                for sid in current.keys() & (after or {}).keys():
                    left, right = current[sid], after[sid]
                    if left == right:
                        continue
                    try:
                        lt, rt = bank.get(left), bank.get(right)
                    except ValueError as exc:
                        missing.append({'essay_id': episode['source_id'], 'sid': sid, 'error': str(exc)})
                        continue
                    a, b = tense_signature(lt), tense_signature(rt)
                    # Compare classes, not past allomorph spelling or span shifts.
                    tense_changed, changed_before, changed_after = changed_tense_markers(left, right, a, b)
                    pronouns = pronoun_replacements(left, right, lt, rt)
                    if tense_changed or pronouns:
                        transitions.append({'essay_id': episode['source_id'], 'sid': sid,
                            'role': event['role'], 'step': action['t'], 'thought': action['thought'],
                            'before': left, 'after': right, 'tense_aspect_changed': tense_changed,
                            'tense_before': a, 'tense_after': b, 'pronoun_replacements': pronouns})
                        transitions[-1].update(changed_tense_before=changed_before, changed_tense_after=changed_after)
            if after is not None:
                current = after
    counts = {}
    for kind in ('tense_aspect', 'pronoun_np', 'either'):
        rows = [r for r in transitions if kind == 'either' or
                (r['tense_aspect_changed'] if kind == 'tense_aspect' else r['pronoun_replacements'])]
        counts[kind] = {'sentence_transitions': len(rows),
            'edit_actions': len({(r['essay_id'], r['role'], r['step']) for r in rows}),
            'unique_sentences': len({(r['essay_id'], r['sid']) for r in rows}),
            'essays': len({r['essay_id'] for r in rows})}
    result = {'label': 'Bareun marker diagnostic, not semantic/human accuracy',
        'essays': len(list((root/'real/episodes').glob('*.json'))),
        'accepted_EDIT_actions': accepted_edits, 'counts': counts,
        'definition': 'Changed surface past/retrospective EP or aspect EC+VX; exclude -겠- alone and tag changes on identical surface; accepted EDIT transitions only',
        'missing_cached_sentences': missing, 'new_API_calls': 0,
        'transitions': transitions}
    write_json(output/'ending_diagnostic.json', result)
    write_jsonl(output/'ending_transitions.jsonl', transitions)
    if missing:
        raise ValueError('Offline diagnostic missing cached sentences; inspect before reporting counts')
    return result
