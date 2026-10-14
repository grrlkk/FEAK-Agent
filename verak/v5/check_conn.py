"""CPU-only L_CONN audit report from saved, explicit linguistic annotations."""
from collections import Counter
import time

from .common import atomic_new, read_json, file_sha
from .check_judge import ROOT, REPORT


def summarize(records):
    counts = Counter(r['label'] for r in records)
    certain = counts['ill_formed_morphotactics'] + counts['ill_formed_predicate_selection']
    uncertain = counts['uncertain_predicate_selection']
    return {'records': len(records), 'source_essays': len({r['source_id'] for r in records}),
        'unique_corrupted_forms': len({r['corrupted'] for r in records}), 'labels': dict(counts),
        'clearly_ungrammatical': certain, 'uncertain': uncertain,
        'clear_share': certain/len(records), 'clear_plus_uncertain_share': (certain+uncertain)/len(records),
        'conjunction_misclassified_as_EC': sum(r.get('conjunction_misclassified_as_EC', False) for r in records)}


def report():
    if (ROOT/'step4/complete.json').exists():
        return
    started = time.monotonic()
    source = ROOT/'step4/audit_input.json'; annotations = read_json(ROOT/'step4/annotations.json')
    if annotations['input_sha256'] != file_sha(source):
        raise ValueError('Audited source changed')
    original = read_json(source)['records']; labels = annotations['records']
    if {r['record_id'] for r in original} != set(labels):
        raise ValueError('Audit has missing or extra labels')
    rows = [dict(r, **labels[r['record_id']]) for r in original]
    stats = {'active_agent_dev': summarize(rows), 'dev100': summarize([r for r in rows if r['dev100']])}
    atomic_new(ROOT/'step4/metrics.json', stats)
    full, dev = stats['active_agent_dev'], stats['dev100']
    lines = ['## 4. CPU L_CONN form audit', '',
        f'All {full["records"]} L_CONN records in the active agent_dev corpus were inspected, covering '
        f'{full["source_essays"]} source essays. The frozen dev100 contains {dev["records"]} L_CONN records. '
        'The full active development split supplies 20 distinct examples; '
        'no test split is used. This is a saved-text Codex linguistic audit, not an independent human annotation or API judgment. '
        'A grammatical ending with a wrong discourse relation is not counted as an ungrammatical form. '
        'Only clear allomorph/past-tense attachment and predicate-selection violations count in the main numerator; '
        f'{full["uncertain"]} cases remain uncertain. Existing source typos and factual errors do not count.', '',
        '| Scope | Clearly ill-formed / records | Share | Uncertain | Clear + uncertain upper bound |',
        '|---|---:|---:|---:|---:|']
    for scope, s in stats.items():
        lines.append(f'| {scope} | {s["clearly_ungrammatical"]}/{s["records"]} | {s["clear_share"]:.1%} | '
            f'{s["uncertain"]} | {s["clear_plus_uncertain_share"]:.1%} |')
    lines += ['', 'The upper bound is a sensitivity analysis, not a second definite estimate. '
        'Records, rather than unique sources or strings, are the primary denominator to match recovery evaluation. '
        'Repeated source variants are retained and source/unique-form counts are saved in `step4/metrics.json`.', '',
        f'Conjunction substitutions marked as EC: {full["conjunction_misclassified_as_EC"]}. '
        'These are not automatically counted as morphologically ill-formed. '
        'The audit does not change any existing record, reward, environment or training data.', '',
        '| Example | Record | Original → corrupted sentence | Audit label | Rationale |',
        '|---|---|---|---|---|']
    cell = lambda s: str(s).replace('|', '\\|').replace('\n', ' ')
    selected = annotations['example_indices']
    if len(selected) != 20 or len(set(selected)) != 20:
        raise ValueError('Expected 20 distinct audit examples')
    for i in selected:
        r = next(r for r in rows if r['index'] == i)
        lines.append('| '+' | '.join(map(cell, [i, r['record_id'], r['original']+' → '+r['corrupted'], r['label'], r['rationale']]))+' |')
    lines += ['', 'Examples are purposefully selected to show clear form errors, uncertainty and grammatical relation changes; '
        f'they are not a random prevalence sample. Every one of the {len(rows)} labels and its rationale is retained.', '',
        f'Wall-clock for the CPU audit (preparation + inspection + annotation): {annotations["audit_wall_seconds"]:.1f} s; '
        f'report rendering {time.monotonic()-started:.1f} s. API calls: 0; GPU calls: 0.', '']
    with REPORT.open('a') as stream:
        stream.write('\n'+'\n'.join(lines))
    atomic_new(ROOT/'step4/complete.json', {'wall_seconds': annotations['audit_wall_seconds'],
        'report_seconds': time.monotonic()-started, 'new_API_calls': 0, 'GPU_calls': 0})


if __name__ == '__main__':
    report()
