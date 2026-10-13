"""Finalize V5_CHECK from completed artifacts without any model call."""
from collections import defaultdict
from datetime import datetime, timezone
import re

from .common import read_json, verified, atomic_new, write_json, ROOT as PIPE_ROOT
from .check_judge import ROOT, REPORT


def edit_type_links(part):
    cases = [verified(p) for p in read_json(ROOT/part/'manifest.json')['cases']]
    stats = read_json(ROOT/part/'metrics.json'); rows = []
    for source, summary in stats.items():
        labels = {r['case_id']: set(r['broken_sids']) for r in summary['cases']}
        types = sorted({a['kind'] for c in cases if c['source'] == source for a in c['actions']})
        for kind in types:
            counts = defaultdict(int)
            for case in [c for c in cases if c['source'] == source]:
                actions = [a for a in case['actions'] if a['kind'] == kind]
                sites = {s for a in actions for s in a['surviving_sites']}
                broken = labels[case['case_id']] & sites
                counts['actions'] += len(actions)
                counts['essays_with_edit_type'] += bool(actions)
                counts['essays_with_broken'] += bool(broken)
                counts['broken_links'] += len(broken)
            rows.append({'part': part, 'source': source, 'edit_type': kind,
                'cohort_essays': summary['essays'], **counts})
    return rows


def finalize():
    for path in ('validation/final.json', 'step1/complete.json', 'step2/complete.json',
                 'step3/inference_complete.json', 'step3/report_complete.json', 'step4/complete.json'):
        if not (ROOT/path).exists():
            raise RuntimeError('Required part unfinished: '+path)
    marker = ROOT/'complete.json'
    if marker.exists():
        return read_json(marker)
    gate = read_json(ROOT/'validation/final.json')
    cost = read_json(ROOT/'api/accounting.json')
    if cost['by_provider']['claude']['confirmed_usd']+cost['by_provider']['claude']['reserved_usd'] > 8:
        raise ValueError('Claude cap exceeded')
    if cost['by_provider']['luna']['confirmed_usd']+cost['by_provider']['luna']['reserved_usd'] > 3:
        raise ValueError('Luna cap exceeded')
    stats = read_json(ROOT/'step2/metrics.json'); combined = {}
    for condition in ('without_notices', 'with_notices'):
        values = [s for k, s in stats.items() if k.endswith('_'+condition)]
        combined[condition] = {k: sum(s[k] for s in values) for k in
            ('essays', 'protocol_complete', 'structural_actions', 'essays_with_broken_link', 'broken_links', 'unjudged_links', 'uncertain_links')}
    a4 = read_json(ROOT/'step3/metrics.json'); a = read_json(PIPE_ROOT/'A/metrics.json')
    policy_repeat = read_json(ROOT/'step2/determinism_audit.json')
    judge_repeat = read_json(ROOT/'step2/judge_repeat_audit.json')
    by_type = edit_type_links('step1')+edit_type_links('step2')
    atomic_new(ROOT/'edit_type_link_counts.json', by_type)
    full, local = (gate['tables'][c]['ALL'] for c in ('full', 'local'))
    observation = read_json(ROOT/'step1/metrics.json')
    old_without = observation['observation_real30_text_only']
    old_with = observation['observation_real30_current']
    conn = read_json(ROOT/'step4/metrics.json')
    dev_conn, all_conn = conn['dev100'], conn['active_agent_dev']
    without, with_ = combined['without_notices'], combined['with_notices']
    paired_description = (
        f'With minus without notices: {with_["essays_with_broken_link"]-without["essays_with_broken_link"]:+d} '
        f'essays with a broken link; {with_["broken_links"]-without["broken_links"]:+d} broken links; '
        f'{with_["protocol_complete"]-without["protocol_complete"]:+d} protocol-complete episodes. '
        'Interpret these differences alongside the recorded execution/judge variability; a single paired run is not a causal estimate.')
    lines = ['## Interpretation and comparison', '',
        f'Broken-record recall: full {full["positive_yes"]}/{full["positive_n"]}; local '
        f'{local["positive_yes"]}/{local["positive_n"]}. Selected context: {gate["validated_setting"]}. '
        f'Full-context false positives: {full["control_yes"]}/{full["control_judged"]} valid controls; '
        f'{full["control_unjudged"]} unjudged controls; bounds '
        f'{full["false_positive_bounds"][0]:.1%}–{full["false_positive_bounds"][1]:.1%}. '
        'This measures sensitivity on selected synthetic damage types; it does not establish a human gold standard '
        'for every natural discourse error.', '',
        '| Local frozen Revision condition | Protocol | Structural actions | Essays with broken structural link | Broken links |',
        '|---|---:|---:|---:|---:|']
    for name, s in combined.items():
        lines.append(f'| {name} | {s["protocol_complete"]}/{s["essays"]} | {s["structural_actions"]} | '
            f'{s["essays_with_broken_link"]}/{s["essays"]} | {s["broken_links"]} |')
    lines += ['', f'Temperature 0 did not make execution bit-identical: {len(policy_repeat["identical_prefix_divergences"])} paired '
        'requests produced different raw actions despite exactly identical saved messages. These divergences are listed in '
        '`step2/determinism_audit.json`. The unchanged Claude judge also gave '
        f'{judge_repeat["label_disagreements"]} different link labels among {judge_repeat["notices"]} notices in '
        f'{judge_repeat["identical_payload_pairs"]} pairs with identical judgment payloads. Those original labels are preserved; '
        'no selective re-judging was performed. '+paired_description, '',
        f'Re-judging the historical observation real30 gave {old_without["essays_with_broken_link"]}/{old_without["essays"]} '
        f'essays with a broken structural link without notices and {old_with["essays_with_broken_link"]}/{old_with["essays"]} '
        'with the current profile/notices; the original conditions differ in more than just next-step notices. '
        'These are final-state Claude link judgments at surviving structural-change sites. The historical 21.7% / 8–12% '
        'figures used a different judge/candidate definition and denominator, so the numbers should not be treated as a direct replication. '
        'In the corrupted dev essays, residual original damage can contribute to these link labels.', '',
        '| Operator | a1 full restoration | a2 oracle | a3 one-pass | a4 rules | a4 − a1 (pp) |',
        '|---|---:|---:|---:|---:|---:|']
    for op, s in a4['operators'].items():
        values = [a[k]['operators'][op] for k in ('a1', 'a2', 'a3')]
        cells = [f'{x["fully_restored"]}/{x["measurable_records"]}' for x in values]
        delta = (s['full_restoration']/s['records']-values[0]['fully_restored']/values[0]['measurable_records'])*100
        lines.append('| '+op+' | '+' | '.join(cells)+f' | {s["full_restoration"]}/{s["records"]} | {delta:+.1f} |')
    lines += ['', f'Overall a4 protocol completion is {a4["protocol_complete"]}/100; '
        f'a1 was {a["a1"]["protocol_complete"]}/100. Full per-operator over-edit, actions and protocol counts are in part 3. '
        'The candidate detector has no L_CONN rule, as specified; L_CONN repair remains dependent on the whole-essay pass. '
        f'CPU inspection found {dev_conn["clearly_ungrammatical"]}/{dev_conn["records"]} clearly ill-formed L_CONN forms '
        f'in dev100 ({dev_conn["uncertain"]} uncertain), and {all_conn["clearly_ungrammatical"]}/{all_conn["records"]} '
        f'in the active development corpus ({all_conn["uncertain"]} uncertain). Record restoration therefore mixes '
        'morphosyntactic repair and discourse-relation repair.', '',
        '## Broken links by structural edit type', '',
        'Each type counts distinct surviving broken sentence sites within an essay. An affected link can belong to '
        'several types, so rows are not additive. The denominator for links/essay is the full source cohort.', '',
        '| Part / source | Type | Actions | Essays with broken link / cohort | Broken links / essay |',
        '|---|---|---:|---:|---:|']
    for r in by_type:
        lines.append(f'| {r["part"]} / {r["source"]} | {r["edit_type"]} | {r["actions"]} | '
            f'{r["essays_with_broken"]}/{r["cohort_essays"]} | {r["broken_links"]/r["cohort_essays"]:.3f} |')
    wall = {
        '0: validation including format retry': gate['wall_seconds'],
        '1: saved-output judgments': read_json(ROOT/'step1/complete.json')['wall_seconds'],
        '2: GPU lifecycle': read_json(ROOT/'step2/server_stopped.json')['wall_seconds'],
        '2: Claude judgments': read_json(ROOT/'step2/complete.json')['wall_seconds'],
        '3: GPU lifecycle': read_json(ROOT/'step3/server_stopped.json')['wall_seconds'],
        '3: CPU measurement': read_json(ROOT/'step3/report_complete.json')['measurement_seconds'],
        '4: CPU linguistic audit': read_json(ROOT/'step4/complete.json')['wall_seconds']}
    # Exact phase-0 preparation/retry times are also reported at its start.
    lines += ['', '## Completion, cost and artifacts', '',
        '| Part | Measured wall-clock seconds |', '|---|---:|']
    for part, seconds in wall.items():
        lines.append(f'| {part} | {seconds:.1f} |')
    lines += ['', 'Parts overlap (a4 GPU execution ran alongside step-2 judging and CPU/report work), so these times '
        'must not be summed as elapsed user waiting time. Part 0 format-retry timing is separately disclosed above.', '',
        f'Final confirmed API usage: **Claude ${cost["by_provider"]["claude"]["confirmed_usd"]:.6f} / $8**, '
        f'**Luna ${cost["by_provider"]["luna"]["confirmed_usd"]:.6f} / $3**; '
        f'reserved/unknown ${cost["reserved_usd"]:.6f}. No further paid calls are scheduled.', '',
        'All requested parts are complete. No training was run, the test cohort was not accessed, and the owned GPU server '
        'was stopped. Frozen prompts, adapters, rewards and prior experiment artifacts remain unchanged. '
        'Local code tests and live SEARCH/Bareun checks are recorded in `checks.json`.', '',
        f'[30-notice human review CSV]({ROOT}/validation/C_notices_human_review_30.csv). '
        f'[All artifact completion metadata]({ROOT}/complete.json).', '']
    body = REPORT.read_text()
    # Parts finished in parallel; arrange existing blocks in requested order.
    sections = re.split(r'(?=^## )', body, flags=re.M)
    intro = sections.pop(0)
    def order(section):
        heading = section.splitlines()[0]
        if heading.startswith('## 1.'): return 1
        if heading.startswith('## 2.'): return 2
        if heading.startswith('## 3.'): return 3
        if heading.startswith('## 4.'): return 4
        return 0
    body = intro+''.join(sorted(sections, key=order))
    REPORT.write_text(body+'\n'+'\n'.join(lines))
    value = {'completed_at': datetime.now(timezone.utc).isoformat(), 'validated_setting': gate['validated_setting'],
        'parts': [0,1,2,3,4], 'cost': cost, 'wall_seconds': wall, 'training': False,
        'test_cohort_used': False, 'step2_combined': combined, 'a4_protocol_complete': a4['protocol_complete']}
    atomic_new(marker, value)
    return value


if __name__ == '__main__':
    print(finalize())
