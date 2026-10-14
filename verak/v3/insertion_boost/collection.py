"""Read-only catalog of original and explicitly versioned teacher slots."""
from ..common import file_sha, read_json
from ..v2_ops.config import PHASE
from ..v2_ops.teacher import attempt_path


def entries(config, *, include_missing=False):
    root = config['paths'][PHASE + '_output']
    original_path = root / 'teacher_design.json'
    original = read_json(original_path)
    variants = {}
    for phase in ('test_v1', 'resume_v1'):
        path = root / 'prompt_fix' / phase / 'design.json'
        if not path.exists():
            continue
        design = read_json(path)
        if design['original_design_sha256'] != file_sha(original_path):
            raise ValueError('Prompt-variant design does not match the original teacher cohort')
        for row in design['tasks']:
            key = (row['attempt'], row['episode_id'])
            if key in variants:
                raise ValueError('Two prompt variants claim the same original teacher slot')
            variants[key] = (phase, path)
    for attempt in (1, 2):
        for eid in original['orders'][str(attempt)]:
            path = attempt_path(root, attempt, eid)
            variant, design_path, base = 'original', original_path, root
            if (attempt, eid) in variants:
                if path.exists():
                    raise ValueError('A new prompt must not overwrite or duplicate an old saved slot')
                variant, design_path = variants[(attempt, eid)]
                base = design_path.parent
                path = attempt_path(base, attempt, eid)
            if not path.exists() and not include_missing:
                continue
            if path.exists() and read_json(path)['v2']['design_sha256'] != file_sha(design_path):
                raise ValueError('Teacher trajectory prompt/design provenance changed')
            yield {'attempt': attempt, 'episode_id': eid, 'path': path, 'variant': variant,
                'design_path': design_path, 'design_sha256': file_sha(design_path),
                'recovery_path': base / f'attempt_{attempt}/recovery' / path.name,
                'provisional_path': root / f'provisional_scored/attempt_{attempt}' / path.name,
                'scored_path': root / f'scored/attempt_{attempt}' / path.name}


def finished(status):
    if status['saved_attempts'] == status['planned_attempts']:
        return True
    if status.get('stop_reason') == 'user_prompt_gate_failed':
        return status.get('prompt_test_denominator') == 20 and status.get('prompt_test_insertions', 20) < 6
    return any(row['type'] == 'CallBudgetExceeded' for row in status.get('errors', []))
