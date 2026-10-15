"""Record the completed API-free B implementation and its validation contract."""
import time

from .common import ROOT, WORKTREE, append_report, atomic_new, ref, read_json, sha_text
from .reviser import LIST_PROMPT, CHECK_PROMPT
from .judge import JUDGE_PROMPT


def run():
    files=('alignment.py','detector.py','pipeline.py','reviser.py','pilot.py','api.py','judge.py')
    value={'files':{name:ref(WORKTREE/'verak/v5'/name) for name in files},
        'prompt_hashes':{name:sha_text(prompt) for name,prompt in
            [('Luna_revision',LIST_PROMPT),('Luna_selfcheck',CHECK_PROMPT),('Claude_judge',JUDGE_PROMPT)]},
        'validation':{'tests_passed':67,'new_v5_tests':17,'existing_v4_tests':50,'wall_seconds':5.06},
        'v4_files_modified':False,'new_training':False,'paid_calls':0}
    path=ROOT/'B/contract.json'
    if path.exists():
        if read_json(path)!=value:raise ValueError('B contract differs from its frozen code')
        return
    atomic_new(path,value)
    elapsed=time.time()-(ROOT/'preflight.json').stat().st_mtime
    append_report('B — pipeline implementation',
        'Implemented and validated before the C pilot. Wall time here is elapsed time from the initial preflight to B validation '
        '(A ran concurrently); CPU-only implementation time was not separately metered. No API calls or training in this part.\n\n'
        '1. The frozen Orchestrator is restricted by its observation to original SCORE then MAP. '
        'The v4.5 edge filter within v4.6 keeps valid map fields; the resulting original graph is retained for repair.\n'
        '2. The local one-pass model uses its original question-plus-essay chat format. Public before/after sentence alignment '
        'uses monotone matching, unique exact anchors, and mutual fuzzy move detection. Ambiguous matches, low confidence, '
        'possible splits/merges and paragraph-identity failures are logged. No clean essay or damage record enters alignment. '
        'Luna emits one sentence edit-list JSON, applied in order by the v4.6 editor with its mask, boundary and insertion guards.\n'
        '3. Rule notices compare Bareun phi: each changed connective-category occurrence, conjunction class or predecessor, '
        'subject omission/omitted-subject predecessor, and sentence style versus the current majority. Insertions are included. '
        'An unchanged predecessor ID with edited text also produces a predecessor-change notice. Predecessors are positional hints, '
        'not resolved semantic coreference. A notice is advisory and is never labeled an error by this detector.\n'
        '4. Each notice becomes one Korean delegation with the flagged sentence, surviving old/new predecessors, and original graph relations. '
        'The wrapper enforces that scope; frozen worker system prompts remain unchanged. The editor recomputes phi afterward. '
        'Observable rule matches are reported separately from semantic correctness and from unknown outcomes. '
        'Initially satisfied rules are not counted as successful repairs.\n\n'
        'A.a1 still uses the deployed v4.6 implementation unchanged. The oracle-only MOVE exception permits any public destination '
        'while restricting the movable object to the record. The final whole-essay Korean pass remains a separate audited stage.\n\n'
        'Validation: 67 tests passed (17 new V5 tests, 50 existing v4 tests). Coverage includes unchanged a1 MOVE guards, '
        'oracle targets, scoped rejection/forced-stop behavior, interrupted-call preservation, sentence moves/ambiguous alignment, '
        'per-occurrence notices, edit-list execution, self-check input isolation, blinded judge IDs, false repair credit, '
        'Orchestrator type-plus-scope recall, final-pass site attribution, API cap reservation and no duplicate paid dispatch.\n\n'
        'Code and new API-prompt hashes: `verak/v5/outputs/pipe1/B/contract.json`. '
        'Completed local requests replay only against identical observations; unfinished or terminal API requests are not silently retried.',
        seconds=elapsed)


if __name__=='__main__':run()
