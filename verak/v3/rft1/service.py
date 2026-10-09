"""Owned detached collection process; lifecycle markers never authorize new rounds."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request
import signal

from ..common import read_json, write_json
from .config import PHASE, WORKTREE


def matches(pid, needle):
    try:
        return needle in Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0', b' ').decode()
    except (OSError, UnicodeError):
        return False


def launch_rollouts(config):
    root = config['paths'][PHASE + '_output']
    path = root / 'rollout_worker.json'
    if path.exists() and matches(read_json(path)['pid'], 'verak.v3.cli.rft1 rollout'):
        return read_json(path)
    with urllib.request.urlopen(config['policy']['base_url'] + '/models', timeout=10) as response:
        models = json.load(response)
    if not {'sft-global-ep2', 'sft-korean-ep2'} <= {row['id'] for row in models['data']}:
        raise RuntimeError('The epoch-2 adapter pair is not served')
    write_json(root / 'rollout_served_models.json', models)
    command = [sys.executable, '-m', 'verak.v3.cli.rft1', 'rollout']
    with (root / 'rollouts.log').open('a') as log:
        child = subprocess.Popen(command, cwd=WORKTREE, stdin=subprocess.DEVNULL,
                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    result = {'pid': child.pid, 'command': command, 'at': time.time(), 'requested': 5720}
    write_json(path, result)
    write_json(root / 'status.json', {'stage': 'rollouts', **result})
    return result


def process_alive(pid):
    try:
        state = Path(f'/proc/{pid}/stat').read_text().split(') ', 1)[1].split()[0]
        return state != 'Z'
    except OSError:
        return False


def stop_owned_server(path):
    """Never terminate an unrelated process or infer ownership just from GPU usage."""
    if not path.exists():
        return
    record = read_json(path)
    pid = record['pid']
    if not process_alive(pid):
        return
    if (pid <= 100 or record.get('owner') != PHASE or
            not matches(pid, 'vllm.entrypoints.openai.api_server') or
            not matches(pid, 'c963a5f4f6496c749f94064a20b33028b0db9f19') or
            os.getpgid(pid) != pid):
        raise RuntimeError('Refusing to stop a process without exact owned-server identity')
    for sig, seconds in ((signal.SIGINT, 60), (signal.SIGTERM, 30)):
        if not process_alive(pid):
            break
        os.killpg(pid, sig)
        deadline = time.monotonic() + seconds
        while process_alive(pid) and time.monotonic() < deadline:
            time.sleep(2)
    if process_alive(pid):
        raise RuntimeError('Owned server did not stop gracefully; inspect before using both GPUs')
    write_json(path.with_name(path.stem + '_stopped.json'), {**record, 'stopped_at': time.time()})


def idle_gpus():
    deadline = time.monotonic() + 120
    while True:
        result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_gpu_memory',
            '--format=csv,noheader,nounits'], check=True, capture_output=True, text=True)
        if not result.stdout.strip():
            return
        if time.monotonic() >= deadline:
            raise RuntimeError('Both GPUs must be free of compute processes before DDP; found ' + result.stdout.strip())
        time.sleep(5)


def start_server(config, condition):
    root = config['paths'][PHASE + '_output']
    adapter_root = root if condition == 'rft1' else root.parent / 'oneshot_baseline'
    marker = root / (condition + '_server.json')
    expected = {'rft1-global', 'rft1-korean'} if condition == 'rft1' else {'oneshot-oneshot'}
    def available():
        try:
            with urllib.request.urlopen(config['policy']['base_url'] + '/models', timeout=5) as response:
                return {r['id'] for r in json.load(response)['data']}
        except (OSError, ValueError):
            return set()
    if marker.exists() and process_alive(read_json(marker)['pid']):
        if not matches(read_json(marker)['pid'], 'vllm.entrypoints.openai.api_server'):
            raise RuntimeError('Server PID identity changed')
        if expected <= available():
            return read_json(marker)
        raise RuntimeError('An owned but unready evaluation server requires inspection')
    if available():
        raise RuntimeError('Another policy service owns port8030; do not replace it')
    idle_gpus()
    executable = '/home/chanwoo/anaconda3/envs/verak_vllm/bin/python'
    command = [executable, '-m', 'verak.v3.cli.serve_rft1', '--root', str(adapter_root), '--condition', condition]
    with (root / (condition + '_vllm.log')).open('a') as log:
        child = subprocess.Popen(command, cwd=WORKTREE, stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    record = {'pid': child.pid, 'command': command, 'at': time.time(), 'owner': PHASE, 'physical_gpu': 0}
    write_json(marker, record)
    deadline = time.monotonic() + 300
    while time.monotonic() < deadline:
        if child.poll() is not None:
            raise RuntimeError('Evaluation policy server exited; inspect ' + condition + '_vllm.log')
        if expected <= available():
            return record
        time.sleep(3)
    raise RuntimeError('Evaluation server was not ready in300 seconds; inspect without duplicate launching')


def continue_work(config):
    """Finish A, honor an explicit C hold, then run C only when authorized."""
    from ..train.teacher_bulk import collection_lock
    from .config import runtime_hashes
    root = config['paths'][PHASE + '_output']
    control = root / 'controller'
    def state(stage, **values):
        record = {'stage': stage, 'controller_pid': os.getpid(), 'at': time.time(), **values}
        write_json(root / 'status.json', record)
        return record
    def execute(stage, arguments, *, ddp=False):
        command = [sys.executable]
        if ddp:
            command += ['-m', 'torch.distributed.run', '--standalone', '--nnodes=1', '--nproc-per-node=2']
        command += ['-m', 'verak.v3.cli.rft1', *arguments]
        environment = {**os.environ, 'HF_HUB_OFFLINE': '1', 'TOKENIZERS_PARALLELISM': 'false',
                       'OMP_NUM_THREADS': '8', 'MKL_NUM_THREADS': '8', 'OPENBLAS_NUM_THREADS': '8'}
        if ddp:
            idle_gpus()
            environment['CUDA_VISIBLE_DEVICES'] = '0,1'
        log_path = root / (stage + '.log')
        with log_path.open('a') as log:
            child = subprocess.Popen(command, cwd=WORKTREE, env=environment, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT)
        started = time.time()
        state(stage, pid=child.pid, command=command, started_at=started, log=str(log_path))
        while True:
            try:
                result = child.wait(timeout=30)
                break
            except subprocess.TimeoutExpired:
                state(stage, pid=child.pid, command=command, started_at=started, log=str(log_path))
        with (control / 'steps.jsonl').open('a') as stream:
            stream.write(json.dumps({'stage': stage, 'command': command, 'pid': child.pid,
                'exit_code': result, 'elapsed_s': time.time() - started, 'log': str(log_path)}) + '\n')
        if result:
            raise RuntimeError(stage + ' exited ' + str(result) + '; inspect ' + str(log_path))

    with collection_lock(control):
        try:
            runtime_hashes(config)
            if not (root / 'a_complete.json').exists():
                worker = read_json(root / 'rollout_worker.json')
                while matches(worker['pid'], 'verak.v3.cli.rft1 rollout') and process_alive(worker['pid']):
                    state('rollouts', rollout_pid=worker['pid'], waiting_for='all5720saved')
                    time.sleep(30)
                result = read_json(root / 'rollout_status.json')
                if not result['sampling_complete'] or result['saved'] != 5720 or result['errors']:
                    raise RuntimeError('Collection not complete; inspect saved errors before resuming')
                stop_owned_server(root / 'rollout_server.json')
                from ..data_boost.rescore import at_boundary
                state('extra_teacher_pre_training_boundary')
                at_boundary(config, 'pre_rft_training')
                execute('exporting', ['export'])
                for role in ('global', 'korean'):
                    output = root / 'adapters' / role
                    if not (output / 'complete.json').exists():
                        args = ['train', '--role', role]
                        if (output / 'recipe.json').exists():
                            args.append('--resume')
                        execute('training_' + role, args, ddp=True)
                evaluation_status = root / 'evaluation/rft1/status.json'
                if not evaluation_status.exists() or not read_json(evaluation_status)['all_attempted']:
                    state('starting_rft1_server')
                    start_server(config, 'rft1')
                    execute('evaluating_rft1', ['evaluate'])
                execute('measuring_rft1', ['measure'])
                execute('checking_rft1_markers', ['markers'])
                execute('reporting_rft1', ['report'])
                stop_owned_server(root / 'rft1_server.json')
                write_json(root / 'a_complete.json', {'completed': True, 'at': time.time(),
                    'report': read_json(root / 'report_status.json')})
                state('a_complete')
            from ..data_boost.rescore import at_boundary
            at_boundary(config, 'post_rft_evaluation')
            # Read this after A, so a hold requested during collection takes effect.
            # A remains authorized and independent of the one-shot comparison.
            hold_path = root / 'oneshot_hold.json'
            if hold_path.exists() and read_json(hold_path).get('hold') is True:
                state('a_complete_c_on_hold', A=str(root / 'report_status.json'),
                      C='held_until_explicit_user_resume', hold=str(hold_path),
                      further_training=False)
                return
            # No C command or GPU use occurs before the complete A marker.
            oneshot = root.parent / 'oneshot_baseline'
            if not (oneshot / 'complete.json').exists():
                execute('oneshot_export', ['oneshot-export'])
                output = oneshot / 'adapters/oneshot'
                if not (output / 'complete.json').exists():
                    args = ['train', '--role', 'oneshot']
                    if (output / 'recipe.json').exists():
                        args.append('--resume')
                    execute('training_oneshot', args, ddp=True)
                evaluation_status = oneshot / 'evaluation/oneshot/status.json'
                if not evaluation_status.exists() or not read_json(evaluation_status)['all_attempted']:
                    state('starting_oneshot_server')
                    start_server(config, 'oneshot')
                    execute('evaluating_oneshot', ['evaluate', '--condition', 'oneshot'])
                execute('measuring_oneshot', ['measure', '--condition', 'oneshot'])
                execute('checking_oneshot_markers', ['markers', '--condition', 'oneshot'])
                execute('reporting_oneshot', ['report', '--condition', 'oneshot'])
                stop_owned_server(root / 'oneshot_server.json')
                write_json(oneshot / 'complete.json', {'completed': True, 'at': time.time(),
                    'report': read_json(oneshot / 'report_status.json')})
            state('complete', A=str(root / 'report_status.json'), C=str(oneshot / 'report_status.json'),
                  further_training=False)
        except Exception as exc:
            previous = read_json(root / 'status.json')
            state('failed', previous_stage=previous['stage'], error={'type': type(exc).__name__, 'message': str(exc)})
            raise


def launch_controller(config):
    root = config['paths'][PHASE + '_output']
    marker = root / 'controller_worker.json'
    if marker.exists() and matches(read_json(marker)['pid'], 'verak.v3.cli.rft1 continue'):
        return read_json(marker)
    command = [sys.executable, '-m', 'verak.v3.cli.rft1', 'continue']
    with (root / 'controller.log').open('a') as log:
        child = subprocess.Popen(command, cwd=WORKTREE, stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    value = {'pid': child.pid, 'command': command, 'at': time.time(), 'owner': PHASE,
             'scope': 'finish A once; honor oneshot_hold.json before C; no further rounds'}
    write_json(marker, value)
    return value


def restart_waiting_controller(config):
    """Load updated orchestration while leaving the independent rollout worker alone."""
    root = config['paths'][PHASE + '_output']
    marker = root / 'controller_worker.json'
    if marker.exists() and process_alive(read_json(marker)['pid']):
        pid = read_json(marker)['pid']
        if (pid <= 100 or read_json(root / 'status.json')['stage'] != 'rollouts' or
                not matches(pid, 'verak.v3.cli.rft1 continue')):
            raise RuntimeError('Only an owned controller waiting for rollouts may be restarted')
        children = Path(f'/proc/{pid}/task/{pid}/children').read_text().strip()
        if children:
            raise RuntimeError('Controller has active stage children; do not interrupt')
        os.kill(pid, signal.SIGTERM)
        deadline = time.monotonic() + 15
        while process_alive(pid) and time.monotonic() < deadline:
            time.sleep(.5)
        if process_alive(pid):
            raise RuntimeError('Controller failed to stop; not launching a duplicate')
    return launch_controller(config)
