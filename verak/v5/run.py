"""Own inference-only GPU jobs and close only processes started here."""
import argparse
import os
import socket
import subprocess
import time
import traceback

from .common import *


def process_handle(pid):
    if hasattr(os,'pidfd_open'):return os.pidfd_open(pid)
    # This interpreter predates the os wrapper; the Linux kernel still exposes
    # the same process-exit descriptor. Limit the fallback to known syscall ABIs.
    import ctypes
    import errno
    if os.uname().machine not in ('x86_64','aarch64'):
        raise RuntimeError('No supported event-based process waiter on this host')
    library=ctypes.CDLL(None,use_errno=True)
    fd=library.syscall(434,ctypes.c_int(pid),ctypes.c_uint(0))
    if fd<0:
        error=ctypes.get_errno()
        if error==errno.ESRCH:raise ProcessLookupError(pid)
        raise OSError(error,os.strerror(error))
    return fd


def server_command():
    return [str(VLLM_PYTHON), '-m', 'vllm.entrypoints.openai.api_server', '--model', str(BASE),
        '--served-model-name', 'kanana-v5', '--host', '127.0.0.1', '--port', str(PORT),
        '--enable-lora', '--max-loras', '4', '--max-cpu-loras', '4', '--max-lora-rank', '16',
        '--dtype', 'bfloat16', '--max-model-len', '16384', '--max-num-seqs', '1',
        '--gpu-memory-utilization', '.90', '--max-num-batched-tokens', '2048', '--enforce-eager',
        '--enable-prefix-caching', '--generation-config', 'vllm', '--seed', '47', '--lora-modules',
        *['v5-'+name+'='+str(path) for name,path in ADAPTERS.items()]]


def run(part):
    from verak.v4.full_run import idle
    complete={'a1':'A/a1/complete.json','a2':'A/a2/complete.json','a3':'A/a3/complete.json',
              'c_local':'C/local_complete.json','c_repair':'C/repair_complete.json'}
    if (ROOT/complete[part]).exists():return
    preflight(); hardware = idle((0, 1))
    with socket.socket() as probe:
        # Match uvicorn's listener behavior after the prior owned server exits.
        # TIME_WAIT connections must not block the next stage; a live listener
        # still makes this bind fail, so no unrelated service is displaced.
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind(('127.0.0.1', PORT))
    environment = {**os.environ, 'CUDA_DEVICE_ORDER': 'PCI_BUS_ID', 'CUDA_VISIBLE_DEVICES': '0',
        'VLLM_NO_USAGE_STATS': '1', 'DO_NOT_TRACK': '1', 'HF_HUB_OFFLINE': '1',
        'TRANSFORMERS_OFFLINE': '1', 'TOKENIZERS_PARALLELISM': 'false', 'OMP_NUM_THREADS': '8'}
    process = None
    started = time.monotonic()
    try:
        import httpx
        with (ROOT/f'{part}_vllm.log').open('a') as log:
            command = server_command()
            process = subprocess.Popen(command, cwd=WORKTREE, env=environment, stdout=log, stderr=subprocess.STDOUT)
            write_json(ROOT/'server.json', {'pid': process.pid, 'physical_gpu': 0, 'command': command,
                                          'part': part, 'at': time.time(), 'hardware': hardware})
            deadline = time.monotonic()+900
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError('vLLM failed; inspect log')
                try:
                    if httpx.get(f'http://127.0.0.1:{PORT}/health', timeout=2).status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
                time.sleep(2)
            else:
                raise RuntimeError('vLLM readiness timeout')
            with (ROOT/f'{part}_inference.log').open('a') as out:
                module='verak.v5.pilot' if part.startswith('c_') else 'verak.v5.diagnosis'
                result = subprocess.run([str(INFERENCE_PYTHON), '-u', '-m', module, part],
                    cwd=WORKTREE, env={**environment, 'CUDA_VISIBLE_DEVICES': '1'}, stdout=out, stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError('V5 inference failed; inspect log')
    except Exception as exc:
        write_json(ROOT/f'{part}_execution_failure.json', {'error': str(exc), 'traceback': traceback.format_exc()})
        raise
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=45)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait(timeout=10)
        write_json(ROOT/f'{part}_server_stopped.json', {'owned_pid': process.pid if process else None,
                    'at': time.time(), 'wall_seconds': time.monotonic()-started})


def follow_local():
    """Wait on the owned a1 server's exit, then finish the API-free sequence.

    pidfd waits for a process event without repeated progress polling. A missing
    a1 completion marker is a hard stop, never authorization to repeat rollouts.
    This entry point cannot dispatch either paid provider.
    """
    import fcntl
    import select
    with (ROOT/'follow_local.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        write_json(ROOT/'local_queue.json',{'sequence':['a1 (already running)','a2','a3','c_local'],
            'API_calls_allowed':False,'training':False,'at':time.time()})
        server=read_json(ROOT/'server.json')
        if server['part']=='a1':
            pid=server['pid']
            try:
                handle=process_handle(pid)
            except ProcessLookupError:handle=None
            if handle is not None:
                try:
                    command=Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode()
                    if 'vllm.entrypoints.openai.api_server' not in command or f'--port {PORT}' not in command:
                        raise RuntimeError('Owned server PID identity no longer matches')
                    event=select.poll();event.register(handle,select.POLLIN);event.poll()
                except FileNotFoundError:pass
                finally:os.close(handle)
        if not (ROOT/'A/a1/complete.json').exists():
            raise RuntimeError('a1 did not finish; no automatic rollout retry')
        for part in ('a2','a3','c_local'):
            write_json(ROOT/'local_queue_stage.json',{'part':part,'at':time.time()})
            run(part)
        atomic_new(ROOT/'local_queue_complete.json',{'API_calls':0,'training':False,'at':time.time()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('part', choices=['preflight', 'a1', 'a2', 'a3', 'c_local', 'c_repair','follow_local'])
    args = parser.parse_args()
    if args.part == 'preflight':
        preflight(); print('V5 frozen models and local SEARCH verified; no training')
    elif args.part=='follow_local':
        try:follow_local()
        except Exception:
            write_json(ROOT/'local_queue_failure.json',{'traceback':traceback.format_exc(),'at':time.time()})
            raise
    else:
        run(args.part)
