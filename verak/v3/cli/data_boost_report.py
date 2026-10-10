"""Assemble data reports and run the explicitly scheduled GPU reference pass."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import signal


def launch(repo, *, restart=False):
    from ..common import read_json, write_json
    root = Path(repo).resolve() / 'verak/v3/outputs/data_boost/report_controller'
    root.mkdir(parents=True, exist_ok=True)
    marker = root / 'worker.json'
    if marker.exists():
        value = read_json(marker)
        try:
            command = Path(f"/proc/{value['pid']}/cmdline").read_bytes()
        except FileNotFoundError:
            command = b''
        if b'verak.v3.cli.data_boost_report' in command and b'watch' in command:
            if not restart:
                return value
            status=read_json(root/'status.json')
            rft=read_json(Path(repo)/'verak/v3/outputs/phase8_rft1/status.json')
            if status.get('stage')!='waiting_for_components' or rft.get('stage')!='rollouts':
                raise RuntimeError('Only an idle report watcher while RFT rollouts own the GPUs may restart')
            os.kill(value['pid'],signal.SIGTERM)
            for _ in range(50):
                path=Path(f'/proc/{value["pid"]}/cmdline')
                if not path.exists() or not path.read_bytes():
                    break
                time.sleep(.1)
            else:
                raise RuntimeError('The owned report watcher did not stop')
    command = [sys.executable, '-m', 'verak.v3.cli.data_boost_report', 'watch', '--repo', str(repo)]
    with (root / 'worker.log').open('a') as stream:
        child = subprocess.Popen(command, cwd=Path(__file__).resolve().parents[3],
            env={**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '1',
                 'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'},
            stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
            start_new_session=True)
    value = {'pid': child.pid, 'command': command, 'at': time.time(),
             'scope': 'final report plus deferred GPU reference pass after RFT evaluation; no APIs/training'}
    write_json(marker, value)
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('finalize', 'watch', 'launch', 'rescore'))
    parser.add_argument('--repo', default='/home/chanwoo/FEAK-Agent')
    parser.add_argument('--component', choices=('global', 'insertion'))
    parser.add_argument('--slot', choices=('pre_rft_training', 'post_rft_evaluation'))
    parser.add_argument('--restart',action='store_true')
    args = parser.parse_args()
    if os.nice(0) < 19:
        os.nice(19 - os.nice(0))
    if args.stage == 'rescore':
        if not args.component or not args.slot:
            parser.error('rescore requires --component and --slot')
        from ..rft1.config import config_for
        from ..data_boost.rescore import score_component
        config = config_for()
        if config['paths']['repo'].resolve() != Path(args.repo).resolve():
            raise ValueError('Reference scorer must use the frozen experiment repository')
        print(json.dumps(score_component(config, args.component, args.slot)))
        return
    from ..data_boost.report import finalize, watch
    if args.restart and args.stage!='launch':
        parser.error('--restart only applies to the idle report watcher launcher')
    result=launch(args.repo,restart=args.restart) if args.stage=='launch' else {'watch':watch,'finalize':finalize}[args.stage](args.repo)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
