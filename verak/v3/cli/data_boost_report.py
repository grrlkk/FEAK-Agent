"""Assemble data reports and run the explicitly scheduled GPU reference pass."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def launch(repo):
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
            return value
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
    action = {'watch': watch, 'finalize': finalize, 'launch': launch}[args.stage]
    print(json.dumps(action(args.repo)))


if __name__ == '__main__':
    main()
