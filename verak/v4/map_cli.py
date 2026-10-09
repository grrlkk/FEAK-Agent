"""Entrypoint for the authorized, isolated C map pilot."""
import argparse
import json


def main():
    from .common import constrain_cpu
    constrain_cpu()
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['run', 'report', 'launch'])
    args = parser.parse_args()
    if args.command == 'launch':
        import os
        from pathlib import Path
        import subprocess
        import sys
        from .common import ROOT, read_json, write_json
        # Run this entrypoint outside the ephemeral sandbox so its owned worker persists.
        if not (ROOT/'design.json').exists():
            raise RuntimeError('Wait for the parent-owned frozen source manifest')
        output=ROOT/'C'; output.mkdir(parents=True,exist_ok=True)
        previous=read_json(output/'launch.json') if (output/'launch.json').exists() else {}
        if previous.get('pid'):
            proc=Path('/proc')/str(previous['pid'])
            try:
                live=(b'verak.v4.map_cli' in (proc/'cmdline').read_bytes().split(b'\0') and
                      (proc/'cwd').resolve()==Path(__file__).resolve().parents[2])
            except OSError:
                live=False
            if live:
                print(json.dumps(previous)); return
        with (output/'worker.log').open('a') as log:
            worker=subprocess.Popen([sys.executable,'-m','verak.v4.map_cli','run'],
                cwd=Path(__file__).resolve().parents[2], env=dict(os.environ),
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        result={'pid':worker.pid,'worktree':str(Path(__file__).resolve().parents[2]),
                'command':'python -m verak.v4.map_cli run','gpu_used':False}
        write_json(output/'launch.json',result)
    elif args.command == 'run':
        from .maps import run
        result = run()
    else:
        from .common import ROOT, read_json, rows_for
        from .map_report import publish
        result = publish(list(rows_for('maps150')), ROOT/'C', read_json(ROOT/'C/design.json'),
                         read_json(ROOT/'C/api/accounting.json'))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
