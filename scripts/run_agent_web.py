#!/usr/bin/env python
"""Open the local FEAK writing studio at http://127.0.0.1:8765."""

import argparse
from pathlib import Path
import signal
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from feak_tc.web.server import RunManager, make_server


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/agent_local.yaml")
    parser.add_argument("--runs-dir", type=Path, default=PROJECT_ROOT / "experiments/results/web_runs")
    parser.add_argument("--history-dir", type=Path, default=PROJECT_ROOT / "experiments/results")
    args = parser.parse_args(argv)
    manager = RunManager(args.runs_dir, args.config, args.history_dir)
    server = make_server(manager, args.host, args.port)

    def stop(*_):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop)
    print(f"FEAK Writing Studio: http://{args.host}:{server.server_address[1]}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        manager.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
