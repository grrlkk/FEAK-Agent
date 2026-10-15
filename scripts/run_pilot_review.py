#!/usr/bin/env python
"""Prepare, serve, and export blind human reviews without any model/API calls."""

import argparse
import json
from pathlib import Path
import signal
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from feak_tc.review.store import ReviewStore, prepare_study
from feak_tc.review.server import make_server


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare", help="Freeze all produced pilot candidates and make reviewer codes")
    prepare.add_argument("--run-dir", type=Path, required=True)
    prepare.add_argument("--study-dir", type=Path, required=True)
    prepare.add_argument("--raters", type=int, default=2)
    prepare.add_argument("--seed", type=int, default=20260929)
    prepare.add_argument("--title", default="글 수정 평가")
    serve = sub.add_parser("serve", help="Serve a prepared study; ratings survive restarts")
    serve.add_argument("--study-dir", type=Path, required=True)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8766)
    links = sub.add_parser("links", help="Owner-only: print private reviewer links")
    links.add_argument("--study-dir", type=Path, required=True)
    links.add_argument("--base-url", default="http://127.0.0.1:8766")
    export = sub.add_parser("export", help="Owner-only: export current reviews, history, and source mapping")
    export.add_argument("--study-dir", type=Path, required=True)
    export.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            result = prepare_study(args.run_dir, args.study_dir, raters=args.raters, seed=args.seed, title=args.title)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            print("준비 완료. serve로 열고 links로 평가자별 접속 링크를 확인하세요.")
        elif args.command == "links":
            ReviewStore(args.study_dir)
            credentials = json.loads((args.study_dir / "reviewers.private.json").read_text(encoding="utf-8"))
            for rater in credentials["reviewers"]:
                print(f"{rater['label']}: {args.base_url.rstrip('/')}/#token={rater['token']}")
        elif args.command == "export":
            print(json.dumps(ReviewStore(args.study_dir).export_all(args.output_dir), ensure_ascii=False, indent=2))
        else:
            store = ReviewStore(args.study_dir)
            server = make_server(store, args.host, args.port)
            def stop(*_):
                raise KeyboardInterrupt
            signal.signal(signal.SIGTERM, stop)
            print(f"FEAK Human Review: http://{args.host}:{server.server_address[1]}", flush=True)
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                pass
            finally:
                server.server_close()
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
