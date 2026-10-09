"""Explicit v2 preparation entry point. No training command and no GPU0 access."""
import argparse
import json
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare', 'build-local', 'replay-v1', 'qc', 'qc-summary', 'teacher', 'score', 'report', 'continue'])
    parser.add_argument('--config', required=True, choices=['v2'])
    parser.add_argument('--operator', choices=['G_DEL_LINK', 'L_FUSE'])
    parser.add_argument('--limit', type=int)
    parser.add_argument('--max-api-calls', type=int, default=0)
    parser.add_argument('--paid-approval', choices=['Proceed'])
    parser.add_argument('--final', action='store_true')
    parser.add_argument('--qc-pid', type=int)
    parser.add_argument('--detached', action='store_true')
    args = parser.parse_args()
    os.environ.update(CUDA_VISIBLE_DEVICES='1' if args.stage == 'score' else '', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                      TOKENIZERS_PARALLELISM='false', HF_HUB_OFFLINE='1')
    os.nice(10)
    from ..v2_ops.config import config_for
    config = config_for(version=args.config)
    if args.stage == 'continue':
        if not args.qc_pid or args.paid_approval != 'Proceed' or args.max_api_calls <= 0:
            parser.error('continue requires existing --qc-pid and approved bounded paid calls')
        from ..v2_ops.continuation import launch, run
        result = (run if args.detached else launch)(config, qc_pid=args.qc_pid,
            max_api_calls=args.max_api_calls, paid_approved=True)
    elif args.stage == 'prepare':
        from ..v2_ops.data import prepare
        result = prepare(config)
        result = {k: result[k] for k in ('sampling', 'budget_usd', 'paid_calls', 'gpu_used')}
    elif args.stage == 'replay-v1':
        from ..v2_ops.replay_v1 import replay
        result = replay(output=config['paths']['v2_ops_output'] / 'v1_replay.json')
        result.pop('details')
    elif args.stage == 'qc-summary':
        from ..v2_ops.qc import summary
        result = summary(config)
    elif args.stage == 'score':
        from ..v2_ops.evaluate import run
        result = run(config)
    elif args.stage == 'report':
        from ..v2_ops.report import report
        result = report(config, final=args.final)
    elif args.stage in ('qc', 'teacher'):
        if args.paid_approval != 'Proceed' or args.max_api_calls <= 0:
            parser.error('User Proceed approval and explicit --max-api-calls are required before paid v2 QC')
        if args.stage == 'qc':
            from ..v2_ops.qc import run
            result = run(config, max_api_calls=args.max_api_calls, paid_approved=True)
        else:
            from ..v2_ops.teacher import run
            result = run(config, max_api_calls=args.max_api_calls, paid_approved=True, limit=args.limit)
    else:
        if not args.operator:
            parser.error('build-local requires --operator')
        from ..v2_ops.candidates import build
        result = build(config, operator=args.operator, limit=args.limit)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
