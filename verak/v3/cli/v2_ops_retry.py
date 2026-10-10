"""CPU/API-only retry, with a separate budget and no scorer/training entry point."""
import argparse
import json
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare', 'build', 'qc', 'summary', 'teacher', 'report', 'continue', 'audit'])
    parser.add_argument('--config', choices=['v2'], required=True)
    parser.add_argument('--max-api-calls', type=int, default=0)
    parser.add_argument('--paid-approval', choices=['Proceed'])
    parser.add_argument('--final', action='store_true')
    parser.add_argument('--wait-pid', type=int)
    parser.add_argument('--detached', action='store_true')
    args = parser.parse_args()
    os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                      OPENBLAS_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false', HF_HUB_OFFLINE='1')
    os.nice(10)
    available = sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0, available[-8:])
    from ..v2_ops.retry import config_for_retry, prepare, build, run_qc
    from ..v2_ops.qc import summary
    config = config_for_retry()
    from ..common import write_json
    write_json(config['paths']['v2_ops_output'] / 'process_contracts' / f'{os.getpid()}.json',
        {'pid': os.getpid(), 'stage': args.stage, 'cpu_affinity': sorted(os.sched_getaffinity(0)),
         'nice': os.getpriority(os.PRIO_PROCESS, 0), 'CUDA_VISIBLE_DEVICES': os.environ['CUDA_VISIBLE_DEVICES'],
         'OMP_NUM_THREADS': os.environ['OMP_NUM_THREADS'], 'MKL_NUM_THREADS': os.environ['MKL_NUM_THREADS'],
         'gpu_used': False, 'training': False})
    if args.stage in {'qc', 'teacher', 'continue'} and (args.paid_approval != 'Proceed' or args.max_api_calls <= 0):
        parser.error('Explicit authorization and bounded --max-api-calls are required')
    if args.stage == 'prepare':
        result = prepare(config)
        result = {'sampling': result['sampling'], 'gpu_used': False, 'paid_calls': 0}
    elif args.stage == 'build':
        result = build(config)
    elif args.stage == 'summary':
        result = summary(config)
    elif args.stage == 'qc':
        result = run_qc(config, max_api_calls=args.max_api_calls, paid_approved=True)
    elif args.stage == 'report':
        from ..v2_ops.retry_report import report
        result = report(config, final=args.final)
    elif args.stage == 'continue':
        from ..v2_ops.retry_continuation import launch, run
        result = (run if args.detached else launch)(config, wait_pid=args.wait_pid, max_api_calls=args.max_api_calls)
    elif args.stage == 'audit':
        from ..v2_ops.retry_audit import run
        result = run(config)
    else:
        from ..v2_ops.teacher import run
        result = run(config, max_api_calls=args.max_api_calls, paid_approved=True)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
