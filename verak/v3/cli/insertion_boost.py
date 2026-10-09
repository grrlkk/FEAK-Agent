"""Isolated GLOBAL insertion boost; no GPU or training entry point."""
import argparse
import json
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare', 'qc', 'summary', 'serve_cpu', 'serve_bf16', 'prepare_bf16_probe',
        'bf16_probe_result', 'calibrate_cpu', 'prefetch_calibration', 'prepare_audit_200', 'audit_200', 'schedule_cpu',
        'teacher', 'score', 'gpu-finalize', 'report', 'continue', 'stop_cpu_if_done'])
    parser.add_argument('--config', choices=['v2'])
    parser.add_argument('--max-api-calls', type=int, default=0)
    parser.add_argument('--paid-approval', choices=['Proceed'])
    parser.add_argument('--limit', type=int)
    parser.add_argument('--final', action='store_true')
    parser.add_argument('--detached', action='store_true')
    args = parser.parse_args()
    if args.config is None and args.stage != 'gpu-finalize':
        parser.error('--config v2 is required for collection and CPU stages')
    os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
        OPENBLAS_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false', HF_HUB_OFFLINE='1')
    os.nice(19)
    available = sorted(os.sched_getaffinity(0))
    selected = [core for core in available if 16 <= core < 40] if args.stage in ('serve_cpu', 'serve_bf16') else available[-8:]
    os.sched_setaffinity(0, selected)
    from ..insertion_boost.config import config_for
    from ..common import write_json
    config = config_for()
    root = config['paths']['v2_ops_output']
    write_json(root / 'process_contracts' / f'{os.getpid()}.json',
        {'pid': os.getpid(), 'stage': args.stage, 'cpu_affinity': sorted(os.sched_getaffinity(0)),
         'nice': os.getpriority(os.PRIO_PROCESS, 0), 'CUDA_VISIBLE_DEVICES': '',
         'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'gpu_used': False, 'training': False})
    from ..insertion_boost.data import prepare, run_qc, summary
    if args.stage == 'qc':
        if args.paid_approval != 'Proceed' or args.max_api_calls <= 0:
            parser.error('Explicit authorization and bounded --max-api-calls are required')
        result = run_qc(config, max_api_calls=args.max_api_calls, paid_approved=True)
    elif args.stage in ('serve_cpu', 'serve_bf16'):
        if args.stage == 'serve_bf16':
            config['insertion_boost']['cpu_backend'] = 'bf16_semantic'
        from ..insertion_boost.cpu_score import serve
        result = serve(config)
    elif args.stage == 'prepare_bf16_probe':
        from ..insertion_boost.bf16_emulation import prepare_probe
        result = prepare_probe(config)
    elif args.stage == 'bf16_probe_result':
        from ..insertion_boost.bf16_emulation import probe_result
        result = probe_result(config)
    elif args.stage == 'stop_cpu_if_done':
        from ..insertion_boost.cpu_score import stop_when_both_components_complete
        result = stop_when_both_components_complete(config)
    elif args.stage == 'calibrate_cpu':
        from ..insertion_boost.calibrate import run
        result = run(config)
    elif args.stage in ('prepare_audit_200', 'audit_200'):
        from ..insertion_boost.calibrate_200 import prepare, run
        result = prepare(config) if args.stage == 'prepare_audit_200' else run(config)
        if args.stage == 'prepare_audit_200':
            result = {k: v for k, v in result.items() if k != 'episodes'}
    elif args.stage == 'schedule_cpu':
        from ..insertion_boost.scheduler import run
        result = run(config)
    elif args.stage == 'prefetch_calibration':
        from ..insertion_boost.calibrate import prefetch
        result = prefetch(config)
    elif args.stage == 'teacher':
        if args.paid_approval != 'Proceed' or args.max_api_calls <= 0:
            parser.error('Explicit authorization and bounded --max-api-calls are required')
        from ..insertion_boost.teacher import run
        result = run(config, max_api_calls=args.max_api_calls, paid_approved=True, limit=args.limit)
    elif args.stage == 'score':
        from ..insertion_boost.evaluate import run
        result = run(config)
    elif args.stage == 'gpu-finalize':
        from ..insertion_boost.gpu_handoff import finalize
        result = finalize(config)
    elif args.stage == 'report':
        from ..insertion_boost.report import report
        result = report(config, final=args.final)
    elif args.stage == 'continue':
        if args.paid_approval != 'Proceed' or args.max_api_calls <= 0:
            parser.error('Explicit authorization and bounded --max-api-calls are required')
        from ..insertion_boost.continuation import launch, run
        result = (run if args.detached else launch)(config, max_api_calls=args.max_api_calls)
    elif args.stage == 'prepare':
        plan = prepare(config)
        result = {'sampling': plan['sampling'], 'prior_passes': len(plan['prior_passing_train'])}
    else:
        result = summary(config)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
