"""Run only the user-authorized first RFT round; the accepted v1 environment is imported."""
import argparse
import json
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare', 'rollout', 'launch-rollouts', 'select', 'export',
        'train', 'evaluate', 'measure', 'markers', 'report', 'context-audit', 'oneshot-export',
        'continue', 'launch-controller', 'restart-controller'])
    parser.add_argument('--limit', type=int)
    parser.add_argument('--role', choices=['global', 'korean', 'oneshot'])
    parser.add_argument('--condition', choices=['rft1', 'oneshot'], default='rft1')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--max-api-calls', type=int, default=2000)
    args = parser.parse_args()
    os.environ.update(HF_HUB_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
                      OMP_NUM_THREADS='8', MKL_NUM_THREADS='8')
    from ..rft1.config import config_for
    from ..rft1.rollout import prepare, run
    config = config_for()
    if args.stage == 'prepare':
        design, _ = prepare(config)
        print(json.dumps({k: design[k] for k in ('requested', 'essays', 'samples_per_essay', 'levels')}))
    elif args.stage == 'launch-rollouts':
        from ..rft1.service import launch_rollouts
        print(json.dumps(launch_rollouts(config)))
    elif args.stage in ('select', 'export'):
        from ..rft1.selection import select, export
        result = (select if args.stage == 'select' else export)(config)
        print(json.dumps(result.get('summaries', result.get('roles')), ensure_ascii=False))
    elif args.stage == 'train':
        if not args.role:
            parser.error('train requires --role')
        from ..rft1.train import train
        train(config, args.role, resume=args.resume, oneshot=args.role == 'oneshot')
    elif args.stage == 'evaluate':
        import torch
        torch.set_num_threads(8)
        if args.condition == 'rft1':
            from ..rft1.evaluate import execute
        else:
            from ..rft1.oneshot import execute
        execute(config, limit=args.limit)
    elif args.stage == 'markers':
        from ..rft1.markers import run as markers
        result = markers(config, args.condition, max_api_calls=args.max_api_calls)
        print(json.dumps({k: v for k, v in result.items() if k != 'details'}))
    elif args.stage == 'measure':
        from ..rft1.evaluate import measure_saved_real
        measure_saved_real(config, args.condition)
    elif args.stage == 'report':
        from ..rft1.report import write_report
        print(json.dumps(write_report(config, args.condition)))
    elif args.stage == 'context-audit':
        from ..rft1.context_audit import audit
        result = audit(config)
        print(json.dumps({eid: {k: v for k, v in record.items() if k != 'conditions'}
                          for eid, record in result['essays'].items()}))
    elif args.stage == 'oneshot-export':
        from ..rft1.oneshot import export
        result = export(config)
        print(json.dumps({k: result[k] for k in ('selection_counts', 'excluded_counts', 'roles')}))
    elif args.stage in ('continue', 'launch-controller', 'restart-controller'):
        from ..rft1.service import continue_work, launch_controller, restart_waiting_controller
        if args.stage == 'continue':
            continue_work(config)
        elif args.stage == 'restart-controller':
            print(json.dumps(restart_waiting_controller(config)))
        else:
            print(json.dumps(launch_controller(config)))
    elif args.stage == 'rollout':
        import torch
        torch.set_num_threads(8)
        run(config, limit=args.limit)


if __name__ == '__main__':
    main()
