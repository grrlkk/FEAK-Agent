"""Authorized warm-start training stages; deliberately no RFT command."""
import argparse
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['export', 'train', 'prepare-eval', 'evaluate', 'markers', 'measure-real', 'report', 'continue'])
    parser.add_argument('--role', choices=['global', 'korean'])
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--condition', choices=['base', 'epoch_1', 'epoch_2', 'luna_low'])
    parser.add_argument('--max-api-calls', type=int, default=0)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--global-pid', type=int)
    parser.add_argument('--luna-pid', type=int)
    args = parser.parse_args()
    if args.stage == 'continue':
        if not args.global_pid or not args.luna_pid:
            parser.error('continue requires the existing GLOBAL and Luna process IDs')
        from ..train.sft_continue import run
        run(args.global_pid, args.luna_pid)
    elif args.stage == 'export':
        from ..train.sft_data import config_for, export
        result = export(config_for())
        print(json.dumps({'roles': result['roles'], 'checks': result['checks']}, ensure_ascii=False))
    elif args.stage == 'train':
        if not args.role:
            parser.error('train requires --role')
        from ..train.sft_train import train
        train(args.role, resume=args.resume)
    elif args.stage == 'prepare-eval':
        from ..train.sft_eval import eval_config, prepare
        design, _, _ = prepare(eval_config())
        print(json.dumps(design, ensure_ascii=False))
    elif args.stage == 'markers':
        if not args.max_api_calls:
            parser.error('Paid Sol checks require --max-api-calls')
        from ..train.sft_eval import eval_config
        from ..train.sft_markers import run
        run(eval_config(), max_api_calls=args.max_api_calls)
    elif args.stage in ('measure-real', 'report'):
        from ..train.sft_eval import eval_config
        from ..train.sft_report import measure_real, report
        (measure_real if args.stage == 'measure-real' else report)(eval_config())
    else:
        if not args.condition:
            parser.error('evaluate requires --condition')
        if args.condition == 'luna_low' and not args.max_api_calls:
            parser.error('Paid teacher evaluation requires --max-api-calls')
        from ..train.sft_eval import eval_config, execute
        execute(eval_config(), args.condition, max_api_calls=args.max_api_calls, limit=args.limit)


if __name__ == '__main__':
    main()
