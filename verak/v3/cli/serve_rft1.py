"""Serve only the authorized completed round-1 or one-shot adapters on GPU0."""
import argparse
import json
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--condition', choices=['rft1', 'oneshot'], required=True)
    args = parser.parse_args()
    if Path(sys.prefix).name != 'verak_vllm':
        raise SystemExit('Use the separate verak_vllm environment')
    revision = 'c963a5f4f6496c749f94064a20b33028b0db9f19'
    path = Path('/home/chanwoo/.cache/huggingface/hub/models--kakaocorp--kanana-1.5-8b-instruct-2505/snapshots') / revision
    modules = []
    for role in (('global', 'korean') if args.condition == 'rft1' else ('oneshot',)):
        adapter = args.root / 'adapters' / role / 'epoch_1'
        if not (adapter.parent / 'complete.json').exists():
            raise SystemExit('Only completed adapters can be evaluated')
        if json.loads((adapter / 'provenance.json').read_text())['base_revision'] != revision:
            raise SystemExit('Wrong pinned base')
        modules.append(f'{args.condition}-{role}={adapter.resolve()}')
    command = [sys.executable, '-m', 'vllm.entrypoints.openai.api_server', '--model', str(path),
        '--served-model-name', 'kanana-policy', '--host', '127.0.0.1', '--port', '8030',
        '--enable-lora', '--max-loras', '2', '--max-cpu-loras', '4', '--max-lora-rank', '16',
        '--dtype', 'bfloat16', '--max-model-len', '8192', '--max-num-seqs', '4',
        '--gpu-memory-utilization', '0.90', '--max-num-batched-tokens', '2048', '--enforce-eager',
        '--enable-prefix-caching', '--generation-config', 'vllm', '--seed', '47', '--lora-modules', *modules]
    os.execvpe(command[0], command, {**os.environ, 'CUDA_VISIBLE_DEVICES': '0', 'HF_HUB_OFFLINE': '1',
                                  'VLLM_NO_USAGE_STATS': '1', 'DO_NOT_TRACK': '1'})


if __name__ == '__main__':
    main()
