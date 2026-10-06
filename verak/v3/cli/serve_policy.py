"""Run only in the separate verak_vllm conda environment."""
import os
from pathlib import Path
import sys

REVISION = 'c963a5f4f6496c749f94064a20b33028b0db9f19'


def main():
    if Path(sys.prefix).name != 'verak_vllm':
        raise SystemExit('Use the separate verak_vllm environment')
    path = Path('/home/chanwoo/.cache/huggingface/hub/models--kakaocorp--kanana-1.5-8b-instruct-2505/snapshots') / REVISION
    if not (path / 'config.json').is_file():
        raise SystemExit('Pinned local snapshot is missing')
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '0', 'HF_HUB_OFFLINE': '1',
           'VLLM_NO_USAGE_STATS': '1', 'DO_NOT_TRACK': '1'}
    command = [sys.executable, '-m', 'vllm.entrypoints.openai.api_server', '--model', str(path),
        '--served-model-name', 'kanana-policy', '--host', '127.0.0.1', '--port', '8030',
        '--enable-lora', '--max-loras', '2', '--max-lora-rank', '64', '--dtype', 'bfloat16',
        '--max-model-len', '32768', '--max-num-seqs', '1', '--gpu-memory-utilization', '0.90',
        '--max-num-batched-tokens', '2048', '--enforce-eager', '--enable-prefix-caching',
        '--generation-config', 'vllm', '--seed', '47']
    os.execvpe(command[0], command, env)


if __name__ == '__main__':
    main()
