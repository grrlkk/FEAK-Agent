# v4.3 incremental GPU reference pass

This is an explicitly dispatched scoring job. It does not change the live B2/B3
collectors, the v4.3 prompts/environment, API budgets, selection rules, or v1
rewards. It does not train a model or call an API/analyzer.

The original batch driver remains unchanged. Its physical-GPU1/all-GPUs-idle
requirements are not used by this new, user-authorized physical-GPU0 entry point.

## Resource and equality contract

- Only `CUDA_VISIBLE_DEVICES=0`, `CUDA_DEVICE_ORDER=PCI_BUS_ID`, local `cuda:0`.
  The actual CUDA UUID must equal `nvidia-smi -i 0`'s UUID. GPU1 may be busy.
- Same frozen RFT1 scorer base, adapter, NF4, seed, deterministic/eager inference,
  no TF32, single greedy score line and eight teacher-forced digit expectations.
- The scorer fingerprint includes its device setting. Both the actual `gpu=0`
  execution fingerprint and the original `gpu=1` reference fingerprint are
  recorded, with both complete settings dictionaries. No difference is hidden.
- First freeze 20 completed, distinct-source RFT1 sample-1 episodes in the saved
  design order, including initial/stage1/final inputs. Duplicate exact inputs are
  scored once. All fresh calls bypass and disable the scorer cache; they use a
  new calibration output directory, never the original score cache.
- Every expected-score float, all eight digits, Q, generated score line and input
  length must match exactly. A nonzero difference, even tiny, fails the gate.
  Per-input/rubric differences, mean/max absolute Q difference and rubric digit
  agreement are saved. Any failed calibration prevents all B3 scoring, including
  on resume. Saved inference errors are preserved rather than retried.
- Shared old `data_boost/gpu_rescore/worker.lock`, plus B3's scorer/dispatch locks.
  Only this same watch process's retained CUDA context may already occupy GPU0;
  foreign or malformed PID entries fail closed. Nothing kills or pauses another
  process. One worker uses eight CPU threads.

## Root-owned dispatch

First review and freeze the file-only contract (no GPU/model load):

```sh
env CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 taskset -c 116-119 nice -n 19 \
  /home/chanwoo/anaconda3/envs/feak_agent/bin/python \
  -m verak.v4.scale3_incremental_reference preflight \
  --root /home/chanwoo/FEAK-Agent/verak/v4/outputs/scale3
```

After root reviews that contract, run from the isolated implementation worktree:

```sh
env CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_ORDER=PCI_BUS_ID HF_HUB_OFFLINE=1 \
  taskset -c 96-103 nice -n 10 \
  /home/chanwoo/anaconda3/envs/feak_agent/bin/python \
  -m verak.v4.scale3_incremental_reference run --watch \
  --root /home/chanwoo/FEAK-Agent/verak/v4/outputs/scale3
```

Without `--watch`, the process handles one snapshot batch and exits. Watch mode
waits 2,700 seconds **after** a batch before its next collection-directory scan;
there are no live progress polls during the wait. GPU weights are released
between batches. SIGINT/SIGTERM wake the wait and stop between score requests.
`B3/gpu_rescore/incremental/stop_requested.json` is also checked between requests
and batches. Its presence does not wake a sleeping process; use an owned signal
when immediate cancellation is necessary.

## Durable outputs and handoff

`B3/gpu_rescore/incremental/` contains the frozen contract, calibration manifest,
fresh responses/comparison evidence, exact raw/prepared byte snapshots, per-episode
input plans and immutable batch summaries. Existing input/response mutations fail
closed. Only atomic saved attempts with `completed=true` enter incremental batches;
no live partial episode is read or modified.

`B3/gpu_rescore/responses/<pair-key>.json` uses the existing consumer schema with
explicit physical/local GPU0, actual execution fingerprint and exact-calibration
proof. No CPU Q or historical unproven score is accepted. Saved responses and
errors resume without new calls; an already scored batch loads no GPU model.

Once the **collector itself** publishes its original `gpu_ready.json` and immutable
`gpu_manifest.json`, the last pass validates every manifest endpoint against exact
saved state text, original question, input key and snapshot hashes. This final
manifest may include a finished GLOBAL stage from an unsuccessful Korean episode;
its original status/termination is preserved, counted as incomplete/global-only,
and never relabeled a successful complete episode. Missing projections/scores
remain unknown under the unchanged finalizer.

The scorer then writes the compatible `B3/gpu_rescore/complete.json`, including the
original manifest, completed one-shot report, reference identity, actual GPU0
mapping and calibration hashes. It **does not** run selection/export/reporting.
Root may subsequently call the existing file-only `corruption-gpu-finalize` from
the bound collection worktree. The old GPU1 reference stage must remain suspended.
