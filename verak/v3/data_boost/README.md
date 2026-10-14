# GLOBAL data boost report assembly

The two independently budgeted data collectors own their candidate, teacher and
provisional CPU reward artifacts. Final rewards and selections require the frozen
GPU reference scorer, even when the CPU audit shows no threshold changes.

`rescore.py` freezes each component's readiness after all 5,720 RFT rollouts. The
RFT controller releases its policy service, scores ready teacher data on GPU1,
closes that scorer and invokes each component's CPU-only `gpu-finalize` consumer.
Only the ready, GPU-rescored v1 GLOBAL selection may enter the imminent RFT1 export.
G_DEL_LINK stays separate for round2. Late components wait until RFT evaluation
and its report finish. No active RFT process is stopped by this package.

Each `global/complete.json` and `insertion/complete.json` must identify immutable
metrics/report hashes, an approved scorer hash, a final or budget-stop status,
finished measurements, stopped component work and no live paid calls. The
assembler verifies the separate $12/$6 commitments, GLOBAL-only selection,
GPU-only final rewards, the audit of at least200 distinct source essays and the
unchanged active RFT rollout corpus/runtime before publishing.
Raw unknown outcomes and retained budget reservations remain explicit.

`python -m verak.v3.cli.data_boost_report launch` starts an owned, file-only
background watcher. Use a host context that preserves child processes. It can
dispatch an explicitly authorized GPU reference pass for a late component only
after RFT A completes; every GPU pass has its own process, slot and manifest hash.
Otherwise it waits for both final component markers and the full CPU/GPU audit.
It writes `report_complete.json` once the report is assembled, then exits. The
shared CPU scorer observes that marker between requests and exits separately.
No training, teacher API call or one-shot work is started by this watcher.
