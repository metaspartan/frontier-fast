# qwen3.6-35b-a3b R9700 — DFlash depth sweep measured (2026-08-20)

## Method
Sanctioned suspend/measure/restore. Box endpoint suspended, GPU lock taken.
Upstream DFlash engine (`/home/ghost/dflash2full/build/bin/llama-server`),
target `Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` (22 GB) + pinned DFlash draft
`Qwen3.6-35B-A3B-DFlash-Q4_K_M.fixed.gguf` (291 MB). `--spec-type draft-dflash`
with sweep over --spec-draft-n-max. 4x 128-token decodes per depth.
Server-internal `eval time` tok/s used (cleanest per-depth).

## Results (server-internal eval tok/s)
| depth | eval tok/s | draft acceptance | mean len |
|-------|-----------|------------------|----------|
| 3     | (control-U/partial) | - | - |
| 5     | 116.47 | 0.426 (86/202) | 3.10 |
| 6     | **118.38** | 0.397 (89/224) | 3.34 |
| 8     | 88.47  | 0.319 (89/279) | 3.41 |

Best ~**118 tok/s at depth 6**; depth 8 drops (cf. pinned-note "depth 8 slower than stock"). Confirms nonlinear verify-width crossover and the depth-8 penalty.

## Against the frontier
Local plain draft-dflash tops out ~118 tok/s over the 82.6 baseline (+43%).
BUT the board spec frontier is **233.3 tok/s (+154.94%)** ("DFlash block drafting,
depth priced as verify width"). That leader is the kernel frontier (162.2 tok/s
MoE) + DFlash + depth-pricing combined, far above a bare draft-dflash run.
My depth sweep alone (this bare engine, no R9700 kernel series) does NOT beat
233.3. It reproduces the mechanism and the depth-8 penalty, but the winning 233.3
requires the kernel frontier layered on, which the leader already has.

## Honest verdict
Depth-sweep of bare draft-dflash on the stock upstream engine: ~118 tok/s max,
below the 233.3 spec frontier. Not a submission (does not beat the leader).
The leader already combined kernel+DFlash+depth. To beat it would need the
R9700 kernel series (162.2) + DFlash at best depth — an integration the leader
has effectively already done.

## Box
Restored (endpoint active, :8080 serving, GPU lock returned, VRAM re-saturated).
Artifacts: /home/ghost/q36-depth-{3,4,5,6,8}.{log,txt}, q36-run.sh, q36-restore.sh.
