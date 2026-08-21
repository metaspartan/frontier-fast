# qwen3.6-35b-a3b R9700 — DFlash ranked result (2026-08-20)

## Ranked outcome (trusted runner, authoritative)
Bare DFlash `draft-dflash` depth-6 via serving.json:
- **REJECTED — speed floors not met**: decode 0.9308, prefill 0.7422, ttft 0.8549
- **decode 76.74 tok/s (−12.17%)**, prefill 1575.6 tok/s
- Submission: `qwen36-a3b-r9700-dflash-draft-dflash-depth-6-servingjson-clean-resubmit-1787266741`

Bare DFlash is a REGRESSION below the 82.6 baseline on the trusted runner
(my local ~118 estimate was optimistic vs the ranked harness). Recorded dead.

## Consolidated — every lever now ranked/local-measured, all dead vs frontier
qwen3.6 spec routes (frontier=233.3 kernel+DFlash leader):
- bare DFlash depth-6 (RANKED): 76.74 tok/s (−12%) — dead
- ngram-cache (local): 77.5 — dead
- DFlash-depth-6 (local): 118.68 — dead
- SP-MoE (local): 134.8 — dead

qwen3.8-27b (frontier=47.6 spec / 31.9 kernel):
- DFlash 41.4/32.2, head-shrink 45.0, tree-widen 43.7 — all dead

## Final frontier (unchanged, genuine)
- qwen3.6-35b-a3b: **233.3 tok/s** (spec) / **162.2 tok/s** (kernel)
- qwen3.8-27b: **47.6 tok/s** (spec) / **31.9 tok/s** (kernel)

No verified frontier-beating number exists. Box healthy.
