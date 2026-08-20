# DFlash2 measured on R9700 — FINAL (2026-08-20)

## Result
DFlash2 (block-diffusion draft, incoai Qwen3.8-27B-DFlash2-GGUF Q4_K_M)
on the R9700, measured on a legitimately-freed GPU:

- **Steady-state decode ≈ 41.4 tok/s** (5 runs: 29.5, 39.9, 41.4, 41.4, 41.4)
- Server-internal timing confirms: **43.4 tok/s** (task 193, 2927.78 ms / 128 tok)
- **draft acceptance 0.596** (81 / 136), mean accepted len 2.76
- block_size = 8, but `n_max` = 3 draft tokens/round
- Both target (17 GB) + draft (1.1 GB) loaded, /completions answered correctly

## Verdict
**BELOW the 47.69 tok/s leader (greedy-argmax MTP chain).** Recorded `dead`
on the ledger with the numbers. The contract's caveat was confirmed: DFlash2's
2.7-3.4x is NVIDIA/TPU; on the R9700 the lightweight block-diffusion draft
cannot out-decode the 47.69 chain.

## Honest conclusion
Every lever on this track is now measured/exhausted:
- Kernel: 31.9 tok/s frontier (Conv-state placement fusion)
- Spec: 47.69 tok/s frontier (greedy-argmax MTP chain); my attempts at MTP
  head-shrink (45.0), tree-widen (43.7), and now DFlash2 (41.4) all lose.
- Frontier UNCHANGED at **47.69 tok/s**.

## Artifacts (kept, in case a knob changes the trade later)
- /home/ghost/dflash2full — full upstream DFlash2 build (v0.1.2-dev)
- /home/ghost/models/Qwen3.8-27B-DFlash2-Q4_K_M.gguf (1.14 GB)
- /home/ghost/dfl-bench.sh, dfl-run.log, dfl-run-c3.log
- GPU restored: guest `gainz-qwen-endpoint.service` restarted; arena intact.

No submission was made (would not beat the leader). Everything restored.
