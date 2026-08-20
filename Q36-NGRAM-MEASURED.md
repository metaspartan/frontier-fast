# qwen3.6-35b-a3b R9700 — ngram-cache alternative-spec measurement (2026-08-20)

- server-internal eval: **77.5 tok/s** (1638.8 ms / 128 tokens, 12.90 ms/token)
- draft acceptance **0.333** (3/9), mean len 2.50
- client-side steady ~72-77 tok/s

## Verdict
ngram-cache (a spec path with NO draft weights) = **77.5 tok/s**, well below
DFlash depth-6 (**118.68 tok/s**) and the **233.3 tok/s** spec frontier
(kernel+DFlash combined). Confirms ngram is not competitive for this MoE.
Recorded dead on the ledger: `ngram-cache-spec-measured-well-below-dflash-and-frontier`.

Combined qwen3.6 a3b R9700 spec-axis status:
- ngram-cache: 77.5 (dead)
- DFlash depth-6: 118.68 (dead vs 233.3)
- frontier 233.3 = kernel series (162.2) + DFlash leader (integration, already landed)

## Box
Restored. Artifacts: /home/ghost/q36-ngram.log, tools/q36-ngram-probe.sh,
tools/q36-restore-ngram.sh.
