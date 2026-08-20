# qwen3.8-27b R9700 — exhaustive lever map (verified 2026-08-20)

Authoritative terminal state for the frontier push. All numbers measured on the
R9700 (trusted runner or sanctioned local build) unless noted.

## Live frontiers (unchanged)
- **Spec: 47.6 tok/s (+49.46%)** — "Greedy MTP draft via GPU argmax (chain series)"
- **Kernel: 31.9 tok/s (+7.84%)** — "Conv-state placement fusion"
- Pinned base: `2b63e0610`

## Levers closed (measured / source-verified / already-claimed)
| Lever | Result | Where recorded |
|-------|--------|----------------|
| DFlash2, n_max=3 | 41.4 tok/s — **dead** (built+measured) | ledger `dflash2-r9700-measured-below-leader`, `DFLASH2-R9700-MEASURED.md` |
| DFlash2, n_max=7 | 32.2 tok/s — **dead** | ledger `dflash2-r9700-full-width-nmax7-also-below-leader`, `DFLASH2-NMAX7.md` |
| MTP head-shrink (Latin 98304) | 45.0 tok/s — dead | ledger + `NOTES-mtp-series.md` |
| Tree-widen / rs-decouple / mmvf-16 | 43.7 tok/s (+crash variant) — dead | ledger `wider-tree-rs-decouple-13nodes-already-lost` |
| q6k merge-scale-chains | **inapplicable** — model is Q4_K_M, zero Q6_K tensors | ledger `q6k-merge-scale-chains-not-applicable-qwen38` |
| l2_norm → GDN fold | **already landed** — it IS the +7.2% "gated_delta_net normalises q and k in register" (#2 kernel) | source: `gated_delta_net.cu` `L2_FOLD` template exists + board #2 |
| DFlash2 engine onto pinned base | **blocked** — requires refactor-window backport past 2b63 | ledger `dflash2-gguf-needs-full-refactor-backport`, `DFLASH2-STATUS.md` |

## Staged, ready artifact (fires if a slot/pin clears)
- `/home/ghost/dflash2full/build/bin/llama-server` (DFlash2 v0.1.2-dev, draft-dflash)
- `/home/ghost/models/Qwen3.8-27B-DFlash2-Q4_K_M.gguf` (1.14 GB)
- Bench: `/home/ghost/dfl-bench.sh`, logs `dfl-run*.log`
- Clean build trees at pinned base: `/home/ghost/rebase27`, `/home/ghost/draftprobe`

## Why no submission landed
None of the above beats the 47.6/31.9 leader. The competition disciplines: a
candidate that loses is correctly rejected. The frontier is genuine and
sustained; shared-GPU ownership prevents further builds without authorization.

## Conclusion
**47.6 tok/s (spec) / 31.9 tok/s (kernel) is the verified real frontier.**
Complete map; no hidden win remains discoverable with approved tooling.
