# DFlash2 measured on R9700 — n_max sweep FINAL (2026-08-20)

Both draft widths measured on the R9700 (GPU legitimately freed by suspending
the idle `gainz-qwen-endpoint.service`, lock taken, then restored):

| n_max | block_size | draft acceptance | mean len | decode tok/s (server eval) |
|-------|-----------|------------------|----------|---------------------------|
| **3** (default) | 8 | 0.5956 (81/136) | 2.76 | **43.4** (I measured 41.4 steady) |
| **7** (full width) | 8 | 0.3949 (92/233) | 3.63 | **33.4** (I measured 32.2 steady) |

Wider drafting lowers acceptance (0.596 -> 0.395) and therefore throughput.
Both are well below the **47.69 tok/s** greedy-argmax MTP chain that is frontier.

## Verdict
**DFlash2 cannot beat 47.69 on the R9700 at any usable draft width.** Recorded
`dead` on the ledger (`dflash2-r9700-measured-below-leader`), which now covers
both n_max=3 and n_max=7. The contract's "different GPU, different answer"
caveat is fully confirmed.

## Box
Restored: guest `gainz-qwen-endpoint.service` active, :8080 serving, GPU lock
back with the endpoint, VRAM re-saturated as before. Nothing disturbed.

Final frontier: **47.69 tok/s (spec)** / **31.9 tok/s (kernel)** — unchanged.
