# qwen3.8-27b-gguf-r9700-v1 — patch series

The series applies to vanilla llama.cpp `2b63e0610bbc2be990ae1360d5256efcdc3f9efb`
in filename order:

- `0001` — 5-warp workgroups for batch-1 K-quant matvec (kernel board, +3.31%).

## The model

Qwen3.8-27B Q4_K_M (`unsloth/Qwen3.8-27B-GGUF`), a **dense** 27B with a hybrid
attention stack: 64 trunk layers, of which **48 are linear-attention and 16 are
full-attention** — one full layer every fourth.

Two things follow from that, and they are what make this track different from the
MoE tracks next to it:

1. **Dense means every weight is read every token.** There is no routing slack.
   Decode is weight-bandwidth bound from the first token, and a patch that wins
   here has to move fewer bytes or spend fewer launches — it cannot win by
   touching less of the model.
2. **The linear-attention layers carry recurrent state**, the way a Mamba stack
   does. On the sibling Nemotron tracks that state turned out to be a large,
   *uncounted* traffic term, and correcting for it moved a published ceiling by
   8.7%. It has since been censused here and it is **not** the large term: it is
   649,199,616 B/token, 4.0% of traffic, and it is served at ~727 GB/s from
   last-level cache rather than from DRAM.

## Engine

Stock pinned engine — no custom pin. The GGUF declares
`general.architecture = qwen35` and llama.cpp `2b63e0610` already implements
`LLM_ARCH_QWEN35`, so the standard build works unmodified.

## The MTP block

The file declares `block_count = 65` against the config's 64 hidden layers,
because `qwen35.nextn_predict_layers = 1` puts a multi-token-prediction block at
`blk.64` (15 tensors, including `nextn.eh_proj`). llama.cpp loads it but executes
it only when the context type is `LLAMA_CONTEXT_TYPE_MTP`, so during a ranked
decode its **424,699,392 params / 289,527,808 bytes never move**.

The track's published parameter and byte figures are the trunk only, for that
reason. If you are computing bytes-per-token, exclude `blk.64.*` — counting it
inflates your denominator and will make your kernel look better than it is.

## Where the token goes

Censused on the box, at the `0001` frontier (28.353 tok/s stock → 29.84 tok/s):

| term | value |
|---|---|
| decode token | 35.27 ms |
| GPU busy | 31.9 ms |
| dispatches per token | 1,743 |
| marginal dispatch cost | 2.08 µs |
| launch structure (1743 × 2.08 µs) | 3.63 ms — **10.3% of the token** |
| `mul_mat_vec_q` share | **79% of the token** |
| weight bytes moved | 16,091,091,776 B/token |
| linear-attention state | 649,199,616 B/token (4.0%), served at ~727 GB/s from cache |
| achievable streaming bandwidth | 639–640 GB/s (`t = 2.3 µs + bytes/640.2`) |

**Compute the launch ratio, do not inherit it.** It is
`0.01 x token_duration / dispatch_cost`, so on this 35.27 ms token **~170
dispatches buy 1% of decode** — not the ~20 that holds on the 6 ms Laguna and
Nemotron tracks on the same box. An earlier version of this file said 20, which
is wrong by 8.5x here and would have you reject every launch-structure idea.

`mul_mat_vec_q` moves 15,986,135,040 B at ~600 GB/s against 639–640 GB/s
achievable: it is already at **~94% of achievable streaming bandwidth**, so the
whole remaining matvec headroom is `0.79 x (1 - 600/640)` ≈ **4.7% of the
token**. The larger untouched pool is the 10.3% in launch structure.

## Measured dead ends — do not re-run these

All measured against the `0001` frontier as control, ABBA-counterbalanced whole
process boots, 2 warmups + 9 measured runs per boot, median of per-boot medians.
Each arm was proven distinct by md5-ing the `libggml-hip.so` listed in
`/proc/<pid>/maps` of the live server, not the 17 KB `llama-server` stub.

| arm | decode | prefill |
|---|---:|---:|
| **K-quant header coalesce** (one wide load for `dm`+`scales[12]`) | **−0.97%** | −1.8% |
| **Non-temporal weight loads** (`TH_LOAD_NT` on the Q4_K/Q5_K payload) | **−0.50%** | −0.3% |
| **2 rows per workgroup** for K-quant `ncols_dst == 1` | **−0.29%** | ~0 |

1. **Header coalesce.** `block_q4_K` / `block_q5_K` open with a contiguous
   16-byte `dm`+`scales[12]` header that stock fetches as 5 narrow 16-bit loads,
   redundantly across the 16-lane group. Replacing them with one wide load is
   real at the ISA level — Q4_K global loads per K-loop body drop 14 → 10
   (24 → 16 with fusion), Q5_K 16 → 12 (28 → 20), and the compiler emits
   `global_load_b96` + `b32`, so it needs no alignment stock did not already
   need. It still **loses**. On RDNA4 those narrow loads are cache hits that
   cost nothing the kernel is short of, while VGPRs rise 24 → 30 (35 → 41 with
   fusion) and the wide load makes all three dwords one `vmcnt` event instead of
   values consumable as they land. Note this is the same edit that measures
   **+0.87% on the GB10 CUDA twin**: it is a genuine backend split, not a port
   error. Also worth knowing: the arm's greedy output was **not** bit-identical
   to control despite identical arithmetic, because the rescheduled expression
   tree contracts FMAs differently.
2. **Non-temporal weight loads.** The mechanism looked right — 16 GB streamed
   once per token past a recurrent state small enough to live in cache — but the
   two payload loads of a Q4_K super-block (`q4[0]` and `q4[4]`) touch the *same*
   64-byte lines, so `nt` evicts what the next instruction wants. Output was
   bit-identical, as a cache hint should be. **Census warning:** the Q6_K half of
   this edit silently did nothing. `get_int_b2` assembles a 32-bit value from two
   16-bit loads because `sizeof(block_q6_K) == 210` leaves block starts only
   2-byte aligned; LLVM widens that pair back into one 32-bit load and **drops
   the non-temporal metadata**, so the Q6_K kernel compiled byte-identical to
   control. Read the ISA before believing this kind of hint landed.
3. **Two rows per workgroup.** Batch-1 K-quant matvec re-reads the whole
   quantized activation once per output row — 144 B of `block_q8_1` against
   144 B of `block_q4_K` per super-block, so activation and weight traffic are
   1:1 in L1/L2 even though the activation is a few KiB that never leaves cache.
   Halving that term also halves the workgroup count, and the lost parallelism
   costs more than the traffic saves. `test-backend-ops -o MUL_MAT` passes, so
   this is a performance answer and not a correctness one.

Taken together: at 94% of achievable bandwidth the batch-1 matvec on this box
appears to be at a local optimum, and three independent ways of touching it all
lose. Bring bytes or dispatches, or do not touch `mul_mat_vec_q`.

## Measuring here

**Do not compile while you measure.** A concurrent `cmake --build -j 24` on the
same host costs, same binary in both arms, ABBA over four boots:

| arm | decode | prefill | TTFT |
|---|---:|---:|---:|
| quiet host | 29.793 | 802.4 | 0.862 s |
| host compiling | 29.440 (**−1.18%**) | 769.1 (**−4.2%**) | 1.040 s (**+20.7%**) |

That is larger than most kernel patches on this track and it also inflates
within-boot spread from 0.24% to 1.6%. The cause is structural: 1,743 dispatches
per token puts the host submit thread on the critical path, so host CPU
contention shows up as GPU throughput.

**On a quiet box this track is much tighter than the 0.86% general floor.**
Thirteen whole-process boots of the same control binary across three sessions
gave 29.798 tok/s ± 0.028 (1 σ), a **0.095%** coefficient of variation. A 0.3%
effect is therefore resolvable here — which is how the three arms above could be
rejected rather than parked as noise.
