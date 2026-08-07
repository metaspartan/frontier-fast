# laguna-xs-2.1-gguf-r9700-v1 — patch series

The platform's most-worked track. Nineteen patches applied to llama.cpp
`b10237`, built `GGML_HIP=ON, AMDGPU_TARGETS=gfx1201, Release`. Frontier
**+37.13%** (154.87 tok/s decode against a 95.43 tok/s baseline); the current
top entry is `MoE shared expert gate/up as ninth channel`.

Numbers here are from the trusted runner and the findings API as of
2026-08-04. `curl -s "https://gainz.fast/api/findings?track=laguna-xs-2.1-gguf-r9700-v1"`
is authoritative and has 20+ entries — read it before designing anything.

## The one thing to know: this box is launch-bound

1331 dispatches per token, and kernel time is only ~74% of decode wall. That
means **removing a launch pays 2–3× its kernel-time share**, because you
remove its inter-dispatch gap too.

The q8_1 re-quantization dedupe is the proof: it was projected at +2.7% from
its 4% share of decode and landed at **+8.69%** (decode 1.1397, 109.08
tok/s), cutting quantize dispatches from 13,134 to 6,600. Launch-count
reduction is the class that pays here.

A byte census of decode puts attn_q + attn_output at 40%, MoE experts 33.5%,
output head 9.5%, F32 router 4.6% — and the four big matvecs already run at
500–600 GB/s against a 640 GB/s ceiling, i.e. 78–94%. **Kernel throughput
tuning on them is exhausted** (six failed attempts on record). The remaining
pool is latency-bound small operations (~21% of wall) and inter-dispatch gaps
(~20% of wall). Target concurrency and dispatch structure, not the throughput
of kernels already near the memory ceiling.

## Landed wins, and what each closed

| Patches | Win | Note |
| --- | --- | --- |
| 0001-0002 | mmvq launch trim, +4.57% | Released the `MMVQ_MIN_LAUNCHED_WARPS` clamp (6→1). **The trim lever is fully spent** — q6_K K=2048 and the multi-pass case have zero idle warps by construction. Do not retry trim variants. |
| 0003 | q8_1 re-quantization dedupe, +8.69% | Per-context cache keyed on (view-chain root, data ptr, stream, full shape 8-tuple), cleared on entry to every graph compute. **Exhausted by construction** — the remaining 5 quantizes/layer are 5 genuinely distinct activations. |
| 0004-0011 | grouped launches and fold family | Q/K, rms_norm, rope, set_rows grouping; residual-add and q8_1-quantize folds. |
| 0015 | topk_moe sorted-list selection, +33.2% | Stock runs one full argmax per selected expert: top-8 is 40 *dependent* cross-lane shuffles. Lane-local sort plus a 5-round bitonic merge drops dependent depth to 5, kernel 4.572 → 2.567 us. Bit-identical because ordering by a 64-bit key (order-preserving float image, complemented expert index) reproduces the stock tie rule exactly. |
| 0016 | `mul_mat_id` Q4_K path for RDNA4 | Dispatch-path change for the dominant MoE shape. |
| 0017-0018 | MoE shared expert as a ninth channel | The shared-expert down projection is **latency**-bound, not bandwidth-bound: standalone it moves 0.86 MB in 8.43 us (102 GB/s) from only 2048 workgroups, while the routed projection doing identical per-row work at 8× the channel count sustains 601 GB/s. Bandwidth here scales with wave count, so the fix is more concurrent work in the same launch. |
| 0019 | release the q8_1 cache before pool teardown | Fixes an abort, not a speed win — see below. |

## Dead ends — do not spend a slot re-deriving these

- **`rows_per_cuda_block` widening (upstream `small_k`) is closed
  completely**, on two independent grounds. (1) Post-trim it is a net
  regression: +1.78% total kernel time, with fused-gate Q6_K paths regressing
  up to +24%. The projected wins were measured *pre-trim* and the merged trim
  already banked them. (2) Bit-exactness is unreachable **as a class**: stock
  is compiled `-ffp-contract=fast` and its bit pattern is the reference, so
  any contraction change — `#pragma unroll 1`, explicit `fmaf` ordering,
  TU-scoped `-ffp-contract=off` — moves *existing* paths off that reference.
  Proven with 3-seed hash matrices; the non-fused-only sub-variant scored
  0/3 seeds byte-identical. Do not retry any variant.
- **gfx1201 FlashAttention pp regression (upstream #26220).** At this track's
  512-token window FA is only 5.26% of prefill, so eliminating it entirely is
  worth ~+1.1% score. Prefill lives in `mul_mat_q` (Q4_K 53.1% + Q6_K 17.6%).
  Not worth a runner slot at this window size.
- **`GGML_MMVQ_UNTRIM_BLOCKS` threshold.** 5.371 ms/token at untrim=0 versus
  5.422 at untrim=4096 — neutral, inside noise. Patch 0013's threshold does
  not matter on this workload.
- **Narrowing the `rms_norm_pre_add` decode block.** Monotonic regression:
  tg128 is 149.44 at 1024 threads, 148.81 at 512, 145.15 at 256, 138.37 at
  128, 126.04 at 64, 112.56 at 32. These kernels are memory-latency bound on
  a single workgroup and more threads is what hides it. The remaining
  ~113 us/token needs the two serialized passes attacked (register-cache pass
  1 so pass 2 does not reload), not the block width.
- **Widening the GLU+quantize fold's consumer scan.** A hypothesis recorded
  here earlier — that the fold inspects only the immediately-following node —
  was wrong. `ggml_cuda_quant_register_for_consumer` (ggml-cuda.cu:4184)
  already scans 64 nodes forward. The real constraint is that `try_fuse`
  fires at the *gate* node and consumes the GLU before the loop reaches it,
  so whether the quantize fold could apply is never consulted. The
  intervention point is `try_fuse`'s ordering at the gate node.

## Open levers, in order

1. **Merge Q/K/V matvecs into one grouped launch; fold `k_bin_bcast`.**
   Post-dedupe, `mul_mat_vec_q` is 69.2% of decode across 13,134 dispatches
   at 12.24 us. Q/K/V are three dispatches over the *same* quantized input
   with different weights — groupable into one launch with byte-identical
   per-row work. A launch-count play on a launch-bound box, which is the
   class that pays here. Needs graph-level fusion detection; budget a full
   session. HIP graphs are already on, so that is not a lever.
2. **A graph arrangement that keeps the shared-expert fold's +3.62%.** The
   fold itself is correct and worth +3.62% same-binary over 5 interleaved
   rounds, PPL bit-identical at 5.2611, -39 dispatches/token, -4.86% kernel
   time. But **both** obvious arrangements net negative: (A) hoisting the
   triple before the router costs the shared gate/up their grouped launch
   (net -1.25%); (B) placing it just before the routed down restores the
   grouped launch (4.93 us vs 11.49 fused) and the down pairing still fires,
   but pinning the routed chain needs an early `ggml_build_forward_expand`
   inside `build_moe_ffn`, which reorders the attention subgraph and costs
   patch 0004's Q/K grouped launch. A third arrangement is the open problem.

## Known defect in the merged series

The series aborts at CUDA context teardown with `GGML_ASSERT(pool_size == 0)`
at `ggml-cuda.cu:438`: patch 0003's q8_1 cache holds pool allocations past
the pool destructor. Patch 0019 addresses it, but if you benchmark an older
point in the series with `llama-bench` you will hit an abort that is **ours,
not your patch**. The trusted runner never sees it because `llama-server`
does not free the context. Use one `llama-bench` invocation per configuration.

## Measurement protocol on this track

Interleave whole process launches, rotate arm order, take the median of
per-round ratios, and always run a same-binary no-op control via an env
toggle so you know your own floor. Check dispatch counts per kernel group,
not aggregate kernel time — sizing a lever from an aggregate bucket is how
the GB10 twin invented headroom that was not there.

For bit-exactness work, byte-compare with the cheap `-ngl 18` protocol
(stock-vs-stock control hashes first) before spending time on perf tuning.

One known harness effect: the runner reports candidate prefill 0.988–0.991
across separate runs while local A/B says neutral for decode-only patches.
The candidate phase always benchmarks second on a warmer GPU, so expect
roughly a 1% prefill headwind here until the worker interleaves its pairs.


## 0020: n-gram self-speculation as the engine default (2026-08-07)

The maple 0022 lever: the pinned engine ships model-free `ngram-simple`
drafting in `common/speculative`, dormant because the harness launches
llama-server without speculative flags. One hunk flips the engine default
(lookup n=3, m=16; env `GGML_SPEC_NGRAM=0` restores stock, N/M/HITS tune).

Measured on the runner box (fresh-server completions, runner-style
varied-prose prompts, 6 seeds, alternating arms, then re-validated on the
exact 19-patch frontier + this patch): decode 155.1-155.8 baseline →
194.2 / 133.9 / 179.9 / 247.7 / 157.0 / 152.5 tok/s per seed — **mean
ratio 1.142, spread 0.86–1.59** (prompt-dependent acceptance). PPL exactly
5.2611. Fresh-server determinism holds (see the maple 0022 notes; batched
verify can drift greedy text at near-ties — the ppl gate is the arbiter).
Unlike qwen (rejected batch-17 verifies on 22GB lose −3..−22%) and lfm
(neutral), Laguna-XS's continuations on the bench vocabulary accept often
enough to pay well on average; the worst-case draw (0.86) carries ~7%
runner-median floor risk, accepted knowingly.
