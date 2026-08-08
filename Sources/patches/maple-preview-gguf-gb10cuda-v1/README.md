# maple-preview-gguf-gb10cuda-v1 — patch series

Two things live in this file: the **numerics investigation** that closed the
GPU-offload lever on this track for good, and the **series** that opens it on
the lever that is actually available.

Read the numerics section before proposing any TQ2_0 CUDA kernel here. It is
not a "we did not try hard enough" dead end; it is a property of the model.

---

## 0001-generation-thread-pool

`common_cpu_get_num_generate()` caps the **generation** thread pool at 7 on
aarch64 Linux while the **batch** pool keeps the full math-core count.

The stock reference on this track computes all seven TQ2_0 matmuls per layer
on the ARM CPU (see the scheduler census below), so the ranked run is
CPU-bound in the decode phase. `common_cpu_get_num_math()` returns 20 on a
GB10 — the x86 path drops efficiency cores explicitly ("efficiency cores harm
lockstep threading") but the aarch64 path falls through to the physical core
count, handing lockstep barrier work to 10 Cortex-A725s that cannot keep pace
with the 10 Cortex-X925s. On top of that, generation is a matvec regime that
saturates memory long before it saturates cores, with a threadpool barrier
after every op, so past the saturation point each extra thread adds barrier
latency and no bandwidth.

`llama-bench` tg128 tok/s over the thread sweep (box parked, SM 2405 MHz):

| threads | 2 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 12 | 20 |
|---|---|---|---|---|---|---|---|---|---|---|
| tg128 | 63.4 | 96.0 | 101.6 | 104.2 | **106.4** | 100.1 | 101.3 | 94.5 | 72.5 | 59.5 |

Prefill runs the other way over the same sweep (pp512: 222 tok/s at 4 → 548 at
20), which is why the cap is applied to the generation pool only. Lowering
both pools trips the 0.95 prefill floor (t=10 measured 0.884, t=8 0.735).

### Measured — runner-shaped server A/B

Candidate built from the patch's own parent commit (`8ce8ca6c6`) against a
stock build of that same commit with identical cmake flags. Both arms run the
runner's exact server flags with **no** `-t`, 5 interleaved whole-process
launches with rotating arm order, median of 5 requests per launch after 2
warmups, box parked.

| round | prefill stock→cand | decode stock→cand |
|---|---|---|
| 1 | 485.42 → 484.18 | 53.79 → 100.87 |
| 2 | 479.81 → 486.63 | 53.37 → 104.38 |
| 3 | 486.33 → 481.91 | 55.33 → 108.68 |
| 4 | 482.52 → 481.78 | 53.26 → 99.92 |
| 5 | 484.54 → 480.52 | 48.55 → 96.73 |

Median per-round ratios: **decode 1.9556**, prefill 0.9975, TTFT 1.0045.
Projected score ≈ 1.9556^0.65 × 0.9975^0.20 × 1.0045^0.15 ≈ **1.546**.

Stock decode of 48.6–55.3 tok/s brackets the pinned 54.03 calibration, so the
baseline arm is healthy. `n_threads` verified from `system_info`: stock
`20 (n_threads_batch = 20)`, candidate `7 (n_threads_batch = 20)`.

### Gate

Perplexity is **bit-identical**, not merely close:

```
llama-perplexity -m maple-preview-TQ2_0-head-Q4_K.gguf -f ~/gainz-ppl-corpus.txt -ngl 99 -c 512 --chunks 8
stock 22.4217   candidate 22.4217   (0.000%, limit 0.1%)
```

That is guaranteed by construction and was checked directly: thread count does
not enter any dot product's arithmetic, only which thread computes which output
chunk. Measured invariance on the stock binary — `-t 20`, `-t 10`, `-t 4` all
return 22.4217 on the CUDA build and 22.9596 on a CPU-only build. On a track
where an f32 ulp is worth 1.5% perplexity (below), that invariance is the whole
reason this lever is available at all.

---

## The numerics investigation: why no TQ2_0 CUDA kernel can pass this gate

An earlier session measured the R9700 series' TQ2_0 CUDA port here — ~9x decode,
~15x prefill, and perplexity 22.4217 → 21.7312 (−3.08%) against a 0.1% gate. It
attributed the shift to activation-quantisation granularity and left the lever
open pending a CPU-parity kernel. **The attribution was right and the conclusion
was still wrong**: matching the quantisation is not sufficient, because the model
amplifies *any* float perturbation far past the gate.

### 1. What stock actually computes

`GGML_SCHED_DEBUG=2 ... -v`, 954-node graph, node→backend census:

| backend | ops |
|---|---|
| CUDA0 | ADD 216, MUL 121, RMS_NORM 97, SET_ROWS 48, ROPE 36, GET_ROWS 26, MUL_MAT 25, FLASH_ATTN 24, SOFT_MAX 24, ARGSORT 24, SUM_ROWS 24, CLAMP 24, DIV 24 |
| CPU | MUL_MAT 96, MUL_MAT_ID 72, CLAMP 48, SWIGLU 24, GET_ROWS 1 |

The CPU nodes are exactly `Qcur`/`Kcur`/`Vcur`/`attn_out` (×24 layers) and
`ffn_moe_gate`/`ffn_moe_up`/`ffn_moe_down` (×24) — the seven TQ2_0 matmuls per
layer, nothing else. Attention, flash-attention, the router (`ffn_moe_logits`)
and the **Q4_K output head (`result_output`) already run on CUDA in stock.** The
earlier note that stock "collapses the whole graph to the ARM CPU" is not right;
only the ternary matmuls fall back, and `-ngl 0` matching `-ngl 99` (22.4228 vs
22.4217) is a coincidence of the two configurations, not evidence of an all-CPU
graph. So the only numerics a TQ2_0 CUDA kernel would change are those seven
matmuls — a much smaller target than assumed, which is what made it worth
re-testing.

### 2. The quantisation term is real, and 5% wide

Instrument: CPU-only builds (`-DGGML_CUDA=OFF -DGGML_CPU_REPACK=OFF`, verified
that repack changes nothing here) with `type_traits_cpu[TQ2_0].vec_dot_type`
switched to F32, so the activations arrive at the dot unquantised and the dot
itself decides how to treat them. Reference 22.9596 in this configuration.

| activations fed to the ternary dot | PPL | vs reference |
|---|---|---|
| Q8_K, one int8 scale per 256 (**stock**) | 22.9596 | — |
| re-implemented per-256 int8, self-check | **22.9596** | 0.000% |
| int8, one scale per 32 (what CUDA q8_1 does) | 22.1107 | −3.70% |
| f32, no activation quantisation at all | 21.7843 | −5.12% |

The self-check row matters: an independent re-implementation of
`quantize_row_q8_K_ref` inside the dot reproduces stock to the last digit, so
the instrument is exact. The mechanism is confirmed — coarse per-256 int8
activations cost this model ~5% of perplexity, the CPU pays that cost, any GPU
path does not, and the sign and magnitude match the −3.08% measured for the
full GPU port. So far the "replicate Q8_K on the GPU" plan looks sound.

### 3. The term that kills it: one f32 ulp is worth 1.5% perplexity

Same build, same Q8_K activations, same exact-integer ternary dot. The only
difference between these arms is the **order of the f32 accumulation** of the
per-block partial products:

| implementation of the identical dot product | PPL | vs stock |
|---|---|---|
| stock ARM NEON kernel | 22.9596 | — |
| generic scalar, sequential accumulate | 22.6118 | **−1.51%** |
| generic scalar, 4 accumulators combined pairwise | 22.8347 | **−0.54%** |

These are not different algorithms. The NEON kernel sums `Σ code·y` with
`vdotq_s32` and subtracts `Σ bsums`; the scalar loop sums `Σ (code−1)·y`
directly. Both integer paths are exact and provably equal, both compute
`d = d_x·d_y` as one f32 multiply, both do `sumf += sumi·d` once per 256-block.

Measured directly rather than argued: an arm that computes both and reports the
deviation gives, over **176.7 million dot products**,

```
TQ2PROBE dots=176717870 differing=167099678 maxrel=3.391e+01 meanrel=1.737e-07
```

— mean relative deviation **1.7e-7**, i.e. one f32 ulp, on 94.6% of dots. (The
`maxrel` outlier is a dot whose true value is ~0; relative error there is
meaningless.) That arm returns the NEON value and reproduces 22.9596 exactly,
so the harness is confirmed to be measuring nothing but the arithmetic.

**One ulp of f32 rounding in the ternary matmuls → 1.5% perplexity.** An
amplification of about 10^5. The cause is structural: top-8-of-256 expert
routing over 24 layers means a 1e-7 perturbation of an attention or FFN output
reroutes an expert for some token, and 4096 corpus tokens are not enough to
average that away. This is the `equivalent-sum-is-not-safe` finding, an order
of magnitude worse.

Everything is deterministic — repeat runs and thread-count changes reproduce to
the last digit — so this is a real arithmetic sensitivity, not measurement noise.

### 4. Therefore

A CUDA TQ2_0 kernel would have to be **bit-identical** to
`ggml_vec_dot_tq2_0_q8_K` on ARM, not merely numerically equivalent. Matching
Q8_K activation quantisation exactly gets you into a ±1.5% band and no further;
where you land in that band is a lottery, and the gate is 0.1%. Bit-identity
would require replicating the NEON kernel's exact f32 accumulation sequence,
which forces one thread per output row with a serial k-loop — uncoalesced,
and slower than the ARM CPU it replaces. The ~2.4x decode / ~15x prefill
sitting behind this gate is not reachable.

Note this also explains the cross-vendor discrepancy that made the R9700 result
look transferable: R9700 stock reads 21.8066 and its port +0.317%, GB10 stock
reads 22.4217 and the same port −3.08%. Those are two draws from the same
lottery, not two measurements of the same effect.

### 5. What is left

The ranked run is CPU-bound by construction, so **CPU-side work on the ternary
path is this track's real lever** — but only changes that are bit-identical.
That rules out new NEON/SVE dot kernels (any different reduction shape is
another lottery draw) and rules in everything that leaves each dot's arithmetic
sequence alone: thread-pool shape (0001), affinity and barrier cost, `nrows > 1`
for activation reuse across output rows, `mul_mat_id` gather/scatter cost, and
prefetch/layout. Stock decode at 53 tok/s against ~100 GB/s of CPU-side
bandwidth for ~250 MB of per-token active weights says there is a lot of room
left in that lane.

Measured but not pursued: thread **affinity** pinned to the ten X925 cores
(`-C f83e0`) is not better than letting the scheduler place 7 threads
(101.3 vs 108.0 tok/s median).
