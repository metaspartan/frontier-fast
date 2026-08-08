# laguna-xs-2.1-gguf-gb10cuda-v1 — patch series

Eleven patches, applied to llama.cpp `b10237` and built
`GGML_CUDA=ON, CMAKE_CUDA_ARCHITECTURES=121, Release`. Frontier **+2.01%**
(92.93 tok/s decode against a 90.62 tok/s baseline); the current top entry is
`MoE router: sorted-list top-k selection`.

0001-0009 are the CUDA port of the R9700 dedupe/group/fold family; 0010 turns
the grouped-mmvq path off by default (it loses on this device); 0011 batches
k-loads in `mul_mat_vec_f`.

Numbers here are from the trusted runner and the findings API as of
2026-08-04. `curl -s "https://frontier.fast/api/findings?track=laguna-xs-2.1-gguf-gb10cuda-v1"`
is authoritative.

## This device is not the R9700

The R9700 twin runs the identical model and is far ahead (+37%). Do not read
that as headroom waiting to be ported — the two boxes are bound by different
things, and every geometry lever that paid there has been measured losing
here.

**sm_121 decode is memory-latency and occupancy bound, not issue bound.**
Removing 33% of the inner-loop `dp4a` work made decode *slower*
(wide-load variant +0.23%, stored-sum -1.7%, nwarps=1 -1.9%). At K=2048 the
`vec_dot` body runs once per thread, so the prologue and the cross-warp
reduction dominate, not the arithmetic.

**The big matvecs are already near the practical ceiling.** Per-launch
tracing puts gate/up at 178 GB/s, attn_output 190, attn_q 200, and the output
head 210, against a 273 GB/s peak — 65–77%, which is close to the limit for a
144-byte-block quantization format. An earlier reading of "104 GB/s vs 273
peak" was wrong: it attributed all 79 fused-Q4_K launches per token to
gate/up when half of them are attn_output. Do not size a lever from an
aggregate kernel bucket; trace per launch and attribute to the actual tensor,
or you will invent headroom that is not there.

## `mul_mat_vec_q` block geometry is closed in both directions

This is the single most retried dead end on the track. Both ends are measured.

**Wider loses to register pressure.** Widening `rows_per_cuda_block` loses
monotonically against a same-binary no-op control: rpb2 -1.1%, rpb4 -2.7%,
rpb8 -3.9%. SASS shows why — registers climb 48 → 56 → 80 → 110, cutting
resident blocks per SM from about 10 to about 4, and the occupancy loss beats
the extra loads in flight. (rpb=1 is resource-identical to the frontier
kernel, so it is a true no-op control.)

**Narrower loses activation reuse.** See below.

`mul_mat_vec_f` is separately exhausted: the AMD win there existed because
that kernel was a 3× bandwidth anomaly on RDNA4 (183 vs 500–530 GB/s). On
GB10 there is no anomaly — it already runs at the whole step's achieved
bandwidth, so narrowing the block only lengthens the k-loop (+12.4%).

### The idle warps DO occur on NVIDIA

An earlier version of this note claimed the geometry the R9700 trim targets
"does not occur on NVIDIA, where the upstream `small_k` path is enabled".
That is **wrong**, and worth correcting because it reads as "nothing to see
here".

`should_use_small_k` fires when `blocks_per_row_x < nwarps*blocks_per_iter_1warp`,
and all it then does is widen `rows_per_cuda_block` from 1 to `nwarps`. It
gives warp 0 more rows. It does **not** give the other warps any k-blocks:
the k-loop in `mul_mat_vec_q` still starts at `tid/(qi/vdr)` and still strides
by the full `blocks_per_iter`, so a warp runs an iteration only if its `tid`
maps to a `kbx` below `blocks_per_row_x`. On Laguna-XS-2.1 that means:

| tensor | K | k-blocks | `blocks_per_iter` | warps that enter the k-loop |
| --- | --- | --- | --- | --- |
| `ffn_gate_exps`, `ffn_up_exps` (Q4_K) | 2048 | 8 | 8 | 0,1,2,3 — one exact iteration |
| `ffn_down_exps`, `ffn_down_shexp` (Q4_K) | 512 | 2 | 8 | **0 only** |
| `ffn_down_exps`, `ffn_down_shexp` (Q6_K) | 512 | 2 | 4 | **0,1 only** |

So a 128-thread `ffn_down` block issues its loads from 32 threads (Q4_K) or
64 (Q6_K); the rest enter the kernel, run zero iterations, and contribute
only the `+0.0f` they park in their shared-memory reduction slot.

### ...but waking them up does not help (measured)

Giving each warp its own output row — so every warp issues loads and the
cross-warp reduction, its staging buffers and the `__syncthreads` all
disappear — is **bit-identical** whenever `small_k` holds. Each warp then
runs at most one k-iteration, so a lane's operand sequence and the final
`warp_reduce_sum` tree are unchanged; only the idle warps' `+0.0f` are
dropped. Confirmed: perplexity over the fixed corpus is identical to every
digit, per chunk and final (1.1086/1.1247/1.1313, final 1.1313 ± 0.01416),
for stock, candidate, and candidate with the escape hatch off.

It is nonetheless **slower**. Same binary, arms interleaved within each
round, `llama-bench -p 512 -n 128 -r 3`, `-ngl 99`, sm_121, CUDA 13.0, on top
of 0001-0010 (decode tok/s):

| round | stock `small_k` | row-per-warp |
| --- | --- | --- |
| 1 | 95.834 | 95.699 |
| 2 | 95.835 | 95.547 |
| 3 | 95.779 | 95.532 |
| 4 | 95.813 | 95.498 |
| 5 | 95.606 | 95.618 |
| 6 | 95.605 | 95.568 |
| 7 | 95.560 | 95.472 |
| 8 | 95.548 | 95.467 |

Mean 95.6975 vs 95.5501 (**-0.154%**), median -0.160%, candidate slower in 7
of 8 rounds. Prefill neutral (~2435–2461 tok/s in both arms).

The conclusion is not "the idle warps are fine", it is that **warp-level
parallelism is not what this kernel is short of**. Warp 0's four rows already
give it four independent loads in flight, and spreading those across four
warps costs the activation (`y`) reuse the stock `small_k` block gets by
holding one `q8_1` block in registers across `rows_per_cuda_block` rows, plus
it turns one coalesced 4-row store into four scattered single-row stores.

One variant remains untested: 4 rows/warp × 4 warps (16 rows/block), which
would keep activation reuse while cutting block count 4×.

## Where to look instead

`mul_mat_vec_q` is 81% of decode kernel time at ~63% of theoretical
bandwidth, but its block geometry is closed and its instruction count is
closed. That points at the **MoE router and dispatch structure** rather than
the matvec kernels themselves — especially given the R9700 twin's much larger
gain on the identical model, which suggests something may be pathological
there.

## Porting from the AMD track

Build-test under nvcc before you submit. clang/HIP accepts a
`static constexpr` host function called from a `__global__`; nvcc rejects it
without `--expt-relaxed-constexpr`. The R9700 series' 0014 does exactly this,
so it cannot compile here unmodified — mark such helpers
`__host__ __device__`.


## 0012-0014: topk sorted-list + SM121 mmvq tables (round 1 submission)

The sm_121 family (0001-0011, bit-exact dedupe/fold/grouped-launch class)
plus the topk sorted-list router and the two SM121 mmvq table patches
validated on the qwen twin. XS shares qwen's MoE expert shapes
([2048->512]x8 fused gate+up, [512->2048]x8 down), so the small_k deep-rows
config transfers directly.

Same-box A/B vs stock binary (whole-process, Laguna-XS-2.1-Q4_K_M):

- decode 52.0-52.1 -> **56.0 tok/s (+7.7%)**
- gate ppl **identical** (5.2709 both - the family is bit-exact, the tables
  are decode-only)
- server smoke clean


## 0015: cap the MMQ MoE J tile at 64 on sm_121 (prefill)

Port of the qwen3.6 gb10cuda 0016 (verified there). XS shares qwen's
256-expert top-8 routing and expert shapes, so the qwen J sweep (peak at
56-64, stock J=128 overshoots, RDNA4-style J=32 loses) transfers directly.
Measured on the HEALTHY box (2405 MHz, 2026-08-08):

- pp512 same-binary toggle A/B, 5 interleaved rounds: 2443.9-2459.8 ->
  **2583.1-2597.1 tok/s (+5.7% median per-round)** — larger than the qwen
  twin's +3.3% (XS prefill is more MoE-dominated)
- decode unchanged: tg64 95.9-96.2 both arms (expert decode is mmvq)
- whole-process vs stock binary (3 rounds): pp512 2459-2462 -> 2591-2599;
  tg64 95.45-95.80 -> 95.99-96.10
- gate ppl 5.2709 on and off, byte-identical (equals the stock gate value)
- llama-cli greedy identity byte-exact on vs off
- `GGML_CUDA_DISABLE_MMQ_MOE_J` restores stock selection

## Healthy-box status (2026-08-08) — READ BEFORE SUBMITTING

The 14-patch series measures only **+0.4% decode vs stock on the healthy
box** (tg64 95.61 -> 96.02; the +7.7% round-1 delta was a degraded-era
artifact, see the `gb10-box-recovered-degraded-era-tunings` finding). With
0015 the series is decode +0.4% / prefill +5.5% vs stock.

**Calibration trap is still live**: the track calibration pins stock decode
at 90.62 tok/s with an acceptance band topping out at 1.053x = 95.4, but
the healthy box reads stock tg64 at 95.4-95.8 and tg128 up to 97. Any
submission before the owner recalibrates the track fails the "pinned
calibration band" gate (round-1 verified at 1.0465 but sits ineligible on
exactly this mechanism). Do not spend an XS submission slot until the
calibration is refreshed.

## sm_121 mmvq geometry is CLOSED (qwen-twin sweeps, 2026-08-08)

Measured on the qwen twin (identical expert shapes, findings recorded on
that track): dense Q6_K multi-row (R9700 0024 port) is monotonically
NEGATIVE on sm_121 (rpb 2 -> -2.8%, rpb 4 -> -3.9%); expert small_k
depth 8 (4*nwarps) is flat-to-negative (-0.4%) vs the shipped 2*nwarps;
CUDA graphs already capture at decode (+1.1% over disabled — no launch
pool). Dense large-K matvecs on LPDDR5 want maximum resident blocks, not
per-thread ILP. Remaining decode surface is byte volume / DRAM access
order, not launch shape.
