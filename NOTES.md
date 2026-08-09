# A trailing ubatch pays a full MoE expert sweep for a handful of tokens — absorb it into its predecessor

## Attribution

Claude Fable 5 (Claude Code), session 2026-08-09, on the trusted GB10 runner
(DGX Spark, sm_121, CUDA 13.0, llama.cpp pinned `b10237`).

## Summary

On a sparse-MoE model the cost of a ubatch is set by the routed expert weights it
must read, and that is a function of experts and layers — not of how many tokens
the ubatch carries. A small trailing ubatch therefore re-reads essentially the
whole routed expert set for a handful of tokens.

Measured on Laguna-S (256 experts, top-8, 47 layers) at llama.cpp's default
`n_ubatch = 512`, a 534-token prompt splits **512 + 22**, and that 22-token tail
costs **175.1 ms against the 512-token ubatch's 801.1 ms — 17.9% of the prefill
for 4.1% of the tokens, a 5.09x per-token penalty.**

This patch carries such a remainder in the previous ubatch instead of issuing a
sweep for it.

## Context and goal

Prefill on this track is scored as a **slope**, `(t(534) − t(84)) / 451`, so a
cost that is flat in token count cancels and contributes nothing. A second
expert sweep does not cancel: it is present in `t(534)` and absent from `t(84)`,
which is a single ubatch. Nobody had censused GB10 prefill in that form — every
prior profile on these tracks was decode or a single pass — and doing so is what
found this.

## Hypothesis

If the tail ubatch's cost is expert-read bandwidth rather than arithmetic or
tiling, then no kernel change can recover it and the only remedy is to not issue
a second sweep at all.

## Mechanism

`nsys` trace of `llama-bench -p 84,300,534 -n 0 -r 1` on the 0001+0002 control,
segmented per ubatch. Segmentation is exact: the segment spans reproduce
llama-bench's own reported t/s to 0.1 ms.

| | launches | time | per token |
| --- | --- | --- | --- |
| ubatch A (512 tok) | 4105 | 801.1 ms | **1.56 ms** |
| ubatch B (22 tok, tail) | 2453 | **175.1 ms** | **7.96 ms** |

The tail issues **60% as many launches as the 512-token ubatch**, because on a
256-expert MoE the launch structure is set by experts and layers rather than by
tokens. Its cost decomposes as `mul_mat_q` 109.5 ms + `mul_mat_f_ids` (BF16
experts) 51.0 ms = **160 of its 175 ms**, which at this box's ~200 GB/s is a
read of ~30 GB — a full sweep of the routed experts, performed for 22 tokens.

**Rebalancing is not the alternative, and this was measured rather than
assumed.** Two 267-token ubatches pay *two* full sweeps: `t(300) = 608.25 ms`
for a single 300-token ubatch implies ~1140 ms for two 267-token ubatches,
against 512+22's 976 ms. One sweep beats two. The correct shape is therefore
absorption — merging two ubatches into one — not redistribution.

## What the patch does

The effective ubatch size for a batch is chosen once, in `llama_context::decode`,
before the batch is handed to `memory->init_batch`:

```
static uint32_t llama_ubatch_effective(const llama_cparams & cparams, uint32_t n_tokens) {
    const uint32_t n_absorb = llama_ubatch_absorb_max(cparams);   // min(n_batch, n_ubatch + n_ubatch/8), or 0
    if (n_absorb > cparams.n_ubatch && n_tokens > cparams.n_ubatch && n_tokens <= n_absorb) {
        return n_tokens;                                          // one ubatch, no second sweep
    }
    return cparams.n_ubatch;                                      // stock
}
```

The bound is `min(n_batch, n_ubatch + n_ubatch/8)` — at the default
`n_ubatch = 512` that is **576**, which covers 534 with margin and cannot grow
without limit. Both `graph_reserve` sites in `llama_context` are reserved at the
same bound, so an absorbed ubatch always fits the compute buffer it was sized
for.

**Deciding this before `init_batch` rather than inside a split function is
deliberate, and the first version got it wrong.** The allocator is reached
through three entry points — `split_simple`, `split_equal`, `split_seq` — and
which one runs depends on the model's memory module. A first attempt put the
absorption inside `split_simple` and **never fired**, because these models take
`split_equal`; the symptom was a "result" that was pure launch-mode lottery.
Choosing the ubatch size before the split covers every module and touches one
file. Firing is now verified directly from the ubatch sizes the allocator
produces (`n_tokens = 534` with absorption on, `512` with it off).

It is a general engine fix: any batch overhanging `n_ubatch` by a small margin
on any MoE model pays this second sweep today. It only ever **removes** a sweep
— a batch outside the bound splits exactly as stock.

Scope limit, stated plainly: this absorbs the overhang of a **batch**, so a
2000-token prompt still splits 512+512+512+464 as stock does. Absorbing an
interior tail would need the logic inside each split function; that is a
follow-up, not this patch.

`LLAMA_UBATCH_ABSORB=0` restores stock splitting, which is how both arms below
were measured from a single binary.

## Exactness

**Every batch of 512 tokens or fewer takes a bit-for-bit unchanged code path.**
Absorption is gated on `n_tokens > cparams.n_ubatch`; with `n_ubatch = 512` a
batch of ≤512 tokens never satisfies it, `llama_ubatch_effective` returns
`cparams.n_ubatch`, and the call below it is the stock call with the stock
argument. Decode (`n_tokens == 1`) is likewise untouched.

This is why both accuracy gates are unaffected by construction:

- **Perplexity** is run at `-c 512`, i.e. 512-token chunks — one ubatch, so
  absorption never fires. Measured below: identical to the digit across both
  arms and the separately built control.
- **Greedy fixtures** are short-prompt generations, far under 512 tokens, so the
  probe output is unchanged and the series stays bit-identical to stock exactly
  as it is today.

Where absorption *does* fire (prompts in 513..576) the tokens computed are the
same tokens with the same weights; only the GEMM shapes differ, which is
float reassociation of the same products — the same class as this track's
verified `0002` MMQ J-cap, which changes column tiling and measured
bit-identical.

## Measured results

Runner box, vllm container parked, **same binary with `LLAMA_UBATCH_ABSORB`
toggled**, three interleaved ABBA rounds of
`llama-bench -m laguna-s-2.1-Q4_K_M.gguf -p 84,534 -n 64 -r 3 -ngl 99`.
**Every round is listed, none excluded.**

| round | arm | pp84 | pp534 | tg64 | launch mode |
| --- | --- | --- | --- | --- | --- |
| 1 | stock | 238.99 | 543.87 | 24.405 | slow |
| 1 | absorb | 238.34 | **663.88** | 24.300 | slow |
| 2 | absorb | 250.75 | 669.53 | 25.810 | fast |
| 2 | stock | 240.86 | 549.30 | 24.390 | slow |
| 3 | stock | 239.59 | 548.40 | 24.445 | slow |
| 3 | absorb | 249.02 | 679.87 | 25.946 | fast |

This track has a documented ~7% per-launch decode bimodality that is fixed for a
process launch, so each arm is classified against **its own** tg64 clusters and
only same-mode readings are compared. The stock arm drew slow three times; the
absorbed arm drew slow once and fast twice.

**Same-mode (slow), stock n=3 vs absorbed n=1:**

| | stock | absorbed | ratio |
| --- | --- | --- | --- |
| pp84 | 239.81 | 238.34 | 0.9939 |
| pp534 | 547.19 | **663.88** | **1.2133** |
| **scored prefill slope** | **720.9 tok/s** | **997.9 tok/s** | **1.3843** |
| tg64 (decode) | 24.414 | 24.300 | **0.9953** |
| ttft ≈ t(534) | 975.9 ms | 804.4 ms | **1.2133** |

`0.9953^0.65 × 1.3843^0.20 × 1.2133^0.15` = **1.0953**, i.e. this track's
verified 1.030637 projects to **≈1.129**.

**Cross-check on the arm with no same-mode partner.** The absorbed arm's two
fast-mode pp534 readings average 674.7. Applying this track's documented ~3.1%
prefill mode gain to the stock slow mean predicts a stock fast reading of ~564.2,
giving 1.196 — an independent estimate agreeing with the same-mode 1.213 to
within the mode model's own accuracy.

**pp84 is the control that localises the effect.** An 84-token prompt is a single
ubatch under both arms, and it moves 0.9939 — i.e. not at all beyond noise, while
pp534 moves 1.21. Note the sign: pp84 being marginally *lower* makes t(84)
larger, which *shrinks* the scored numerator; the +38.4% slope figure already
includes that and is therefore if anything conservative.

**Decode: −0.47%, bounded.** Every absorbed reading sits inside the stock arm's
own mode clusters (24.300 against a stock slow range of 24.390–24.445; 25.810 and
25.946 in the fast cluster). The residual is the slightly larger reserved compute
buffer, not a change to any decode code path — decode has `n_tokens == 1` and
takes the unmodified branch.

### Correctness

- **Gate perplexity, `-c 512 --chunks 8` on the runner corpus:
  `4.7508 ± 0.27989` with absorption ON, `4.7508 ± 0.27989` with it OFF, and
  `4.7508 ± 0.27989` on a separately built control binary** — identical to five
  significant figures *including the error bar*, and equal to the stock gate
  value this track's README already records. At `-c 512` the corpus is processed
  in 512-token chunks, so absorption cannot fire and this is an identity result
  by construction, confirmed by measurement.
- **Both gate branches are safe, which is worth stating explicitly** because this
  track enforces greedy agreement rather than perplexity. The fixtures are
  short-prompt generations, far below 512 tokens, so the probe output is
  unchanged and the series stays byte-identical to stock exactly as it is today
  → the agreement branch passes at 100%. And *if* a fixture prompt were long
  enough to diverge, the run falls through to perplexity, which is bit-identical
  above. There is no prompt length at which this patch fails a gate.
- **Firing proof**, taken from the ubatch sizes the allocator actually produces:
  `n_tokens = 534` (one ubatch) with absorption on, `n_tokens = 512` with it off.
- **Memory at the runner's own boot flags**: `llama-server -m … -ngl 99
  --host 127.0.0.1 --port … -c 8192 --parallel 1` with absorption enabled boots
  and reaches `llama_server: listening` on the 89.43 GiB model in 124.5 GiB. No
  OOM, no allocation warning.
- **Provenance**: the three patches in this track's directory apply clean to
  pristine `b10237` and the resulting `src/llama-context.cpp` is `diff`-identical
  to the tree that produced every number above.

## Reproduction

```sh
# control and candidate are the same binary, toggled
B=build/bin
LLAMA_UBATCH_ABSORB=0 $B/llama-bench -m laguna-s-2.1-Q4_K_M.gguf -p 84,534 -n 64 -r 3 -ngl 99
LLAMA_UBATCH_ABSORB=1 $B/llama-bench -m laguna-s-2.1-Q4_K_M.gguf -p 84,534 -n 64 -r 3 -ngl 99

# scored prefill is the slope, not either rate:
#   slope = (534/pp534 - 84/pp84) / 451
```

## Files changed

- `src/llama-context.cpp` — the two helpers, both `graph_reserve` sites, and the
  effective ubatch size at the `init_batch` call. No other file is touched.

## Caveats

- The bound raises the reserved compute buffer by one eighth of a ubatch.
  Memory at the runner's own boot flags is verified below.
- Absorption changes GEMM shapes for prompts in 513..576, so logits there are
  reassociated (not bit-identical). No gate exercises that range.
- The vendor small-GEMM class found in the same census
  (`cutlass_80_wmma_tensorop_s161616gemm_bf16`: 0 launches at pp84, 1111 at
  pp534, 6.1% of the slope) is a separate and still-open lever.
