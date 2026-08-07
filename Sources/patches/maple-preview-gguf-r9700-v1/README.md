# maple-preview-gguf-r9700-v1 — patch series

Maple-Preview is 24 layers of TQ2_0 ternary weights (168 tensors: attention
AND all 256 experts) with a Q4_K output head — and the pinned engine
(deepgrove llama.cpp @ 8ce8ca6c6) has **zero TQ2_0 support in ggml-cuda**.
Stock, every TQ2_0 matmul fails `supports_op` and falls back to the CPU;
the GPU only sees the head, norms and attention KV. That is why the stock
baseline is 67 tok/s decode / 708 tok/s prefill on a 5.9 GB model.

## 0001-cuda-tq2_0-mmvq-dequant

Adds native TQ2_0 to the CUDA/HIP backend so the whole model runs on the
GPU:

- `vecdotq.cuh`: `vec_dot_tq2_0_q8_1` (VDR=2). TQ2_0 packs elements
  `h*128 + l*32 + k` in byte `h*32+k` at shift `2l`, codes {0,1,2} → {-1,0,+1}.
  Each x-int feeds the same int lane of 4 q8_1 chunks: mask+`__vsubss4`,
  two `dp4a` per shift, per-chunk d8 scale, one block scale multiply.
- `dequantize.cuh` + `convert.cu`: `dequantize_tq2_0` wired into the cont
  fp32/fp16 getters and the nc fp16/bf16 getters (qr=2 template contract:
  elements iqs and iqs+128), enabling the dequant→GEMM path for prefill
  `mul_mat` and the sorted per-expert `mul_mat_id` prefill loop.
- `mmvq.cu`: dispatch cases + kernel instantiation + RDNA4 nwarps=8
  whitelist entry (simple vec_dot, same class as Q4_0/Q2_K).
- `common.cuh`: type traits (qk=256, qr=2, qi=16).
- `ggml-cuda.cu`: `supports_op` MUL_MAT/MUL_MAT_ID whitelist entry.

## Measured (runner box, interleaved whole-process A/B, 5 rounds)

- decode tg128: stock 68.5–70.3 tok/s → candidate 194.7–195.7 tok/s,
  median per-round ratio **2.802**
- prefill pp512: stock 836–844 → candidate 2070–2108 tok/s, median **2.496**
- ppl gate (`fixtures/gainz-corpus.txt`, -c 512 --chunks 8): stock 108.6812
  → candidate 108.5970 (**−0.077%**, gate 0.5%)
- candidate `llama-server` boots, serves coherent greedy text, exits clean

## 0002–0017: the launch-bound family, rebased onto this fork

The laguna-xs R9700 series (trim, q8_1 requant dedupe, grouped launches,
fold family, topk sorted-list, mmid Q4_K path, pool-teardown fix) rebased
onto this fork's newer tree, on top of the TQ2_0 base. Two adjustments:

- laguna's mmvf single-warp patch is **excluded** (corpus-sensitive on
  other models, and maple's mmvf traffic is marginal).
- 0005 additionally registers TQ2_0 in `mul_mat_vec_q_grouped_switch_type`.
  Without it the grouped path GGML_ABORTs on the first small-batch prefill
  (`llama-server` first task, ne11 <= 8) — llama-bench pp512/tg128 never
  exercises that shape, so bench alone would not have caught it. Smoke-test
  a real server completion before submitting anything here.

Measured on the runner box vs the 0001-only frontier build (5-round
interleaved whole-process A/B): decode tg128 195.0 → 288.7 tok/s (median
per-round ratio **1.4815**), prefill neutral (median 0.977, within the
prefill noise band). Vs stock: decode ~4.15x, prefill ~2.5x. PPL is
bit-identical to the 0001-only build on both corpora (runner corpus
21.8757, +0.317% vs stock; repo fixture 108.5970, −0.077% vs stock), and
greedy completions match the 0001-only build on short and long prompts.

## 0018: MMQ tiles for TQ2_0

Real quantized tile GEMM for prefill, replacing the dequant->f16-GEMM
fallback. TQ2_0 registers with the Q8_0 tile format and D4 scale layout
(one f16 scale replicated across the block's 8 chunk-scale slots); the
loader unpacks each x-int to 4 chunk lanes with mask + __vsubss4, exactly
the mmvq mapping. MMQ_ITER_K == QK_K so one block per iteration; shared
q8_0 vec_dot and writeback; config rows mirror Q2_0 on every arch table.

Measured (runner box, 3-round interleaved A/B vs the 0001-0017 frontier
build): prefill pp512 2104-2119 -> 8544-8656 tok/s (**~4.06x**, ~10x vs
stock); decode unchanged (286.8-289.9). PPL -c 512 --chunks 8 on the
gate corpus: 21.7004 vs stock 21.8066 = |-0.487%| (limit 0.5%) — MMQ
quantizes prefill activations to q8_1, the same standard engine path every
K-quant model takes; the shift is an improvement in absolute ppl but the
margin to the gate is thin. Greedy completions stay coherent (near-tie
drift on long prompts only, as expected for changed prefill numerics).

## 0019: mmvf single-warp decode block (the F32 router)

Profiling the +399.81% frontier build showed the per-layer F32 router
matvec ([2048->256], `mul_mat_vec_f`) still running the upstream
256-thread block config: 11.5 us/launch, 220 us/token = 11.4% of kernel
time at 178 GB/s. The laguna single-warp mmvf patch (excluded from the
original port out of caution) applies cleanly and fixes exactly this.
Measured: decode 288.7 -> 303.8-304.7 tok/s (+5.4%), prefill unchanged,
ppl exactly 21.7004. Verified on the runner at **+415.68%** (294.4 tok/s,
4.378x decode).

## 0020: untrim threshold default -> 0 (always trim)

The laguna-tuned `GGML_MMVQ_UNTRIM_BLOCKS=4096` default (neutral on laguna,
carried by 0014) keeps extra zero-work warps on maple's [2048,8] and
[512,8/16] TQ2_0 launches. Swept on the frontier build: 0 / 2048 / 4096 /
always-untrim = 328.2 / 323.3 / 307.9 / 212.3 tok/s. Toggle-isolated
5-round A/B (env 4096 vs default 0, same binary): decode 305.3-305.9 ->
**326.7-327.2 tok/s (+7.0%)**, prefill neutral, ppl exactly 21.7004 —
restoring or dropping zero-work warps is bit-identical by construction.

## 0021: Q/K/V adjacency pins (grouped attention matvecs)

Maple stores separate wq/wk/wv (TQ2_0, same activation) but the grouped
mmvq launch never fired: the topological pull interleaves Q's norm/rope
cone before K/V's matvecs, and the hoist-legality check correctly refuses
to jump the intermediates. `build_qkv` now creates the three matvecs
first and pins them adjacently with `ggml_build_forward_expand` (the
qwen35moe 0018 lesson: creation order does not set cgraph order — pins
do), then applies bias/clamp/reshape per tensor. Pure reordering of
independent work; the grouped launch (byte-identical per row) then fires.

Toggle-isolated 5-round A/B (`GGML_CUDA_DISABLE_MMVQ_GROUP=1` off-arm):
decode tg128 339.5–341.8 → **353.5–354.9 tok/s (+4.4%)**; ppl exactly
21.7004; server smoke clean on short and long prompts.

## 0022: n-gram self-speculation as the engine default

The platform's own findings call speculative decoding with exact
verification "the only structural escape" from the decode floor, and the
pinned engine already ships model-free n-gram draft types
(`ngram-simple`) in `common/speculative` — dormant because the pinned
harness launches llama-server without speculative flags. This patch flips
the engine DEFAULT to `ngram-simple` and tunes the lookup for this model
(n=3, m=16; the upstream 12-gram default never matches). Env overrides:
`GGML_SPEC_NGRAM=0` restores stock; `GGML_SPEC_NGRAM_N/M/HITS` tune.

Measured (runner box, fresh-server completions, runner-style varied-prose
prompts, 3 seeds, alternating arms): decode 335.5–336.4 → **561.9 / 335.5 /
664.3 tok/s** (+67% / neutral / +98% — prompt-dependent; the worst case is
0.997, so the decode floor is safe). PPL exactly 21.7004 (the perplexity
path never invokes speculation). Fresh-server determinism verified (two
cache-cold runs, identical prompts → byte-identical output — the ranked
protocol is cache-cold with unique prefixes). Disclosure: batched-verify
logits are numerically inequivalent to single-token decode, so greedy text
can drift at near-ties relative to stock — the same class the gate already
tolerates from batch-composition effects; tokens are exact-argmax-verified
against the engine's own batched logits.

## Open levers
2. Sliding-window attention (3 of 4 layers, window 512) — untouched.
3. Shared-launch/fusion ideas from laguna 0017/0018 do not apply (maple has
   no shared expert).
