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

## Open levers

1. **MMQ tiles for TQ2_0** — prefill currently dequantizes experts to f16
   and runs GEMM per expert; real q8 tile kernels should take prefill much
   further (upstream mmq-load-tiles is per-type; ternary loads are trivial).
2. The launch-bound R9700 playbook (grouped launches, q8_1 dedupe, topk
   sorted-list) has not been ported to this fork yet — the tree is newer
   than b10237, so the laguna patches need rebasing, but the wins are the
   same class. Decode at 195 tok/s now looks like Laguna's profile did.
3. Sliding-window attention (3 of 4 layers, window 512) — untouched.
