# nemotron-3.5-lightning-gguf-gb10cuda-v1

Patch series for Nemotron 3.5 Lightning 30B-A3B (Q4_K_M) on the DGX Spark GB10,
via llama.cpp CUDA. Empty today — newly commissioned.

**This track does not build the platform default engine.** It pins llama.cpp at
`7a20b417f452` rather than `b10237`, for the same reason its R9700 twin does:
the ggml-org GGUF was converted after upstream reworked the `nemotron_h_moe`
tensor layout, and the older tree refuses it with

```
done_getting_tensors: wrong number of tensors; expected 417, got 408
```

`curl -s "https://frontier.fast/api/recipe?track=nemotron-3.5-lightning-gguf-gb10cuda-v1"`
prints the exact clone and checkout.

## Why there is no vLLM track for this model

There was going to be. It cannot work: with NVIDIA's own recommended DGX Spark
configuration the model serves, but five identical greedy prompts return five
different completions, and adding `VLLM_BATCH_INVARIANT=1` — which this
platform requires — refuses to start with `CutlassNvFp4LinearKernel does not
support W4A16`. A deterministic engine that will not load it, or a loading
engine that is not reproducible. Neither can pass the correctness gate.

## What is different here

First hybrid **Mamba2 + attention** model on the platform. Every other
llama.cpp track is pure attention, so the SSM/state layers are unprofiled
surface, as are the 128-expert MoE matvecs with 6 active. Its R9700 twin runs
the identical model and quantization, so this pair is the platform's
portability probe for hybrid kernels across two vendors — check the sibling's
findings before assuming a win transfers.
