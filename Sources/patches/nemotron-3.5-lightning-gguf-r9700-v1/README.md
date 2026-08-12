# nemotron-3.5-lightning-gguf-r9700-v1

Patch series for Nemotron 3.5 Lightning 30B-A3B (Q4_K_M) on the Radeon AI PRO
R9700. Empty today — the track is newly commissioned and its reference row is
the pinned tree measured against itself.

**This track does not build the platform default engine.** It pins llama.cpp
at `7a20b417f452` (10 August, "model: add MTP support for Nemotron model")
rather than `b10237`. The ggml-org GGUF was converted after upstream reworked
the `nemotron_h_moe` tensor layout, and the older tree refuses it outright:

```
done_getting_tensors: wrong number of tensors; expected 417, got 408
```

Rebase your patches on that commit, not on `b10237`. `curl -s
"https://frontier.fast/api/recipe?track=nemotron-3.5-lightning-gguf-r9700-v1"`
prints the exact clone and checkout.

## What is different here

This is the platform's first hybrid **Mamba2 + attention** model. Every other
llama.cpp track is pure attention, so the SSM/state layers are kernel surface
nobody here has profiled yet — as are the 128-expert MoE matvecs with 6 active.
Measured at commissioning: 24.6 GB of the 34.2 GB VRAM, ~128 tok/s decode.
