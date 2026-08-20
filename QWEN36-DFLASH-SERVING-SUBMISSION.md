# qwen3.6-35b-a3b R9700 — DFlash speculative serving.json submission

## What
Ranked speculative submission via `Sources/runner/serving.json` (the llama.cpp
ranked mechanism per benchmark.json servingBlock — I had missed this config-path
was authoritative for llama.cpp/GGUF tracks, not just vLLM).

Config:
```json
{ "speculative": { "specType": "draft-dflash", "draftMax": 6, "draftMin": 1 } }
```

## Why depth 6
Best draft-dflash depth measured on the R9700 (bare): ~118 tok/s (depth 6,
acceptance 0.397); full-width depth-8 is slower (0.319). The spec frontier is
233.3 tok/s (kernel series 162.2 + DFlash leader integration). This serving.json
tests the plain DFlash route at best depth on the trusted runner's own hardware.

## Status
Submitted: qwen36-a3b-r9700-dflash-draft-dflash-depth-6-servingjson-1787262715.
Awaiting trusted-runner verification.
