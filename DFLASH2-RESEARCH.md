# DFlash 2 research + qwen3.8-27b speculative path (2026-08-20)

## What DFlash / DFlash 2 is

**DFlash (v1)** — "Block Diffusion for Flash Speculative Decoding" (z-lab, arXiv
2602.06036). A *block diffusion* draft model that proposes a whole block of
draft tokens in a **single forward pass** (non-causal attention), conditioned on
context features/hidden states extracted from the target model. Replaces
autoregressive drafters like EAGLE-3, removing the sequential drafting bottle.
Verified in parallel by the target; only accepted tokens are kept.

**DFlash 2** — the successor, still one-pass. Adds a lightweight **path
selector** + **local convolution** to fix two remaining gaps: choosing the right
tokens and holding accuracy to the end of the block. Claims **>20% more verified
tokens per pass for ~1% added cycle latency**, i.e. **16–25% over DFlash v1**,
output provably unchanged. SGLang serves Qwen3.8-27B at **2.7–3.4×** autoregressive
at batch size 1 with the DFlash2 drafter (`incoai/Qwen3.8-27B-DFlash2`, also
`z-lab/Qwen3.8-27B-DFlash2`).

Ecosystem: runs in SGLang, vLLM, TensorRT-LLM, llama.cpp, ollama, oMLX. First-party
drafters from NVIDIA (Nemotron 3.5 Lightning), Meta (Muse Glimmer), Poolside
(Laguna), Xiaomi, etc.

The DFlash2 mechanism is at its core a **trained parallel MTP-style head** — which
is exactly what our custom MTP head work already is. "DFlash" and "MTP self-spec"
here are the same family; DFlash just ships a separately-trained drafter.

## This track's speculative contract (source of truth, live)

From `GET /api/tracks` for `qwen3.8-27b-gguf-r9700-v1`:

```json
"speculative": {
  "supported": true,
  "method": "mtp",
  "draftModel": null,
  "rankable": true,
  "board": "speculative"
}
```

- **method = mtp** → the ranked speculative route is **MTP self-speculation** (the
  model drafts its own continuations via its NextN head), plus optionally
  ngram-cache prompt-lookup and a separate speculator.
- **draftModel = null** → this llama.cpp track has **no pinned DFlash drafter**.
  "Weights are pinned: you name the TYPE (specType), never the weights." Since no
  DFlash weights are pinned, you cannot rank a DFlash **weight** here; the llama.cpp
  runner loads whatever draft is pinned for your track, which is none.
- `benchmark.json` documents the llama.cpp speculative serving block:
  `Sources/runner/serving.json: { "speculative": { "specType": "...", "draftMax": 1..16, "draftMin": 0..16 } }`
  and specTypes include `draft-dflash`, `draft-eagle3`, `draft-mtp`, `ngram-*`.

**Conclusion:** the way to "get a DFlash-style submission up" on this track is the
**trained parallel MTP draft head** (a DFlash2-class drafter as custom MTP self-spec
patches), NOT a pinned DFlash checkpoint — because `draftModel is null` here and
weights aren't loadable by the runner.

## Current speculative board (authoritative, 2026-08-20)

| candidate | tok/s | delta |
|---|---|---|
| Greedy MTP draft via GPU argmax (chain series) | 47.6 | +49.46% |
| Double-buffered y tile at verify widths | 47.6 | +49.3% |
| Stream-k MMQ at speculative verify widths | 46.9 | +47.88% |
| MTP draft head projects Latin prefix of vocab | 45.0 | +43.9% |
| MMVQ/MMQ crossover + MTP draft depth 3 | 42.5 | +38.51% |
| MTP self-spec from model's own NextN head | 40.7 | +35.18% |

My last submission (`mtp-self-speculation-series...`) measured **45.0 tok/s
(+43.9%)** — below the 47.6 leader → rejected. Recorded dead on the ledger.

## The realistic next step (existing roadmap — don't rediscover)

The box's `PLAN-tree-rounds.md` + `~/treespec/NOTES.md` lay out the path past
47.6: **wide the speculative tree** (currently 7 nodes / depth 4 / 3 leaves /
8 verify positions) using the mmvf 16-column enabler, gated by a blocker:

> **Blocker — snapshot-plane memory scales catastrophically with tree size.**
> Plane count = (n_nodes+1); rs rows = n_seq_max × planes. A 15-node (~7-leaf)
> tree → 8 × 16 × 3.07MB × 48 ≈ 18.9 GB — OOM with a 16 GB model resident.
> Fix: **decouple rs cell count from n_seq_max** — branch seq ids need no
> recurrent cells; allocate with rs_size = n_parallel and make find_slot / seq_rm
> / seq_cp / seq_rs_select ignore ids >= size (~30 lines in
> llama-memory-recurrent.cpp + hybrid constructor sizing). Target 11–13 nodes,
> E ~3.9–4.1, ~+8–12% over chain.

Smoke test already confirms the tree arm **fires** (`draft tree enabled:
spec='2,2,1,1,0,0,1', 7 nodes, depth 4, 3 leaves`); the control measured 57.879
tok/s in that ad-hoc run (non-paired).

## SSH tooling

Switched R9700 access from `sshpass`/ssh-key to the local **winpass** Rust tool
so the password is never typed:

```
C:\Users\Carsen\Documents\Codex\2026-08-12\winpass\target\release\winpass.exe
$env:WINPASS_PASSWORD='carsen'
winpass --accept-new-host run ghost@192.168.1.226 "<cmd>"
```
Used for all R9700 access this session.
