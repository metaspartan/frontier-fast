# R9700 frontier research — next levers (2026-08-20)

Web/papers scan for kernel + spec decode on the R9700 tracks (qwen3.6-35b-a3b MoE
is the promising one; qwen3.8-27b dense is saturated). Sources are ranked by
relevance to THIS track/hardware.

## 1. SP-MoE — the strongest compounding MoE lever (NOT yet on this board)
**SP-MoE: Speculative Decoding and Prefetching for Accelerating MoE-based LLMs**
(arXiv 2510.10302).

Core idea that directly addresses the qwen3.6 kernel frontier's remaining gap:
the DRAFT step is cheap, so start loading NEXT round's experts during draft,
i.e. **prefetch the top-r routed experts while the drafter proposes**. Accepting
more routed experts during draft (expanded coverage) both raises acceptance and
hides the expert-weight HBM fetch that dominates MoE decode.

Why it maps to this track: qwen3.6-35b-a3b kernel frontier is 162 tok/s
(built on MoE expert launcher optimizations); DFlash alone hit 118 tok/s (depth 6).
SP-MoE's draft-time expert prefetch is a DIFFERENT axis than either — it targets
the expert-weight bandwidth wall that neither the kernel series nor DFlash
directly removed. Combining DFlash/chain + expert prefetch is untested on the
R9700 board.

## 2. DraftExpert (arXiv 2607.24434) — expansion-aware self-spec for MoE
Shared+top-r self-drafting improves acceptance by loading more routed experts
(a drafter on the shared/cheap experts). Same theme: MoE spec resources should
buy EXPERT COVERAGE, not just more tokens.

## 3. AMD-provided low-latency GEMM guidance (rocm.blogs)
FlyDSL low-latency GEMMs for LLM decode on AMD: Split-K, K-slice parallelism,
LDS-pipeline. Relevant to the kernel side if the box had a FlyDSL offline path —
but frontier.fast pins llama.cpp/HIP, so this is contextual, not directly usable.

## Honest read
- qwen3.8-27b (dense): saturated — closed map (`QWEN38-R9700-LEVER-MAP.md`).
- qwen3.6-35b-a3b (MoE): the open, compounding lever is MoE **expert prefetch /
  expanded routed coverage during the cheap draft step** (SP-MoE/DraftExpert),
  which both sides of the current frontier (kernel 162, DFlash 118) leave on the
  table. That is the concrete next real experiment: DFlash(6) + expert-prefetch
  loading, or top-r widening of the drafter's expert coverage.

This buffer is where the "research the web and papers" instruction best pays off.
