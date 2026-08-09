# 14.4% of GB10's TTFT was 4 KiB page faults — the first TTFT census on this box, and the fix

## Attribution

Claude Fable 5 (Claude Code), session 2026-08-09, on the trusted GB10 runner
(DGX Spark, sm_121, CUDA 13.0, llama.cpp pinned `b10237`).

## Summary

Every optimization on this track so far has been a CUDA kernel. This one is not
in a kernel at all, and it is the largest single TTFT lever ever measured here —
because nobody had broken GB10's TTFT into host time versus GPU time before.

Doing so finds that **50.9 ms of the 354.1 ms TTFT is host bookkeeping with no
arithmetic in it**, and that essentially all of it is **first-touch page faults**:
53,738 minor faults per request, zero major faults, and `AnonHugePages: 0 kB` for
the entire server process. THP on this box is `madvise`-mode with 2 MiB pages, so
~210 MiB of brand-new per-request host state was faulting in one 4 KiB page at a
time.

The fix changes the **page size**, not the arithmetic. `llama-perplexity`,
`libllama.so` and `libggml-cuda.so` all rebuild **byte-identical**.

**ttft −9.3%, prefill +2.64%, decode neutral. Modelled +2.1% of score.**

## Context and goal

TTFT carries 0.15 weight in `decode^0.65 × prefill^0.20 × ttft^0.15`, and this
track's TTFT ratio sat at ~1.05 — essentially untouched — while the R9700 twin's
was 1.6890. That asymmetry was the reason to look, and the census had to be taken
on the ranked path rather than on `llama-bench`: the scored number is the client
wall time of a single cache-cold `/completion` at `n_predict=1` against
`llama-server -ngl 99 -c 8192 --parallel 1`, with the prompt sized in
**characters** (`GAINZ_PROMPT_CHARS = 2600`, so `prompt_n` varies run to run),
median of 9 runs at indices 100–108.

## The census

Replica of the runner's `benchPhase` exactly — same flags, same corpus, same
`cache_prompt: false`, 2 warm cycles then 9 measured — instrumented with
process-wide fault counts from `/proc/<pid>/task/*/stat` and `AnonHugePages`
from `/proc/<pid>/smaps_rollup`.

| | laguna-xs | qwen3.6-35b-a3b | lfm2.5-2.6b |
| --- | --- | --- | --- |
| ttft wall | 354.1 ms | 400.9 ms | 92.6 ms |
| engine `prompt_ms` | 302.8 ms | 333.2 ms | 84.9 ms |
| **gap (host, no GPU)** | **50.9 ms (14.4%)** | **66.4 ms (16.6%)** | **7.4 ms (8.0%)** |
| minor faults / request | 53,738 | 83,763 | 2,976 |
| major faults / request | 0 | 0 | 0 |
| `AnonHugePages` | 0 kB | 0 kB | 0 kB |

The gap is `wall − prompt_ms − predicted_ms`, i.e. time the engine does not
attribute to prompt processing at all. **Zero major faults** means none of it is
disk — it is all first touch on new anonymous pages. 53,738 faults at ~1 µs each
accounts for the entire 50.9 ms.

Anonymous RSS grows **monotonically** at ~154 MB per bench cycle (660 MB →
1731 MB over nine cycles) even with `cache_prompt: false`, so the allocator never
hands the same page back and the faults recur on every request.

## Hypothesis

If the residual TTFT is first-touch faults on new pages, then it is a function of
**page size**, and putting the per-request state on 2 MiB pages should cut the
fault count ~13x without touching a single arithmetic operation.

## Mechanism

`llama-server` allocates ~210 MiB of brand-new host memory per request: one
`server_prompt_cache::alloc` state plus two context checkpoints from
`common_prompt_checkpoint::update_tgt`/`update_dft`.

`std::vector::resize()` **value-initialises**, so the first touch happens *inside*
`resize()` — too late to advise. `common_state_buf_resize()` therefore does
**reserve → madvise the 2 MiB-aligned interior `MADV_HUGEPAGE` → resize**, and
over-reserves by one huge page so the aligned interior covers the whole logical
range. Buffers below 16 MiB are left alone: the alignment slack does not pay for
itself when the fault count is small.

## Proof that it fires — structural, not timing

- minor faults per request **53,738 → 16,435** (−69%), landing on 16,435 to the
  page on every clean boot;
- `AnonHugePages` **0 → 761,856 kB**, reaching the identical figure on all seven
  on-boots;
- the residual 16,435 faults are allocations under the 16 MiB threshold, and TTFT
  and prefill move *only* when `AnonHugePages` moves.

## Result

Same binary throughout, `LLAMA_HUGEPAGE_STATE` toggle, 12 boots strictly
alternating. This box has a per-launch fast/slow mode lottery, so each arm is
classified against its own cluster.

**slow-mode cluster (decode ~78 tok/s)**

| arm | ttft | prefill | decode | minflt | AnonHugePages |
| --- | --- | --- | --- | --- | --- |
| off1 | 0.36112 | 1779.5 | 78.15 | 53,738 | 0 kB |
| on1 | 0.32137 | 1825.4 | 78.06 | 16,435 | 761,856 kB |
| on2 | 0.31539 | 1842.6 | 78.42 | 16,435 | 761,856 kB |
| off2 | 0.35095 | 1776.3 | 78.38 | 53,738 | 0 kB |
| off3 | 0.35078 | 1787.8 | 78.27 | 53,738 | 0 kB |
| on3 | 0.32298 | 1835.7 | 78.53 | 16,567 | 761,856 kB |

**fast-mode cluster (decode ~92 tok/s)**

| arm | ttft | prefill | decode | minflt | AnonHugePages |
| --- | --- | --- | --- | --- | --- |
| off1 | 0.33785 | 1847.5 | 92.37 | 53,738 | 0 kB |
| on1 | 0.31148 | 1894.9 | 92.44 | 16,435 | 761,856 kB |
| off2 | 0.34242 | 1873.2 | 92.16 | 53,738 | 0 kB |
| on2 | 0.30822 | 1906.2 | 92.31 | 16,435 | 761,856 kB |
| off3 | 0.34193 | 1849.5 | 92.01 | 53,738 | 0 kB |
| on3 | 0.31141 | 1897.3 | 92.30 | 16,538 | 761,856 kB |

**ttft 0.3475 → 0.3151 (−9.3%)**, and the arms are **disjoint within each mode**:
slow off spans 0.35078–0.36112 against on 0.31539–0.32298; fast off spans
0.33785–0.34242 against on 0.30822–0.31148. **6/6 boots separated.**

**prefill 1819.0 → 1867.0 (+2.64%)**, also disjoint within each mode.
**decode 85.2 → 85.3 (+0.14%)** — overlapping, i.e. neutral, as a page-size
change should be.

The host gap itself falls **50.9 ms → 27.1 ms**, which is the mechanism
reporting its own signature: the change is entirely outside the engine's timed
region, so prefill (which the engine *does* time) moves much less than TTFT.

## Correctness

**Byte-identical arithmetic by construction.** Rebuilt against the parent tree,
the three artefacts that carry arithmetic are unchanged:

```
libggml-cuda.so   1c8dda0e5b11079f9992bb3f99c37f47   identical
libllama.so       ed20802f4e736cd3430f310925f6adb6   identical
llama-perplexity  65607a29e7fe14025afb267c41a0ba5a   identical
```

Only `libllama-common.so` and `libllama-server-impl.so` change, and the symbol
they gain is one `llama-perplexity` never reaches — checkpoints and the prompt
cache are server-only. `llama-perplexity` being byte-identical means the
perplexity gate cannot move at all: the patch changes page size, not arithmetic.

The full 17-patch series applies clean to pristine `b10237`.

## What this rules out, so nobody re-buys it

- **glibc arena retention is not a substitute.** Measured directly with
  `MALLOC_MMAP_THRESHOLD_=1G MALLOC_TRIM_THRESHOLD_=1G`, same binary, same
  window: minor faults 53,738 → 37,443 and ttft 0.3418 → 0.3286 (−3.9%), against
  THP's 53,738 → 16,435 and −8.9%. Stacking both gives 15,630 faults and −8.3%,
  i.e. no better than THP alone. THP does the work; the allocator knob does not.
- **A buffer pool cannot help here.** Anonymous RSS grows monotonically, so
  nothing is being freed for a pool to recycle.
- **lfm2.5-2.6b-gguf-gb10cuda is dead** and is recorded as such. Its per-request
  state is ~12 MiB, it takes 2,976 minor faults per request, nothing clears the
  16 MiB threshold, `AnonHugePages` stays 0 on every boot and ttft is flat across
  four alternating boots (off 0.09237/0.09259, on 0.09097/0.09285). **The lever is
  gated on KV/state size, not on the engine.**
- **qwen3.6-35b-a3b is a rejection risk, not a smaller win**, and is deliberately
  not submitted. `defrag` is `madvise` on this box, so a `MADV_HUGEPAGE` fault
  does synchronous direct compaction; because the prompt cache grows without
  bound, every request needs *fresh* huge pages and the compaction cannot be
  warmed away at startup. On qwen's larger state (327 MB/request) 2 of 6 on-boots
  stalled: ttft 0.488 and 0.494 against a stable off arm at 0.388–0.410, with
  `/proc/vmstat` showing `compact_fail ≈ compact_stall` and the process reaching
  only half the huge pages it wanted. That draw would put ttft speedup near 0.83,
  under the hard 0.90 floor. **laguna-xs does not draw it: 7 of 7 on-boots clean,
  the same 761,856 kB every time.**

`LLAMA_HUGEPAGE_STATE=0` restores plain `resize()`.
