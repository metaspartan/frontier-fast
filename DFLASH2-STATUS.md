# DFlash2 on qwen3.8-27b r9700 — status + blocker (2026-08-20)

## What is DONE / staged
- **Weights on box**: `/home/ghost/models/Qwen3.8-27B-DFlash2-Q4_K_M.gguf` (1,143,006,752 B) from `incoai/Qwen3.8-27B-DFlash2-GGUF`. Z-lab mirror = same. Q8_0/BF16 variants exist too.
- **Engine confirmed real + portable in principle**: PR #27342 (`refs/heads/dflash2`, head `5ecbe1a`) adds DFlash2 to llama.cpp. Fetched into `/home/ghost/llama.cpp-dflash-0b1bad1`.
- **Applied the DFlash2 slice** to a fresh worktree at the pinned base `2b63e0610` (`/home/ghost/dflash2w`). Conv/selector tensors present (13 refs; `dflash_selector_*`, `build_dflash2_conv`, `dflash.block_size/conv/selector` KV keys).

## The blocker (proven by a real build)
The DFlash2 PR sits ~1–2 weeks of llama.cpp commits after the **pinned base `2b63e061061`**. My slice diff applied, but the dependency window is missing:
- `llama-kv-cache-msa.h` not in the pinned tree
- `hparams.n_embd_k_idx` / `n_embd_k_idx` member absent
- `LLAMA_LOAD_MODE_AUTO` absent
- `ggml_ssm_scan` has 8 vs 9 args in the pin

⇒ **Porting DFlash2 onto the exact pinned base is a full refactor-window backport**, not a config or a small patch. Ranked runner (which builds at 2b63) cannot run it no matter the weights.

## Files / locations for a resumer
- Worktree: `/home/ghost/dflash2w` (applied slice, failed build; logs `cmake.log`, `build.log`; 211/... objects built)
- Slice diffs: `/tmp/dfl/dflash2-on-pinned.diff` (2711 lines, applies clean) and `/tmp/dfl/dflash2-slice.diff`
- Build recipe that configures: `cmake -B build -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1201 -DCMAKE_BUILD_TYPE=Release -DCMAKE_HIP_COMPILER=/opt/rocm/core-7.14/lib/llvm/bin/clang -DCMAKE_C_COMPILER=/usr/bin/gcc -DCMAKE_CXX_COMPILER=/usr/bin/g++`  (make, not ninja; HIP `core-7.14`)

## Two real paths forward (both need arena/maintainer or a heavy commitment)
1. **Arena bumps the pinned base** to a tree that already includes PR #27342 ⇒ then DFlash2 = a `draftModel` pin (weights) + `specType: draft-dflash` in serving.json. Clean, no kernel porting.
2. **Backport the whole 0b1bad..dflash2 refactor window** onto 2b63 as one patch series ⇒ multi-hour/multi-session engineering on the shared box.

## Frontier (unchanged)
Spec board champion **47.69 tok/s (+49.46%)** — greedy-argmax MTP chain. No new number produced.
