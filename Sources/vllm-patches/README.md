# Sources/vllm-patches — deep engine surgery for the GB10 vLLM tracks

Set `"vllmSource": true` in `Sources/runner/serving.json` and put an ordered
git patch series here (`0001-*.patch`, `0002-*.patch`, …). The trusted
runner applies it to a pristine copy of the pinned image's OWN vllm package
and overlays the result into the candidate container — you are editing the
real engine the benchmark runs, llama.cpp-track style.

**One series, both vLLM tracks.** Unlike `Sources/patches/`, this directory
is *not* split per track: `laguna-xs-2.1-nvfp4-gb10-v1` and
`laguna-s-2.1-nvfp4-gb10-v1` share it. Do not create a per-track
subdirectory here — the runner will not look in one. If a change helps XS and
hurts S, gate it on the model or shape inside the patch.

## S-track validation (2026-08-05)

The merged frontier tree (kernels: NVFP4 MoE M-sub-tile + sign-folded nibble
decode + 2-warp decode schedule + stock-geometry prefill signfold; vllmSource:
0001 batch-invariant dense decode tile) measured on the GB10 box at the
ranked-style S window (fresh 512-token prompt + 128 decode steps, median of 5):
**17.46 tok/s decode** (57.3 ms/token) against the S frontier's 16.06 — the
MoE decode wins transfer to Laguna-S. Served with the pinned baseline flags
(util 0.88, KV blocks pinned to the baseline's 4,785).

## Series log

- `0001` — dense decode tile for the batch-invariant matmul (XS-verified
  +12.4%); decode branch fires at M <= 16 with BLOCK_SIZE_M=16, N=64, nw=4,
  ns=4, registered as an opaque custom op so the runtime-shape branch is live
  under torch.compile.


## Scope

- Everything in vLLM's **Python/Triton layer** is fair game: the NVFP4 MoE
  emulation, attention orchestration, model runner, scheduler, sampling —
  ~2,100 files, including every Triton kernel. On these tracks the hot decode
  kernels are already Triton and editable as plain Python, so reach for CUDA
  only when Triton cannot express what you want.
- **CUDA and C++ are in scope too**: `.cu`, `.cuh`, `.cpp`, `.cc`, `.h`
  and `.hpp` under `vllm/` are accepted, and you may ADD new source files, not
  just edit existing ones. The overlay mount becomes writable when your series
  touches any native source, so a JIT path
  (`torch.utils.cpp_extension.load` / `load_inline`, or a `nvcc` call from a
  module initializer) can compile and cache next to the sources it builds.
  `nvcc` ships in the image.
- **Caveat, and it decides your design: there is no ahead-of-time rebuild of
  the image's prebuilt `_C.so`.** Editing a `.cu` that is already compiled
  into `_C.so` changes nothing at runtime. For a custom kernel to actually
  run, **your Python must load it and dispatch to it**. The clean pattern is
  a new `vllm/gainz_kernels/` directory holding the `.cu` plus a loader, and
  a one-line dispatch edit at the call site.

  (`/api/tracks` currently describes these tracks as rebuilding the CUDA
  extensions for sm_121. That description is ahead of the implementation — a
  true from-source rebuild is being brought up separately. Until it lands,
  assume the JIT-and-dispatch pattern above.)
- Anything outside the `vllm/` package is still rejected at deploy time, as
  are non-source files (`.so`, `.sh`, archives) — ship source, not binaries.
- Generate patches against the image's installed package: copy it out
  (`docker cp $(docker create vllm/vllm-openai:v0.25.1):/usr/local/lib/python3.12/dist-packages/vllm ./vllm`),
  `git init && git add -A && git commit`, edit, then `git diff > 0001-<name>.patch`.
- Combines with `"kernels": true` (the `Sources/kernels/` plugin package) if
  you need both.

## Two things that do not work, both measured

- **Monkeypatching anything inside vLLM's compiled region does not take
  effect.** That is what this surface exists to replace.
- **A GEMM benchmarked outside the engine does not predict the engine.** More
  than one apparent win here was real offline and gone in-process.

Also worth knowing before you design: making a runtime-shape branch actually
execute under `@support_torch_compile` is its own problem, and it is written
up in the findings API — `curl -s "https://frontier.fast/api/findings?track=laguna-xs-2.1-nvfp4-gb10-v1"`.

Same gates as always: teacher-forced correctness (≥ 90% agreement), the
floors, the calibration band, and the frontier rule. Validate with
`./tools/preflight.sh` before spending a runner slot — it boots an
identically-configured control beside your candidate and runs the same check.

This surface exists because the profiled headroom on these tracks lives in
engine code a plugin cannot cleanly reach. Use it to remove provably-dead
work — never to change values.
