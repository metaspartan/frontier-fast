# Sources/vllm-patches — deep engine surgery for the GB10 vLLM tracks

Set `"vllmSource": true` in `Sources/runner/serving.json` and put an ordered
git patch series here (`0001-*.patch`, `0002-*.patch`, …). The trusted
runner applies it to a pristine copy of the pinned image's OWN vllm package
and overlays the result into the candidate container — you are editing the
real engine the benchmark runs, llama.cpp-track style.

## Scope

- Everything in vLLM's **Python/Triton layer** is fair game: the NVFP4 MoE
  emulation, attention orchestration, model runner, scheduler, sampling —
  ~2,100 files, including every Triton kernel.
- **CUDA and C++ are now in scope too**: `.cu`, `.cuh`, `.cpp`, `.cc`, `.h`
  and `.hpp` under `vllm/` are accepted, and you may ADD new source files, not
  just edit existing ones. The overlay mount becomes writable when your series
  touches any native source, so a JIT path
  (`torch.utils.cpp_extension.load` / `load_inline`, or a `nvcc` call from a
  module initializer) can compile and cache next to the sources it builds.
  There is no ahead-of-time rebuild of the image's prebuilt `_C.so`: if you
  want a custom kernel to run, your Python must load it and dispatch to it.
  The clean pattern is a new `vllm/gainz_kernels/` directory holding the `.cu`
  plus a loader, and a one-line dispatch edit at the call site.
- Anything outside the `vllm/` package is still rejected at deploy time, as
  are non-source files (`.so`, `.sh`, archives) — ship source, not binaries.
- Generate patches against the image's installed package: copy it out
  (`docker cp $(docker create vllm/vllm-openai:v0.25.1):/usr/local/lib/python3.12/dist-packages/vllm ./vllm`),
  `git init && git add -A && git commit`, edit, then `git diff > 0001-<name>.patch`.
- Combines with `"kernels": true` (plugin package) if you need both.

Same gates as always: teacher-forced correctness, floors, band, frontier.
This surface exists because the profiled headroom on these tracks lives in
engine code a plugin cannot cleanly reach. Use it to remove provably-dead
work — never to change values.
