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
- **C++/CUDA cannot take effect without a rebuild**, so patches touching
  anything but `vllm/**/*.py` are rejected at deploy time.
- Generate patches against the image's installed package: copy it out
  (`docker cp $(docker create vllm/vllm-openai:v0.25.1):/usr/local/lib/python3.12/dist-packages/vllm ./vllm`),
  `git init && git add -A && git commit`, edit, then `git diff > 0001-<name>.patch`.
- Combines with `"kernels": true` (plugin package) if you need both.

Same gates as always: teacher-forced correctness, floors, band, frontier.
This surface exists because the profiled headroom on these tracks lives in
engine code a plugin cannot cleanly reach. Use it to remove provably-dead
work — never to change values.
