"""gainz.fast participant kernel package — Fable 5 (Claude Code) submission.

Loaded into the candidate vLLM v0.25.1 engine via the ``vllm.general_plugins``
entry point before model load. This submission deliberately ships NO custom
math kernels: a Triton replacement for an existing op cannot be credibly
guaranteed bit-exact against the pinned CUDA kernels within this iteration
budget, and any token flip is an automatic rejection.

Instead ``register()`` applies engine-process init optimizations that cannot
change numerics — they touch only Python-runtime scheduling and CUDA module
loading, never a tensor value:

1. Cut Python GC pressure in the serial decode hot loop. The ranked window
   is one sequence decoding 128 steps; each step allocates short-lived
   Python objects (scheduler outputs, sampling metadata, request state).
   Raising the gen-0 threshold and freezing the imported baseline keeps
   stop-the-world young-gen scans out of the measured window.

2. Slightly raise the GIL switch interval so the single hot engine thread
   is preempted less often by housekeeping threads.

3. Ask CUDA to load module code eagerly at init instead of lazily at first
   kernel call, so no lazy-load stall can land inside a measured run.

Expected effect is honest and small (CPU-side overhead shaving, low
single-digit percent at best); the paired GB10 measurement is the judge.
"""

import gc
import os
import sys


def register() -> None:
    """Engine-process init optimizations. No numeric path is touched."""
    # (3) Must be set before CUDA context creation to matter; harmless later.
    os.environ.setdefault("CUDA_MODULE_LOADING", "EAGER")

    # (1) Collect import-time garbage once, then freeze the surviving
    # baseline into the permanent generation so full collections never
    # re-scan it. Raise gen-0 threshold: fewer young-gen pauses per decode
    # step. Affects only *when* GC runs, never what the model computes.
    gc.collect()
    gc.freeze()
    gc.set_threshold(50_000, 50, 50)

    # (2) Default is 5 ms; 20 ms changes nothing numerically — it only
    # reduces GIL handoff frequency for the serial decode thread.
    sys.setswitchinterval(0.02)
