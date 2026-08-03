"""gainz.fast participant kernel package.

Set ``"kernels": true`` in ``Sources/runner/serving.json`` and the trusted
runner pip-installs this package into the candidate engine before serving
(read-only mount of your exact submitted commit, HF offline). vLLM calls
``register()`` at engine startup via the ``vllm.general_plugins`` entry
point — patch in your Triton/CUDA kernels there.

Rules of the road: exact greedy output must survive your kernels (the
paired correctness probe rejects any token flip), the calibration band caps
a single submission's gain at ~5.3%, and torch/triton ship in the pinned
vllm/vllm-openai:v0.25.1 image — no new dependencies.
"""


def register() -> None:
    """Called by vLLM at engine startup. Replace ops/kernels here."""
    # Example shape (fill in with real Triton kernels):
    #   import torch, triton
    #   from my_fused_ops import faster_rms_norm
    #   torch.library.opcheck(...)  # keep numerics exact
    pass
