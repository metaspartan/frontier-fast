"""gainz.fast participant kernel package — NO-OP CONTROL.

Installs nothing. Used to measure the local-harness boot-artifact floor:
the same teacher-forced comparison run with a candidate that differs from
the control by an empty plugin should reproduce the ~9/128 mismatch floor
documented in the findings ledger (no-op-control-run, won).
"""


def register() -> None:
    """No-op: do not install any kernels."""
    return
