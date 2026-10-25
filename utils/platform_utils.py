"""Small platform choices for the reproduction entry points."""
import os
import warnings


WINDOWS = os.name == "nt"


def runner_resource_limits(limits):
    """Windows mode runs without the Linux affinity and memory controls."""
    if WINDOWS and limits is not None:
        warnings.warn(
            "Windows mode: resource_limits is disabled (Linux CPU affinity and "
            "guard/strict RAM limits). Tasks can use all available CPU and RAM.",
            RuntimeWarning,
            stacklevel=2,
        )
        return None
    return limits
