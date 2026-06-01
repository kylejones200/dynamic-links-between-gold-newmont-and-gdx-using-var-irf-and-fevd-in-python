"""VAR(1) impulse response."""

from __future__ import annotations

import numpy as np


def var1_irf(phi: float, shock: float, steps: int) -> np.ndarray:
    out = np.empty(steps, dtype=float)
    y = float(shock)
    for i in range(steps):
        out[i] = y
        y *= phi
    return out
