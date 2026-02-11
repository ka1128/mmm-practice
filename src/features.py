# src/features.py
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def geometric_adstock(x: NDArray | list[float], decay: float) -> NDArray:
    """
    Geometric adstock:
      out[t] = x[t] + decay * out[t-1]

    Notes:
      - NaN in x is treated as 0.
      - decay is expected in [0, 1].
    """
    if not (0.0 <= float(decay) <= 1.0):
        raise ValueError(f"decay must be in [0, 1]. got {decay}")

    x_arr = np.asarray(x, dtype=float)
    x_arr = np.nan_to_num(x_arr, nan=0.0)

    out = np.zeros_like(x_arr, dtype=float)
    carry = 0.0
    for i in range(len(x_arr)):
        carry = float(x_arr[i]) + float(decay) * carry
        out[i] = carry
    return out


def build_log_ad_from_spend(spend: NDArray | list[float], decay: float) -> NDArray:
    """
    Convenience: log1p(adstock(spend)).
    """
    adstock = geometric_adstock(spend, decay)
    return np.log1p(adstock)
