# src/optimization.py
from __future__ import annotations

from typing import Callable, Sequence, Any, Dict, List
import pandas as pd
import numpy as np

# 依存注入：statsmodels でも sklearn でも OK にする
# model_fit_fn は少なくとも .aic を持つ fitted_model を返す想定
ModelFitFn = Callable[..., Any]


def decay_grid_search_aic(
    df_w: pd.DataFrame,
    decay_grid: Dict[str, Sequence[float]],
    model_fit_fn: ModelFitFn,
    exog_cols: Sequence[str] | None = None,
) -> pd.DataFrame:
    """
    戻り値:
      df_decay: columns=[d_online, d_broadcast, d_ooh, AIC]
    """
    required = ["d_online", "d_broadcast", "d_ooh"]
    missing = [k for k in required if k not in decay_grid]
    if missing:
        raise ValueError(f"decay_grid is missing keys: {missing}")

    rows: List[dict] = []

    for d_ooh in decay_grid["d_ooh"]:
        for d_broadcast in decay_grid["d_broadcast"]:
            for d_online in decay_grid["d_online"]:
                fitted = model_fit_fn(
                    df_w=df_w,
                    d_online=float(d_online),
                    d_broadcast=float(d_broadcast),
                    d_ooh=float(d_ooh),
                    exog_cols=exog_cols,
                )

                aic = getattr(fitted, "aic", None)
                if aic is None:
                    raise AttributeError(
                        "model_fit_fn must return an object with attribute .aic"
                    )

                rows.append(
                    {
                        "d_online": float(d_online),
                        "d_broadcast": float(d_broadcast),
                        "d_ooh": float(d_ooh),
                        "AIC": float(aic),
                    }
                )

    df_decay = (
        pd.DataFrame(rows)
        .sort_values("AIC", ascending=True)
        .reset_index(drop=True)
    )
    return df_decay


def select_best_decay(df_decay: pd.DataFrame) -> dict:
    """
    戻り値:
      {"d_online": ..., "d_broadcast": ..., "d_ooh": ..., "AIC": ...}
    """
    required_cols = ["d_online", "d_broadcast", "d_ooh", "AIC"]
    missing = [c for c in required_cols if c not in df_decay.columns]
    if missing:
        raise ValueError(f"df_decay is missing columns: {missing}")
    if len(df_decay) == 0:
        raise ValueError("df_decay is empty")

    best_row = df_decay.loc[df_decay["AIC"].astype(float).idxmin(), required_cols]
    return {k: float(best_row[k]) for k in required_cols}
