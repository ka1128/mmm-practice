# src/modeling.py
from __future__ import annotations

from typing import Sequence, Optional, Any
import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.features import build_log_ad_from_spend

from src.config import (
    EXOG,
    COL_ONLINE_SPEND, COL_BROADCAST_SPEND, COL_OOH_SPEND,
    COL_LOG_SALES,
    COL_LOG_AD_ONLINE, COL_LOG_AD_BROADCAST, COL_LOG_AD_OOH,
    COL_T, FOURIER_COLS,
)

def build_X_for_model(df: pd.DataFrame, exog_cols: Sequence[str]) -> pd.DataFrame:
    """
    Build design matrix X aligned to exog_cols.
    This project assumes an intercept ('const') is always included.
    """
    X = df.copy()
    exog_cols = list(exog_cols)

    # Ensure const exists (project convention)
    if "const" not in exog_cols:
        exog_cols = ["const"] + exog_cols
    if "const" not in X.columns:
        X["const"] = 1.0

    # Align columns (missing filled with 0)
    X = X.reindex(columns=exog_cols, fill_value=0.0)
    return X


def fit_model_for_decays(
    df_w: pd.DataFrame,
    d_online: float,
    d_broadcast: float,
    d_ooh: float,
    exog_cols: Optional[Sequence[str]] = None,
):
    """
    Fit OLS for a given set of adstock decays and return fitted_model (AIC via fitted_model.aic).

    Schema control:
      - If exog_cols is None, use src.config.EXOG (single source of truth).
      - Otherwise, align X to the provided schema.
    """
    log_ad_online = build_log_ad_from_spend(df_w[COL_ONLINE_SPEND].values, d_online)
    log_ad_broadcast = build_log_ad_from_spend(df_w[COL_BROADCAST_SPEND].values, d_broadcast)
    log_ad_ooh = build_log_ad_from_spend(df_w[COL_OOH_SPEND].values, d_ooh)

    X_raw = pd.DataFrame(
        {
            "const": 1.0,
            COL_LOG_AD_ONLINE: log_ad_online,
            COL_LOG_AD_BROADCAST: log_ad_broadcast,
            COL_LOG_AD_OOH: log_ad_ooh,
            COL_T: df_w[COL_T].values,
            # Fourier terms (sin/cos)
            **{col: df_w[col].values for col in FOURIER_COLS},
        }
    )

    schema = EXOG if exog_cols is None else list(exog_cols)
    X = build_X_for_model(X_raw, schema)
    y = df_w[COL_LOG_SALES].values

    fitted_model = sm.OLS(y, X).fit()
    return fitted_model



def predict_annual_sales_52(model: Any, X_52: pd.DataFrame) -> float:
    """
    model.predict(X_52) returns log1p(sales).
    Return annual sales by summing weekly expm1(pred).
    """
    pred_log_sales = model.predict(X_52)
    return float(np.expm1(pred_log_sales).sum())


def predict_annual_sales_52_from_df(
    model: Any,
    df_52: pd.DataFrame,
    exog_cols: Optional[Sequence[str]] = None,
) -> float:
    """
    df_52 already contains required feature columns.
    Build X aligned to exog_cols (default: src.config.EXOG) and predict annual sales.
    """
    schema = EXOG if exog_cols is None else list(exog_cols)
    X_52 = build_X_for_model(df_52, schema)
    return predict_annual_sales_52(model, X_52)

