# src/config.py
from __future__ import annotations

from typing import List


# =========================================================
# Column conventions (single source of truth)
# =========================================================

# Spend columns (weekly)
COL_ONLINE_SPEND = "online_spend"
COL_BROADCAST_SPEND = "broadcast_spend"
COL_OOH_SPEND = "ooh_print_spend"

MEDIA_SPEND_COLS: List[str] = [
    COL_ONLINE_SPEND,
    COL_BROADCAST_SPEND,
    COL_OOH_SPEND,
]

# Target
COL_LOG_SALES = "log_sales"  # log1p(sales)


# =========================================================
# Feature conventions
# =========================================================

# Log-adstock feature columns (created from spend + decay)
COL_LOG_AD_ONLINE = "log_ad_online"
COL_LOG_AD_BROADCAST = "log_ad_broadcast"
COL_LOG_AD_OOH = "log_ad_ooh"

LOG_AD_COLS: List[str] = [
    COL_LOG_AD_ONLINE,
    COL_LOG_AD_BROADCAST,
    COL_LOG_AD_OOH,
]

# Time + seasonality (Fourier order=2)
COL_T = "t"
FOURIER_COLS: List[str] = ["sin_1", "cos_1", "sin_2", "cos_2"]


# =========================================================
# Model schema (EXOG)
# =========================================================
# Note: build_X_for_model() will ensure 'const' exists, but we keep it explicit.
EXOG: List[str] = [
    "const",
    COL_LOG_AD_ONLINE,
    COL_LOG_AD_BROADCAST,
    COL_LOG_AD_OOH,
    COL_T,
    *FOURIER_COLS,
]


# =========================================================
# Decay grid (for AIC grid search)
# =========================================================
# Matches your notebook04 grid definition.
DECAY_GRID = {
    "d_online": [0.0, 0.3, 0.5, 0.7, 0.85, 0.95],
    "d_broadcast": [0.0, 0.2, 0.4, 0.6, 0.7, 0.85],
    "d_ooh": [0.0, 0.1, 0.3, 0.5, 0.7, 0.85],
}
