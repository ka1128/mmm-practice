# src/simulate.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np
import pandas as pd

from src.config import (
    MEDIA_SPEND_COLS,
    COL_ONLINE_SPEND, COL_BROADCAST_SPEND, COL_OOH_SPEND,
)

# =========================================================
# Local column name constants (kept here for now)
# (These are not in src.config yet, so we define locally.)
# =========================================================
COL_WEEK = "Week"
COL_FY_YEAR = "fy_year"
COL_ISO_WEEK = "iso_week"
COL_WEEK_OF_YEAR = "week_of_year"
COL_SALES = "sales"
COL_TOTAL_SPEND = "total_spend"


# =========================================================
# Step08 helpers: ws_52 / df_sim_52 / avg_df / weights / ROI score
# =========================================================

REQ_WS_COLS = ["week_of_year", "baseline_hat", "roi_online", "roi_broadcast"]


def build_ws_52(weekly_score: pd.DataFrame) -> pd.DataFrame:
    """
    weekly_score(week_of_year単位) を 1..52 の ws_52 に正規化する。
    - week_of_year を int 化
    - 53週があれば 52 に fold
    - 重複週があれば mean で集約
    - baseline_hat 欠損はエラー（重み生成に必須）
    - ROI欠損は 0 扱い（保守的）
    """
    ws_tmp = weekly_score.copy()

    # normalize week_of_year to int and fold 53 -> 52
    ws_tmp["week_of_year"] = ws_tmp["week_of_year"].astype(int)
    ws_tmp.loc[ws_tmp["week_of_year"] == 53, "week_of_year"] = 52

    missing_cols = [c for c in REQ_WS_COLS if c not in ws_tmp.columns]
    if missing_cols:
        raise KeyError(f"weekly_score is missing required columns: {missing_cols}")

    ws_agg = (
        ws_tmp[REQ_WS_COLS]
        .groupby("week_of_year", as_index=False)
        .mean()
    )

    ws_52 = pd.DataFrame({"week_of_year": np.arange(1, 53)}).merge(
        ws_agg,
        on="week_of_year",
        how="left"
    )

    if ws_52["baseline_hat"].isna().any():
        miss = ws_52.loc[ws_52["baseline_hat"].isna(), "week_of_year"].tolist()
        raise ValueError(f"ws_52 has missing baseline_hat for weeks: {miss}")

    for c in ["roi_online", "roi_broadcast"]:
        ws_52[c] = ws_52[c].fillna(0.0)

    if len(ws_52) != 52:
        raise ValueError(f"ws_52 must be 52 rows. got {len(ws_52)}")

    return ws_52


def build_df_sim_52(df_w: pd.DataFrame, sim_fy: int) -> pd.DataFrame:
    """
    df_w（週次データ）から、指定FYの 52週テンプレ df_sim_52 を作る。
    - ISO週（df_w['iso_week']）を week_of_year として採用
    - 53週があれば 52 に fold
    - week_of_year で重複があれば drop（先に来る週を採用）
    """
    tmp = df_w[df_w[COL_FY_YEAR] == sim_fy].copy()

    tmp[COL_WEEK_OF_YEAR] = tmp[COL_ISO_WEEK]
    tmp.loc[tmp[COL_WEEK_OF_YEAR] == 53, COL_WEEK_OF_YEAR] = 52

    df_sim_52 = (
        tmp.sort_values(COL_WEEK)
           .drop_duplicates(COL_WEEK_OF_YEAR)
           .query("1 <= week_of_year <= 52")   # ←ここは後述の置換
           .reset_index(drop=True)
           .copy()
    )

    if len(df_sim_52) != 52:
        miss = sorted(set(range(1, 53)) - set(df_sim_52[COL_WEEK_OF_YEAR]))
        raise ValueError(
            f"FY{sim_fy} cannot build 52-week ISO template. "
            f"len={len(df_sim_52)} missing_weeks={miss}"
        )

    return df_sim_52


def build_avg_df_as_is(
    df_w: pd.DataFrame,
    media_cols: list[str] | None = None,
    min_weeks: int = 49,
) -> pd.DataFrame:
    """
    Step08の As-Is年平均（FY平均）を作る。
    - FYごとの週数が min_weeks 以上のFYのみ採用（欠損FY排除）
    - FYごとに sales/total_spend/media_cols を年合計→年平均
    戻り値: avg_df（columns: metric, avg_value）
    """
    if media_cols is None:
        media_cols = ["broadcast_spend", "ooh_print_spend", "online_spend"]

    df_tmp = df_w.copy()

    weeks_per_fy = df_tmp.groupby("fy_year")["week_of_year"].nunique()
    valid_fy_years = weeks_per_fy[weeks_per_fy >= min_weeks].index.tolist()
    if len(valid_fy_years) == 0:
        raise ValueError(f"No FY has >= {min_weeks} weeks. Check df_w completeness.")

    df_base = df_tmp[df_tmp["fy_year"].isin(valid_fy_years)].copy()

    cols_sum = ["sales", "total_spend"] + media_cols
    df_year = (
        df_base
        .groupby("fy_year", as_index=False)[cols_sum]
        .sum()
        .sort_values("fy_year")
    )

    avg_series = df_year.drop(columns=["fy_year"]).mean()
    avg_df = avg_series.reset_index().rename(columns={"index": "metric", 0: "avg_value"})

    return avg_df


def _get_avg(avg_df: pd.DataFrame, metric: str) -> float:
    return float(avg_df.loc[avg_df["metric"] == metric, "avg_value"].iloc[0])


def build_online_as_is_paths_52(
    avg_df: pd.DataFrame,
    n_weeks: int = 52,
) -> tuple[float, float, float, float, np.ndarray, np.ndarray, np.ndarray]:
    """
    As-Is年平均から、52週均等配賦の各媒体pathを作る。
    戻り値:
      total_budget_as_is, avg_online, avg_br, avg_ooh,
      online_as_is_52, broadcast_as_is_52, ooh_as_is_52
    """
    total_budget_as_is = _get_avg(avg_df, "total_spend")
    avg_online = _get_avg(avg_df, "online_spend")
    avg_br = _get_avg(avg_df, "broadcast_spend")
    avg_ooh = _get_avg(avg_df, "ooh_print_spend")

    online_as_is_52 = np.full(n_weeks, avg_online / n_weeks)
    broadcast_as_is_52 = np.full(n_weeks, avg_br / n_weeks)
    ooh_as_is_52 = np.full(n_weeks, avg_ooh / n_weeks)

    return (
        total_budget_as_is, avg_online, avg_br, avg_ooh,
        online_as_is_52, broadcast_as_is_52, ooh_as_is_52
    )


def build_w_online_52(
    ws_52: pd.DataFrame,
    always_on_share: float,
    boost_quantile: float = 0.6,
    bot_mode: str = "baseline",  # "baseline" or "equal"
) -> np.ndarray:
    """
    Online週重み w_online_52（Always-on + Boost）
    - Boost: ROI上位（quantile以上）の週に baseline_hat * max(roi,0) を付与
    - Always-on: それ以外の週に baseline_hat（or equal）を付与
    - always_on_share は Online内の配分比率（0..1）
    """
    tmp = ws_52.copy()

    if not (0.0 <= always_on_share <= 1.0):
        raise ValueError("always_on_share must be in [0, 1].")
    for c in ["week_of_year", "baseline_hat", "roi_online"]:
        if c not in tmp.columns:
            raise KeyError("ws_52 must have columns: week_of_year, baseline_hat, roi_online")
    if tmp["roi_online"].isna().any() or tmp["baseline_hat"].isna().any():
        raise ValueError("ws_52 contains NaN in roi_online or baseline_hat")

    thr = tmp["roi_online"].quantile(boost_quantile)
    mask_top = (tmp["roi_online"] >= thr).values
    mask_bot = ~mask_top

    # guard: all top / none top の場合、rankで上位40%をtop扱い
    if mask_top.sum() in (0, len(tmp)):
        k = max(1, int(np.ceil(len(tmp) * 0.4)))
        top_idx = tmp["roi_online"].rank(method="first", ascending=False).values <= k
        mask_top = top_idx
        mask_bot = ~mask_top

    top_weight = np.where(
        mask_top,
        tmp["baseline_hat"].values * np.clip(tmp["roi_online"].values, 0, None),
        0.0,
    )
    if top_weight.sum() <= 0:
        top_weight = mask_top.astype(float)
    top_weight = top_weight / top_weight.sum()

    if bot_mode == "equal":
        bot_weight = mask_bot.astype(float)
    elif bot_mode == "baseline":
        bot_weight = np.where(mask_bot, tmp["baseline_hat"].values, 0.0)
    else:
        raise ValueError("bot_mode must be 'equal' or 'baseline'")

    if bot_weight.sum() <= 0:
        bot_weight = mask_bot.astype(float)
    bot_weight = bot_weight / bot_weight.sum()

    w_online_52 = (1.0 - always_on_share) * top_weight + always_on_share * bot_weight
    w_online_52 = w_online_52 / w_online_52.sum()

    return w_online_52


def build_w_br_52(ws_52: pd.DataFrame, pulse_weeks: Iterable[int] = (16, 20)) -> np.ndarray:
    """
    Broadcast週重み w_br_52（pulse weeksに均等配賦）
    """
    if "week_of_year" not in ws_52.columns:
        raise KeyError("ws_52 must have week_of_year")

    w = np.zeros(len(ws_52), dtype=float)
    mask = ws_52["week_of_year"].isin(tuple(pulse_weeks)).values
    if mask.sum() == 0:
        raise ValueError(f"pulse_weeks={tuple(pulse_weeks)} not found in ws_52['week_of_year']")
    w[mask] = 1.0 / mask.sum()
    return w


def calc_roi_score(
    ws_52: pd.DataFrame,
    online_share: float,
    w_online_52: np.ndarray,
    w_br_52: np.ndarray,
) -> float:
    """
    ROIスコア（会計的ROIではない）：週次ROIの重み付き平均
    """
    roi_online = ws_52["roi_online"].values
    roi_br = ws_52["roi_broadcast"].values

    score = (
        online_share * float(np.sum(roi_online * w_online_52)) +
        (1.0 - online_share) * float(np.sum(roi_br * w_br_52))
    )
    return float(score)


# =========================================================
# Step08 main: scenario table builder
# =========================================================

def build_step8_scenario_table_52(
    df_sim_52: pd.DataFrame,
    ws_52: pd.DataFrame,
    avg_df: pd.DataFrame,
    online_share_list: list[float],
    always_on_share_list: list[float],
    boost_quantile: float,
    pulse_weeks: Iterable[int],
    bot_mode: str,
    predict_annual_sales_52_fn: Callable[[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray], float],
) -> pd.DataFrame:
    """
    Step08のシナリオ表を生成する（52週ベース）
    - total_budget は As-Is total 固定
    - OOHは0として Online/Broadcastへ再配分（現行08の仕様を踏襲）
    - As-Is=100 の sales_index / sales_lift_pct、ROI score と ROI lift を出す
    """
    n = 52
    if len(df_sim_52) != n:
        raise ValueError(f"df_sim_52 must be 52 rows. got {len(df_sim_52)}")
    if len(ws_52) != n:
        raise ValueError(f"ws_52 must be 52 rows. got {len(ws_52)}")

    (
        total_budget_as_is, avg_online, avg_br, avg_ooh,
        online_as_is_52, br_as_is_52, ooh_as_is_52
    ) = build_online_as_is_paths_52(avg_df, n_weeks=n)

    # As-Is sales（基準）
    sales_as_is = float(
        predict_annual_sales_52_fn(df_sim_52, online_as_is_52, br_as_is_52, ooh_as_is_52)
    )

    # As-Is ROI score（基準）
    w_online_as_is_52 = (ws_52["baseline_hat"] * ws_52["roi_online"].clip(lower=0)).values
    if w_online_as_is_52.sum() <= 0:
        w_online_as_is_52 = np.ones(n, dtype=float)
    w_online_as_is_52 = w_online_as_is_52 / w_online_as_is_52.sum()

    w_br_as_is_52 = build_w_br_52(ws_52, pulse_weeks=pulse_weeks)

    # As-Isのonline_share（OOH込み total に対する online）
    online_share_as_is = avg_online / total_budget_as_is if total_budget_as_is != 0 else np.nan
    as_is_roi_score = calc_roi_score(ws_52, float(online_share_as_is), w_online_as_is_52, w_br_as_is_52)

    rows: list[dict] = []

    for online_share in online_share_list:
        if not (0.0 <= online_share <= 1.0):
            raise ValueError(f"online_share must be in [0,1]. got {online_share}")
        broadcast_share = 1.0 - online_share

        for always_on_share in always_on_share_list:
            w_online_52 = build_w_online_52(
                ws_52,
                always_on_share=always_on_share,
                boost_quantile=boost_quantile,
                bot_mode=bot_mode,
            )
            w_br_52 = build_w_br_52(ws_52, pulse_weeks=pulse_weeks)

            to_be_roi_score = calc_roi_score(ws_52, online_share, w_online_52, w_br_52)
            roi_lift_pct = (to_be_roi_score / as_is_roi_score - 1) * 100 if as_is_roi_score != 0 else np.nan

            # 予算（OOH=0として再配分：現行08の踏襲）
            online_budget = total_budget_as_is * online_share
            br_budget = total_budget_as_is * broadcast_share

            online_path_52 = online_budget * w_online_52
            br_path_52 = br_budget * w_br_52
            ooh_path_52 = np.zeros(n)

            annual_sales_hat = float(
                predict_annual_sales_52_fn(df_sim_52, online_path_52, br_path_52, ooh_path_52)
            )

            sales_index = (annual_sales_hat / sales_as_is) * 100 if sales_as_is != 0 else np.nan
            sales_lift_pct = (annual_sales_hat / sales_as_is - 1) * 100 if sales_as_is != 0 else np.nan

            rows.append({
                "online_share": float(online_share),
                "broadcast_share": float(broadcast_share),
                "always_on_share": float(always_on_share),     # ★命名統一
                "boost_quantile": float(boost_quantile),
                "pulse_weeks": str(tuple(pulse_weeks)),
                "bot_mode": bot_mode,

                "as_is_roi_score": float(as_is_roi_score),
                "to_be_roi_score": float(to_be_roi_score),
                "roi_lift_pct": float(roi_lift_pct),

                "sales_as_is": float(sales_as_is),
                "annual_sales_hat": float(annual_sales_hat),
                "sales_index_as_is_100": float(sales_index),
                "sales_lift_pct": float(sales_lift_pct),

                "total_budget_as_is": float(total_budget_as_is),
            })

    return (
        pd.DataFrame(rows)
        .sort_values(["online_share", "always_on_share"])
        .reset_index(drop=True)
    )
