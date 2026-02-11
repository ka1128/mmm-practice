# src/calendar_utils.py
from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Optional, Union

import pandas as pd

DateLike = Union[str, date, datetime, pd.Timestamp]


# ---------------------------
# 基本：型変換
# ---------------------------
def to_date(x: DateLike) -> date:
    """Convert input to python date."""
    if isinstance(x, date) and not isinstance(x, datetime):
        return x
    return pd.to_datetime(x).date()


# ---------------------------
# 週の開始（日曜始まり）
# ---------------------------
def week_start_sunday(d: DateLike) -> date:
    """
    Return Sunday of the week containing d (Sunday-start weeks).
    Python weekday: Mon=0 ... Sun=6
    Sunday-start => subtract (weekday+1)%7
    """
    dd = to_date(d)
    delta = (dd.weekday() + 1) % 7
    return dd - timedelta(days=delta)


# ---------------------------
# FY（7/1始まり）ラベルと開始日
# FYラベルは「期末年」: 2022/7/1〜2023/6/30 => FY2023
# ---------------------------
def fy_year_of(d: DateLike, fy_start_month: int = 7, fy_start_day: int = 1) -> int:
    dd = to_date(d)
    fy_start_this_year = date(dd.year, fy_start_month, fy_start_day)
    return dd.year + (dd >= fy_start_this_year)


def fy_start_date(fy_year: int, fy_start_month: int = 7, fy_start_day: int = 1) -> date:
    return date(fy_year - 1, fy_start_month, fy_start_day)


def fy_week1_sunday(fy_year: int, fy_start_month: int = 7, fy_start_day: int = 1) -> date:
    """
    Define FY week 1 as the Sunday of the week that contains FY start date.
    """
    return week_start_sunday(fy_start_date(fy_year, fy_start_month, fy_start_day))


def fy_week_of(d: DateLike, fy_start_month: int = 7, fy_start_day: int = 1) -> int:
    """
    Fiscal week number (1-based), using Sunday-start weeks.
    """
    dd = to_date(d)
    fy = fy_year_of(dd, fy_start_month, fy_start_day)
    w1 = fy_week1_sunday(fy, fy_start_month, fy_start_day)
    ws = week_start_sunday(dd)
    return ((ws - w1).days // 7) + 1


# ---------------------------
# DataFrameへ列追加（共通の単一ソース）
# ---------------------------
def add_calendar_columns(
    df: pd.DataFrame,
    date_col: str = "Week",
    fy_start_month: int = 7,
    fy_start_day: int = 1,
) -> pd.DataFrame:
    """
    Add consistent calendar fields:
      - week_start (Sunday)
      - iso_year, iso_week (参照用：週番号を説明に使う場合に便利)
      - fy_year, fy_week (意思決定・最適化の主キーに推奨)
      - month (YYYY-MM)
    """
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col])

    out["week_start"] = out[date_col].dt.date.apply(week_start_sunday)

    iso = out[date_col].dt.isocalendar()
    out["iso_year"] = iso.year.astype(int)
    out["iso_week"] = iso.week.astype(int)

    out["fy_year"] = out[date_col].dt.date.apply(lambda x: fy_year_of(x, fy_start_month, fy_start_day)).astype(int)
    out["fy_week"] = out[date_col].dt.date.apply(lambda x: fy_week_of(x, fy_start_month, fy_start_day)).astype(int)

    out["month"] = out[date_col].dt.to_period("M").astype(str)
    return out


def make_fy_calendar(
    fy_year: int,
    fy_start_month: int = 7,
    fy_start_day: int = 1,
    n_weeks: Optional[int] = None,
) -> pd.DataFrame:
    """
    Build FY week calendar with Sunday-start weeks.
    If n_weeks is None => 52 or 53 depending on alignment (事実ベース)
    """
    start_sun = fy_week1_sunday(fy_year, fy_start_month, fy_start_day)
    fy_end_date = fy_start_date(fy_year + 1, fy_start_month, fy_start_day) - timedelta(days=1)
    end_sun = week_start_sunday(fy_end_date)

    if n_weeks is None:
        weeks = ((end_sun - start_sun).days // 7) + 1
    else:
        weeks = int(n_weeks)

    rows = []
    for w in range(1, weeks + 1):
        ws = start_sun + timedelta(days=7 * (w - 1))
        rows.append({"fy_year": fy_year, "fy_week": w, "week_start": ws})

    cal = pd.DataFrame(rows)
    cal["week_start"] = pd.to_datetime(cal["week_start"])
    iso = cal["week_start"].dt.isocalendar()
    cal["iso_year"] = iso.year.astype(int)
    cal["iso_week"] = iso.week.astype(int)
    cal["month"] = cal["week_start"].dt.to_period("M").astype(str)
    return cal
