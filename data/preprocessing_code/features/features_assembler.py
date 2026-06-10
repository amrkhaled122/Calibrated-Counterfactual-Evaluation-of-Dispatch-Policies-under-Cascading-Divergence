"""
Assemble final features for each offer row.

Produces / attaches:
 - eta_seconds_current
 - tod_sin, tod_cos (Asia/Shanghai) computed from actual_dispatch_time
 - income_cur (income estimate for this order)  [uses income_utils.compute_income_row]
 - income_rate_for_the_day (if available on main_df join)  -- best-effort
 - courier_active_seconds_so_far (should already exist from active_seconds)
 - and ensures type safety + uses with_columns(...) (polars compatibility)

This file is defensive: it tolerates missing columns and None values and logs as needed.
"""
from typing import List
import polars as pl
from math import sin, cos, pi
from datetime import datetime, timezone
import pytz
from tqdm import tqdm
from agents.utils.logger_setup import get_logger

logger = get_logger("assembler")
SH_TZ = pytz.timezone("Asia/Shanghai")


def _safe_int(x):
    try:
        if x is None:
            return None
        if isinstance(x, str) and x.strip() == "":
            return None
        return int(x)
    except Exception:
        return None


def _epoch_to_tod_sin_cos(ts):
    """Convert epoch seconds to tod sin/cos in Asia/Shanghai. Returns (sin, cos) or (0.,1.) on failure."""
    try:
        if ts is None:
            return 0.0, 1.0
        dt = datetime.fromtimestamp(int(ts), tz=timezone.utc).astimezone(SH_TZ)
        seconds_of_day = dt.hour * 3600 + dt.minute * 60 + dt.second
        cyc = 2 * pi * (seconds_of_day / 86400.0)
        return sin(cyc), cos(cyc)
    except Exception:
        return 0.0, 1.0


def assemble_features(offers_df: pl.DataFrame) -> pl.DataFrame:
    """
    Attach engineered features required by the observation vector.
    Returns augmented offers_df (polars DataFrame).
    
    Features added:
        - eta_seconds_current: Platform ETA (estimate_arrived_time - actual_dispatch_time)
        - tod_sin, tod_cos: Cyclical time-of-day encoding (Asia/Shanghai timezone)
        - income_cur: Estimated income for this order
        - order_income_value: Alias for income_cur
        - income_rate_for_the_day: Running income rate (CNY/hour)
        - courier_active_seconds_so_far: Cumulative working time
    """
    logger.info(f"Assembling features for {offers_df.height} rows...")

    if "actual_dispatch_time" in offers_df.columns:
        offers_df = offers_df.with_columns([pl.col("actual_dispatch_time").cast(pl.Int64)])
    else:
        offers_df = offers_df.with_columns([pl.lit(None).alias("actual_dispatch_time")])

    # Compute ETA (vectorized approach preferred, fallback to row-wise)
    eta_list: List[int] = []
    for row in offers_df.iter_rows(named=True):
        est = row.get("estimate_arrived_time") if "estimate_arrived_time" in offers_df.columns else None
        adt = row.get("actual_dispatch_time")
        try:
            est_i = _safe_int(est)
            adt_i = _safe_int(adt)
            if (est_i is None) or (adt_i is None):
                eta_list.append(-1)
            else:
                eta_list.append(max(0, int(est_i - adt_i)))
        except Exception:
            eta_list.append(-1)

    try:
        offers_df = offers_df.with_columns([pl.Series("eta_seconds_current", [int(x) for x in eta_list])])
    except Exception:
        logger.warning("with_columns failed for eta attachment; using concat fallback.")
        offers_df = pl.concat([offers_df.reset_index(drop=True), pl.DataFrame({"eta_seconds_current": eta_list}).reset_index(drop=True)], how="horizontal")

    # Compute time-of-day encoding
    tod_sin = []
    tod_cos = []
    for row in offers_df.iter_rows(named=True):
        adt = row.get("actual_dispatch_time")
        s, c = _epoch_to_tod_sin_cos(adt)
        tod_sin.append(float(s))
        tod_cos.append(float(c))
    try:
        offers_df = offers_df.with_columns([pl.Series("tod_sin", tod_sin), pl.Series("tod_cos", tod_cos)])
    except Exception:
        offers_df = pl.concat([offers_df.reset_index(drop=True), pl.DataFrame({"tod_sin": tod_sin, "tod_cos": tod_cos}).reset_index(drop=True)], how="horizontal")

    income_cur = []
    try:
        from agents.utils.income_utils import compute_income_row
        for row in tqdm(offers_df.iter_rows(named=True), desc="compute income_cur"):
            try:
                inc = compute_income_row(
                    row.get("sender_lat"),
                    row.get("sender_lng"),
                    row.get("recipient_lat"),
                    row.get("recipient_lng"),
                    row.get("arrive_time"),
                    row.get("estimate_arrived_time"),
                )
                income_cur.append(float(inc) if inc is not None else 0.0)
            except Exception:
                income_cur.append(0.0)
    except Exception:
        logger.warning("income_utils.compute_income_row not available; filling income_cur with 0.0")
        income_cur = [0.0] * offers_df.height

    try:
        offers_df = offers_df.with_columns([pl.Series("income_cur", income_cur)])
    except Exception:
        offers_df = pl.concat([offers_df.reset_index(drop=True), pl.DataFrame({"income_cur": income_cur}).reset_index(drop=True)], how="horizontal")

    # Provide a stable, descriptive alias for downstream consumers
    try:
        offers_df = offers_df.with_columns([pl.col("income_cur").alias("order_income_value")])
    except Exception:
        try:
            offers_df = pl.concat([
                offers_df.reset_index(drop=True),
                pl.DataFrame({"order_income_value": income_cur}).reset_index(drop=True)
            ], how="horizontal")
        except Exception:
            pass

    if "courier_active_seconds_so_far" not in offers_df.columns:
        offers_df = offers_df.with_columns([pl.Series("courier_active_seconds_so_far", [0] * offers_df.height)])

    if "income_rate_for_the_day" not in offers_df.columns:
        try:
            # Build daily income totals per courier
            totals = {}
            for r in offers_df.iter_rows(named=True):
                cid = r.get("courier_id")
                day = r.get("local_day")
                inc = r.get("income_cur") or 0.0
                key = (cid, day)
                totals[key] = totals.get(key, 0.0) + float(inc)
            inc_rate_list = []
            for r in offers_df.iter_rows(named=True):
                key = (r.get("courier_id"), r.get("local_day"))
                total_inc = totals.get(key, 0.0)
                active_sec = r.get("courier_active_seconds_so_far") or 0
                hours = max(1.0, float(active_sec) / 3600.0)
                inc_rate_list.append(float(total_inc) / hours)
            offers_df = offers_df.with_columns([pl.Series("income_rate_for_the_day", inc_rate_list)])
        except Exception:
            logger.warning("Failed to compute income_rate_for_the_day vectorizedly; filling zeros.")
            offers_df = offers_df.with_columns([pl.Series("income_rate_for_the_day", [0.0] * offers_df.height)])

    cast_cols = {
        "eta_seconds_current": pl.Int64,
        "tod_sin": pl.Float32,
        "tod_cos": pl.Float32,
        "income_cur": pl.Float32,
        "order_income_value": pl.Float32,
        "income_rate_for_the_day": pl.Float32,
        "courier_active_seconds_so_far": pl.Int64,
    }
    casts = []
    for cname, ctype in cast_cols.items():
        if cname in offers_df.columns:
            casts.append(pl.col(cname).cast(ctype).alias(cname))
    if casts:
        offers_df = offers_df.with_columns(casts)

    logger.info("Features assembled")
    return offers_df
