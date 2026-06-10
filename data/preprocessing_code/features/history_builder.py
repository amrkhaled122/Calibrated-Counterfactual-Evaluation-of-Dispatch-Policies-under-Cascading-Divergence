"""
Build courier-day cumulative history features using polars.

This version is compatible with multiple polars versions: it will attempt to call
`groupby(...).apply(...)` and fall back to `group_by(...).apply(...)` or a pure-Python
group loop if needed.

Produces:
 - offers_so_far
 - accepts_so_far
 - completed_so_far
 - acceptance_rate_so_far
 - late_ratio_so_far
 - time_since_last_accept_s
"""
import polars as pl
from agents.utils.logger_setup import get_logger
from tqdm import tqdm
import math
from collections import defaultdict

logger = get_logger("history")


def _has_attr(obj, name):
    return hasattr(obj, name)


def _group_apply_compat(df: pl.DataFrame, group_cols, func):
    """
    Apply `func` to groups of df defined by group_cols using whatever GroupBy API exists.
    Returns concatenated DataFrame.
    """
    try:
        if _has_attr(df, "groupby"):
            gb = df.groupby(group_cols, maintain_order=True)
            if _has_attr(gb, "apply"):
                return gb.apply(func)
    except Exception:
        logger.debug("groupby(...).apply(...) not available or failed, falling back")

    try:
        if _has_attr(df, "group_by"):
            gb = df.group_by(group_cols, maintain_order=True)
            if _has_attr(gb, "apply"):
                return gb.apply(func)
    except Exception:
        logger.debug("group_by(...).apply(...) not available or failed, falling back")

    logger.info("Falling back to Python-loop grouping (slower).")
    pieces = []
    groups = defaultdict(list)
    for row in df.iter_rows(named=True):
        key = tuple(row[c] for c in group_cols)
        groups[key].append(row)
    for key, rows in tqdm(groups.items(), desc="manual groups"):
        small_df = pl.DataFrame(rows)
        out = func(small_df)
        pieces.append(out)
    if pieces:
        return pl.concat(pieces)
    else:
        return df


def build_courier_day_history(offers_df: pl.DataFrame) -> pl.DataFrame:
    logger.info("Building courier-day history features...")

    if "actual_dispatch_time" in offers_df.columns:
        offers_df = offers_df.with_columns([pl.col("actual_dispatch_time").cast(pl.Int64)])
    else:
        offers_df = offers_df.with_columns([pl.lit(None).alias("actual_dispatch_time")])

    if "courier_id" in offers_df.columns:
        offers_df = offers_df.with_columns([pl.col("courier_id").cast(pl.Int64)])
    else:
        offers_df = offers_df.with_columns([pl.lit(None).alias("courier_id")])

    if "waybill_id" in offers_df.columns:
        offers_df = offers_df.with_columns([pl.col("waybill_id").cast(pl.Int64)])
    else:
        offers_df = offers_df.with_columns([pl.lit(None).alias("waybill_id")])

    sort_cols = [c for c in ["courier_id", "local_day", "actual_dispatch_time"] if c in offers_df.columns]
    if sort_cols:
        offers_df = offers_df.sort(sort_cols)

    offers_df = offers_df.with_columns([
        pl.when(pl.col("is_assigned_courier_accepted").is_null()).then(0).otherwise(pl.col("is_assigned_courier_accepted").cast(pl.Int64)).alias("is_accept"),
        pl.when(pl.col("arrive_time").is_null()).then(0).otherwise(1).alias("is_completed"),
        pl.when((pl.col("arrive_time").is_null()) | (pl.col("estimate_arrived_time").is_null())).then(0)
          .otherwise((pl.col("arrive_time") > pl.col("estimate_arrived_time")).cast(pl.Int64)).alias("is_late"),
    ])

    def _group_cums(df: pl.DataFrame) -> pl.DataFrame:
        """Given a sub-DataFrame for a courier/day, compute cumulative counts."""
        if "actual_dispatch_time" in df.columns:
            df = df.sort("actual_dispatch_time")
        df = df.with_columns([
            (pl.arange(0, df.height) + 1).alias("offers_cum"),
            (pl.col("is_accept").cum_sum()).alias("accepts_cum"),
            (pl.col("is_completed").cum_sum()).alias("completed_cum"),
            (pl.col("is_late").cum_sum()).alias("late_cum"),
        ])
        return df

    if "courier_id" in offers_df.columns and "local_day" in offers_df.columns:
        offers_df = _group_apply_compat(offers_df, ["courier_id", "local_day"], _group_cums)
    else:
        offers_df = _group_cums(offers_df)

    offers_df = offers_df.with_columns([
        (pl.col("offers_cum").shift(1).fill_null(0)).alias("offers_so_far"),
        (pl.col("accepts_cum").shift(1).fill_null(0)).alias("accepts_so_far"),
        (pl.col("completed_cum").shift(1).fill_null(0)).alias("completed_so_far"),
        (pl.col("late_cum").shift(1).fill_null(0)).alias("late_so_far"),
    ])

    offers_df = offers_df.with_columns([
        pl.when(pl.col("offers_so_far") == 0).then(0.0).otherwise(pl.col("accepts_so_far") / pl.col("offers_so_far")).alias("acceptance_rate_so_far"),
        pl.when(pl.col("completed_so_far") == 0).then(0.0).otherwise(pl.col("late_so_far") / pl.col("completed_so_far")).alias("late_ratio_so_far"),
    ])

    def _attach_last_accept(df_group: pl.DataFrame) -> pl.DataFrame:
        g = df_group.sort("actual_dispatch_time") if "actual_dispatch_time" in df_group.columns else df_group
        last = None
        last_list = []
        for row in g.iter_rows(named=True):
            last_list.append(last)
            if int(row.get("is_accept", 0)) == 1:
                adt = row.get("actual_dispatch_time")
                if adt is not None:
                    last = int(adt)
        return g.with_columns([pl.Series("last_accept_dispatch_time_prev", last_list)])

    if "courier_id" in offers_df.columns and "local_day" in offers_df.columns:
        offers_df = _group_apply_compat(offers_df, ["courier_id", "local_day"], _attach_last_accept)
    else:
        offers_df = _attach_last_accept(offers_df)

    offers_df = offers_df.with_columns([
        pl.when(pl.col("last_accept_dispatch_time_prev").is_null()).then(-1).otherwise(pl.col("actual_dispatch_time") - pl.col("last_accept_dispatch_time_prev")).alias("time_since_last_accept_s")
    ])

    for c in ["offers_so_far", "accepts_so_far", "completed_so_far", "time_since_last_accept_s"]:
        if c in offers_df.columns:
            offers_df = offers_df.with_columns([pl.col(c).cast(pl.Int64)])

    if "acceptance_rate_so_far" in offers_df.columns:
        offers_df = offers_df.with_columns([pl.col("acceptance_rate_so_far").cast(pl.Float32)])
    if "late_ratio_so_far" in offers_df.columns:
        offers_df = offers_df.with_columns([pl.col("late_ratio_so_far").cast(pl.Float32)])

    logger.info("History features built")
    return offers_df
