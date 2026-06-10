"""
Compute courier_active_seconds_so_far per offer using batch algorithm.

This implementation avoids calling Polars grouping APIs that differ across versions.
Instead it builds groups using a Python pass over completed rows, which is robust
and avoids the 'groupby' / 'group_by' and '.list()' API incompatibilities.
"""
from typing import Tuple, Dict, List, Any
import polars as pl
from tqdm import tqdm
from agents.utils.logger_setup import get_logger
from datetime import datetime, timezone
import pytz

logger = get_logger("active_seconds")
SH_TZ = pytz.timezone("Asia/Shanghai")


def _safe_int(x: Any):
    try:
        if x is None:
            return None
        if isinstance(x, str) and x.strip() == "":
            return None
        return int(x)
    except Exception:
        return None


def _epoch_to_local_day(ts):
    """Convert epoch seconds (int) to 'YYYY-MM-DD' in Asia/Shanghai; returns None on failure."""
    try:
        if ts is None:
            return None
        dt = datetime.fromtimestamp(int(ts), tz=timezone.utc).astimezone(SH_TZ)
        return dt.strftime("%Y-%m-%d")
    except Exception:
        return None


def compute_active_seconds(main_df: pl.DataFrame, offers_df: pl.DataFrame) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """
    Args:
      main_df: original waybills dataset (polars DataFrame)
      offers_df: exploded offers dataset (polars DataFrame)

    Returns:
      offers_df augmented with courier_active_seconds_so_far (polars DataFrame),
      batch_df listing the detected batches per courier/day (polars DataFrame)
    """
    logger.info("Computing active seconds per courier-day...")

    if "income" not in main_df.columns:
        logger.info("Income column missing in main_df — computing income per row (this may take a while).")
        from agents.utils.income_utils import compute_income_row

        incs: List[float] = []
        for r in tqdm(main_df.iter_rows(named=True), desc="compute income main"):
            try:
                inc = compute_income_row(
                    r.get("sender_lat"),
                    r.get("sender_lng"),
                    r.get("recipient_lat"),
                    r.get("recipient_lng"),
                    r.get("arrive_time"),
                    r.get("estimate_arrived_time"),
                )
                incs.append(float(inc) if inc is not None else 0.0)
            except Exception:
                incs.append(0.0)
        try:
            main_df = main_df.with_columns([pl.Series("income", incs)])
        except Exception:
            logger.warning("with_columns failed attaching income; using concat fallback.")
            main_df = pl.concat([main_df.reset_index(drop=True), pl.DataFrame({"income": incs}).reset_index(drop=True)], how="horizontal")

    if "local_day" not in main_df.columns:
        logger.info("local_day missing in main_df — deriving from dispatch_time (Asia/Shanghai).")
        if "dispatch_time" in main_df.columns:
            local_days = []
            for ts in main_df.select("dispatch_time").to_series().to_list():
                local_days.append(_epoch_to_local_day(_safe_int(ts)))
            try:
                main_df = main_df.with_columns([pl.Series("local_day", local_days)])
            except Exception:
                main_df = pl.concat([main_df.reset_index(drop=True), pl.DataFrame({"local_day": local_days}).reset_index(drop=True)], how="horizontal")
        else:
            main_df = main_df.with_columns([pl.lit(None).alias("local_day")])

    if "grab_time" in main_df.columns and "arrive_time" in main_df.columns:
        groups_map: Dict[Tuple[int, str], List[Dict]] = {}
        for r in tqdm(main_df.iter_rows(named=True), desc="group completed rows"):
            try:
                g = _safe_int(r.get("grab_time"))
                a = _safe_int(r.get("arrive_time"))
                if g is None or a is None:
                    continue
                courier = r.get("courier_id")
                day = r.get("local_day")
                key = (int(courier) if courier is not None else None, day)
                inc = r.get("income") or 0.0
                groups_map.setdefault(key, []).append({"grab_time": int(g), "arrive_time": int(a), "income": float(inc)})
            except Exception:
                logger.debug("Skipping malformed main_df row while grouping completed tasks.")
                continue
    else:
        groups_map = {}

    batch_records: List[Dict] = []
    for key, tasks in tqdm(groups_map.items(), desc="build batches"):
        try:
            tasks = sorted(tasks, key=lambda x: x["grab_time"])
            if not tasks:
                continue
            current = [tasks[0]]
            earliest_arr = tasks[0]["arrive_time"]
            max_arr = tasks[0]["arrive_time"]
            batches = []
            for task in tasks[1:]:
                fetch = task["grab_time"]
                arrive = task["arrive_time"]
                if fetch <= earliest_arr or arrive <= max_arr or fetch <= max_arr:
                    current.append(task)
                    earliest_arr = min(earliest_arr, arrive)
                    max_arr = max(max_arr, arrive)
                else:
                    batch_start = min(t["grab_time"] for t in current)
                    batch_end = max(t["arrive_time"] for t in current)
                    duration = batch_end - batch_start
                    batches.append((batch_start, batch_end, duration))
                    current = [task]
                    earliest_arr = arrive
                    max_arr = arrive
            if current:
                batch_start = min(t["grab_time"] for t in current)
                batch_end = max(t["arrive_time"] for t in current)
                duration = batch_end - batch_start
                batches.append((batch_start, batch_end, duration))
            for b in batches:
                batch_records.append({
                    "courier_id": key[0],
                    "local_day": key[1],
                    "batch_start": int(b[0]),
                    "batch_end": int(b[1]),
                    "batch_duration": int(b[2]),
                })
        except Exception:
            logger.exception("Failed to build batches for one group; skipping.")

    batch_df = pl.DataFrame(batch_records) if batch_records else pl.DataFrame([])

    batches_by_key: Dict = {}
    for r in batch_df.iter_rows(named=True):
        try:
            key = (int(r.get("courier_id")), r.get("local_day"))
            batches_by_key.setdefault(key, []).append((int(r.get("batch_start")), int(r.get("batch_end")), int(r.get("batch_duration"))))
        except Exception:
            continue

    active_seconds: List[int] = []
    # Process offers without progress bar for cleaner output
    for r in offers_df.iter_rows(named=True):
        try:
            courier_id = r.get("courier_id")
            if courier_id is None:
                active_seconds.append(0)
                continue
            key = (int(courier_id), r.get("local_day"))
            offer_time = _safe_int(r.get("actual_dispatch_time")) or 0
            secs = 0
            if key in batches_by_key:
                for b in batches_by_key[key]:
                    if b[1] < offer_time:
                        secs += b[2]
            active_seconds.append(int(secs))
        except Exception:
            active_seconds.append(0)

    try:
        offers_df = offers_df.with_columns([pl.Series("courier_active_seconds_so_far", active_seconds)])
    except Exception:
        logger.warning("with_columns failed when attaching active seconds; using concat fallback.")
        offers_df = pl.concat([offers_df.reset_index(drop=True), pl.DataFrame({"courier_active_seconds_so_far": active_seconds}).reset_index(drop=True)], how="horizontal")

    logger.info("Active seconds assigned")
    return offers_df, batch_df
