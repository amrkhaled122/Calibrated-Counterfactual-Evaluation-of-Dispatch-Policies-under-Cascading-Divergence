"""
Load CSVs using polars with robust dtype coercion and basic validation.
"""
from typing import Tuple
import polars as pl
from agents.utils.logger_setup import get_logger

logger = get_logger("loader")

def load_main_csv(path: str) -> pl.DataFrame:
    logger.info(f"Loading main CSV: {path}")
    try:
        df = pl.read_csv(path, try_parse_dates=False)
    except Exception as e:
        logger.exception("Failed to read main CSV")
        raise
    df = df.rename({c: c.strip() for c in df.columns})
    int_cols = [
        "order_id",
        "waybill_id",
        "courier_id",
        "da_id",
        "is_courier_grabbed",
        "is_weekend",
        "estimate_arrived_time",
        "is_prebook",
        "poi_id",
        "dispatch_time",
        "grab_time",
        "fetch_time",
        "arrive_time",
        "estimate_meal_prepare_time",
        "order_push_time",
        "platform_order_time",
    ]
    for c in int_cols:
        if c in df.columns:
            df = df.with_columns([pl.col(c).cast(pl.Int64, strict=False).alias(c)])
    
    # Coordinate columns need special handling:
    # The raw CSV stores coordinates as scaled integers (e.g., 174579111 = 174.579111 degrees)
    # We need to convert them to proper decimal degrees by dividing by 1,000,000
    coord_cols = ["sender_lng", "sender_lat", "recipient_lng", "recipient_lat", "grab_lng", "grab_lat"]
    for c in coord_cols:
        if c in df.columns:
            # First cast to Float64, then scale down by 1,000,000 to get decimal degrees
            df = df.with_columns([
                (pl.col(c).cast(pl.Float64, strict=False) / 1_000_000).alias(c)
            ])
            logger.debug(f"Scaled coordinate column {c} from integers to decimal degrees")

    return df


def load_cycles_csv(path: str) -> pl.DataFrame:
    logger.info(f"Loading cycles CSV: {path}")
    try:
        df = pl.read_csv(path, try_parse_dates=False)
    except Exception:
        logger.exception("Failed to read cycles CSV")
        raise
    df = df.rename({c: c.strip() for c in df.columns})
    logger.info(f"Cycles CSV loaded with {df.height} rows and {len(df.columns)} columns")
    return df

