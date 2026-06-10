"""
Join offers with main dataset to bring timestamps and arrival/estimate times.
"""
import polars as pl
from agents.utils.logger_setup import get_logger

logger = get_logger("join")


def join_main_info(offers_df: pl.DataFrame, main_df: pl.DataFrame) -> pl.DataFrame:
    logger.info("Joining offers with main waybill info...")
    if "waybill_id" not in main_df.columns:
        logger.warning("main_df lacks 'waybill_id', skipping join.")
        return offers_df
    keep_cols = [
        "waybill_id",
        "estimate_arrived_time",
        "arrive_time",
        "fetch_time",
        "grab_time",
        "sender_lng",
        "sender_lat",
        "recipient_lng",
        "recipient_lat",
    ]
    exist_cols = [c for c in keep_cols if c in main_df.columns]
    main_sub = main_df.select(exist_cols).unique(subset=["waybill_id"]) if "waybill_id" in exist_cols else main_df
    joined = offers_df.join(main_sub, on="waybill_id", how="left")
    logger.info("Join complete")
    return joined
