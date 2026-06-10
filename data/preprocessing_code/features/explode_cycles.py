"""
Explode cycles into per-offer rows preserving alignment.

This version is defensive: it tolerates None entries and mismatched parallel-array lengths.
"""
from typing import List
import ast
import json
import polars as pl
from tqdm import tqdm
from agents.utils.logger_setup import get_logger

logger = get_logger("explode")


def parse_json_list(cell):
    """Robustly parse JSON-like array string into Python list.

    Handles:
      - Python-list string: "[1, 2, 3]"
      - JSON array string
      - actual Python list already
      - None / empty -> []
    """
    if cell is None:
        return []
    if isinstance(cell, list):
        return cell
    try:
        s = str(cell).strip()
        if s == "":
            return []
        return json.loads(s)
    except Exception:
        try:
            return ast.literal_eval(str(cell))
        except Exception:
            return []


def _safe_float(x):
    """Return float(x) or None if x is None/invalid."""
    try:
        if x is None:
            return None
        if isinstance(x, str):
            s = x.strip()
            if s == "" or s.lower() in ("none", "null"):
                return None
        return float(x)
    except Exception:
        return None


def _safe_int(x):
    """Return int(x) or None if x is None/invalid."""
    try:
        if x is None:
            return None
        if isinstance(x, str):
            s = x.strip()
            if s == "" or s.lower() in ("none", "null"):
                return None
        return int(x)
    except Exception:
        return None


def explode_cycles(cycles_df: pl.DataFrame) -> pl.DataFrame:
    """
    Explode a cycles dataframe into per-offer rows.
    Each cycle row may contain parallel JSON arrays (waybill_ids, dispatch_courier_ids, etc).
    We preserve alignment; if an array is shorter, treat missing entries as None.
    """
    logger.info("Exploding cycles into offers...")
    offers = []

    total = cycles_df.height
    for row in tqdm(cycles_df.iter_rows(named=True), total=total, desc="Exploding cycles"):
        courier_ids = parse_json_list(row.get("dispatch_courier_ids"))
        waybill_ids = parse_json_list(row.get("waybill_ids"))
        order_ids = parse_json_list(row.get("order_ids"))
        sender_lngs = parse_json_list(row.get("sender_lngs"))
        sender_lats = parse_json_list(row.get("sender_lats"))
        proxy_lngs = parse_json_list(row.get("proxy_lngs"))
        proxy_lats = parse_json_list(row.get("proxy_lats"))
        actual_dispatch_times = parse_json_list(row.get("actual_dispatch_times"))
        capacities = parse_json_list(row.get("capacities_at_dispatch"))
        recipient_lngs = parse_json_list(row.get("recipient_lngs"))
        recipient_lats = parse_json_list(row.get("recipient_lats"))
        assigned_courier_arr = parse_json_list(row.get("assigned_courier_id")) if row.get("assigned_courier_id") is not None else []
        accepted_arr = parse_json_list(row.get("is_assigned_courier_accepted")) if row.get("is_assigned_courier_accepted") is not None else []

        n = max(len(waybill_ids), len(courier_ids))

        try:
            unique_couriers_in_cycle = len(set(int(x) for x in courier_ids if x is not None))
        except Exception:
            unique_couriers_in_cycle = len(set(courier_ids))
        try:
            unique_waybills_in_cycle = len(set(int(x) for x in waybill_ids if x is not None))
        except Exception:
            unique_waybills_in_cycle = len(set(waybill_ids))

        for i in range(n):
            try:
                offer = {
                    "dispatch_cycle_id": row.get("dispatch_cycle_id"),
                    "local_day": row.get("local_day"),
                    "dispatch_start_time": _safe_int(row.get("dispatch_start_time")),
                    "dispatch_end_time": _safe_int(row.get("dispatch_end_time")),
                    "offer_index_in_cycle": int(i),
                    "unique_couriers_in_cycle": int(unique_couriers_in_cycle),
                    "unique_waybills_in_cycle": int(unique_waybills_in_cycle),
                }

                offer["courier_id"] = _safe_int(courier_ids[i]) if i < len(courier_ids) else None
                offer["waybill_id"] = _safe_int(waybill_ids[i]) if i < len(waybill_ids) else None
                offer["order_id"] = _safe_int(order_ids[i]) if i < len(order_ids) else None

                offer["sender_lng"] = _safe_float(sender_lngs[i]) if i < len(sender_lngs) else None
                offer["sender_lat"] = _safe_float(sender_lats[i]) if i < len(sender_lats) else None
                offer["proxy_lng"] = _safe_float(proxy_lngs[i]) if i < len(proxy_lngs) else None
                offer["proxy_lat"] = _safe_float(proxy_lats[i]) if i < len(proxy_lats) else None
                offer["recipient_lng"] = _safe_float(recipient_lngs[i]) if i < len(recipient_lngs) else None
                offer["recipient_lat"] = _safe_float(recipient_lats[i]) if i < len(recipient_lats) else None

                offer["actual_dispatch_time"] = _safe_int(actual_dispatch_times[i]) if i < len(actual_dispatch_times) else None
                cap = _safe_int(capacities[i]) if i < len(capacities) else None
                offer["capacity_at_dispatch"] = int(cap) if cap is not None else 0

                offer["assigned_courier_id"] = _safe_int(assigned_courier_arr[i]) if i < len(assigned_courier_arr) else None
                ia = _safe_int(accepted_arr[i]) if i < len(accepted_arr) else None
                offer["is_assigned_courier_accepted"] = int(ia) if ia is not None else 0

                offers.append(offer)
            except Exception:
                logger.exception("Failed to explode one element; skipping")
                continue

    offers_df = pl.DataFrame(offers)
    logger.info(f"Exploded into {offers_df.height} offers.")
    return offers_df
