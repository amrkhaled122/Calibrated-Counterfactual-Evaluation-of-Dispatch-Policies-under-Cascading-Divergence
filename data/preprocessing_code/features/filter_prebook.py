"""
Filter prebooked waybills from cycles data using main waybill info.
"""
from typing import Set, Tuple
import polars as pl
import json
import ast
from agents.utils.logger_setup import get_logger
from .loader import load_main_csv

logger = get_logger("filter_prebook")


def parse_json_list(cell):
    """Robustly parse JSON-like array string into a Python list without using eval.

    Strategy:
      - If already a list -> return as-is
      - Try json.loads on the string form
      - Fallback to ast.literal_eval for single-quoted lists
      - On failure or empty/None -> []
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


def filter_prebooked_waybills(main_df: pl.DataFrame, cycles_df: pl.DataFrame) -> Tuple[pl.DataFrame, Set[int]]:
    logger.info("Filtering prebooked waybills...")
    if "is_prebook" not in main_df.columns or "waybill_id" not in main_df.columns:
        logger.warning("main_df missing is_prebook or waybill_id columns. Skipping filter.")
        return cycles_df, set()

    prebook_df = main_df.filter(pl.col("is_prebook") == 1)
    prebook_waybills = set(prebook_df.select("waybill_id").drop_nulls().to_series().to_list())
    logger.info(f"Found {len(prebook_waybills)} prebooked waybills in main data.")
    if len(prebook_waybills) == 0:
        return cycles_df, set()

    rows = []
    removed_count = 0
    total_entries = 0
    problem_rows = 0

    parallel_cols = [
        "dispatch_courier_ids",
        "waybill_ids",
        "order_ids",
        "sender_lngs",
        "sender_lats",
        "proxy_lngs",
        "proxy_lats",
        "actual_dispatch_times",
        "capacities_at_dispatch",
        "recipient_lngs",
        "recipient_lats",
        "is_assigned_courier_accepted",
        "assigned_courier_id",
    ]

    for idx, row in enumerate(cycles_df.iter_rows(named=True)):
        try:
            waybills = parse_json_list(row.get("waybill_ids"))
            if not waybills:
                rows.append(row)
                continue

            total_entries += len(waybills)
            keep_idx = [i for i, w in enumerate(waybills) if int(w) not in prebook_waybills]

            if len(keep_idx) == 0:
                removed_count += len(waybills)
                continue

            if len(keep_idx) < len(waybills):
                removed_count += len(waybills) - len(keep_idx)
                new_row = dict(row)

                for col in parallel_cols:
                    if col in row and row[col] is not None:
                        lst = parse_json_list(row[col])
                        new_lst = []
                        if len(lst) != len(waybills):
                            logger.warning(
                                f"Cycle {row.get('dispatch_cycle_id', '(unknown)')} parallel column length mismatch: "
                                f"waybill_ids length={len(waybills)} vs {col} length={len(lst)}"
                            )
                            problem_rows += 1
                        for i in keep_idx:
                            if i < len(lst):
                                new_lst.append(lst[i])
                            else:
                                new_lst.append(None)
                        new_row[col] = json.dumps(new_lst)
                rows.append(new_row)
            else:
                rows.append(row)
        except Exception as e:
            logger.exception(f"Exception while processing cycles row index={idx}, dispatch_cycle_id={row.get('dispatch_cycle_id')}")
            rows.append(row)

    logger.info(f"Removed {removed_count} prebooked entries out of {total_entries} total entries.")
    if problem_rows > 0:
        logger.info(f"Encountered {problem_rows} cycles with parallel-array length mismatches (logged warnings).")

    new_df = pl.DataFrame(rows)
    return new_df, prebook_waybills
