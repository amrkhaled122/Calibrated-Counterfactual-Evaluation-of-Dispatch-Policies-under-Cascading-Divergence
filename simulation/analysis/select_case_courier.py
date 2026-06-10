"""
Run simulation and pick a high-divergence courier case study.

This script runs a simulation window, aggregates per-courier divergence statistics,
and selects one courier that best illustrates cascading divergence.
"""

import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import pytz

from simulation.simulator import Simulator


def _log(msg: str, enabled: bool = True) -> None:
    if not enabled:
        return
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {msg}", flush=True)


def _ts_to_shanghai(ts: int) -> str:
    tz = pytz.timezone("Asia/Shanghai")
    return datetime.fromtimestamp(int(ts), tz=tz).strftime("%Y-%m-%d %H:%M:%S")


def _safe_div(n: float, d: float) -> float:
    return n / d if d else 0.0


def _build_simulator(args: argparse.Namespace) -> Simulator:
    max_hours = None
    if args.end_hour is not None:
        max_hours = args.end_hour - args.start_hour
    elif args.hours is not None:
        max_hours = args.hours

    sim = Simulator(
        agent_ckpt=args.agent_ckpt,
        scaler_json=args.scaler,
        thresholds_json=args.thresholds,
        threshold_name=args.threshold_name,
        parquet_path=args.parquet,
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        baseline_mode=False,
        agent_type="ddqn",
        max_hours=max_hours,
        start_hour=args.start_hour,
        verbose=args.sim_verbose,
        travel_time_model_path=args.travel_time_model,
    )
    return sim


def _aggregate_case_stats(sim: Simulator, min_offers: int) -> tuple[list[dict], dict]:
    if sim.dispatcher is None:
        raise RuntimeError("Simulator dispatcher is not initialized. Did setup() run?")
    events = sim.dispatcher.get_events()

    hist = defaultdict(
        lambda: {
            "offers": 0,
            "hist_accepts": 0,
            "hist_rejects": 0,
        }
    )
    for ev in events:
        h = hist[ev.courier_id]
        h["offers"] += 1
        h["hist_accepts"] += int(ev.historical_decision == 1)
        h["hist_rejects"] += int(ev.historical_decision == 0)

    div_origin = defaultdict(int)
    div_assigned = defaultdict(int)
    div_type_counts = defaultdict(lambda: defaultdict(int))
    reassigned_out = defaultdict(int)
    reassigned_in = defaultdict(int)
    lost_from_origin = defaultdict(int)
    divergence_events_by_courier = defaultdict(list)

    for d in sim.metrics.divergences:
        orig = d.courier_id_original
        assg = d.courier_id_assigned
        dtype = d.divergence_type

        div_origin[orig] += 1
        div_type_counts[orig][dtype] += 1
        divergence_events_by_courier[orig].append(
            {
                "role": "original",
                "timestamp": int(d.timestamp),
                "timestamp_shanghai": _ts_to_shanghai(d.timestamp),
                "waybill_id": int(d.waybill_id),
                "divergence_type": dtype,
                "courier_id_original": int(orig),
                "courier_id_assigned": int(assg),
            }
        )

        if assg != -1:
            div_assigned[assg] += 1
            divergence_events_by_courier[assg].append(
                {
                    "role": "assigned",
                    "timestamp": int(d.timestamp),
                    "timestamp_shanghai": _ts_to_shanghai(d.timestamp),
                    "waybill_id": int(d.waybill_id),
                    "divergence_type": dtype,
                    "courier_id_original": int(orig),
                    "courier_id_assigned": int(assg),
                }
            )

        if assg == -1:
            lost_from_origin[orig] += 1
        elif assg != orig:
            reassigned_out[orig] += 1
            reassigned_in[assg] += 1

    rows = []
    for cid, state in sim.courier_states.items():
        h = hist[cid]
        offers = h["offers"]
        if offers < min_offers:
            continue

        hist_accepts = h["hist_accepts"]
        sim_accepts = int(state.waybills_accepted)
        hist_accept_rate = _safe_div(hist_accepts, offers)
        sim_accept_rate = _safe_div(sim_accepts, max(1, int(state.waybills_offered)))

        origin_div = div_origin[cid]
        assigned_div = div_assigned[cid]
        invalidated_count = len(getattr(state, "invalidated_historical_orders", set()))
        late = int(state.late_deliveries)
        delivered = int(state.orders_delivered)
        late_rate = _safe_div(late, delivered)

        score = (
            origin_div * 3.0
            + assigned_div * 1.2
            + reassigned_out[cid] * 2.0
            + lost_from_origin[cid] * 4.0
            + invalidated_count * 1.0
            + abs(sim_accept_rate - hist_accept_rate) * 100.0
            + late_rate * 25.0
        )

        rows.append(
            {
                "courier_id": int(cid),
                "score": round(score, 4),
                "offers": int(offers),
                "historical_accepts": int(hist_accepts),
                "historical_rejects": int(h["hist_rejects"]),
                "historical_accept_rate": round(hist_accept_rate, 4),
                "sim_offers_seen": int(state.waybills_offered),
                "sim_accepts": int(state.waybills_accepted),
                "sim_rejects": int(state.waybills_rejected),
                "sim_delivered": delivered,
                "sim_late_deliveries": late,
                "sim_late_rate": round(late_rate, 4),
                "sim_on_time_deliveries": int(state.on_time_deliveries),
                "sim_total_active_hours": round(
                    float(state.total_active_time) / 3600.0, 4
                ),
                "sim_total_block_hours": round(
                    float(state.total_block_time) / 3600.0, 4
                ),
                "sim_utilization_rate": round(float(state.utilization_rate), 4),
                "divergence_originated": int(origin_div),
                "divergence_assigned": int(assigned_div),
                "reassigned_out": int(reassigned_out[cid]),
                "reassigned_in": int(reassigned_in[cid]),
                "lost_from_origin": int(lost_from_origin[cid]),
                "invalidated_historical_orders": int(invalidated_count),
                "divergence_type_counts": {
                    k: int(v) for k, v in div_type_counts[cid].items()
                },
            }
        )

    rows.sort(key=lambda x: x["score"], reverse=True)
    chosen = rows[0] if rows else {}

    if chosen:
        chosen_cid = chosen["courier_id"]
        timeline = sorted(
            divergence_events_by_courier.get(chosen_cid, []),
            key=lambda x: x["timestamp"],
        )
        chosen["divergence_timeline_sample"] = timeline[:40]

        # Add courier's own historical offer timeline (for narrative figure construction)
        courier_events = [e for e in events if e.courier_id == chosen_cid]
        courier_events.sort(key=lambda e: e.actual_dispatch_time)
        chosen["historical_offer_timeline_sample"] = [
            {
                "timestamp": int(e.actual_dispatch_time),
                "timestamp_shanghai": _ts_to_shanghai(e.actual_dispatch_time),
                "waybill_id": int(e.waybill_id),
                "order_id": int(e.order_id),
                "historical_decision": int(e.historical_decision),
                "dispatch_cycle_id": str(e.dispatch_cycle_id),
                "offer_index_in_cycle": int(e.offer_index_in_cycle),
            }
            for e in courier_events[:80]
        ]

    return rows, chosen


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run simulation and select divergence case courier"
    )
    parser.add_argument(
        "--agent_ckpt",
        default=str(
            Path("agents") / "outputs" / "ddqn_model_integrated" / "ddqn_model.pt"
        ),
    )
    parser.add_argument(
        "--scaler",
        default=str(
            Path("agents")
            / "outputs"
            / "ddqn_model_integrated"
            / "ddqn_scaler.json"
        ),
    )
    parser.add_argument(
        "--thresholds",
        default=str(
            Path("agents") / "outputs" / "bc_model" / "thresholds.json"
        ),
    )
    parser.add_argument("--threshold_name", default="f1_opt")
    parser.add_argument(
        "--parquet",
        default=str(Path("data") / "features" / "offers_observations.parquet"),
    )
    parser.add_argument(
        "--manifest",
        default=str(Path("data") / "features" / "manifest.json"),
    )
    parser.add_argument(
        "--travel_time_model",
        default=str(Path("data") / "travel_time_model.json"),
    )
    parser.add_argument(
        "--output_dir",
        default=str(Path("results") / "ddqn_integrated_case_selection"),
    )
    parser.add_argument("--start_hour", type=int, default=96)
    parser.add_argument("--end_hour", type=int, default=120)
    parser.add_argument("--hours", type=int, default=None)
    parser.add_argument("--min_offers", type=int, default=20)
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print step-level progress messages with timestamps",
    )
    parser.add_argument(
        "--sim_verbose",
        action="store_true",
        help="Enable simulator internal verbose logs",
    )

    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _log("Starting case-courier selection run", args.verbose)
    _log(
        f"Config: start_hour={args.start_hour}, end_hour={args.end_hour}, min_offers={args.min_offers}",
        args.verbose,
    )
    _log(f"Output dir: {out_dir}", args.verbose)

    _log("Building simulator", args.verbose)
    sim = _build_simulator(args)
    _log(
        "Running simulator setup() (loads model, parquet, events, couriers)",
        args.verbose,
    )
    sim.setup()
    _log("Running simulator run() (this is usually the longest step)", args.verbose)
    sim.run()

    _log("Aggregating per-courier divergence statistics", args.verbose)
    ranked, chosen = _aggregate_case_stats(sim, min_offers=args.min_offers)
    _log(f"Ranked couriers: {len(ranked)}", args.verbose)

    metadata = {
        "agent_type": "ddqn",
        "agent_ckpt": args.agent_ckpt,
        "scaler": args.scaler,
        "start_hour": args.start_hour,
        "end_hour": args.end_hour,
        "hours": args.hours,
        "min_offers": args.min_offers,
        "total_ranked_couriers": len(ranked),
    }

    with open(out_dir / "courier_case_ranking.json", "w", encoding="utf-8") as f:
        json.dump({"metadata": metadata, "ranked_couriers": ranked[:200]}, f, indent=2)
    _log("Wrote courier_case_ranking.json", args.verbose)

    with open(out_dir / "chosen_courier_case_study.json", "w", encoding="utf-8") as f:
        json.dump({"metadata": metadata, "chosen_courier": chosen}, f, indent=2)
    _log("Wrote chosen_courier_case_study.json", args.verbose)

    if chosen:
        print("\nChosen courier for case study:")
        print(f"  courier_id: {chosen['courier_id']}")
        print(f"  score: {chosen['score']}")
        print(f"  offers: {chosen['offers']}")
        print(f"  divergence_originated: {chosen['divergence_originated']}")
        print(f"  reassigned_out: {chosen['reassigned_out']}")
        print(f"  lost_from_origin: {chosen['lost_from_origin']}")
        print(f"  sim_late_rate: {chosen['sim_late_rate']}")

    print(f"\nSaved ranking to: {out_dir / 'courier_case_ranking.json'}")
    print(f"Saved selected case study to: {out_dir / 'chosen_courier_case_study.json'}")


if __name__ == "__main__":
    main()
