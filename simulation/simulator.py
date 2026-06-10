"""
Main Simulator: Orchestrates all components for agent evaluation.

Runs event-driven simulation:
1. Load historical offers (Dispatcher)
2. For each offer event:
   - Agent decides (keep/prune)
   - If aligned with history → log replay (no simulation)
   - If diverged → re-score next courier, simulate routing
3. Track metrics (divergences, capacity, income)
"""

import json
import time as time_module
from pathlib import Path
from typing import Dict, Optional, List, Tuple
import numpy as np
import pandas as pd
from tqdm import tqdm

from .dispatcher import Dispatcher, OfferEvent
from .scorer import Scorer, CourierState, haversine_km
from .router import Router, Order
from .capacity_tracker import CapacityTracker
from .metrics import MetricsLogger
from .baseline_agent import BaselineAgent
from .order_tracker import OrderTracker, OrderState
from .utilization_tracker import UtilizationTracker
from .travel_time_model import TravelTimeModel

# Import agents and income utils from main project
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from agents.agents.bc_agent import BCPruningAgent
from agents.agents.ddqn_agent import DDQNAgent
from agents.utils.income_utils import compute_income_row


class SimulationProfiler:
    """
    Profiler to measure time spent in different simulation operations.
    Helps identify bottlenecks.
    """

    def __init__(self):
        self.timings: Dict[str, float] = {}
        self.counts: Dict[str, int] = {}
        self._start_times: Dict[str, float] = {}

    def start(self, operation: str):
        """Start timing an operation."""
        self._start_times[operation] = time_module.perf_counter()

    def stop(self, operation: str):
        """Stop timing an operation and accumulate."""
        if operation in self._start_times:
            elapsed = time_module.perf_counter() - self._start_times[operation]
            self.timings[operation] = self.timings.get(operation, 0.0) + elapsed
            self.counts[operation] = self.counts.get(operation, 0) + 1
            del self._start_times[operation]

    def get_summary(self) -> Dict:
        """Get profiling summary."""
        total_time = sum(self.timings.values())
        summary = {}
        for op, elapsed in sorted(self.timings.items(), key=lambda x: -x[1]):
            count = self.counts.get(op, 1)
            avg_per_call = elapsed / count if count > 0 else 0
            pct = (elapsed / total_time * 100) if total_time > 0 else 0
            summary[op] = {
                "total_seconds": round(elapsed, 3),
                "count": count,
                "avg_ms": round(avg_per_call * 1000, 4),
                "pct_of_total": round(pct, 1),
            }
        summary["_total_profiled_seconds"] = round(total_time, 3)
        return summary

    def print_summary(self):
        """Print profiling summary to console."""
        summary = self.get_summary()
        total = summary.pop("_total_profiled_seconds", 0)

        print("\n" + "=" * 80)
        print("SIMULATION PROFILING SUMMARY")
        print("=" * 80)
        print(
            f"{'Operation':<40} {'Total(s)':<12} {'Count':<10} {'Avg(ms)':<12} {'%':<8}"
        )
        print("-" * 80)

        for op, data in summary.items():
            print(
                f"{op:<40} {data['total_seconds']:<12} {data['count']:<10} {data['avg_ms']:<12} {data['pct_of_total']:<8}"
            )

        print("-" * 80)
        print(f"{'TOTAL PROFILED TIME':<40} {total:<12}")
        print("=" * 80)


class Simulator:
    """
    Main simulation orchestrator.

    Flow:
    - Load agent, historical data, manifest
    - For each offer event:
      - Query agent decision
      - Compare with historical decision
      - If aligned: log replay
      - If diverged: re-score courier, simulate routing, track metrics
    """

    def __init__(
        self,
        agent_ckpt: str,
        scaler_json: str,
        thresholds_json: str,
        threshold_name: str,
        parquet_path: str,
        manifest_path: str,
        output_dir: str,
        baseline_mode: bool = False,
        agent_type: str = "bc",
        max_hours: Optional[int] = None,
        start_hour: int = 0,
        verbose: bool = False,
        travel_time_model_path: Optional[str] = None,
        scorer_mode: str = "distance_only",
        scorer_distance_weight: float = 1.0,
        scorer_load_penalty_weight: float = 0.35,
        scorer_late_penalty_weight: float = 0.35,
        scorer_idle_bonus_weight: float = 0.15,
        max_courier_distance_km: float = 3.0,
        block_buffer_seconds: int = 60,
        max_pending_per_cycle: int = 50,
        pending_cycles_limit: int = 5,
        divergence_window_seconds: int = 40 * 60,
        travel_time_add_noise: bool = True,
    ):
        self.agent_ckpt = agent_ckpt
        self.scaler_json = scaler_json
        self.thresholds_json = thresholds_json
        self.threshold_name = threshold_name
        self.parquet_path = parquet_path
        self.manifest_path = manifest_path
        self.output_dir = output_dir
        self.baseline_mode = baseline_mode
        self.agent_type = agent_type
        self.max_hours = max_hours
        self.start_hour = start_hour  # Skip first N hours (for train/eval split)
        self.verbose = verbose
        self.travel_time_model_path = travel_time_model_path
        self.scorer_mode = scorer_mode
        self.scorer_distance_weight = scorer_distance_weight
        self.scorer_load_penalty_weight = scorer_load_penalty_weight
        self.scorer_late_penalty_weight = scorer_late_penalty_weight
        self.scorer_idle_bonus_weight = scorer_idle_bonus_weight
        self.max_courier_distance_km = max_courier_distance_km
        self.block_buffer_seconds = block_buffer_seconds
        self.max_pending_per_cycle = max_pending_per_cycle
        self.pending_cycles_limit = pending_cycles_limit
        self.divergence_window_seconds = divergence_window_seconds
        self.travel_time_add_noise = travel_time_add_noise

        # Logging setup
        self.event_log = []  # Will store detailed event logs
        self.log_file_handle = None  # File handle for verbose logging

        # Load manifest
        with open(manifest_path, "r") as f:
            self.manifest = json.load(f)

        # Initialize components
        self.agent = None
        self.dispatcher = None
        self.scorer = None
        self.router = Router()
        self.capacity_tracker = CapacityTracker()
        self.metrics = MetricsLogger()
        self.order_tracker = OrderTracker()
        self.utilization_tracker = UtilizationTracker()
        self.travel_time_model = None  # Will be loaded in setup() if path provided

        # State tracking
        self.courier_states: Dict[int, CourierState] = {}
        self.courier_next_dispatch: Dict[
            int, int
        ] = {}  # courier_id -> next dispatch time

        # Actual ETA lookup (order_id -> actual eta in seconds)
        self.actual_eta_lookup: Dict[int, int] = {}

        # Diverged couriers tracking: couriers whose trajectory differs from history
        # These couriers' subsequent offers need re-scoring since their location is no longer
        # what the historical proxy_lng/lat suggests
        self.diverged_couriers: Dict[
            int, Dict
        ] = {}  # courier_id -> {reason, timestamp, simulated_location}

        # Cycle tracking
        self.current_cycle_id: int = 0
        self.current_cycle_start_time: int = 0
        self.current_cycle_end_time: int = 0
        self.cycle_events: List = []  # Events in current cycle

        # Pending orders for reassignment (rejected orders waiting for next cycles)
        # Structure: {waybill_id: {'order': OrderState, 'event': OfferEvent, 'cycles_remaining': int, 'original_cycle': int}}
        self.pending_reassignments: Dict[int, Dict] = {}

        # Track the latest delivery completion time (for extending simulation beyond time window)
        # This eliminates artificial idle time when couriers have orders in transit at end of simulation
        self.latest_delivery_completion_time: int = 0
        self.simulation_end_time: int = 0  # Actual end of simulation cycles
        self.extended_end_time: int = 0  # Extended end time based on pending deliveries

        # Profiler for performance analysis
        self.profiler = SimulationProfiler()

    def setup(self):
        """Initialize all components."""
        print("Setting up simulation...")

        # Load agent (skip if already injected, e.g., during training)
        if self.agent is not None:
            print(f"Using pre-injected agent: {type(self.agent).__name__}")
        elif self.baseline_mode:
            print(
                "Running in BASELINE mode (no agent intervention - pure historical replay)"
            )
            self.agent = BaselineAgent()
        elif self.agent_type == "ddqn":
            print(f"Loading DDQN agent from {self.agent_ckpt}...")
            self.agent = DDQNAgent(
                model_ckpt=self.agent_ckpt, scaler_json=self.scaler_json
            )
            print(
                f"  DDQN agent loaded with {len(self.agent.feature_names)} features, gamma={self.agent.gamma}"
            )
        else:  # default to BC
            print(f"Loading BC (Behavior Cloning) agent from {self.agent_ckpt}...")
            self.agent = BCPruningAgent(
                model_ckpt=self.agent_ckpt,
                scaler_json=self.scaler_json,
                thresholds_json=self.thresholds_json,
                threshold_name=self.threshold_name,
            )
            print(f"  BC agent loaded with threshold={self.agent.threshold:.4f}")

        # Load dispatcher
        self.dispatcher = Dispatcher(
            self.parquet_path,
            self.manifest,
            max_hours=self.max_hours,
            start_hour=self.start_hour,
        )
        self.dispatcher.load()

        # Get relevant order IDs from loaded events for filtered data loading
        events = self.dispatcher.get_events()
        relevant_order_ids = set(ev.waybill_id for ev in events)

        # Load actual ETA data (filtered to relevant orders only)
        self._load_actual_eta(relevant_order_ids)

        # Initialize courier states from historical data
        self._initialize_courier_states()

        # Create scorer
        self.scorer = Scorer(
            self.courier_states,
            max_courier_distance_km=self.max_courier_distance_km,
            block_buffer_seconds=self.block_buffer_seconds,
            scorer_mode=self.scorer_mode,
            distance_weight=self.scorer_distance_weight,
            load_penalty_weight=self.scorer_load_penalty_weight,
            late_penalty_weight=self.scorer_late_penalty_weight,
            idle_bonus_weight=self.scorer_idle_bonus_weight,
        )

        # Load travel time model if path provided
        if self.travel_time_model_path:
            print(f"Loading travel time model from {self.travel_time_model_path}...")
            self.travel_time_model = TravelTimeModel.load(self.travel_time_model_path)
            print(
                "  ✓ Travel time model loaded - will use empirical pickup/delivery times"
            )
        else:
            print("  ⚠ No travel time model provided - using historical ETA fallback")

        print("Setup complete.")

    def _load_actual_eta(self, relevant_order_ids: set = None):
        """Load actual ETA data from extracted CSV, optionally filtered to relevant orders."""
        eta_path = (
            Path(__file__).parent.parent / "data" / "actual_eta_by_order.csv"
        )

        if not eta_path.exists():
            raise FileNotFoundError(
                f"Actual ETA file not found at {eta_path}. "
                "Please run: python -m data.preprocessing_code.extract_actual_eta"
            )

        print(f"Loading actual ETA data from {eta_path}...")
        df = pd.read_csv(eta_path)
        total_in_file = len(df)

        # Filter to relevant orders if specified
        if relevant_order_ids is not None:
            df = df[df["order_id"].isin(relevant_order_ids)]
            print(
                f"  Total in file: {total_in_file:,}, filtered to {len(df):,} relevant orders"
            )

        # Build lookup: order_id -> eta_seconds_actual
        self.actual_eta_lookup = dict(
            zip(df["order_id"].astype(int), df["eta_seconds_actual"].astype(int))
        )
        print(f"Loaded actual ETA for {len(self.actual_eta_lookup):,} orders")

        # Stats
        if len(self.actual_eta_lookup) > 0:
            etas = list(self.actual_eta_lookup.values())
            print(f"  Mean: {np.mean(etas):.0f}s ({np.mean(etas) / 60:.1f} min)")
            print(f"  Median: {np.median(etas):.0f}s ({np.median(etas) / 60:.1f} min)")
        else:
            print(
                "  Warning: No matching ETAs found for the events in this time window"
            )

    def _get_actual_eta(self, order_id: int, fallback: int = 1800) -> int:
        """Get actual ETA for an order, with fallback if not found."""
        return self.actual_eta_lookup.get(order_id, fallback)

    def _load_courier_blocks(
        self, relevant_courier_ids: set, sim_start_time: int, sim_end_time: int
    ) -> Dict[int, List[Tuple[int, int]]]:
        """
        Load courier working blocks from CSV, filtered to relevant couriers and time window.

        Args:
            relevant_courier_ids: Set of courier IDs that appear in the simulation events
            sim_start_time: Start of simulation time window (earliest dispatch time)
            sim_end_time: End of simulation time window (latest dispatch time + buffer)
        """
        blocks_path = Path(__file__).parent.parent / "data" / "courier_working_blocks.csv"
        if not blocks_path.exists():
            # Try common local fallback locations for ad hoc experiments.
            candidates = [
                Path(self.output_dir) / "courier_working_blocks.csv",
                Path(self.output_dir).parent / "courier_working_blocks.csv",
                Path("data") / "courier_working_blocks.csv",
                Path("courier_working_blocks.csv"),
            ]
            for p in candidates:
                if p.exists():
                    blocks_path = p
                    break

        if not blocks_path.exists():
            print(
                f"Warning: Courier blocks file not found. Using simple shift windows."
            )
            return {}

        print(f"Loading courier blocks from {blocks_path}...")
        # Add 1800 seconds (30 min) buffer after sim_end_time to catch blocks that extend slightly beyond
        max_block_end = sim_end_time + 0

        try:
            df = pd.read_csv(blocks_path)
            blocks = {}
            total_blocks = 0
            filtered_blocks = 0

            for _, row in df.iterrows():
                cid = int(row["courier_id"])
                start = int(row["block_start_time"])
                end = int(row["block_end_time"])
                total_blocks += 1

                # Filter 1: Only include couriers that appear in our simulation events
                if cid not in relevant_courier_ids:
                    continue

                # Filter 2: Only include blocks that overlap with our simulation time window
                # Block overlaps if: block_start <= sim_end_time AND block_end >= sim_start_time
                # Also allow blocks that end within 1800s after sim_end_time
                if start > max_block_end or end < sim_start_time:
                    continue

                if cid not in blocks:
                    blocks[cid] = []
                blocks[cid].append((start, end))
                filtered_blocks += 1

            print(f"  Total blocks in file: {total_blocks}")
            print(f"  Relevant couriers: {len(relevant_courier_ids)}")
            print(f"  Filtered blocks (overlapping time window): {filtered_blocks}")
            print(f"  Couriers with blocks: {len(blocks)}")

            return blocks
        except Exception as e:
            print(f"Error loading blocks: {e}")
            return {}

    def _initialize_courier_states(self):
        """
        Build initial courier state dict from historical data.

        Assumptions:
        - Courier starts at first known sender location
        - Available during their active time windows
        - Shift window = min/max dispatch times they appear in data
        - Only loads courier blocks for couriers that appear in simulation events
        - Only loads blocks that overlap with the simulation time window
        """
        print("Initializing courier states...")
        events = self.dispatcher.get_events()

        # First pass: Gather all courier IDs and determine time window from events
        courier_events = {}
        all_dispatch_times = []

        for ev in events:
            if ev.courier_id not in courier_events:
                courier_events[ev.courier_id] = []
            courier_events[ev.courier_id].append(ev)
            all_dispatch_times.append(ev.actual_dispatch_time)

        # Determine simulation time window from events
        relevant_courier_ids = set(courier_events.keys())
        sim_start_time = min(all_dispatch_times) if all_dispatch_times else 0
        sim_end_time = max(all_dispatch_times) if all_dispatch_times else 0

        # Convert to Shanghai time for display
        from datetime import datetime
        import pytz

        try:
            shanghai_tz = pytz.timezone("Asia/Shanghai")
            start_dt = datetime.fromtimestamp(sim_start_time, tz=shanghai_tz)
            end_dt = datetime.fromtimestamp(sim_end_time, tz=shanghai_tz)
            start_str = start_dt.strftime("%Y-%m-%d %H:%M:%S")
            end_str = end_dt.strftime("%Y-%m-%d %H:%M:%S")
        except:
            # Fallback to local time if pytz not available
            start_str = datetime.fromtimestamp(sim_start_time).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            end_str = datetime.fromtimestamp(sim_end_time).strftime("%Y-%m-%d %H:%M:%S")

        print(f"  Couriers in events: {len(relevant_courier_ids)}")
        print(f"  Simulation time window:")
        print(f"    Start: {sim_start_time} ({start_str} Shanghai)")
        print(f"    End:   {sim_end_time} ({end_str} Shanghai)")

        # Load blocks only for relevant couriers and time window
        blocks = self._load_courier_blocks(
            relevant_courier_ids, sim_start_time, sim_end_time
        )

        for cid, evs in courier_events.items():
            # Find first location and time window
            first_ev = evs[0]
            sender_lat = first_ev.order_data.get("sender_lat", 0.0)
            sender_lng = first_ev.order_data.get("sender_lng", 0.0)

            shift_start = min(e.actual_dispatch_time for e in evs)
            shift_end = max(e.actual_dispatch_time for e in evs)

            courier_blocks = blocks.get(cid, [])

            self.courier_states[cid] = CourierState(
                courier_id=cid,
                current_lat=sender_lat,
                current_lng=sender_lng,
                available=True,
                shift_start=shift_start,
                shift_end=shift_end,
                blocks=courier_blocks,
            )

            # Initialize utilization tracking for this courier's blocks
            if courier_blocks:
                self.utilization_tracker.initialize_courier_blocks(cid, courier_blocks)

        print(f"Initialized {len(self.courier_states)} couriers")

    def _track_delivery_completion(
        self, courier_id: int, waybill_id: int, arrive_time: int
    ):
        """
        Track a delivery completion time for simulation extension.

        When couriers have orders in transit at the end of the simulation time window,
        we extend the effective simulation end time to include those deliveries.
        This eliminates artificial idle time caused by the simulation ending while
        orders are still being delivered.
        """
        if arrive_time > self.latest_delivery_completion_time:
            self.latest_delivery_completion_time = arrive_time

    def _mark_delivery_complete(self, courier_id: int, waybill_id: int):
        """Mark a delivery as complete on the courier state."""
        courier_state = self.courier_states.get(courier_id)
        if courier_state:
            courier_state.complete_delivery(waybill_id)

    def _count_active_couriers(self) -> int:
        """Count couriers that still have pending deliveries."""
        return sum(1 for c in self.courier_states.values() if c.is_active)

    def _count_deliveries_beyond_window(self, simulation_end_time: int) -> int:
        """Count deliveries that complete after the simulation's original end time."""
        count = 0
        for courier in self.courier_states.values():
            for detail in courier.delivered_waybills:
                # We don't track individual delivery times in courier state
                # So we'll use the utilization tracker which has detailed info
                pass
        # Use utilization tracker for accurate count
        return self.utilization_tracker.count_deliveries_beyond_time(
            simulation_end_time
        )

    def _format_timestamp(self, ts: int) -> str:
        """Format unix timestamp to readable datetime string."""
        from datetime import datetime
        import pytz

        try:
            shanghai_tz = pytz.timezone("Asia/Shanghai")
            dt = datetime.fromtimestamp(ts, tz=shanghai_tz)
            return dt.strftime("%H:%M:%S")
        except:
            return datetime.fromtimestamp(ts).strftime("%H:%M:%S")

    def _mark_courier_diverged(
        self,
        courier_id: int,
        reason: str,
        timestamp: int,
        simulated_lat: float,
        simulated_lng: float,
        pending_delivery_info: dict = None,
    ):
        """
        Mark a courier as having diverged from their historical trajectory.

        Once diverged, all subsequent events involving this courier need special handling:
        - Their actual location differs from historical proxy_lng/lat
        - Future orders may need re-scoring based on simulated position
        - Historical orders within 40-minute window are invalidated

        Args:
            courier_id: The courier who diverged
            reason: Why they diverged (e.g., 'rejected_historical_accept', 'accepted_historical_reject')
            timestamp: When the divergence occurred
            simulated_lat/lng: Their new simulated position
            pending_delivery_info: Optional dict with delivery timing info:
                - sender_lat, sender_lng: Restaurant location
                - recipient_lat, recipient_lng: Customer location
                - fetch_time: When courier will reach restaurant
                - arrive_time: When courier will complete delivery
                - waybill_id: Order being delivered
        """
        self.diverged_couriers[courier_id] = {
            "reason": reason,
            "timestamp": timestamp,
            "simulated_lat": simulated_lat,
            "simulated_lng": simulated_lng,
        }

        # Update courier state
        courier_state = self.courier_states.get(courier_id)
        if courier_state:
            courier_state.current_lat = simulated_lat
            courier_state.current_lng = simulated_lng
            courier_state.mark_diverged(timestamp, reason)

            # Set pending delivery info if provided
            if pending_delivery_info:
                courier_state.set_pending_delivery(
                    waybill_id=pending_delivery_info.get("waybill_id"),
                    sender_lat=pending_delivery_info.get("sender_lat", 0.0),
                    sender_lng=pending_delivery_info.get("sender_lng", 0.0),
                    recipient_lat=pending_delivery_info.get("recipient_lat", 0.0),
                    recipient_lng=pending_delivery_info.get("recipient_lng", 0.0),
                    fetch_time=pending_delivery_info.get("fetch_time", 0),
                    arrive_time=pending_delivery_info.get("arrive_time", 0),
                    grab_time=pending_delivery_info.get("grab_time"),
                )

            # Invalidate historical orders within 40-minute window
            self._invalidate_historical_orders_in_window(courier_id, timestamp)

    def _invalidate_historical_orders_in_window(
        self, courier_id: int, divergence_time: int
    ):
        """
        Invalidate historical orders for this courier within 40-minute window.

        When a courier diverges, their historical orders that fall within 40 minutes
        of the divergence need to be invalidated because:
        - The courier is now at a different location
        - A full delivery cycle takes ~30 minutes on average
        - We add 10 minutes buffer

        These orders will need to be re-dispatched when they come up.
        """
        courier_state = self.courier_states.get(courier_id)
        if not courier_state:
            return

        window_end = divergence_time + self.divergence_window_seconds

        # Look through dispatcher events to find orders for this courier
        events = self.dispatcher.get_events()
        invalidated_count = 0

        for event in events:
            if event.courier_id != courier_id:
                continue

            # Check if this event falls within the invalidation window
            if divergence_time <= event.actual_dispatch_time <= window_end:
                # Only invalidate if it was a historical accept
                if event.historical_decision == 1:
                    courier_state.invalidate_historical_order(event.waybill_id)
                    invalidated_count += 1

                    if self.verbose:
                        self._log_event(
                            event,
                            "INVALIDATED_BY_DIVERGENCE",
                            f"Historical order invalidated due to divergence at {divergence_time}. "
                            f"Order dispatch time {event.actual_dispatch_time} within 40-min window.",
                        )

        if invalidated_count > 0 and self.verbose and self.log_file_handle:
            self.log_file_handle.write(
                f"\n[DIVERGENCE] Courier {courier_id} diverged at {divergence_time}. "
                f"Invalidated {invalidated_count} historical orders within 40-min window.\n"
            )
            self.log_file_handle.flush()

    def _is_courier_diverged(self, courier_id: int) -> bool:
        """Check if a courier has diverged from historical trajectory."""
        courier_state = self.courier_states.get(courier_id)
        if courier_state:
            return courier_state.is_diverged
        return courier_id in self.diverged_couriers

    def _get_courier_position(
        self, courier_id: int, event: "OfferEvent"
    ) -> Tuple[float, float]:
        """
        Get courier's current position, accounting for divergence.

        If courier has diverged, use INTERPOLATED position based on:
        - grab_time → fetch_time: interpolate between start → sender
        - fetch_time → arrive_time: interpolate between sender → recipient
        - after arrive_time: at recipient

        For non-diverged couriers, use historical proxy_lng/lat from event.

        Returns:
            (lat, lng) tuple
        """
        courier_state = self.courier_states.get(courier_id)

        if courier_state and courier_state.is_diverged:
            # Use interpolated location based on current time
            current_time = event.actual_dispatch_time
            return courier_state.get_interpolated_location(current_time)

        # Use historical position
        return (
            event.order_data.get("proxy_lat", 0.0),
            event.order_data.get("proxy_lng", 0.0),
        )

    def _update_diverged_couriers_for_time(self, current_time: int):
        """
        Update all diverged couriers' locations based on current simulation time.

        At fetch_time: courier is at sender location
        At arrive_time: courier is at recipient location and becomes available
        """
        for courier_id, courier_state in self.courier_states.items():
            if courier_state.is_diverged:
                courier_state.update_location_for_time(current_time)

    def run(self):
        """Run the simulation with enhanced 4-case logic and cycle-based reassignment."""
        from datetime import datetime
        import pytz
        import time

        mode_desc = (
            "BASELINE (historical replay)" if self.baseline_mode else "AGENT-DRIVEN"
        )
        print(f"Starting simulation in {mode_desc} mode...")

        # Track wall-clock execution time
        wall_clock_start = time.time()

        # Open log file for verbose mode
        if self.verbose:
            output_path = Path(self.output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            log_file_path = output_path / "verbose_simulation_log.txt"
            self.log_file_handle = open(log_file_path, "w", encoding="utf-8")
            print(f"VERBOSE MODE ENABLED - Writing detailed logs to: {log_file_path}")
            self.log_file_handle.write(f"=== Simulation Log ({mode_desc} mode) ===\n")
            self.log_file_handle.write(f"Started at: {datetime.now()}\n\n")

        events = self.dispatcher.get_events()

        # Build courier next dispatch time map
        self._build_courier_dispatch_schedule(events)

        # Build cycle boundaries (group events by dispatch time)
        cycles = self._build_cycles(events)
        print(f"Identified {len(cycles)} dispatch cycles")

        # Get simulation time boundaries for progress bar
        if cycles:
            sim_start_time = cycles[0]["start_time"]
            sim_end_time = cycles[-1]["end_time"]
            total_duration = sim_end_time - sim_start_time

            # Format start and end times for display (full date+time for both)
            try:
                shanghai_tz = pytz.timezone("Asia/Shanghai")
                start_dt = datetime.fromtimestamp(sim_start_time, tz=shanghai_tz)
                end_dt = datetime.fromtimestamp(sim_end_time, tz=shanghai_tz)
                start_str = start_dt.strftime("%Y-%m-%d %H:%M:%S")
                end_str = end_dt.strftime("%Y-%m-%d %H:%M:%S")
            except:
                start_str = datetime.fromtimestamp(sim_start_time).strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
                end_str = datetime.fromtimestamp(sim_end_time).strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
        else:
            sim_start_time = 0
            sim_end_time = 0
            total_duration = 1
            start_str = "N/A"
            end_str = "N/A"

        # Helper function to format elapsed time
        def format_elapsed(seconds):
            """Format elapsed seconds as MM:SS or HH:MM:SS."""
            if seconds < 3600:
                return f"{int(seconds // 60):02d}:{int(seconds % 60):02d}"
            else:
                hours = int(seconds // 3600)
                mins = int((seconds % 3600) // 60)
                secs = int(seconds % 60)
                return f"{hours:02d}:{mins:02d}:{secs:02d}"

        # Create progress bar with timeline display
        mode_short = "BASELINE" if self.baseline_mode else "AGENT"
        pbar = tqdm(
            total=len(cycles),
            desc=f"Simulating ({mode_short})",
            bar_format="{desc}: {percentage:3.0f}%|{bar}| {n}/{total} cycles [{postfix}]",
            postfix=f"{start_str} --> {end_str}",
        )

        for cycle_id, cycle_info in enumerate(cycles):
            self.current_cycle_id = cycle_id
            self.current_cycle_start_time = cycle_info["start_time"]
            self.current_cycle_end_time = cycle_info["end_time"]
            cycle_events = cycle_info["events"]

            # OPTIMIZATION: Invalidate scorer cache at start of each cycle
            self.scorer.invalidate_cache()

            # Update progress bar with current timestamp and elapsed wall-clock time
            current_time_str = self._format_timestamp(cycle_info["start_time"])
            elapsed = time.time() - wall_clock_start
            elapsed_str = format_elapsed(elapsed)
            pbar.set_postfix_str(
                f"{current_time_str} --> {end_str} | elapsed: {elapsed_str}"
            )

            if self.verbose:
                self._log_cycle_start(cycle_id, cycle_info)

            # First, try to reassign pending orders from previous cycles
            if not self.baseline_mode:
                self._process_pending_reassignments(cycle_info)

            # Process events in this cycle
            for event in cycle_events:
                self._process_event(event)

            if self.verbose:
                self._log_cycle_end(cycle_id, cycle_info)

            pbar.update(1)

        pbar.close()

        # Print total execution time
        total_elapsed = time.time() - wall_clock_start
        print(
            f"\nSimulation completed in {format_elapsed(total_elapsed)} (wall-clock time)"
        )

        # Track simulation timing for extended delivery handling
        if cycles:
            self.simulation_end_time = cycles[-1]["end_time"]
        else:
            self.simulation_end_time = 0

        # Extend simulation end time if there are deliveries still in progress
        if self.latest_delivery_completion_time > self.simulation_end_time:
            self.extended_end_time = self.latest_delivery_completion_time
            time_extension = self.extended_end_time - self.simulation_end_time

            # Count deliveries that complete after the original end time
            pending_deliveries, pending_couriers = (
                self.utilization_tracker.count_deliveries_beyond_time(
                    self.simulation_end_time
                )
            )

            print(
                f"Simulation complete. Extended end time by {time_extension}s ({time_extension / 60:.1f}min) "
                f"to account for {pending_deliveries} deliveries ({pending_couriers} couriers) in transit."
            )

            # Extend utilization tracker blocks for couriers with pending deliveries
            # This ensures active time is properly counted for deliveries that complete
            # after the simulation's original time window
            self.utilization_tracker.extend_blocks_for_pending_deliveries(
                self.simulation_end_time, self.extended_end_time
            )
        else:
            self.extended_end_time = self.simulation_end_time
            print("Simulation complete.")

        # Mark remaining pending orders as lost
        if not self.baseline_mode:
            self._finalize_pending_orders()

        self._generate_report()

        # Save verbose log if enabled
        if self.verbose and self.event_log:
            self._save_verbose_log()

        # Close log file handle
        if self.log_file_handle:
            self.log_file_handle.close()
            self.log_file_handle = None

    def _build_cycles(self, events) -> List[Dict]:
        """
        Group events into cycles based on dispatch_cycle_id from parquet.
        Uses actual cycle boundaries (dispatch_start_time, dispatch_end_time) from data.
        """
        if not events:
            return []

        from collections import defaultdict

        # Group events by dispatch_cycle_id
        cycle_events = defaultdict(list)
        cycle_info = {}

        for event in events:
            cycle_id = event.dispatch_cycle_id
            cycle_events[cycle_id].append(event)

            # Store cycle metadata (same for all events in cycle)
            if cycle_id not in cycle_info:
                cycle_info[cycle_id] = {
                    "dispatch_cycle_id": cycle_id,
                    "start_time": event.dispatch_start_time,
                    "end_time": event.dispatch_end_time,
                }

        # Sort cycles by start_time and build output
        sorted_cycle_ids = sorted(
            cycle_info.keys(), key=lambda cid: cycle_info[cid]["start_time"]
        )

        cycles = []
        for cycle_id in sorted_cycle_ids:
            info = cycle_info[cycle_id]
            evts = cycle_events[cycle_id]
            # Sort events within cycle by offer_index_in_cycle
            evts.sort(key=lambda e: e.offer_index_in_cycle)

            cycles.append(
                {
                    "dispatch_cycle_id": cycle_id,
                    "start_time": info["start_time"],
                    "end_time": info["end_time"],
                    "events": evts,
                    "num_events": len(evts),
                }
            )

        return cycles

    def _log_cycle_start(self, cycle_id: int, cycle_info: Dict):
        """Log the start of a new cycle."""
        from datetime import datetime

        start_ts = datetime.fromtimestamp(cycle_info["start_time"]).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        end_ts = datetime.fromtimestamp(cycle_info["end_time"]).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        dispatch_cycle_id = cycle_info.get("dispatch_cycle_id", cycle_id)

        if self.log_file_handle:
            self.log_file_handle.write(f"\n{'=' * 80}\n")
            self.log_file_handle.write(
                f"CYCLE {cycle_id} (dispatch_cycle_id={dispatch_cycle_id})\n"
            )
            self.log_file_handle.write(f"  Time range: {start_ts} - {end_ts}\n")
            self.log_file_handle.write(
                f"  Unix time: {cycle_info['start_time']} - {cycle_info['end_time']}\n"
            )
            self.log_file_handle.write(
                f"  Events in cycle: {cycle_info['num_events']}\n"
            )
            self.log_file_handle.write(
                f"  Pending reassignments: {len(self.pending_reassignments)}\n"
            )
            self.log_file_handle.write(f"{'=' * 80}\n")
            self.log_file_handle.flush()

    def _log_cycle_end(self, cycle_id: int, cycle_info: Dict):
        """Log the end of a cycle with summary."""
        if self.log_file_handle:
            self.log_file_handle.write(f"\n--- End of Cycle {cycle_id} ---\n")
            self.log_file_handle.write(
                f"  Pending reassignments remaining: {len(self.pending_reassignments)}\n"
            )
            for waybill_id, info in self.pending_reassignments.items():
                self.log_file_handle.write(
                    f"    waybill={waybill_id}, cycles_remaining={info['cycles_remaining']}\n"
                )
            self.log_file_handle.flush()

    def _process_pending_reassignments(self, cycle_info: Dict):
        """
        Try to reassign pending orders at the start of each cycle.

        OPTIMIZATION: Pre-compute available couriers once per cycle and reuse for all pending orders.
        This reduces complexity from O(pending × couriers) to O(couriers + pending × available).
        """
        if not self.pending_reassignments:
            return

        self.profiler.start("process_pending_reassignments")

        # OPTIMIZATION: Pre-compute available couriers ONCE for this cycle
        # Estimate max completion time for cache (use 30 min as typical max ETA)
        cycle_start = cycle_info["start_time"]
        max_completion_time = cycle_start + 1800  # 30 minutes

        self.profiler.start("precompute_available_couriers")
        available_couriers = self.scorer.get_available_couriers_for_time(
            cycle_start, max_completion_time
        )
        self.profiler.stop("precompute_available_couriers")

        # Decrement cycles remaining and try to reassign
        to_remove = []

        # OPTIMIZATION: Limit number of pending orders to process per cycle
        # to prevent runaway growth
        max_pending_per_cycle = self.max_pending_per_cycle
        pending_items = list(self.pending_reassignments.items())[
            : max_pending_per_cycle + 10
        ]

        for waybill_id, info in pending_items:
            info["cycles_remaining"] -= 1

            if info["cycles_remaining"] <= 0:
                # Order has been pending for configured cycles, mark as lost
                if self.verbose:
                    self._log_pending_lost(waybill_id, info)
                self.metrics.log_system_order(
                    accepted=False,
                    late=False,
                    courier_id=info["event"].courier_id,
                    capacity=0,
                    lost=True,
                )
                # Track not assigned in order_tracker
                self.order_tracker.mark_not_assigned(waybill_id)
                to_remove.append(waybill_id)
                continue

            # Try to find a new courier in this cycle using FAST scoring
            event = info["event"]
            order = info["order"]
            eta = self._get_actual_eta(event.order_id, fallback=1800)

            self.profiler.start("pending_scorer_score_couriers")
            # Use optimized fast scoring with pre-computed available couriers
            result = self.scorer.score_couriers_fast(
                waybill_id=event.waybill_id,
                sender_lat=event.order_data.get("sender_lat", 0.0),
                sender_lng=event.order_data.get("sender_lng", 0.0),
                dispatch_time=cycle_start,
                estimated_duration=eta,
                excluded_couriers=order.rejecting_couriers,
                available_couriers=available_couriers,  # Reuse pre-computed list
            )
            self.profiler.stop("pending_scorer_score_couriers")

            if result is None:
                # No courier available, keep pending
                if self.verbose:
                    self._log_pending_attempt(waybill_id, info, None, None, [])
                continue

            new_courier_id, distance = result

            if self.verbose:
                self._log_pending_attempt(
                    waybill_id, info, new_courier_id, distance, []
                )

            # Build fresh features for the NEW courier (not the stale original features)
            reassign_features = self._build_features_for_courier(
                event, new_courier_id, cycle_info["start_time"]
            )

            # Check agent decision for new courier with correct features
            agent_decision, agent_prob = self.agent.act(reassign_features)

            if agent_decision == 1:
                # Accept - assign to new courier
                order.assign_to_courier(new_courier_id)
                order.mark_accepted(new_courier_id)

                # Track acceptance on order_tracker for unique order counting
                self.order_tracker.mark_accepted(event.waybill_id)

                # Mark courier as diverged - taking a reassigned order
                recipient_lat = order.recipient_lat
                recipient_lng = order.recipient_lng
                self._mark_courier_diverged(
                    courier_id=new_courier_id,
                    reason="accepted_reassigned_order",
                    timestamp=cycle_info["start_time"],
                    simulated_lat=recipient_lat,
                    simulated_lng=recipient_lng,
                )

                is_late, realized_capacity = self._simulate_order_delivery(
                    order, new_courier_id, event
                )

                # Record assignment on new courier state
                new_courier_state = self.courier_states.get(new_courier_id)
                if new_courier_state:
                    new_courier_state.record_assignment(
                        event.waybill_id, timestamp=cycle_info["start_time"]
                    )

                self.metrics.log_system_order(
                    accepted=True,
                    late=is_late,
                    courier_id=new_courier_id,
                    capacity=realized_capacity,
                )

                if self.verbose:
                    self._log_pending_assigned(
                        waybill_id, info, new_courier_id, distance
                    )

                to_remove.append(waybill_id)
            else:
                # Reject this courier too
                order.add_rejection(new_courier_id)
                if self.verbose:
                    self._log_pending_rejected(waybill_id, info, new_courier_id)

        # Remove processed orders
        for waybill_id in to_remove:
            del self.pending_reassignments[waybill_id]
        self.profiler.stop("process_pending_reassignments")

    def _log_pending_lost(self, waybill_id: int, info: Dict):
        """Log when a pending order is lost after 5 cycles."""
        if self.log_file_handle:
            self.log_file_handle.write(f"\n[PENDING_LOST] waybill={waybill_id}\n")
            self.log_file_handle.write(
                f"  Original cycle: {info['original_cycle']}, Lost after 5 cycles of attempts\n"
            )
            self.log_file_handle.write(
                f"  Rejected by couriers: {list(info['order'].rejecting_couriers)}\n"
            )
            self.log_file_handle.flush()

    def _log_pending_attempt(
        self, waybill_id: int, info: Dict, new_courier_id, distance, top_5
    ):
        """Log an attempt to reassign a pending order."""
        if self.log_file_handle:
            self.log_file_handle.write(
                f"\n[PENDING_REASSIGN_ATTEMPT] waybill={waybill_id}\n"
            )
            self.log_file_handle.write(
                f"  Cycles remaining: {info['cycles_remaining']}\n"
            )
            if new_courier_id is not None:
                self.log_file_handle.write(
                    f"  Best candidate: courier={new_courier_id}, dist={distance:.3f}km\n"
                )
                if top_5:
                    self.log_file_handle.write(f"  Top 5 candidates:\n")
                    for i, c in enumerate(top_5):
                        self.log_file_handle.write(
                            f"    #{i + 1}: courier={c['courier_id']}, dist={c['distance_km']}km\n"
                        )
            else:
                self.log_file_handle.write(
                    f"  No couriers available for reassignment\n"
                )
            self.log_file_handle.flush()

    def _log_pending_assigned(
        self, waybill_id: int, info: Dict, new_courier_id: int, distance: float
    ):
        """Log when a pending order is successfully assigned."""
        if self.log_file_handle:
            self.log_file_handle.write(f"\n[PENDING_ASSIGNED] waybill={waybill_id}\n")
            self.log_file_handle.write(
                f"  Assigned to courier {new_courier_id} (dist={distance:.3f}km)\n"
            )
            self.log_file_handle.write(
                f"  After {5 - info['cycles_remaining']} cycles of waiting\n"
            )
            self.log_file_handle.flush()

    def _log_pending_rejected(self, waybill_id: int, info: Dict, new_courier_id: int):
        """Log when agent rejects another courier for a pending order."""
        if self.log_file_handle:
            self.log_file_handle.write(f"\n[PENDING_REJECTED] waybill={waybill_id}\n")
            self.log_file_handle.write(
                f"  Agent rejected courier {new_courier_id}, will try again next cycle\n"
            )
            self.log_file_handle.flush()

    def _finalize_pending_orders(self):
        """Mark all remaining pending orders as lost at end of simulation."""
        for waybill_id, info in self.pending_reassignments.items():
            if self.verbose:
                self._log_pending_lost(waybill_id, info)
            self.metrics.log_system_order(
                accepted=False,
                late=False,
                courier_id=info["event"].courier_id,
                capacity=0,
                lost=True,
            )
            # Track not assigned in order_tracker
            self.order_tracker.mark_not_assigned(waybill_id)
        self.pending_reassignments.clear()

    def _process_event(self, event):
        """Process a single event within a cycle."""
        # Check if order is already handled (accepted/active/delivered)
        # This prevents double-counting orders if multiple offers exist
        # We check both waybill_id (specific offer) and order_id (unique order)
        # The accepted_order_ids check is critical for Scenario A (hist reject → agent accept):
        # once the agent accepts an order early, all subsequent historical waybills for the
        # same order_id must be skipped even if delivery hasn't completed yet.
        self.profiler.start("check_order_handled")
        if (
            event.waybill_id in self.order_tracker.completed_orders
            or event.waybill_id in self.order_tracker.active_orders
            or event.order_id in self.order_tracker.completed_order_ids
            or event.order_id in self.order_tracker.accepted_order_ids
        ):
            self.profiler.stop("check_order_handled")
            if self.verbose:
                self._log_event(
                    event, "SKIP", "Order already handled (waybill or order_id seen)"
                )
            return
        self.profiler.stop("check_order_handled")

        # Update all diverged couriers' locations based on current time
        self.profiler.start("update_diverged_couriers")
        self._update_diverged_couriers_for_time(event.actual_dispatch_time)
        self.profiler.stop("update_diverged_couriers")

        # Record that this waybill was offered to the courier
        courier_state = self.courier_states.get(event.courier_id)
        if courier_state:
            courier_state.record_offer(event.waybill_id)

        # Check if this historical order has been invalidated due to divergence
        # This happens when the courier diverged within 40 minutes of this order
        if courier_state and courier_state.is_order_invalidated(event.waybill_id):
            if self.verbose:
                self._log_event(
                    event,
                    "ORDER_INVALIDATED",
                    f"Historical order invalidated due to courier {event.courier_id} divergence. "
                    f"Treating as Case 1.1 (needs reassignment).",
                )
            # Treat as if agent rejected (even if history accepted)
            # This forces reassignment since courier is at different location
            self._handle_invalidated_order(event)
            return

        # Check if courier has diverged from historical trajectory
        # If diverged, their position is different from what the historical data shows
        # This affects distance calculations and may require re-scoring
        is_courier_diverged = self._is_courier_diverged(event.courier_id)
        if is_courier_diverged and self.verbose:
            div_info = self.diverged_couriers.get(event.courier_id, {})
            self._log_event(
                event,
                "DIVERGED_COURIER",
                f"Courier {event.courier_id} has diverged (reason: {div_info.get('reason', 'unknown')}). "
                f"Using simulated position instead of historical proxy.",
            )

        hist_decision = event.historical_decision

        # In BASELINE mode, skip all simulation logic and just replay historical decisions
        if self.baseline_mode:
            agent_action, agent_prob = self.agent.act(
                event.features, historical_decision=hist_decision
            )
        else:
            # AGENT mode: Check availability of courier based on blocks
            # Use actual ETA from historical data (arrive_time - fetch_time)
            self.profiler.start("get_eta")
            eta = self._get_actual_eta(event.order_id, fallback=1800)
            self.profiler.stop("get_eta")
            completion_time = event.actual_dispatch_time + eta

            courier_state = self.courier_states.get(event.courier_id)
            is_available = False
            block_matched = None

            # Allow configurable buffer for block boundaries
            block_buffer_seconds = self.block_buffer_seconds

            # KEY FIX: If historical decision was ACCEPT, the courier was definitely available
            # for this specific order - the dispatch time might be before block start but
            # the actual work (grab -> arrive) happened within the block
            self.profiler.start("check_availability")
            if hist_decision == 1:
                # Historical accept = courier was available for this order
                is_available = True
                block_matched = "historical_accept"
                if self.verbose and courier_state and courier_state.blocks:
                    # Log which block the work actually fell into
                    grab_time = event.order_data.get("grab_time")
                    arrive_time = event.order_data.get("arrive_time")
                    for start, end in courier_state.blocks:
                        if grab_time and start <= int(grab_time) <= end:
                            block_matched = (start, end)
                            break
            elif courier_state:
                if courier_state.blocks:
                    for start, end in courier_state.blocks:
                        # Add buffer: allow dispatch up to 60s before block start
                        # and completion up to 60s after block end
                        if (
                            start - block_buffer_seconds
                        ) <= event.actual_dispatch_time and completion_time <= (
                            end + block_buffer_seconds
                        ):
                            is_available = True
                            block_matched = (start, end)
                            break
                else:
                    # Fallback if no blocks defined for this courier
                    is_available = True
            self.profiler.stop("check_availability")

            # If courier is not available, force rejection
            if not is_available:
                agent_action = 0
                agent_prob = 0.0
                if self.verbose:
                    self._log_event(
                        event,
                        "FORCE_REJECT",
                        f"Courier {event.courier_id} outside working hours (with {block_buffer_seconds}s buffer). "
                        f"Dispatch={event.actual_dispatch_time}, Completion={completion_time}, "
                        f"Blocks={courier_state.blocks[:3] if courier_state and courier_state.blocks else 'None'}...",
                    )
            else:
                # Agent decision
                self.profiler.start("agent_inference")
                agent_action, agent_prob = self.agent.act(event.features)
                self.profiler.stop("agent_inference")
                if self.verbose:
                    self._log_event(
                        event,
                        "AGENT_DECISION",
                        f"Agent={agent_action} (prob={agent_prob:.3f}), Historical={hist_decision}, "
                        f"Courier {event.courier_id} in block {block_matched}",
                    )

        # ============ CASE LOGIC ============
        # All system metrics logging happens inside case handlers

        # CASE 1: Both accept (aligned accept)
        if agent_action == 1 and hist_decision == 1:
            self._handle_case_1_aligned_accept(event, agent_action)

        # CASE 1.1: Historical accept, Agent reject (divergence: accept→reject)
        elif agent_action == 0 and hist_decision == 1:
            self._handle_case_1_1_accept_to_reject(event, agent_action, hist_decision)

        # CASE 1.2: Historical reject, Agent accept (divergence: reject→accept)
        elif agent_action == 1 and hist_decision == 0:
            self._handle_case_1_2_reject_to_accept(event, agent_action, hist_decision)

        # CASE 0: Both reject (aligned reject)
        else:  # agent_action == 0 and hist_decision == 0
            self._handle_case_0_aligned_reject(event, agent_action)

    def _log_event(self, event: OfferEvent, action: str, details: str):
        """Log an event for verbose mode - writes to file instead of terminal."""
        from datetime import datetime

        ts = datetime.fromtimestamp(event.actual_dispatch_time).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        log_entry = {
            "timestamp": ts,
            "unix_time": event.actual_dispatch_time,
            "dispatch_cycle_id": event.dispatch_cycle_id,
            "offer_index_in_cycle": event.offer_index_in_cycle,
            "waybill_id": event.waybill_id,
            "order_id": event.order_id,
            "courier_id": event.courier_id,
            "action": action,
            "details": details,
        }
        self.event_log.append(log_entry)

        # Write to file instead of terminal
        if self.log_file_handle:
            self.log_file_handle.write(
                f"\n[{ts}] {action}: waybill={event.waybill_id}, courier={event.courier_id}, offer_idx={event.offer_index_in_cycle}\n"
            )
            self.log_file_handle.write(f"  {details}\n")
            self.log_file_handle.flush()  # Flush so we can tail the file

    def _save_verbose_log(self):
        """Save verbose log to JSON file (in addition to .txt)."""
        output_path = Path(self.output_dir)
        log_path = output_path / "verbose_simulation_log.json"
        with open(log_path, "w") as f:
            json.dump(self.event_log, f, indent=2)
        print(f"\nVerbose log saved to:")
        print(f"  - {output_path / 'verbose_simulation_log.txt'} (human-readable)")
        print(f"  - {log_path} (JSON, {len(self.event_log)} entries)")

    def _build_courier_dispatch_schedule(self, events):
        """Build map of courier -> next dispatch time for lookahead."""
        from collections import defaultdict

        courier_times = defaultdict(list)
        for ev in events:
            courier_times[ev.courier_id].append(ev.actual_dispatch_time)

        # For each courier, sort times and store
        for cid, times in courier_times.items():
            times_sorted = sorted(set(times))
            self.courier_next_dispatch[cid] = times_sorted

    def _get_next_dispatch_time(
        self, courier_id: int, current_time: int
    ) -> Optional[int]:
        """Get the next dispatch time for a courier after current_time."""
        if courier_id not in self.courier_next_dispatch:
            return None
        times = self.courier_next_dispatch[courier_id]
        for t in times:
            if t > current_time:
                return t
        return None

    def _build_features_for_courier(
        self, event: OfferEvent, new_courier_id: int, dispatch_time: int
    ) -> Dict[str, float]:
        """
        Build a fresh feature dict for a NEW courier + existing order.

        This replaces the stale event.features when reassigning an order to a
        different courier. Order-specific features (sender/recipient coords,
        ETA, income) come from the original event; courier-specific features
        (location, capacity, late ratio, etc.) come from live CourierState.

        Args:
            event: original OfferEvent (order data source)
            new_courier_id: ID of the candidate courier
            dispatch_time: current timestamp (for tod and time_since_last_accept)

        Returns:
            Dict[str, float] suitable for self.agent.act()
        """
        from math import sin, cos, pi
        from datetime import datetime, timezone
        import pytz

        courier_state = self.courier_states.get(new_courier_id)

        # --- Order-specific features (unchanged) ---
        sender_lng = float(event.order_data.get("sender_lng", 0.0))
        sender_lat = float(event.order_data.get("sender_lat", 0.0))
        recipient_lng = float(event.order_data.get("recipient_lng", 0.0))
        recipient_lat = float(event.order_data.get("recipient_lat", 0.0))
        eta_seconds_current = float(event.order_data.get("eta_seconds_current", 0.0))
        order_income_value = float(event.order_data.get("order_income_value", 0.0))

        # --- Time-of-day (recomputed for current dispatch_time) ---
        try:
            sh_tz = pytz.timezone("Asia/Shanghai")
            dt = datetime.fromtimestamp(int(dispatch_time), tz=timezone.utc).astimezone(
                sh_tz
            )
            seconds_of_day = dt.hour * 3600 + dt.minute * 60 + dt.second
            cyc = 2.0 * pi * (seconds_of_day / 86400.0)
            tod_sin_val = sin(cyc)
            tod_cos_val = cos(cyc)
        except Exception:
            tod_sin_val = 0.0
            tod_cos_val = 1.0

        # --- Courier-specific features (from live state) ---
        if courier_state is not None:
            proxy_lng = float(courier_state.current_lng)
            proxy_lat = float(courier_state.current_lat)
            capacity_at_dispatch = float(len(courier_state.pending_deliveries))
            offers_so_far = float(courier_state.waybills_offered)
            accepts_so_far = float(courier_state.waybills_accepted)
            completed_so_far = float(courier_state.orders_delivered)

            # Late ratio
            total_delivered = (
                courier_state.late_deliveries + courier_state.on_time_deliveries
            )
            if total_delivered > 0:
                late_ratio_so_far = float(courier_state.late_deliveries) / float(
                    total_delivered
                )
            else:
                late_ratio_so_far = 0.0

            # Time since last accept
            if courier_state.last_accept_time is not None:
                time_since_last_accept_s = float(
                    dispatch_time - courier_state.last_accept_time
                )
            else:
                time_since_last_accept_s = -1.0  # sentinel: never accepted yet

            # Courier active seconds
            courier_active_seconds_so_far = float(courier_state.total_active_time)
        else:
            # Fallback to original event features for unknown courier
            proxy_lng = float(event.features.get("proxy_lng", 0.0))
            proxy_lat = float(event.features.get("proxy_lat", 0.0))
            capacity_at_dispatch = float(
                event.features.get("capacity_at_dispatch", 0.0)
            )
            offers_so_far = float(event.features.get("offers_so_far", 0.0))
            accepts_so_far = float(event.features.get("accepts_so_far", 0.0))
            completed_so_far = float(event.features.get("completed_so_far", 0.0))
            late_ratio_so_far = float(event.features.get("late_ratio_so_far", 0.0))
            time_since_last_accept_s = float(
                event.features.get("time_since_last_accept_s", -1.0)
            )
            courier_active_seconds_so_far = float(
                event.features.get("courier_active_seconds_so_far", 0.0)
            )

        features = {
            "sender_lng": sender_lng,
            "sender_lat": sender_lat,
            "proxy_lng": proxy_lng,
            "proxy_lat": proxy_lat,
            "recipient_lng": recipient_lng,
            "recipient_lat": recipient_lat,
            "tod_sin": tod_sin_val,
            "tod_cos": tod_cos_val,
            "capacity_at_dispatch": capacity_at_dispatch,
            "offers_so_far": offers_so_far,
            "accepts_so_far": accepts_so_far,
            "completed_so_far": completed_so_far,
            "late_ratio_so_far": late_ratio_so_far,
            "time_since_last_accept_s": time_since_last_accept_s,
            "courier_active_seconds_so_far": courier_active_seconds_so_far,
            "eta_seconds_current": eta_seconds_current,
            "order_income_value": order_income_value,
        }
        return features

    def _handle_case_1_aligned_accept(self, event: OfferEvent, agent_action: int):
        """
        CASE 1: Both agent and history accept.

        Log replay until next dispatch cycle or order arrival (whichever comes first).
        Update courier location from historical data.
        """
        if self.verbose:
            self._log_event(
                event,
                "CASE_1_ALIGNED_ACCEPT",
                f"Both agent and history accept. Courier {event.courier_id}. Using historical routing.",
            )

        self.metrics.log_aligned()

        # Create/update order in tracker
        order = self._create_order_from_event(event)
        order.assign_to_courier(event.courier_id)
        order.mark_accepted(event.courier_id)
        order.historical_decision = 1
        order.agent_decision = 1
        order.is_divergence = False

        # Track in order_tracker for unique order counting
        self.order_tracker.mark_accepted(event.waybill_id)

        # Set time attributes from event
        grab_time = event.order_data.get("grab_time")
        fetch_time = event.order_data.get("fetch_time")
        arrive_time = event.order_data.get("arrive_time")
        order.set_times(
            grab_time=int(grab_time) if grab_time else None,
            fetch_time=int(fetch_time) if fetch_time else None,
            arrive_time=int(arrive_time) if arrive_time else None,
        )

        # Mark as fetched if fetch_time is available
        if fetch_time:
            self.order_tracker.mark_fetched(event.waybill_id)

        # Set capacity attributes
        order.capacity_at_dispatch = int(
            event.order_data.get("capacity_at_dispatch", 0)
        )

        # Log replay: mark delivered at historical arrival time
        if arrive_time:
            arrive_time_int = int(arrive_time)
            self.order_tracker.mark_delivered(
                event.waybill_id,
                arrive_time_int,
                event.order_data.get("recipient_lat", 0.0),
                event.order_data.get("recipient_lng", 0.0),
            )
            # Track utilization for aligned accept delivery
            # Use grab_time (when courier accepted) to arrive_time (when delivered)
            grab_time_int = int(grab_time) if grab_time else event.actual_dispatch_time
            self.utilization_tracker.record_delivery(
                courier_id=event.courier_id,
                waybill_id=event.waybill_id,
                grab_time=grab_time_int,
                arrive_time=arrive_time_int,
            )

            # Track delivery completion for simulation extension
            self._track_delivery_completion(
                event.courier_id, event.waybill_id, arrive_time_int
            )

            # Record delivery on courier state
            est_arrive = event.order_data.get("estimate_arrived_time")
            is_late = False
            if est_arrive:
                try:
                    is_late = arrive_time_int > int(est_arrive)
                except:
                    pass

            courier_state = self.courier_states.get(event.courier_id)
            if courier_state:
                income = float(event.order_data.get("order_income_value", 0.0))
                courier_state.record_delivery(
                    waybill_id=event.waybill_id,
                    grab_time=grab_time_int,
                    arrive_time=arrive_time_int,
                    is_late=is_late,
                    income=income,
                )
                courier_state.record_assignment(
                    event.waybill_id, timestamp=event.actual_dispatch_time
                )

        # Update courier location to recipient (historical routing)
        self._update_courier_location_from_event(event)

        # Log system metrics (use historical outcomes since both accept)
        capacity = int(event.order_data.get("capacity_at_dispatch", 0))
        est_arrive = event.order_data.get("estimate_arrived_time")
        is_late = False
        if arrive_time and est_arrive:
            try:
                is_late = int(arrive_time) > int(est_arrive)
            except:
                pass
        self.metrics.log_system_order(
            accepted=True, late=is_late, courier_id=event.courier_id, capacity=capacity
        )

    def _handle_case_1_1_accept_to_reject(
        self, event: OfferEvent, agent_action: int, hist_decision: int
    ):
        """
        CASE 1.1: Historical accept, Agent reject.

        - Try to find a new courier in this cycle
        - If no courier found or all reject, add to pending queue for next 5 cycles
        - After 5 cycles of attempts, order is lost
        """
        self.profiler.start("case_1_1_total")
        if self.verbose:
            self._log_event(
                event,
                "CASE_1.1_DIVERGENCE",
                f"Historical ACCEPT, Agent REJECT. Original courier {event.courier_id} rejected by agent. "
                f"Cycle {self.current_cycle_id}. Will try reassignment...",
            )

        order = self._create_order_from_event(event)
        order.add_rejection(event.courier_id)  # Original courier rejected
        order.set_divergence(agent_decision=0, historical_decision=1)

        # Track rejection on order_tracker for unique order counting
        self.order_tracker.mark_rejected(event.waybill_id)

        # Record rejection on original courier
        courier_state = self.courier_states.get(event.courier_id)
        if courier_state:
            courier_state.record_rejection(event.waybill_id)

        # Mark original courier as diverged - they were supposed to take this order but didn't
        # Their location will NOT be at the recipient location after this delivery
        # Keep them at their current position (they didn't move for this order)
        self.profiler.start("mark_courier_diverged")
        if courier_state:
            self._mark_courier_diverged(
                courier_id=event.courier_id,
                reason="rejected_historical_accept",
                timestamp=event.actual_dispatch_time,
                simulated_lat=courier_state.current_lat,
                simulated_lng=courier_state.current_lng,
            )
        self.profiler.stop("mark_courier_diverged")

        # Use actual ETA from historical data
        eta = self._get_actual_eta(event.order_id, fallback=1800)

        # Try to find a courier in this cycle
        self.profiler.start("scorer_score_couriers")
        if self.verbose:
            result = self.scorer.score_couriers(
                waybill_id=event.waybill_id,
                sender_lat=event.order_data.get("sender_lat", 0.0),
                sender_lng=event.order_data.get("sender_lng", 0.0),
                dispatch_time=event.actual_dispatch_time,
                estimated_duration=eta,
                excluded_couriers=order.rejecting_couriers,
                verbose=True,
            )
            if result[0] is None:
                self._log_event(
                    event,
                    "NO_COURIER_THIS_CYCLE",
                    f"No couriers available in cycle {self.current_cycle_id}. "
                    f"Adding to pending queue for {self.pending_cycles_limit} more cycles.",
                )
                new_courier_id = None
            else:
                new_courier_id, distance, top_5, rejection_reasons = result
                self._log_event(
                    event,
                    "SCORER_RESULT",
                    f"Top candidate: courier={new_courier_id}, dist={distance:.3f}km\n"
                    + f"  Top 5: "
                    + ", ".join(
                        [f"{c['courier_id']}({c['distance_km']}km)" for c in top_5]
                    ),
                )
        else:
            result = self.scorer.score_couriers(
                waybill_id=event.waybill_id,
                sender_lat=event.order_data.get("sender_lat", 0.0),
                sender_lng=event.order_data.get("sender_lng", 0.0),
                dispatch_time=event.actual_dispatch_time,
                estimated_duration=eta,
                excluded_couriers=order.rejecting_couriers,
            )
            new_courier_id = result[0] if result else None
            distance = result[1] if result else None
        self.profiler.stop("scorer_score_couriers")

        assigned = False
        is_late = False
        realized_capacity = 0
        final_courier_id = -1

        if new_courier_id is not None:
            # Build fresh features for the NEW courier (not the stale original features)
            reassign_features = self._build_features_for_courier(
                event, new_courier_id, event.actual_dispatch_time
            )

            # Ask agent about this courier with correct features
            self.profiler.start("case_1_1_agent_decision")
            agent_decision_new, agent_prob_new = self.agent.act(reassign_features)
            self.profiler.stop("case_1_1_agent_decision")

            if agent_decision_new == 1:
                # Accept this courier
                order.assign_to_courier(new_courier_id)
                order.mark_accepted(new_courier_id)
                assigned = True
                final_courier_id = new_courier_id

                # Track acceptance on order_tracker for unique order counting
                self.order_tracker.mark_accepted(event.waybill_id)

                # Mark new courier as diverged - taking a reassigned order
                recipient_lat = order.recipient_lat
                recipient_lng = order.recipient_lng
                self._mark_courier_diverged(
                    courier_id=new_courier_id,
                    reason="accepted_reassigned_order",
                    timestamp=event.actual_dispatch_time,
                    simulated_lat=recipient_lat,
                    simulated_lng=recipient_lng,
                )

                if self.verbose:
                    self._log_event(
                        event,
                        "REASSIGNED",
                        f"Agent accepts courier {new_courier_id} (dist={distance:.3f}km, prob={agent_prob_new:.3f})",
                    )
                is_late, realized_capacity = self._simulate_order_delivery(
                    order, new_courier_id, event
                )

                # Record on new courier state
                new_courier_state = self.courier_states.get(new_courier_id)
                if new_courier_state:
                    new_courier_state.record_assignment(
                        event.waybill_id, timestamp=event.actual_dispatch_time
                    )
            else:
                # Agent rejects this courier too
                order.add_rejection(new_courier_id)
                new_courier_state = self.courier_states.get(new_courier_id)
                if new_courier_state:
                    new_courier_state.record_rejection(event.waybill_id)
                if self.verbose:
                    self._log_event(
                        event,
                        "REASSIGNMENT_REJECTED",
                        f"Agent rejects courier {new_courier_id} (prob={agent_prob_new:.3f}). Adding to pending queue.",
                    )

        if assigned:
            # Log success
            self.metrics.log_system_order(
                accepted=True,
                late=is_late,
                courier_id=final_courier_id,
                capacity=realized_capacity,
            )
            if self.verbose:
                self._log_event(
                    event,
                    "DELIVERY_SIMULATED",
                    f"Order delivered by courier {final_courier_id}. Late={is_late}, Capacity={realized_capacity}",
                )
        else:
            # Add to pending queue for next configured cycles
            self.pending_reassignments[event.waybill_id] = {
                "order": order,
                "event": event,
                "cycles_remaining": self.pending_cycles_limit,
                "original_cycle": self.current_cycle_id,
            }
            if self.verbose:
                self._log_event(
                    event,
                    "ADDED_TO_PENDING",
                    f"Order added to pending queue. Will attempt reassignment for {self.pending_cycles_limit} more cycles. "
                    f"Rejected couriers so far: {list(order.rejecting_couriers)}",
                )

        # Log divergence
        final_courier = order.current_courier_id if assigned else -1
        self.metrics.log_divergence(
            waybill_id=event.waybill_id,
            courier_id_original=event.courier_id,
            courier_id_assigned=final_courier,
            agent_decision=agent_action,
            historical_decision=hist_decision,
            timestamp=event.actual_dispatch_time,
        )
        self.profiler.stop("case_1_1_total")

    def _handle_case_1_2_reject_to_accept(
        self, event: OfferEvent, agent_action: int, hist_decision: int
    ):
        """
        CASE 1.2: Historical reject, Agent accept.

        - Accept order on original courier
        - Check if arrival time < next dispatch time for this courier
        - If yes: simulate routing (can't use proxy location at next dispatch)
        - If no: log replay (order delivered before next dispatch)
        - Potentially rescore next dispatch cycle if order not yet delivered
        """
        if self.verbose:
            self._log_event(
                event,
                "CASE_1.2_DIVERGENCE",
                f"Historical REJECT, Agent ACCEPT. Agent wants courier {event.courier_id} to take this order.",
            )

        order = self._create_order_from_event(event)
        order.assign_to_courier(event.courier_id)
        order.mark_accepted(event.courier_id)
        order.set_divergence(agent_decision=1, historical_decision=0)

        # Track acceptance on order_tracker for unique order counting
        self.order_tracker.mark_accepted(event.waybill_id)

        # Record assignment on courier state
        courier_state = self.courier_states.get(event.courier_id)
        if courier_state:
            courier_state.record_assignment(
                event.waybill_id, timestamp=event.actual_dispatch_time
            )

        # Mark courier as diverged - they accepted an order they historically rejected
        # Their trajectory will now be different (delivering this order)
        recipient_lat = event.order_data.get("recipient_lat", 0.0)
        recipient_lng = event.order_data.get("recipient_lng", 0.0)
        self._mark_courier_diverged(
            courier_id=event.courier_id,
            reason="accepted_historical_reject",
            timestamp=event.actual_dispatch_time,
            simulated_lat=recipient_lat,  # Will end up at recipient location
            simulated_lng=recipient_lng,
        )

        # Estimate delivery time
        est_arrive = event.order_data.get("estimate_arrived_time")
        next_dispatch = self._get_next_dispatch_time(
            event.courier_id, event.actual_dispatch_time
        )

        use_simulation = False
        is_late = False
        realized_capacity = 0

        if est_arrive and next_dispatch:
            try:
                est_arrive_int = int(est_arrive)

                if est_arrive_int < next_dispatch:
                    # Order should be delivered before next dispatch
                    # Log replay: use historical routing
                    if self.verbose:
                        self._log_event(
                            event,
                            "LOG_REPLAY",
                            f"Est arrival ({est_arrive_int}) < next dispatch ({next_dispatch}). "
                            f"Using historical routing.",
                        )
                    self.order_tracker.mark_delivered(
                        event.waybill_id,
                        est_arrive_int,
                        event.order_data.get("recipient_lat", 0.0),
                        event.order_data.get("recipient_lng", 0.0),
                    )
                    # Mark as fetched if fetch_time is available in historical data
                    fetch_time = event.order_data.get("fetch_time")
                    if fetch_time:
                        self.order_tracker.mark_fetched(event.waybill_id)

                    # Track utilization for log replay delivery
                    # Use grab_time (when courier accepted) to arrive_time (when delivered)
                    grab_time = event.order_data.get("grab_time")
                    grab_time_int = (
                        int(grab_time) if grab_time else event.actual_dispatch_time
                    )
                    arrive_time = event.order_data.get("arrive_time")
                    arrive_time_for_util = (
                        int(arrive_time) if arrive_time else est_arrive_int
                    )
                    self.utilization_tracker.record_delivery(
                        courier_id=event.courier_id,
                        waybill_id=event.waybill_id,
                        grab_time=grab_time_int,
                        arrive_time=arrive_time_for_util,
                    )

                    # Track delivery completion for simulation extension
                    self._track_delivery_completion(
                        event.courier_id, event.waybill_id, arrive_time_for_util
                    )

                    # Record delivery on courier state
                    if courier_state:
                        courier_state.record_delivery(
                            waybill_id=event.waybill_id,
                            grab_time=grab_time_int,
                            arrive_time=arrive_time_for_util,
                            is_late=False,
                            income=float(
                                event.order_data.get("order_income_value", 0.0)
                            ),
                        )

                    self._update_courier_location_from_event(event)
                    # Use historical metrics for agent logging
                    is_late = False  # Assume on-time since it was historically rejected
                    realized_capacity = int(
                        event.order_data.get("capacity_at_dispatch", 0)
                    )
                else:
                    # Order delivery time overlaps with next dispatch
                    # Must simulate routing; courier location diverges from historical
                    use_simulation = True
                    if self.verbose:
                        self._log_event(
                            event,
                            "SIMULATE_ROUTING",
                            f"Est arrival ({est_arrive_int}) >= next dispatch ({next_dispatch}). "
                            f"Simulating delivery - courier location will diverge.",
                        )
                    is_late, realized_capacity = self._simulate_order_delivery(
                        order, event.courier_id, event
                    )
                    # Mark courier as needing rescoring at next dispatch
                    # (handled implicitly: scorer uses current_lat/lng from courier_states)
            except:
                # Fallback: simulate delivery
                use_simulation = True
                if self.verbose:
                    self._log_event(
                        event,
                        "SIMULATE_ROUTING_FALLBACK",
                        f"Could not parse times. Simulating delivery.",
                    )
                is_late, realized_capacity = self._simulate_order_delivery(
                    order, event.courier_id, event
                )
        else:
            # No next dispatch info; just simulate delivery
            use_simulation = True
            if self.verbose:
                self._log_event(
                    event,
                    "SIMULATE_ROUTING_NO_NEXT",
                    f"No next dispatch time known for courier {event.courier_id}. Simulating delivery.",
                )
            is_late, realized_capacity = self._simulate_order_delivery(
                order, event.courier_id, event
            )

        # Log system metrics (agent accepted, simulated delivery)
        # In baseline mode, this case never happens (would be aligned reject)
        self.metrics.log_system_order(
            accepted=True,
            late=is_late,
            courier_id=event.courier_id,
            capacity=realized_capacity,
        )

        if self.verbose:
            self._log_event(
                event,
                "DELIVERY_OUTCOME",
                f"Order assigned to original courier {event.courier_id}. "
                f"Late={is_late}, Capacity={realized_capacity}, Simulated={use_simulation}",
            )

        # Log divergence
        self.metrics.log_divergence(
            waybill_id=event.waybill_id,
            courier_id_original=event.courier_id,
            courier_id_assigned=event.courier_id,  # Same courier, but decision changed
            agent_decision=agent_action,
            historical_decision=hist_decision,
            timestamp=event.actual_dispatch_time,
        )

    def _handle_case_0_aligned_reject(self, event: OfferEvent, agent_action: int):
        """
        CASE 0: Both agent and history reject.

        Pure log replay; no action needed.
        """
        if self.verbose:
            self._log_event(
                event,
                "CASE_0_ALIGNED_REJECT",
                f"Both agent and history reject. Courier {event.courier_id}. No action taken.",
            )

        self.metrics.log_aligned()

        # Create order and set attributes
        order = self._create_order_from_event(event)
        order.mark_rejected()
        order.historical_decision = 0
        order.agent_decision = 0
        order.is_divergence = False

        # Track rejection on order_tracker for unique order counting
        self.order_tracker.mark_rejected(event.waybill_id)

        # Record rejection on courier state
        courier_state = self.courier_states.get(event.courier_id)
        if courier_state:
            courier_state.record_rejection(event.waybill_id)

        # Log system metrics (both rejected)
        self.metrics.log_system_order(
            accepted=False, late=False, courier_id=event.courier_id, capacity=0
        )

    def _handle_invalidated_order(self, event: OfferEvent):
        """
        Handle an order that was invalidated due to courier divergence.

        When a courier diverges (accepts/rejects differently than history),
        their historical orders within 40 minutes are invalidated because:
        - The courier is at a different location than historical data suggests
        - They may be busy delivering a divergent order

        This order needs to be reassigned to a different courier.
        """
        if self.verbose:
            self._log_event(
                event,
                "HANDLE_INVALIDATED",
                f"Processing invalidated order. Courier {event.courier_id} has diverged. "
                f"Will attempt to reassign to available courier.",
            )

        order = self._create_order_from_event(event)
        order.add_rejection(event.courier_id)  # Original courier can't take it
        order.set_divergence(
            agent_decision=0, historical_decision=1
        )  # Treat as agent rejection

        # Track rejection on order_tracker
        self.order_tracker.mark_rejected(event.waybill_id)

        # Use actual ETA from historical data
        eta = self._get_actual_eta(event.order_id, fallback=1800)

        # Try to find a courier - prioritize diverged couriers who are available
        if self.verbose:
            result = self.scorer.score_couriers(
                waybill_id=event.waybill_id,
                sender_lat=event.order_data.get("sender_lat", 0.0),
                sender_lng=event.order_data.get("sender_lng", 0.0),
                dispatch_time=event.actual_dispatch_time,
                estimated_duration=eta,
                excluded_couriers=order.rejecting_couriers,
                verbose=True,
            )
            if result[0] is None:
                self._log_event(
                    event,
                    "NO_COURIER_FOR_INVALIDATED",
                    f"No couriers available. Adding to pending queue.",
                )
                new_courier_id = None
            else:
                new_courier_id, distance, top_5, _ = result
        else:
            result = self.scorer.score_couriers(
                waybill_id=event.waybill_id,
                sender_lat=event.order_data.get("sender_lat", 0.0),
                sender_lng=event.order_data.get("sender_lng", 0.0),
                dispatch_time=event.actual_dispatch_time,
                estimated_duration=eta,
                excluded_couriers=order.rejecting_couriers,
            )
            new_courier_id = result[0] if result else None
            distance = result[1] if result else None

        assigned = False
        is_late = False
        realized_capacity = 0

        if new_courier_id is not None:
            # Check if new courier is available (not busy with diverged delivery)
            new_courier_state = self.courier_states.get(new_courier_id)
            if new_courier_state and not new_courier_state.is_available_at_time(
                event.actual_dispatch_time
            ):
                # Courier is busy, try next
                if self.verbose:
                    self._log_event(
                        event,
                        "COURIER_BUSY",
                        f"Courier {new_courier_id} is busy until {new_courier_state.is_busy_until}. Will add to pending.",
                    )
            else:
                # Build fresh features for the NEW courier (not the stale original features)
                reassign_features = self._build_features_for_courier(
                    event, new_courier_id, event.actual_dispatch_time
                )

                # Ask agent about this courier with correct features
                agent_decision_new, agent_prob_new = self.agent.act(reassign_features)

                if agent_decision_new == 1:
                    # Accept this courier
                    order.assign_to_courier(new_courier_id)
                    order.mark_accepted(new_courier_id)
                    assigned = True

                    self.order_tracker.mark_accepted(event.waybill_id)

                    if self.verbose:
                        self._log_event(
                            event,
                            "INVALIDATED_REASSIGNED",
                            f"Assigned to courier {new_courier_id} (dist={distance:.3f}km)",
                        )

                    is_late, realized_capacity = self._simulate_order_delivery(
                        order, new_courier_id, event
                    )

                    if new_courier_state:
                        new_courier_state.record_assignment(
                            event.waybill_id, timestamp=event.actual_dispatch_time
                        )
                else:
                    order.add_rejection(new_courier_id)

        if assigned:
            self.metrics.log_system_order(
                accepted=True,
                late=is_late,
                courier_id=new_courier_id,
                capacity=realized_capacity,
            )
        else:
            # Add to pending queue for reassignment in subsequent cycles
            self.pending_reassignments[event.waybill_id] = {
                "order": order,
                "event": event,
                "cycles_remaining": self.pending_cycles_limit,
                "original_cycle": self.current_cycle_id,
            }

        # Log divergence (since this was a historical accept that we're now rejecting)
        self.metrics.log_divergence(
            waybill_id=event.waybill_id,
            courier_id_original=event.courier_id,
            courier_id_assigned=new_courier_id if assigned else -1,
            agent_decision=0,  # Effectively rejected
            historical_decision=1,  # Was accepted in history
            timestamp=event.actual_dispatch_time,
        )

    def _create_order_from_event(self, event: OfferEvent) -> OrderState:
        """Create or retrieve order from event."""
        order = self.order_tracker.get_order(event.waybill_id)
        if order is None:
            order = self.order_tracker.create_order(
                waybill_id=event.waybill_id,
                order_id=event.order_id,
                original_courier_id=event.courier_id,
                sender_lat=event.order_data.get("sender_lat", 0.0),
                sender_lng=event.order_data.get("sender_lng", 0.0),
                recipient_lat=event.order_data.get("recipient_lat", 0.0),
                recipient_lng=event.order_data.get("recipient_lng", 0.0),
                estimate_arrived_time=event.order_data.get("estimate_arrived_time"),
                dispatch_time=event.actual_dispatch_time,
            )
        return order

    def _simulate_order_delivery(
        self, order: OrderState, courier_id: int, event: OfferEvent
    ):
        """
        Simulate order delivery: routing, capacity tracking, income calculation.

        If travel_time_model is loaded, uses empirical pickup/delivery times:
        - Pickup time: courier location → restaurant (based on distance + time of day)
        - Fetch time: max(courier_arrival, estimate_meal_prepare_time)
        - Delivery time: restaurant → customer (based on distance + time of day)
        - Arrive time: fetch_time + delivery_time

        Otherwise, falls back to historical ETA.

        Returns (is_late, realized_capacity) for agent metrics logging.
        """
        self.profiler.start("simulate_order_delivery")
        courier_state = self.courier_states.get(courier_id)
        if courier_state is None:
            return (False, 0)

        # Get order timestamps and locations
        dispatch_time = order.dispatch_time
        estimate_meal_prepare_time = event.order_data.get(
            "estimate_meal_prepare_time", 0
        )
        estimate_arrived_time = event.order_data.get("estimate_arrived_time")

        # Calculate completion time using travel time model if available
        if self.travel_time_model is not None:
            # Use empirical travel time model for accurate simulation
            from datetime import datetime

            hour = datetime.fromtimestamp(dispatch_time).hour

            # Calculate pickup distance (courier → restaurant)
            pickup_distance_km = haversine_km(
                courier_state.current_lat,
                courier_state.current_lng,
                order.sender_lat,
                order.sender_lng,
            )

            # Calculate delivery distance (restaurant → customer)
            delivery_distance_km = haversine_km(
                order.sender_lat,
                order.sender_lng,
                order.recipient_lat,
                order.recipient_lng,
            )

            # Get empirical travel times from model
            pickup_time_seconds = self.travel_time_model.get_pickup_time(
                pickup_distance_km,
                hour,
                use_median=True,
                add_noise=self.travel_time_add_noise,
            )
            delivery_time_seconds = self.travel_time_model.get_delivery_time(
                delivery_distance_km,
                hour,
                use_median=True,
                add_noise=self.travel_time_add_noise,
            )

            # Calculate timeline
            grab_time = dispatch_time
            courier_arrival_at_restaurant = grab_time + pickup_time_seconds

            # Fetch time: courier must wait if food isn't ready
            if estimate_meal_prepare_time and estimate_meal_prepare_time > 0:
                fetch_time = max(
                    courier_arrival_at_restaurant, int(estimate_meal_prepare_time)
                )
            else:
                fetch_time = courier_arrival_at_restaurant

            # Final arrival at customer
            completion_time = fetch_time + delivery_time_seconds

            # Calculate wait times for logging
            courier_waited = (
                max(0, int(estimate_meal_prepare_time) - courier_arrival_at_restaurant)
                if estimate_meal_prepare_time
                else 0
            )
            food_waited = (
                max(0, courier_arrival_at_restaurant - int(estimate_meal_prepare_time))
                if estimate_meal_prepare_time
                else 0
            )

            if self.verbose:
                self._log_event(
                    event,
                    "SIMULATED_DELIVERY_TIMES",
                    f"Pickup: {pickup_distance_km:.2f}km → {pickup_time_seconds}s, "
                    f"Delivery: {delivery_distance_km:.2f}km → {delivery_time_seconds}s, "
                    f"Courier wait: {courier_waited}s, Food wait: {food_waited}s, "
                    f"Total: {completion_time - dispatch_time}s",
                )
        else:
            # Fallback: Estimate times using simple distance-based model
            # This is used when no travel_time_model is loaded
            actual_eta = self._get_actual_eta(event.order_id, fallback=1800)
            grab_time = dispatch_time

            # Estimate pickup time: courier → restaurant
            # Assume average speed of 20 km/h for courier on scooter in city
            pickup_distance_km = haversine_km(
                courier_state.current_lat,
                courier_state.current_lng,
                order.sender_lat,
                order.sender_lng,
            )
            pickup_time_seconds = int(
                pickup_distance_km * 180
            )  # 3 min per km = 20 km/h
            pickup_time_seconds = max(
                60, min(pickup_time_seconds, 1200)
            )  # Clamp 1-20 min

            courier_arrival_at_restaurant = grab_time + pickup_time_seconds

            # Fetch time: courier waits if food isn't ready
            if estimate_meal_prepare_time and estimate_meal_prepare_time > 0:
                fetch_time = max(
                    courier_arrival_at_restaurant, int(estimate_meal_prepare_time)
                )
            else:
                fetch_time = courier_arrival_at_restaurant

            # Total completion time uses actual ETA from data
            completion_time = dispatch_time + actual_eta

        # Create order object for router (still used for capacity tracking)
        router_order = Order(
            waybill_id=order.waybill_id,
            sender_lat=order.sender_lat,
            sender_lng=order.sender_lng,
            recipient_lat=order.recipient_lat,
            recipient_lng=order.recipient_lng,
            eta_seconds=completion_time - dispatch_time,
            dispatch_time=order.dispatch_time,
        )

        # Route (for capacity calculation)
        route_result = self.router.route_batch(
            courier_start_lat=courier_state.current_lat,
            courier_start_lng=courier_state.current_lng,
            orders=[router_order],
        )

        # Track capacity
        original_capacity = int(event.order_data.get("capacity_at_dispatch", 0))
        realized_capacity = route_result["realized_capacity"]
        self.capacity_tracker.observe_capacity(
            courier_id=courier_id,
            timestamp=order.dispatch_time,
            realized=realized_capacity,
            original=original_capacity,
        )

        # Mark delivered
        self.order_tracker.mark_delivered(
            order.waybill_id, completion_time, order.recipient_lat, order.recipient_lng
        )

        # Track utilization - record this delivery for the courier
        self.utilization_tracker.record_delivery(
            courier_id=courier_id,
            waybill_id=order.waybill_id,
            grab_time=grab_time,
            arrive_time=completion_time,
        )

        # Track delivery completion for simulation extension
        self._track_delivery_completion(courier_id, order.waybill_id, completion_time)

        # Update courier location
        courier_state.current_lat = order.recipient_lat
        courier_state.current_lng = order.recipient_lng

        # Compute income using SIMULATED completion time
        simulated_income = compute_income_row(
            order.sender_lat,
            order.sender_lng,
            order.recipient_lat,
            order.recipient_lng,
            completion_time,  # Use simulated completion time
            estimate_arrived_time,
        )

        historical_income = float(event.order_data.get("order_income_value", 0.0))

        self.metrics.log_income(
            waybill_id=order.waybill_id,
            courier_id=courier_id,
            historical_income=historical_income,
            simulated_income=simulated_income,
            timestamp=order.dispatch_time,
        )

        # Check if order was late
        is_late = False
        if estimate_arrived_time:
            try:
                is_late = completion_time > int(estimate_arrived_time)
            except:
                pass

        # Set order attributes for KPI tracking
        order.simulated_income = simulated_income
        order.historical_income = historical_income
        order.realized_capacity = realized_capacity
        order.set_times(
            grab_time=grab_time,
            fetch_time=fetch_time,  # Now always calculated (either from model or distance-based)
            arrive_time=completion_time,
        )
        order.is_simulated_delivery = True  # Mark timestamps as simulated (divergence)

        # Mark as fetched - all simulated deliveries are fetched
        # (courier accepted, picked up food, and delivered)
        self.order_tracker.mark_fetched(order.waybill_id)

        # Mark courier as diverged with pending delivery info
        # This tracks where courier will be at fetch_time and arrive_time
        self._mark_courier_diverged(
            courier_id=courier_id,
            reason="simulated_delivery",
            timestamp=dispatch_time,
            simulated_lat=order.sender_lat,  # Start at sender
            simulated_lng=order.sender_lng,
            pending_delivery_info={
                "waybill_id": order.waybill_id,
                "sender_lat": order.sender_lat,
                "sender_lng": order.sender_lng,
                "recipient_lat": order.recipient_lat,
                "recipient_lng": order.recipient_lng,
                "grab_time": grab_time,
                "fetch_time": fetch_time,
                "arrive_time": completion_time,
            },
        )

        # Record delivery on courier state
        courier_state.record_delivery(
            waybill_id=order.waybill_id,
            grab_time=grab_time,
            arrive_time=completion_time,
            is_late=is_late,
            income=simulated_income,
        )
        courier_state.record_capacity(realized_capacity)

        self.profiler.stop("simulate_order_delivery")
        return (is_late, realized_capacity)

    def _update_courier_location_from_event(self, event: OfferEvent):
        """Update courier location after completing historical order."""
        courier_id = event.courier_id
        recipient_lat = event.order_data.get("recipient_lat", 0.0)
        recipient_lng = event.order_data.get("recipient_lng", 0.0)

        if courier_id in self.courier_states:
            self.courier_states[courier_id].current_lat = recipient_lat
            self.courier_states[courier_id].current_lng = recipient_lng

    def _generate_idle_blocks_log(self, output_path, mode_suffix: str):
        """
        Generate detailed log for idle courier blocks showing:
        - Block time ranges and utilization
        - Preceding and following block utilization for context
        """
        from datetime import datetime
        import pytz

        log_path = output_path / f"log{mode_suffix}_idle_blocks.txt"

        try:
            shanghai_tz = pytz.timezone("Asia/Shanghai")
            use_shanghai = True
        except:
            use_shanghai = False

        def ts_to_str(ts):
            """Convert timestamp to Shanghai time readable format."""
            try:
                if use_shanghai:
                    dt = datetime.fromtimestamp(ts, tz=shanghai_tz)
                else:
                    dt = datetime.fromtimestamp(ts)
                return dt.strftime("%Y-%m-%d %H:%M:%S")
            except:
                return str(ts)

        # Collect idle blocks (utilization < 50%) with context
        idle_blocks_data = []

        for cid, blocks in self.utilization_tracker.courier_blocks.items():
            # Sort blocks by start time
            sorted_blocks = sorted(blocks, key=lambda b: b.block_start)

            for i, block in enumerate(sorted_blocks):
                if block.utilization_rate < 0.5:
                    # Get preceding block info
                    prev_block = sorted_blocks[i - 1] if i > 0 else None
                    # Get following block info
                    next_block = (
                        sorted_blocks[i + 1] if i < len(sorted_blocks) - 1 else None
                    )

                    idle_blocks_data.append(
                        {
                            "courier_id": cid,
                            "block": block,
                            "block_index": i,
                            "total_blocks": len(sorted_blocks),
                            "prev_block": prev_block,
                            "next_block": next_block,
                        }
                    )

        # Sort by utilization (lowest first)
        idle_blocks_data.sort(key=lambda x: x["block"].utilization_rate)

        with open(log_path, "w", encoding="utf-8") as f:
            f.write(f"=== Idle Blocks Analysis Log ({mode_suffix.upper()}) ===\n")
            f.write(f"Generated: {datetime.now()}\n")
            f.write(f"Timezone: Shanghai (Asia/Shanghai)\n")
            f.write(f"Idle Block Threshold: <50% utilization\n")
            f.write(f"Total idle blocks found: {len(idle_blocks_data)}\n")
            f.write(
                f"Unique couriers with idle blocks: {len(set(d['courier_id'] for d in idle_blocks_data))}\n\n"
            )

            f.write(f"{'=' * 100}\n")
            f.write(f"IDLE BLOCKS DETAIL (sorted by utilization, lowest first)\n")
            f.write(f"{'=' * 100}\n\n")

            for entry in idle_blocks_data:
                cid = entry["courier_id"]
                block = entry["block"]
                prev_block = entry["prev_block"]
                next_block = entry["next_block"]

                f.write(f"\n{'-' * 80}\n")
                f.write(
                    f"COURIER: {cid} | Block {entry['block_index'] + 1} of {entry['total_blocks']}\n"
                )
                f.write(f"{'-' * 80}\n")

                # Current idle block
                f.write(f"\n>>> IDLE BLOCK (Current):\n")
                f.write(
                    f"    Time:        {ts_to_str(block.block_start)} - {ts_to_str(block.block_end)}\n"
                )
                f.write(
                    f"    Duration:    {block.total_block_time / 60:.1f} minutes ({block.total_block_time / 3600:.2f} hours)\n"
                )
                f.write(f"    Active Time: {block.active_time / 60:.1f} minutes\n")
                f.write(
                    f"    Idle Time:   {(block.total_block_time - block.active_time) / 60:.1f} minutes\n"
                )
                f.write(f"    UTILIZATION: {block.utilization_rate:.1%}\n")
                f.write(f"    Orders:      {block.orders_delivered}\n")

                if block.delivery_spans:
                    f.write(f"    Delivery Spans:\n")
                    for span in block.delivery_spans:
                        f.write(
                            f"      [{ts_to_str(span.start_time)} - {ts_to_str(span.end_time)}] "
                            f"({span.duration / 60:.1f} min)\n"
                        )

                # Previous block context
                f.write(f"\n<<< PRECEDING BLOCK:\n")
                if prev_block:
                    f.write(
                        f"    Time:        {ts_to_str(prev_block.block_start)} - {ts_to_str(prev_block.block_end)}\n"
                    )
                    f.write(
                        f"    Duration:    {prev_block.total_block_time / 60:.1f} minutes\n"
                    )
                    f.write(
                        f"    Active Time: {prev_block.active_time / 60:.1f} minutes\n"
                    )
                    f.write(f"    UTILIZATION: {prev_block.utilization_rate:.1%}\n")
                    f.write(f"    Orders:      {prev_block.orders_delivered}\n")
                else:
                    f.write(f"    (No preceding block - this is the first block)\n")

                # Next block context
                f.write(f"\n>>> FOLLOWING BLOCK:\n")
                if next_block:
                    f.write(
                        f"    Time:        {ts_to_str(next_block.block_start)} - {ts_to_str(next_block.block_end)}\n"
                    )
                    f.write(
                        f"    Duration:    {next_block.total_block_time / 60:.1f} minutes\n"
                    )
                    f.write(
                        f"    Active Time: {next_block.active_time / 60:.1f} minutes\n"
                    )
                    f.write(f"    UTILIZATION: {next_block.utilization_rate:.1%}\n")
                    f.write(f"    Orders:      {next_block.orders_delivered}\n")
                else:
                    f.write(f"    (No following block - this is the last block)\n")

            f.write(f"\n\n{'=' * 100}\n")
            f.write(f"SUMMARY BY COURIER\n")
            f.write(f"{'=' * 100}\n\n")

            # Group by courier
            courier_idle_counts = {}
            for entry in idle_blocks_data:
                cid = entry["courier_id"]
                if cid not in courier_idle_counts:
                    courier_idle_counts[cid] = {
                        "idle_blocks": 0,
                        "total_blocks": entry["total_blocks"],
                        "total_idle_time": 0,
                    }
                courier_idle_counts[cid]["idle_blocks"] += 1
                block = entry["block"]
                courier_idle_counts[cid]["total_idle_time"] += (
                    block.total_block_time - block.active_time
                )

            # Sort by total idle time
            sorted_couriers = sorted(
                courier_idle_counts.items(),
                key=lambda x: x[1]["total_idle_time"],
                reverse=True,
            )

            f.write(
                f"{'Courier ID':<15} {'Idle Blocks':<15} {'Total Blocks':<15} {'Total Idle Time':<20}\n"
            )
            f.write(f"{'-' * 65}\n")
            for cid, data in sorted_couriers:
                idle_hours = data["total_idle_time"] / 3600
                f.write(
                    f"{cid:<15} {data['idle_blocks']:<15} {data['total_blocks']:<15} {idle_hours:.2f} hours\n"
                )

            f.write(f"\n{'=' * 100}\n")
            f.write(f"END OF LOG\n")
            f.write(f"{'=' * 100}\n")

        print(f"Idle blocks log saved to {log_path}")

    def _generate_underutilization_log(self, output_path, mode_suffix: str):
        """
        Generate detailed log for underutilized couriers showing:
        - Courier blocks and their time ranges
        - Orders assigned to each courier from offers_observations
        - grab_time, fetch_time, arrive_time for each order
        - Any gaps or anomalies
        """
        from datetime import datetime

        log_path = output_path / f"log{mode_suffix}_underutilization.txt"

        # Get all events and group by courier
        events = self.dispatcher.get_events()
        courier_events = {}
        for ev in events:
            if ev.courier_id not in courier_events:
                courier_events[ev.courier_id] = []
            courier_events[ev.courier_id].append(ev)

        # Get underutilized couriers (< 50%)
        underutilized = []
        for cid, blocks in self.utilization_tracker.courier_blocks.items():
            total_block_time = sum(b.total_block_time for b in blocks)
            total_active_time = sum(b.active_time for b in blocks)
            if total_block_time > 0:
                util = total_active_time / total_block_time
                if util < 0.5:
                    underutilized.append(
                        {
                            "courier_id": cid,
                            "utilization": util,
                            "blocks": blocks,
                            "total_block_time": total_block_time,
                            "total_active_time": total_active_time,
                        }
                    )

        underutilized.sort(key=lambda x: x["utilization"])

        def ts_to_str(ts):
            """Convert timestamp to readable format."""
            try:
                return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
            except:
                return str(ts)

        with open(log_path, "w", encoding="utf-8") as f:
            f.write(f"=== Underutilization Analysis Log ({mode_suffix.upper()}) ===\n")
            f.write(f"Generated: {datetime.now()}\n")
            f.write(f"Total underutilized couriers (<50%): {len(underutilized)}\n\n")

            for courier_info in underutilized:
                cid = courier_info["courier_id"]
                util = courier_info["utilization"]
                blocks = courier_info["blocks"]

                f.write(f"\n{'=' * 80}\n")
                f.write(f"COURIER {cid}\n")
                f.write(f"{'=' * 80}\n")
                f.write(f"Overall Utilization: {util:.1%}\n")
                f.write(
                    f"Total Block Time: {courier_info['total_block_time'] / 3600:.2f} hours\n"
                )
                f.write(
                    f"Total Active Time: {courier_info['total_active_time'] / 3600:.2f} hours\n\n"
                )

                # Show blocks
                f.write(f"--- WORKING BLOCKS ({len(blocks)}) ---\n")
                for i, block in enumerate(blocks):
                    f.write(f"\nBlock {i + 1}:\n")
                    f.write(
                        f"  Start: {ts_to_str(block.block_start)} ({block.block_start})\n"
                    )
                    f.write(
                        f"  End:   {ts_to_str(block.block_end)} ({block.block_end})\n"
                    )
                    f.write(f"  Duration: {block.total_block_time / 60:.1f} minutes\n")
                    f.write(f"  Active Time: {block.active_time / 60:.1f} minutes\n")
                    f.write(f"  Utilization: {block.utilization_rate:.1%}\n")
                    f.write(f"  Orders Delivered: {block.orders_delivered}\n")

                    # Show delivery spans
                    if block.delivery_spans:
                        f.write(f"  Delivery Spans:\n")
                        for span in block.delivery_spans:
                            f.write(
                                f"    [{ts_to_str(span.start_time)} - {ts_to_str(span.end_time)}] "
                                f"({span.duration / 60:.1f} min) waybills: {span.waybill_ids}\n"
                            )

                    # Show order details recorded in block
                    if block.order_details:
                        f.write(f"  Order Details Recorded:\n")
                        for od in block.order_details:
                            f.write(
                                f"    Waybill {od['waybill_id']}: grab={od.get('grab_time', 'N/A')} -> "
                                f"arrive={od.get('arrive_time', 'N/A')} ({od['duration'] / 60:.1f} min)\n"
                            )

                # Show orders from offers_observations for this courier
                courier_evs = courier_events.get(cid, [])
                f.write(
                    f"\n--- ORDERS FROM OFFERS_OBSERVATIONS ({len(courier_evs)} events) ---\n"
                )

                for ev in courier_evs:
                    od = ev.order_data
                    hist_dec = ev.historical_decision
                    decision_str = "ACCEPTED" if hist_dec == 1 else "REJECTED"

                    grab_time = od.get("grab_time")
                    fetch_time = od.get("fetch_time")
                    arrive_time = od.get("arrive_time")
                    is_completed = od.get("is_completed", 0)

                    f.write(f"\n  Waybill {ev.waybill_id} (Order {ev.order_id}):\n")
                    f.write(f"    Historical Decision: {decision_str}\n")
                    f.write(f"    is_completed: {is_completed}\n")
                    f.write(
                        f"    dispatch_time: {ts_to_str(ev.actual_dispatch_time)} ({ev.actual_dispatch_time})\n"
                    )
                    f.write(
                        f"    grab_time:     {ts_to_str(grab_time) if grab_time else 'NULL'} ({grab_time})\n"
                    )
                    f.write(
                        f"    fetch_time:    {ts_to_str(fetch_time) if fetch_time else 'NULL'} ({fetch_time})\n"
                    )
                    f.write(
                        f"    arrive_time:   {ts_to_str(arrive_time) if arrive_time else 'NULL'} ({arrive_time})\n"
                    )

                    if grab_time and arrive_time:
                        duration = arrive_time - grab_time
                        f.write(
                            f"    Duration (grab->arrive): {duration / 60:.1f} minutes\n"
                        )

                    # Check which block this order falls into
                    if grab_time:
                        for i, block in enumerate(blocks):
                            if (
                                block.block_start - 60
                                <= grab_time
                                <= block.block_end + 60
                            ):
                                f.write(f"    -> Falls within Block {i + 1}\n")
                                break
                        else:
                            f.write(
                                f"    -> WARNING: Does not fall within any block!\n"
                            )
                            f.write(
                                f"       Blocks: {[(b.block_start, b.block_end) for b in blocks]}\n"
                            )

            f.write(f"\n\n{'=' * 80}\n")
            f.write(f"END OF LOG\n")
            f.write(f"{'=' * 80}\n")

        print(f"Underutilization log saved to {log_path}")

    def _generate_report(self):
        """
        Generate and save final report.

        All KPIs are calculated from order and courier attributes stored during simulation.
        All JSON outputs are consolidated into a single system_simulation_metrics_{mode}.json file.
        """
        from datetime import datetime
        import pytz

        # Determine mode label and suffix for output files
        if self.baseline_mode:
            mode_label = "BASELINE"
            mode_suffix = "baseline"
        elif self.agent_type == "ddqn":
            mode_label = "DDQN AGENT"
            mode_suffix = "ddqn_agent"
        else:
            mode_label = "BC AGENT"
            mode_suffix = "bc_agent"

        print(f"\n=== Simulation Summary ({mode_label} mode) ===")

        # Get dispatcher statistics
        dispatcher_stats = self.dispatcher.get_stats()

        # Calculate KPIs from order attributes (single source of truth)
        order_kpis = self.order_tracker.get_summary()

        # Get utilization tracker data (for block-level detail) - this is the accurate source
        utilization_summary = self.utilization_tracker.get_summary()

        # Calculate courier delivery KPIs from scorer (for order counts, not utilization)
        courier_delivery_kpis = self.scorer.calculate_courier_kpis()

        # Get capacity stats
        capacity_summary = self.capacity_tracker.get_stats()

        # Get legacy metrics (for divergence tracking)
        metrics_summary = self.metrics.get_summary()

        # Print summary
        print(f"\nInput Data:")
        print(f"  Total waybills (events) loaded: {dispatcher_stats['total_events']}")
        print(f"  Unique couriers: {dispatcher_stats['unique_couriers']}")
        print(f"  Unique orders: {dispatcher_stats['unique_orders']}")

        print(f"\nOrder KPIs (from order tracker):")
        print(f"  Total waybills processed: {order_kpis['total_waybills']}")
        print(f"  Total unique orders: {order_kpis['total_unique_orders']}")
        print(
            f"  Orders delivered: {order_kpis['delivered_count']} ({order_kpis['delivery_rate']:.2%} of unique orders)"
        )
        print(
            f"  Orders fetched: {order_kpis['fetched_count']} ({order_kpis['fetched_rate']:.2%} of unique orders)"
        )
        print(
            f"  Orders accepted: {order_kpis['accepted_count']} ({order_kpis['acceptance_rate']:.2%} of unique orders)"
        )
        print(
            f"  Orders with rejection(s): {order_kpis['orders_rejected_count']} ({order_kpis['orders_rejection_rate']:.2%} of unique orders)"
        )
        print(
            f"  Waybills rejected: {order_kpis['waybills_rejected_count']} ({order_kpis['waybills_rejection_rate']:.2%} of waybills)"
        )
        print(
            f"  Total rejection events: {order_kpis['total_rejection_events']} (sum of all courier rejections)"
        )
        print(
            f"  Not assigned (exhausted reassign): {order_kpis['not_assigned_count']} ({order_kpis['not_assigned_rate']:.2%} of unique orders)"
        )
        print(
            f"  Orders lost: {order_kpis['lost_count']} ({order_kpis['lost_rate']:.2%} of unique orders)"
        )
        print(
            f"  Late deliveries: {order_kpis['late_count']} ({order_kpis['lateness_rate']:.2%} of delivered)"
        )
        print(
            f"  On-time deliveries: {order_kpis['on_time_count']} ({order_kpis['on_time_rate']:.2%} of delivered)"
        )
        print(
            f"  Divergences: {order_kpis['divergence_count']} ({order_kpis['divergence_rate']:.2%} of waybills)"
        )
        print(
            f"  Diverged couriers: {len(self.diverged_couriers)} (couriers whose trajectory differs from history)"
        )

        if self.baseline_mode:
            print(
                "  (Note: Baseline mode expects 0 divergences - pure historical replay)"
            )

        print(f"\nCourier KPIs (from utilization tracker):")
        print(f"  Total couriers tracked: {utilization_summary['total_couriers']}")
        print(f"  Total blocks: {utilization_summary['total_blocks']}")
        print(
            f"  Orders delivered (unique): {utilization_summary['total_orders_delivered']}"
        )
        print(
            f"  Total block time: {utilization_summary['total_block_time_hours']:.2f} hours"
        )
        print(
            f"  Total active time: {utilization_summary['total_active_time_hours']:.2f} hours"
        )
        print(
            f"  Total idle time: {utilization_summary['total_idle_time_hours']:.2f} hours"
        )
        print(
            f"  Overall utilization: {utilization_summary['overall_utilization']:.2%}"
        )
        print(f"  Avg block utilization: {utilization_summary['avg_utilization']:.2%}")
        print(
            f"  Underutilized couriers (<50%): {utilization_summary['underutilized_couriers_count']}"
        )

        print(f"\nCourier Delivery KPIs (from courier state):")
        print(f"  Waybills offered: {courier_delivery_kpis['total_waybills_offered']}")
        print(
            f"  Waybills accepted: {courier_delivery_kpis['total_waybills_accepted']}"
        )
        print(
            f"  Waybills rejected: {courier_delivery_kpis['total_waybills_rejected']}"
        )
        print(
            f"  On-time deliveries: {courier_delivery_kpis['total_on_time_deliveries']}"
        )
        print(f"  Late deliveries: {courier_delivery_kpis['total_late_deliveries']}")

        print(f"\nCapacity stats:")
        print(f"  Overshoot count: {capacity_summary['overshoot_count']}")
        print(f"  Overshoot rate: {capacity_summary['overshoot_rate']:.2%}")
        print(f"  Avg realized capacity: {capacity_summary['avg_realized']:.2f}")
        print(f"  Max realized capacity: {capacity_summary['max_realized']}")

        # Print simulation timing info
        if self.extended_end_time > self.simulation_end_time:
            extension_secs = self.extended_end_time - self.simulation_end_time
            pending_deliveries, pending_couriers = (
                self.utilization_tracker.count_deliveries_beyond_time(
                    self.simulation_end_time
                )
            )
            print(f"\nSimulation Timing:")
            print(
                f"  Original end time: {self._format_timestamp(self.simulation_end_time)}"
            )
            print(
                f"  Extended end time: {self._format_timestamp(self.extended_end_time)}"
            )
            print(f"  Extension: {extension_secs}s ({extension_secs / 60:.1f}min)")
            print(
                f"  Deliveries in transit at end: {pending_deliveries} ({pending_couriers} couriers)"
            )
            print(
                f"  (Extension ensures all in-transit deliveries are fully counted as active time)"
            )

        # Save all reports to single consolidated file
        from pathlib import Path

        output_path = Path(self.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Get current timestamp in Shanghai timezone
        try:
            shanghai_tz = pytz.timezone("Asia/Shanghai")
            timestamp = datetime.now(shanghai_tz).strftime("%Y-%m-%d %H:%M:%S %Z")
        except:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Build consolidated metrics file
        consolidated_metrics = {
            "metadata": {
                "mode": mode_label,
                "timestamp": timestamp,
                "agent_ckpt": self.agent_ckpt if not self.baseline_mode else "N/A",
                "threshold_name": self.threshold_name
                if not self.baseline_mode
                else "N/A",
                "manifest_path": self.manifest_path,
                "parquet_path": self.parquet_path,
                "max_hours": self.max_hours,
                "simulation_end_time": self.simulation_end_time,
                "extended_end_time": self.extended_end_time,
                "simulation_extended": self.extended_end_time
                > self.simulation_end_time,
                "extension_seconds": self.extended_end_time - self.simulation_end_time,
                "scorer_mode": self.scorer_mode,
                "scorer_distance_weight": self.scorer_distance_weight,
                "scorer_load_penalty_weight": self.scorer_load_penalty_weight,
                "scorer_late_penalty_weight": self.scorer_late_penalty_weight,
                "scorer_idle_bonus_weight": self.scorer_idle_bonus_weight,
                "max_courier_distance_km": self.max_courier_distance_km,
                "block_buffer_seconds": self.block_buffer_seconds,
                "max_pending_per_cycle": self.max_pending_per_cycle,
                "pending_cycles_limit": self.pending_cycles_limit,
                "divergence_window_seconds": self.divergence_window_seconds,
                "travel_time_add_noise": self.travel_time_add_noise,
            },
            "input_data": {
                "total_events": dispatcher_stats["total_events"],
                "unique_couriers": dispatcher_stats["unique_couriers"],
                "unique_orders": dispatcher_stats["unique_orders"],
            },
            "order_kpis": order_kpis,
            "courier_kpis": {
                "utilization_metrics": utilization_summary,
                "delivery_metrics": courier_delivery_kpis,
            },
            "capacity_metrics": capacity_summary,
            "divergence_metrics": {
                "total_events": metrics_summary["total_events"],
                "aligned_events": metrics_summary["aligned_events"],
                "divergence_count": metrics_summary["divergence_count"],
                "divergence_rate": metrics_summary["divergence_rate"],
                "divergence_types": metrics_summary.get("divergence_types", {}),
                "divergences_sample": metrics_summary.get("divergences", [])[:20],
            },
            "income_metrics": {
                "total_income_delta": metrics_summary["total_income_delta"],
                "avg_income_delta": metrics_summary["avg_income_delta_per_order"],
                "income_events_count": metrics_summary["income_events_count"],
            },
        }

        # Save consolidated metrics file
        metrics_filename = f"system_simulation_metrics_{mode_suffix}.json"
        with open(output_path / metrics_filename, "w") as f:
            json.dump(consolidated_metrics, f, indent=2, default=str)

        # Generate detailed underutilization log
        self._generate_underutilization_log(output_path, f"_{mode_suffix}")

        # Generate idle blocks log with before/after context
        self._generate_idle_blocks_log(output_path, f"_{mode_suffix}")

        # Print profiling summary if available
        print("\n" + "=" * 60)
        print("PROFILING SUMMARY")
        print("=" * 60)
        self.profiler.print_summary()

        # Save profiling data to file
        profiling_filename = f"profiling_{mode_suffix}.json"
        with open(output_path / profiling_filename, "w") as f:
            json.dump(self.profiler.get_summary(), f, indent=2)

        print(f"\nReports saved to {self.output_dir}")
        print(f"  - {metrics_filename} (consolidated metrics)")
        print(f"  - {profiling_filename} (profiling data)")
        print(f"  - log_{mode_suffix}_underutilization.txt")
        print(f"  - log_{mode_suffix}_idle_blocks.txt")
