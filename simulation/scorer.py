"""
Scorer: Re-match orders to next-best courier on divergence.

Uses greedy distance-based logic to assign orders to available couriers.
CourierState tracks all courier-level attributes for KPI calculation.
"""

from dataclasses import dataclass, field
from typing import Dict, Set, Optional, Tuple, List
import numpy as np


def haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Haversine distance in km."""
    lat1, lng1, lat2, lng2 = map(np.radians, [lat1, lng1, lat2, lng2])
    dlat = lat2 - lat1
    dlng = lng2 - lng1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlng / 2) ** 2
    a = np.clip(a, 0.0, 1.0)
    c = 2 * np.arcsin(np.sqrt(a))
    return 6371.0 * c


@dataclass
class CourierState:
    """
    Track courier availability, location, and KPI attributes.
    All courier-level KPIs are calculated from these attributes at the end.

    Divergence tracking:
    - is_diverged: True if courier's trajectory differs from historical
    - divergence_timestamp: When the divergence occurred
    - is_busy_until: Timestamp when courier will be free (after diverged delivery)
    - current_phase: 'idle', 'to_sender', 'to_recipient' - what courier is currently doing
    - current_order_waybill: waybill_id of order currently being handled (if any)
    """

    courier_id: int
    current_lat: float
    current_lng: float
    available: bool
    shift_start: Optional[int] = None
    shift_end: Optional[int] = None
    blocks: List[Tuple[int, int]] = None  # List of (start, end) timestamps
    rejected_waybills: Set[int] = None  # orders this courier has rejected

    # Divergence tracking
    is_diverged: bool = False
    divergence_timestamp: Optional[int] = None
    is_busy_until: Optional[int] = (
        None  # When courier will be free from current diverged delivery
    )
    current_phase: str = "idle"  # 'idle', 'to_sender', 'to_recipient'
    current_order_waybill: Optional[int] = None  # waybill currently being handled
    pending_sender_lat: Optional[float] = None  # Sender location of current order
    pending_sender_lng: Optional[float] = None
    pending_recipient_lat: Optional[float] = None  # Recipient location of current order
    pending_recipient_lng: Optional[float] = None
    pending_fetch_time: Optional[int] = None  # When courier will arrive at sender
    pending_arrive_time: Optional[int] = None  # When courier will arrive at recipient
    pending_grab_time: Optional[int] = None  # When courier accepted (start of delivery)
    delivery_start_lat: Optional[float] = None  # Courier location when delivery started
    delivery_start_lng: Optional[float] = None  # (for interpolation)

    # Invalidated historical orders (due to divergence within 40-min window)
    invalidated_waybills: Set[int] = None

    # KPI tracking attributes
    orders_delivered: int = 0
    waybills_offered: int = 0  # Total waybills offered to this courier
    waybills_accepted: int = 0  # Waybills this courier accepted
    waybills_rejected: int = 0  # Waybills this courier rejected
    total_active_time: int = 0  # Total seconds actively delivering
    total_block_time: int = 0  # Total seconds in working blocks
    delivered_waybills: Set[int] = None  # Unique waybills delivered
    late_deliveries: int = 0
    on_time_deliveries: int = 0
    total_income: float = 0.0

    # Capacity tracking
    max_concurrent_orders: int = 0
    capacity_sum: int = 0
    capacity_observations: int = 0

    # Temporal tracking for feature recomputation
    last_accept_time: Optional[int] = None  # timestamp of last accepted assignment

    # Active delivery tracking (for simulation extension)
    pending_deliveries: Set[int] = None  # waybill_ids currently being delivered
    is_active: bool = False  # True if courier has pending deliveries

    def __post_init__(self):
        if self.rejected_waybills is None:
            self.rejected_waybills = set()
        if self.blocks is None:
            self.blocks = []
        if self.delivered_waybills is None:
            self.delivered_waybills = set()
        if self.pending_deliveries is None:
            self.pending_deliveries = set()
        if self.invalidated_waybills is None:
            self.invalidated_waybills = set()
        # Calculate total block time from blocks
        if self.blocks:
            self.total_block_time = sum(end - start for start, end in self.blocks)

    def mark_diverged(self, timestamp: int, reason: str = "unknown"):
        """Mark this courier as having diverged from historical trajectory."""
        if not self.is_diverged:
            self.is_diverged = True
            self.divergence_timestamp = timestamp

    def set_pending_delivery(
        self,
        waybill_id: int,
        sender_lat: float,
        sender_lng: float,
        recipient_lat: float,
        recipient_lng: float,
        fetch_time: int,
        arrive_time: int,
        grab_time: int = None,
    ):
        """
        Set pending delivery info for a diverged courier.
        This tracks where the courier will be at different times for interpolation.

        Args:
            waybill_id: Order being delivered
            sender_lat/lng: Restaurant location
            recipient_lat/lng: Customer location
            fetch_time: When courier arrives at restaurant
            arrive_time: When courier delivers to customer
            grab_time: When courier accepted (defaults to fetch_time - 600)
        """
        # Store starting location BEFORE updating anything (for interpolation)
        self.delivery_start_lat = self.current_lat
        self.delivery_start_lng = self.current_lng

        self.current_order_waybill = waybill_id
        self.pending_sender_lat = sender_lat
        self.pending_sender_lng = sender_lng
        self.pending_recipient_lat = recipient_lat
        self.pending_recipient_lng = recipient_lng
        self.pending_fetch_time = fetch_time
        self.pending_arrive_time = arrive_time
        self.pending_grab_time = (
            grab_time if grab_time else (fetch_time - 600)
        )  # Default 10min pickup
        self.is_busy_until = arrive_time
        self.current_phase = "to_sender"

    def update_location_for_time(self, current_time: int):
        """
        Update courier location based on current time and pending delivery.

        For diverged couriers with pending deliveries:
        - If current_time < fetch_time: courier is traveling to sender
        - If fetch_time <= current_time < arrive_time: courier is at sender or traveling to recipient
        - If current_time >= arrive_time: courier is at recipient (delivery complete)
        """
        if not self.is_diverged or self.current_order_waybill is None:
            return

        if self.pending_fetch_time and current_time >= self.pending_fetch_time:
            # Courier has reached sender (fetch_time), now at sender or heading to recipient
            if self.pending_sender_lat is not None:
                self.current_lat = self.pending_sender_lat
                self.current_lng = self.pending_sender_lng
            self.current_phase = "to_recipient"

        if self.pending_arrive_time and current_time >= self.pending_arrive_time:
            # Courier has completed delivery, now at recipient
            if self.pending_recipient_lat is not None:
                self.current_lat = self.pending_recipient_lat
                self.current_lng = self.pending_recipient_lng
            self.current_phase = "idle"
            self.current_order_waybill = None
            self.is_busy_until = None

    def is_available_at_time(self, current_time: int) -> bool:
        """Check if courier is available at a given time (not busy with diverged delivery)."""
        if not self.is_diverged:
            return True  # Non-diverged couriers use historical availability
        if self.is_busy_until is None:
            return True
        return current_time >= self.is_busy_until

    def get_interpolated_location(self, query_time: int) -> tuple:
        """
        Get interpolated courier location at any timestamp.

        For diverged couriers with pending delivery:
        - Before fetch_time: interpolate between start location → sender
        - Between fetch_time and arrive_time: interpolate between sender → recipient
        - After arrive_time: at recipient location

        Uses linear interpolation (acceptable for <10km urban distances).

        Timeline:
            grab_time ──────────► fetch_time ──────────► arrive_time
            [at start]            [at sender]            [at recipient]
                  └── to_sender ──►└── to_recipient ────►

        Returns:
            (latitude, longitude) tuple
        """
        # If not diverged or no pending delivery, return current discrete location
        if not self.is_diverged or self.current_order_waybill is None:
            return (self.current_lat, self.current_lng)

        # Need all waypoints
        if self.pending_fetch_time is None or self.pending_arrive_time is None:
            return (self.current_lat, self.current_lng)

        # Start location (where courier was when they started this delivery)
        # Use delivery_start_lat/lng if available, otherwise fall back to current
        if self.delivery_start_lat is not None and self.delivery_start_lng is not None:
            start_lat, start_lng = self.delivery_start_lat, self.delivery_start_lng
        else:
            start_lat, start_lng = self.current_lat, self.current_lng

        sender_lat, sender_lng = self.pending_sender_lat, self.pending_sender_lng
        recipient_lat, recipient_lng = (
            self.pending_recipient_lat,
            self.pending_recipient_lng,
        )

        # Grab time (when courier accepted, start of travel)
        grab_time = (
            self.pending_grab_time
            or self.divergence_timestamp
            or self.pending_fetch_time - 600
        )
        fetch_time = self.pending_fetch_time
        arrive_time = self.pending_arrive_time

        # Phase 1: Traveling to sender (grab_time → fetch_time)
        if query_time < fetch_time:
            # Interpolate between start and sender
            if fetch_time <= grab_time:
                return (sender_lat, sender_lng)
            progress = max(
                0.0, min(1.0, (query_time - grab_time) / (fetch_time - grab_time))
            )
            lat = start_lat + progress * (sender_lat - start_lat)
            lng = start_lng + progress * (sender_lng - start_lng)
            return (lat, lng)

        # Phase 2: Traveling to recipient (fetch_time → arrive_time)
        elif query_time < arrive_time:
            # Interpolate between sender and recipient
            if arrive_time <= fetch_time:
                return (recipient_lat, recipient_lng)
            progress = max(
                0.0, min(1.0, (query_time - fetch_time) / (arrive_time - fetch_time))
            )
            lat = sender_lat + progress * (recipient_lat - sender_lat)
            lng = sender_lng + progress * (recipient_lng - sender_lng)
            return (lat, lng)

        # Phase 3: Delivery complete, at recipient
        else:
            return (recipient_lat, recipient_lng)

    def invalidate_historical_order(self, waybill_id: int):
        """Mark a historical order as invalidated due to divergence."""
        self.invalidated_waybills.add(waybill_id)

    def is_order_invalidated(self, waybill_id: int) -> bool:
        """Check if a historical order has been invalidated."""
        return waybill_id in self.invalidated_waybills

    def start_delivery(self, waybill_id: int):
        """Mark that a delivery has started (order accepted, in transit)."""
        self.pending_deliveries.add(waybill_id)
        self.is_active = True

    def complete_delivery(self, waybill_id: int):
        """Mark that a delivery has completed."""
        self.pending_deliveries.discard(waybill_id)
        self.is_active = len(self.pending_deliveries) > 0

    def record_delivery(
        self,
        waybill_id: int,
        grab_time: int,
        arrive_time: int,
        is_late: bool,
        income: float = 0.0,
    ):
        """Record a delivery for this courier."""
        if waybill_id not in self.delivered_waybills:
            self.delivered_waybills.add(waybill_id)
            self.orders_delivered += 1

            if is_late:
                self.late_deliveries += 1
            else:
                self.on_time_deliveries += 1

            # Track active time (grab_time to arrive_time)
            if grab_time and arrive_time:
                self.total_active_time += arrive_time - grab_time

            self.total_income += income

        # Mark delivery as complete
        self.complete_delivery(waybill_id)

    def record_offer(self, waybill_id: int):
        """Record that a waybill was offered to this courier."""
        self.waybills_offered += 1

    def record_assignment(self, waybill_id: int, timestamp: Optional[int] = None):
        """Record a waybill acceptance by this courier."""
        self.waybills_accepted += 1
        if timestamp is not None:
            self.last_accept_time = timestamp

    def record_rejection(self, waybill_id: int):
        """Record that this courier rejected a waybill."""
        self.rejected_waybills.add(waybill_id)
        self.waybills_rejected += 1

    def record_capacity(self, realized_capacity: int):
        """Record a capacity observation."""
        self.capacity_sum += realized_capacity
        self.capacity_observations += 1
        self.max_concurrent_orders = max(self.max_concurrent_orders, realized_capacity)

    @property
    def utilization_rate(self) -> float:
        """Calculate utilization rate from active time / block time."""
        if self.total_block_time == 0:
            return 0.0
        return min(1.0, self.total_active_time / self.total_block_time)

    @property
    def avg_capacity(self) -> float:
        """Calculate average capacity."""
        if self.capacity_observations == 0:
            return 0.0
        return self.capacity_sum / self.capacity_observations

    @property
    def acceptance_rate(self) -> float:
        """Calculate acceptance rate (accepted / offered)."""
        if self.waybills_offered == 0:
            return 0.0
        return self.waybills_accepted / self.waybills_offered

    @property
    def on_time_rate(self) -> float:
        """Calculate on-time delivery rate."""
        if self.orders_delivered == 0:
            return 0.0
        return self.on_time_deliveries / self.orders_delivered

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "courier_id": self.courier_id,
            "orders_delivered": self.orders_delivered,
            "waybills_offered": self.waybills_offered,
            "waybills_accepted": self.waybills_accepted,
            "waybills_rejected": self.waybills_rejected,
            "late_deliveries": self.late_deliveries,
            "on_time_deliveries": self.on_time_deliveries,
            "total_active_time_sec": self.total_active_time,
            "total_block_time_sec": self.total_block_time,
            "utilization_rate": self.utilization_rate,
            "avg_capacity": self.avg_capacity,
            "max_concurrent_orders": self.max_concurrent_orders,
            "acceptance_rate": self.acceptance_rate,
            "on_time_rate": self.on_time_rate,
            "total_income": self.total_income,
            "num_blocks": len(self.blocks) if self.blocks else 0,
        }


class Scorer:
    """
    Greedy distance-based courier matching.

    Assumptions:
    - Couriers available if they appear in historical data during relevant time window
    - Courier location = last known location or sender location if no prior info
    - Nearest available courier who hasn't rejected this order wins
    - No capacity constraint enforcement

    Optimizations:
    - Pre-compute available couriers once per cycle
    - Distance radius filter to avoid checking distant couriers
    - Batch scoring for pending reassignments
    """

    # Maximum distance in km to consider a courier for reassignment.
    # Paper specifies 3km radius (Section 2.3.3): "located more than 3 km from the restaurant"
    # This reflects the effective operational range observed in historical dispatch data.
    MAX_COURIER_DISTANCE_KM = 3.0

    def __init__(
        self,
        historical_data: Dict[int, CourierState],
        max_courier_distance_km: float = MAX_COURIER_DISTANCE_KM,
        block_buffer_seconds: int = 60,
        scorer_mode: str = "distance_only",
        distance_weight: float = 1.0,
        load_penalty_weight: float = 0.35,
        late_penalty_weight: float = 0.35,
        idle_bonus_weight: float = 0.15,
    ):
        """
        Args:
            historical_data: dict[courier_id] -> CourierState with initial positions
        """
        self.courier_states = historical_data
        self.max_courier_distance_km = max_courier_distance_km
        self.block_buffer_seconds = block_buffer_seconds
        self.scorer_mode = scorer_mode
        self.distance_weight = distance_weight
        self.load_penalty_weight = load_penalty_weight
        self.late_penalty_weight = late_penalty_weight
        self.idle_bonus_weight = idle_bonus_weight
        # Cache for available couriers per cycle (optimization)
        self._available_couriers_cache: Dict[int, List[Tuple[int, "CourierState"]]] = {}
        self._cache_timestamp: int = 0

    def _late_ratio(self, state: CourierState) -> float:
        delivered = state.late_deliveries + state.on_time_deliveries
        if delivered <= 0:
            return 0.0
        return state.late_deliveries / delivered

    def _idle_bonus(self, state: CourierState, dispatch_time: int) -> float:
        if state.last_accept_time is None:
            return 1.0
        idle_seconds = max(0, dispatch_time - state.last_accept_time)
        return min(1.0, idle_seconds / 1800.0)

    def _load_level(self, state: CourierState) -> float:
        return min(1.0, len(state.pending_deliveries) / 8.0)

    def _candidate_score(
        self, state: CourierState, dist_km: float, dispatch_time: int
    ) -> float:
        if self.scorer_mode == "distance_only":
            return dist_km

        normalized_dist = min(1.0, dist_km / max(1e-6, self.max_courier_distance_km))
        load_level = self._load_level(state)
        late_ratio = self._late_ratio(state)
        idle_bonus = self._idle_bonus(state, dispatch_time)

        score = (
            self.distance_weight * normalized_dist
            + self.load_penalty_weight * load_level
            + self.late_penalty_weight * late_ratio
            - self.idle_bonus_weight * idle_bonus
        )
        return score

    def get_available_couriers_for_time(
        self, dispatch_time: int, completion_time: int
    ) -> List[Tuple[int, "CourierState"]]:
        """
        Get list of couriers available at the given time window.
        Uses caching to avoid recomputation within same cycle.

        Returns:
            List of (courier_id, CourierState) tuples for available couriers
        """
        # Cache key based on time window (1-minute granularity for caching)
        cache_key = dispatch_time // 60

        if cache_key in self._available_couriers_cache:
            return self._available_couriers_cache[cache_key]

        # Clear old cache entries (keep only recent)
        if len(self._available_couriers_cache) > 100:
            self._available_couriers_cache.clear()

        available = []
        block_buffer_seconds = self.block_buffer_seconds

        for cid, state in self.courier_states.items():
            if not state.available:
                continue

            # Check if diverged courier is busy
            if state.is_diverged and not state.is_available_at_time(dispatch_time):
                continue

            # Check working blocks
            if state.blocks:
                in_block = False
                for start, end in state.blocks:
                    if (
                        start - block_buffer_seconds
                    ) <= dispatch_time and completion_time <= (
                        end + block_buffer_seconds
                    ):
                        in_block = True
                        break
                if not in_block:
                    continue
            else:
                # Fallback to shift window
                if state.shift_start is not None and dispatch_time < state.shift_start:
                    continue
                if state.shift_end is not None and dispatch_time > state.shift_end:
                    continue

            available.append((cid, state))

        self._available_couriers_cache[cache_key] = available
        return available

    def score_couriers_fast(
        self,
        waybill_id: int,
        sender_lat: float,
        sender_lng: float,
        dispatch_time: int,
        estimated_duration: int = 0,
        excluded_couriers: Set[int] = None,
        available_couriers: List[Tuple[int, "CourierState"]] = None,
        max_distance_km: float = None,
    ) -> Optional[Tuple[int, float]]:
        """
        Fast courier scoring using pre-computed available couriers and distance filter.

        Args:
            waybill_id: order ID
            sender_lat, sender_lng: restaurant location
            dispatch_time: current timestamp
            estimated_duration: estimated seconds to complete order
            excluded_couriers: set of courier_ids to exclude
            available_couriers: pre-computed list from get_available_couriers_for_time
            max_distance_km: maximum distance to consider (default: MAX_COURIER_DISTANCE_KM)

        Returns:
            (courier_id, distance_km) or None if no courier available
        """
        if excluded_couriers is None:
            excluded_couriers = set()
        if max_distance_km is None:
            max_distance_km = self.max_courier_distance_km

        completion_time = dispatch_time + estimated_duration

        # Use pre-computed available couriers if provided
        if available_couriers is None:
            available_couriers = self.get_available_couriers_for_time(
                dispatch_time, completion_time
            )

        best_courier = None
        best_distance = float("inf")
        best_score = float("inf")

        for cid, state in available_couriers:
            # Skip excluded
            if cid in excluded_couriers:
                continue
            # Skip if courier already rejected this order
            if waybill_id in state.rejected_waybills:
                continue

            # For diverged couriers, update location
            if state.is_diverged:
                state.update_location_for_time(dispatch_time)

            # Compute distance
            dist = haversine_km(
                state.current_lat, state.current_lng, sender_lat, sender_lng
            )

            # Distance filter - skip if too far
            if dist > max_distance_km:
                continue

            candidate_score = self._candidate_score(state, dist, dispatch_time)

            # Track best
            if candidate_score < best_score:
                best_score = candidate_score
                best_distance = dist
                best_courier = cid

        if best_courier is None:
            return None

        return best_courier, best_distance

    def invalidate_cache(self):
        """Clear the available couriers cache (call when cycle changes)."""
        self._available_couriers_cache.clear()

    def score_couriers(
        self,
        waybill_id: int,
        sender_lat: float,
        sender_lng: float,
        dispatch_time: int,
        estimated_duration: int = 0,
        excluded_couriers: Set[int] = None,
        verbose: bool = False,
    ) -> Optional[Tuple[int, float]]:
        """
        Find next-best courier for this order.

        Args:
            waybill_id: order ID
            sender_lat, sender_lng: restaurant location
            dispatch_time: current timestamp
            estimated_duration: estimated seconds to complete order
            excluded_couriers: set of courier_ids to exclude (e.g., original courier who was pruned)
            verbose: if True, return detailed scoring info

        Returns:
            (courier_id, distance_km) or None if no courier available
            If verbose=True, returns (courier_id, distance_km, top_5_candidates, rejection_reasons)
        """
        if excluded_couriers is None:
            excluded_couriers = set()

        completion_time = dispatch_time + estimated_duration

        candidates = []
        rejection_reasons = {}  # courier_id -> reason for rejection

        for cid, state in self.courier_states.items():
            # Skip excluded or unavailable
            if cid in excluded_couriers:
                rejection_reasons[cid] = "excluded (already tried/rejected)"
                continue
            if not state.available:
                rejection_reasons[cid] = "not available"
                continue
            # Skip if courier already rejected this order
            if waybill_id in state.rejected_waybills:
                rejection_reasons[cid] = "already rejected this waybill"
                continue

            # Check if diverged courier is busy with a delivery
            if state.is_diverged and not state.is_available_at_time(dispatch_time):
                rejection_reasons[cid] = (
                    f"diverged courier busy until {state.is_busy_until}"
                )
                continue

            # For diverged couriers, update their location based on current time
            if state.is_diverged:
                state.update_location_for_time(dispatch_time)

            # Allow 60-second buffer for block boundaries
            block_buffer_seconds = self.block_buffer_seconds

            # Check blocks if available
            if state.blocks:
                in_block = False
                block_info = None
                for start, end in state.blocks:
                    # Check if dispatch time is within block AND completion time is within block
                    # Add buffer: allow dispatch up to 60s before block start
                    # and completion up to 60s after block end
                    if (
                        start - block_buffer_seconds
                    ) <= dispatch_time and completion_time <= (
                        end + block_buffer_seconds
                    ):
                        in_block = True
                        block_info = (start, end)
                        break
                if not in_block:
                    rejection_reasons[cid] = (
                        f"outside working blocks (dispatch={dispatch_time}, completion={completion_time}, blocks={state.blocks[:3]}...)"
                    )
                    continue
            else:
                # Fallback to simple shift window if no blocks defined
                if state.shift_start is not None and dispatch_time < state.shift_start:
                    rejection_reasons[cid] = (
                        f"before shift start ({dispatch_time} < {state.shift_start})"
                    )
                    continue
                if state.shift_end is not None and dispatch_time > state.shift_end:
                    rejection_reasons[cid] = (
                        f"after shift end ({dispatch_time} > {state.shift_end})"
                    )
                    continue

            # Compute distance
            dist = haversine_km(
                state.current_lat, state.current_lng, sender_lat, sender_lng
            )
            if dist > self.max_courier_distance_km:
                rejection_reasons[cid] = (
                    f"outside distance radius ({dist:.3f}km > {self.max_courier_distance_km:.3f}km)"
                )
                continue

            score = self._candidate_score(state, dist, dispatch_time)
            candidates.append((cid, dist, score, state))

        if not candidates:
            if verbose:
                return None, None, [], rejection_reasons
            return None

        # Sort by configured scorer objective
        if self.scorer_mode == "distance_only":
            candidates.sort(key=lambda x: x[1])
        else:
            candidates.sort(key=lambda x: x[2])

        if verbose:
            # Return top 5 with details
            top_5 = []
            for cid, dist, score, state in candidates[:5]:
                top_5.append(
                    {
                        "courier_id": cid,
                        "distance_km": round(dist, 3),
                        "score": round(score, 4),
                        "lat": state.current_lat,
                        "lng": state.current_lng,
                        "num_blocks": len(state.blocks) if state.blocks else 0,
                    }
                )
            return candidates[0][0], candidates[0][1], top_5, rejection_reasons

        return candidates[0][0], candidates[0][1]

    def update_courier_location(self, courier_id: int, lat: float, lng: float):
        """Update courier's current location after completing an order."""
        if courier_id in self.courier_states:
            self.courier_states[courier_id].current_lat = lat
            self.courier_states[courier_id].current_lng = lng

    def mark_rejection(self, courier_id: int, waybill_id: int):
        """Mark that a courier rejected/pruned an order."""
        if courier_id in self.courier_states:
            self.courier_states[courier_id].rejected_waybills.add(waybill_id)

    def set_availability(self, courier_id: int, available: bool):
        """Set courier availability (e.g., busy with current batch)."""
        if courier_id in self.courier_states:
            self.courier_states[courier_id].available = available

    def calculate_courier_kpis(self) -> Dict:
        """
        Calculate all courier-related KPIs from courier state attributes.
        This is the single source of truth for courier metrics.
        """
        couriers = list(self.courier_states.values())
        total_couriers = len(couriers)

        if total_couriers == 0:
            return {
                "total_couriers": 0,
                "total_orders_delivered": 0,
                "total_waybills_offered": 0,
                "total_waybills_accepted": 0,
                "total_waybills_rejected": 0,
                "total_active_time_hours": 0.0,
                "total_block_time_hours": 0.0,
                "overall_utilization": 0.0,
                "avg_utilization": 0.0,
                "min_utilization": 0.0,
                "max_utilization": 0.0,
                "avg_capacity": 0.0,
                "max_capacity": 0,
                "total_late_deliveries": 0,
                "total_on_time_deliveries": 0,
                "overall_on_time_rate": 0.0,
                "underutilized_couriers_count": 0,
            }

        total_orders_delivered = sum(c.orders_delivered for c in couriers)
        total_waybills_offered = sum(c.waybills_offered for c in couriers)
        total_waybills_accepted = sum(c.waybills_accepted for c in couriers)
        total_waybills_rejected = sum(c.waybills_rejected for c in couriers)
        total_active_time = sum(c.total_active_time for c in couriers)
        total_block_time = sum(c.total_block_time for c in couriers)
        total_late = sum(c.late_deliveries for c in couriers)
        total_on_time = sum(c.on_time_deliveries for c in couriers)

        # Utilization calculations
        utilizations = [c.utilization_rate for c in couriers if c.total_block_time > 0]
        avg_utilization = sum(utilizations) / len(utilizations) if utilizations else 0.0
        min_utilization = min(utilizations) if utilizations else 0.0
        max_utilization = max(utilizations) if utilizations else 0.0
        overall_utilization = (
            total_active_time / total_block_time if total_block_time > 0 else 0.0
        )

        # Capacity calculations
        capacities = [c.avg_capacity for c in couriers if c.capacity_observations > 0]
        avg_capacity = sum(capacities) / len(capacities) if capacities else 0.0
        max_capacity = max(c.max_concurrent_orders for c in couriers) if couriers else 0

        # Underutilized couriers
        underutilized = [
            c for c in couriers if c.total_block_time > 0 and c.utilization_rate < 0.5
        ]

        return {
            "total_couriers": total_couriers,
            "total_orders_delivered": total_orders_delivered,
            "total_waybills_offered": total_waybills_offered,
            "total_waybills_accepted": total_waybills_accepted,
            "total_waybills_rejected": total_waybills_rejected,
            "total_active_time_hours": total_active_time / 3600,
            "total_block_time_hours": total_block_time / 3600,
            "total_idle_time_hours": max(0, total_block_time - total_active_time)
            / 3600,
            "overall_utilization": min(1.0, overall_utilization),
            "avg_utilization": avg_utilization,
            "min_utilization": min_utilization,
            "max_utilization": max_utilization,
            "avg_capacity": avg_capacity,
            "max_capacity": max_capacity,
            "total_late_deliveries": total_late,
            "total_on_time_deliveries": total_on_time,
            "overall_on_time_rate": total_on_time / total_orders_delivered
            if total_orders_delivered > 0
            else 0.0,
            "underutilized_couriers_count": len(underutilized),
            "underutilized_couriers": [
                {
                    "courier_id": c.courier_id,
                    "utilization": c.utilization_rate,
                    "active_hours": c.total_active_time / 3600,
                    "total_hours": c.total_block_time / 3600,
                }
                for c in sorted(underutilized, key=lambda x: x.utilization_rate)[:10]
            ],
        }

    def get_all_courier_details(self) -> List[Dict]:
        """Get details for all couriers."""
        return [c.to_dict() for c in self.courier_states.values()]
