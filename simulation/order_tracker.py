"""
Order State Tracker: Manage orders across assignment attempts and delivery.

Tracks:
- Assignment attempts and rejections
- Current courier assignment
- Delivery status and location
- Routing simulation state
- All KPIs are calculated from order attributes at the end
"""
from dataclasses import dataclass, field
from typing import Dict, Optional, Set, List


@dataclass
class OrderState:
    """State of a single order throughout its lifecycle."""
    waybill_id: int
    order_id: int  # Added unique order ID
    original_courier_id: int
    current_courier_id: Optional[int] = None
    assignment_attempts: int = 0
    rejecting_couriers: Set[int] = field(default_factory=set)
    
    # Delivery tracking - is_delivered is the primary flag
    is_delivered: bool = False
    delivery_time: Optional[int] = None
    delivery_lat: Optional[float] = None
    delivery_lng: Optional[float] = None
    
    # Time tracking for utilization
    grab_time: Optional[int] = None
    fetch_time: Optional[int] = None
    arrive_time: Optional[int] = None
    is_simulated_delivery: bool = False  # True if timestamps are simulated (due to divergence)
    
    # Order details
    sender_lat: float = 0.0
    sender_lng: float = 0.0
    recipient_lat: float = 0.0
    recipient_lng: float = 0.0
    estimate_arrived_time: Optional[int] = None
    dispatch_time: int = 0
    
    # Decision tracking
    is_accepted: bool = False
    is_rejected: bool = False
    is_lost: bool = False  # Could not be assigned after all attempts
    
    # Historical tracking
    historical_decision: Optional[int] = None  # 1=accept, 0=reject
    agent_decision: Optional[int] = None  # 1=keep, 0=prune
    is_divergence: bool = False
    divergence_type: Optional[str] = None
    
    # Income tracking
    historical_income: float = 0.0
    simulated_income: float = 0.0
    
    # Capacity tracking
    capacity_at_dispatch: int = 0
    realized_capacity: int = 0
    
    # Forced acceptance flag
    force_accept: bool = False
    
    @property
    def delivered(self) -> bool:
        """Alias for is_delivered for backward compatibility."""
        return self.is_delivered
    
    @property
    def income_delta(self) -> float:
        """Calculate income difference from historical."""
        return self.simulated_income - self.historical_income
    
    @property
    def delivery_duration(self) -> Optional[int]:
        """Calculate grab_time to arrive_time duration."""
        if self.grab_time and self.arrive_time:
            return self.arrive_time - self.grab_time
        return None
    
    def add_rejection(self, courier_id: int):
        """Record a rejection by a courier."""
        self.rejecting_couriers.add(courier_id)
        self.assignment_attempts += 1
        
        # After 5 rejections, force next accept
        if self.assignment_attempts >= 5:
            self.force_accept = True
    
    def assign_to_courier(self, courier_id: int):
        """Assign order to a courier."""
        self.current_courier_id = courier_id
        self.assignment_attempts += 1
    
    def mark_accepted(self, courier_id: int):
        """Mark order as accepted by a courier."""
        self.is_accepted = True
        self.is_rejected = False
        self.current_courier_id = courier_id
    
    def mark_rejected(self):
        """Mark order as rejected."""
        self.is_rejected = True
        self.is_accepted = False
    
    def mark_lost(self):
        """Mark order as lost (could not be assigned)."""
        self.is_lost = True
        self.is_rejected = True
        self.is_accepted = False
    
    def mark_delivered(self, actual_time: int, lat: float, lng: float):
        """Mark order as delivered."""
        self.is_delivered = True
        self.delivery_time = actual_time
        self.delivery_lat = lat
        self.delivery_lng = lng
    
    def set_times(self, grab_time: Optional[int], fetch_time: Optional[int], arrive_time: Optional[int]):
        """Set time tracking fields."""
        self.grab_time = grab_time
        self.fetch_time = fetch_time
        self.arrive_time = arrive_time
    
    def set_divergence(self, agent_decision: int, historical_decision: int):
        """Record divergence information."""
        self.agent_decision = agent_decision
        self.historical_decision = historical_decision
        self.is_divergence = agent_decision != historical_decision
        
        if self.is_divergence:
            if agent_decision == 0 and historical_decision == 1:
                self.divergence_type = 'agent_prune_hist_keep'
            elif agent_decision == 1 and historical_decision == 0:
                self.divergence_type = 'agent_keep_hist_prune'
            else:
                self.divergence_type = 'unknown'
    
    def is_late(self) -> bool:
        """Check if order was delivered late."""
        if not self.is_delivered or self.delivery_time is None or self.estimate_arrived_time is None:
            return False
        return self.delivery_time > self.estimate_arrived_time
    
    def is_on_time(self) -> bool:
        """Check if order was delivered on time."""
        return self.is_delivered and not self.is_late()
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'waybill_id': self.waybill_id,
            'order_id': self.order_id,
            'original_courier_id': self.original_courier_id,
            'current_courier_id': self.current_courier_id,
            'is_delivered': self.is_delivered,
            'is_accepted': self.is_accepted,
            'is_rejected': self.is_rejected,
            'is_lost': self.is_lost,
            'is_late': self.is_late(),
            'is_on_time': self.is_on_time(),
            'is_divergence': self.is_divergence,
            'is_simulated_delivery': self.is_simulated_delivery,
            'divergence_type': self.divergence_type,
            'dispatch_time': self.dispatch_time,
            'grab_time': self.grab_time,
            'fetch_time': self.fetch_time,
            'arrive_time': self.arrive_time,
            'delivery_duration': self.delivery_duration,
            'income_delta': self.income_delta,
            'assignment_attempts': self.assignment_attempts
        }


class OrderTracker:
    """
    Track all orders throughout simulation.
    
    Manages:
    - Active orders (assigned but not delivered)
    - Completed orders (delivered or failed)
    - Assignment attempts and rejections
    
    All KPIs are calculated at the end from order attributes.
    
    Terminology:
    - waybill_id: Individual dispatch event/offer (can have multiple per order)
    - order_id: Unique order identifier (one per actual delivery)
    - An order can have multiple waybills if it's rejected and re-offered
    """
    
    def __init__(self):
        self.orders: Dict[int, OrderState] = {}  # waybill_id -> OrderState
        self.active_orders: Set[int] = set()  # waybill_ids
        self.completed_orders: Set[int] = set()  # waybill_ids
        self.completed_order_ids: Set[int] = set()  # Track unique order_ids completed
        self.delivered_order_ids: Set[int] = set()  # Track unique order_ids delivered
        self.accepted_order_ids: Set[int] = set()  # Track unique order_ids accepted
        self.rejected_order_ids: Set[int] = set()  # Track unique order_ids with at least one rejection
        self.fetched_order_ids: Set[int] = set()  # Track unique order_ids that were fetched (courier picked up food)
        self.not_assigned_order_ids: Set[int] = set()  # Track orders that exhausted all reassignment attempts
    
    def create_order(
        self,
        waybill_id: int,
        order_id: int,
        original_courier_id: int,
        sender_lat: float,
        sender_lng: float,
        recipient_lat: float,
        recipient_lng: float,
        estimate_arrived_time: Optional[int],
        dispatch_time: int
    ) -> OrderState:
        """Create and register a new order."""
        order = OrderState(
            waybill_id=waybill_id,
            order_id=order_id,
            original_courier_id=original_courier_id,
            sender_lat=sender_lat,
            sender_lng=sender_lng,
            recipient_lat=recipient_lat,
            recipient_lng=recipient_lng,
            estimate_arrived_time=estimate_arrived_time,
            dispatch_time=dispatch_time
        )
        self.orders[waybill_id] = order
        self.active_orders.add(waybill_id)
        return order
    
    def get_order(self, waybill_id: int) -> Optional[OrderState]:
        """Retrieve an order by ID."""
        return self.orders.get(waybill_id)
    
    def mark_delivered(self, waybill_id: int, actual_time: int, lat: float, lng: float):
        """Mark an order as delivered and move to completed.
        
        Note: An order cannot be delivered without being fetched first,
        so this also marks the order as fetched.
        """
        if waybill_id in self.orders:
            order = self.orders[waybill_id]
            order.mark_delivered(actual_time, lat, lng)
            if waybill_id in self.active_orders:
                self.active_orders.remove(waybill_id)
            self.completed_orders.add(waybill_id)
            self.completed_order_ids.add(order.order_id)  # Track completion by order_id
            self.delivered_order_ids.add(order.order_id)  # Track delivery by order_id
            # An order must be fetched before it can be delivered
            self.fetched_order_ids.add(order.order_id)
    
    def mark_accepted(self, waybill_id: int):
        """Mark that an order (by waybill) was accepted - tracks unique order_id."""
        if waybill_id in self.orders:
            order = self.orders[waybill_id]
            self.accepted_order_ids.add(order.order_id)
    
    def mark_rejected(self, waybill_id: int):
        """Mark that an order (by waybill) had a rejection - tracks unique order_id."""
        if waybill_id in self.orders:
            order = self.orders[waybill_id]
            self.rejected_order_ids.add(order.order_id)
    
    def mark_fetched(self, waybill_id: int):
        """Mark that an order was fetched (courier picked up food from restaurant)."""
        if waybill_id in self.orders:
            order = self.orders[waybill_id]
            self.fetched_order_ids.add(order.order_id)
    
    def mark_not_assigned(self, waybill_id: int):
        """
        Mark that an order exhausted all reassignment attempts and could not be assigned.
        These orders are LOST - they will never be delivered.
        """
        if waybill_id in self.orders:
            order = self.orders[waybill_id]
            order.mark_lost()  # Also mark the order object as lost
            self.not_assigned_order_ids.add(order.order_id)
            # Move from active to completed (as lost)
            if waybill_id in self.active_orders:
                self.active_orders.remove(waybill_id)
            self.completed_orders.add(waybill_id)
    
    def is_active(self, waybill_id: int) -> bool:
        """Check if order is still active (not delivered)."""
        return waybill_id in self.active_orders
    
    def get_active_orders_for_courier(self, courier_id: int) -> list:
        """Get all active orders assigned to a courier."""
        return [
            self.orders[wid]
            for wid in self.active_orders
            if self.orders[wid].current_courier_id == courier_id
        ]
    
    # ==================== KPI Calculation Methods ====================
    
    def get_delivered_orders(self) -> List[OrderState]:
        """Get all delivered orders."""
        return [o for o in self.orders.values() if o.is_delivered]
    
    def get_accepted_orders(self) -> List[OrderState]:
        """Get all accepted orders."""
        return [o for o in self.orders.values() if o.is_accepted]
    
    def get_rejected_orders(self) -> List[OrderState]:
        """Get all rejected orders."""
        return [o for o in self.orders.values() if o.is_rejected]
    
    def get_lost_orders(self) -> List[OrderState]:
        """Get all lost orders (could not be assigned)."""
        return [o for o in self.orders.values() if o.is_lost]
    
    def get_late_orders(self) -> List[OrderState]:
        """Get all late deliveries."""
        return [o for o in self.orders.values() if o.is_delivered and o.is_late()]
    
    def get_on_time_orders(self) -> List[OrderState]:
        """Get all on-time deliveries."""
        return [o for o in self.orders.values() if o.is_delivered and o.is_on_time()]
    
    def get_divergence_orders(self) -> List[OrderState]:
        """Get all orders with divergences."""
        return [o for o in self.orders.values() if o.is_divergence]
    
    def get_fetched_orders(self) -> List[OrderState]:
        """Get all orders that were fetched (courier picked up food)."""
        return [o for o in self.orders.values() if o.fetch_time is not None and o.fetch_time > 0]
    
    def get_not_assigned_orders(self) -> List[OrderState]:
        """Get all orders that exhausted reassignment attempts (lost orders)."""
        return [o for o in self.orders.values() if o.is_lost]
    
    def calculate_kpis(self) -> Dict:
        """
        Calculate all order-related KPIs from order attributes.
        This is the single source of truth for order metrics.
        
        Distinction:
        - total_waybills: Number of dispatch events processed
        - total_unique_orders: Number of unique order_ids
        - waybills_rejected: Sum of all rejection events (can be >1 per order)
        - orders_rejected: Unique orders that had at least one rejection
        """
        all_waybills = list(self.orders.values())
        total_waybills = len(all_waybills)
        
        # Get unique order_ids
        unique_order_ids = set(o.order_id for o in all_waybills)
        total_unique_orders = len(unique_order_ids)
        
        if total_waybills == 0:
            return {
                'total_waybills': 0,
                'total_unique_orders': 0,
                'delivered_count': 0,
                'fetched_count': 0,
                'accepted_count': 0,
                'orders_rejected_count': 0,
                'waybills_rejected_count': 0,
                'not_assigned_count': 0,
                'simulated_delivery_count': 0,
                'lost_count': 0,
                'late_count': 0,
                'on_time_count': 0,
                'divergence_count': 0,
                'delivery_rate': 0.0,
                'fetched_rate': 0.0,
                'acceptance_rate': 0.0,
                'orders_rejection_rate': 0.0,
                'waybills_rejection_rate': 0.0,
                'not_assigned_rate': 0.0,
                'simulated_delivery_rate': 0.0,
                'lost_rate': 0.0,
                'lateness_rate': 0.0,
                'on_time_rate': 0.0,
                'divergence_rate': 0.0,
                'total_income_delta': 0.0,
                'avg_income_delta': 0.0,
                'avg_delivery_duration_sec': 0.0,
                'divergence_types': {}
            }
        
        delivered = self.get_delivered_orders()
        accepted = self.get_accepted_orders()
        rejected_waybills = self.get_rejected_orders()  # waybills with is_rejected=True
        lost = self.get_lost_orders()
        late = self.get_late_orders()
        on_time = self.get_on_time_orders()
        divergences = self.get_divergence_orders()
        
        # Count simulated deliveries (orders with is_simulated_delivery=True)
        simulated_deliveries = [o for o in self.orders.values() if o.is_simulated_delivery]
        simulated_delivery_count = len(simulated_deliveries)
        
        # Unique order_id counts
        delivered_order_ids = len(self.delivered_order_ids)
        accepted_order_ids = len(self.accepted_order_ids)
        rejected_order_ids = len(self.rejected_order_ids)  # unique orders with at least 1 rejection
        fetched_order_ids = len(self.fetched_order_ids)  # unique orders that were fetched
        not_assigned_order_ids = len(self.not_assigned_order_ids)  # unique orders never assigned
        
        # Waybill counts
        waybills_rejected = len(rejected_waybills)
        waybills_accepted = len(accepted)
        
        # Count total rejection events (sum of all rejections across all orders)
        total_rejection_events = sum(len(o.rejecting_couriers) for o in all_waybills)
        
        lost_count = len(lost)
        late_count = len(late)
        on_time_count = len(on_time)
        divergence_count = len(divergences)
        
        # Divergence types breakdown
        div_types = {}
        for o in divergences:
            dt = o.divergence_type or 'unknown'
            div_types[dt] = div_types.get(dt, 0) + 1
        
        # Income calculations
        total_income_delta = sum(o.income_delta for o in all_waybills)
        avg_income_delta = total_income_delta / total_waybills if total_waybills > 0 else 0.0
        
        # Delivery duration calculations
        durations = [o.delivery_duration for o in delivered if o.delivery_duration is not None]
        avg_duration = sum(durations) / len(durations) if durations else 0.0
        
        return {
            # Counts
            'total_waybills': total_waybills,
            'total_unique_orders': total_unique_orders,
            'delivered_count': delivered_order_ids,  # Unique orders delivered
            'fetched_count': fetched_order_ids,  # Unique orders fetched (food picked up)
            'accepted_count': accepted_order_ids,  # Unique orders accepted
            'orders_rejected_count': rejected_order_ids,  # Unique orders with at least 1 rejection
            'waybills_rejected_count': waybills_rejected,  # Total waybills marked rejected
            'total_rejection_events': total_rejection_events,  # Sum of all courier rejections
            'not_assigned_count': not_assigned_order_ids,  # Orders that exhausted all reassignment attempts
            'simulated_delivery_count': simulated_delivery_count,  # Orders with simulated timestamps (divergence)
            'lost_count': lost_count,
            'late_count': late_count,
            'on_time_count': on_time_count,
            'divergence_count': divergence_count,
            # Rates (based on unique orders)
            'delivery_rate': delivered_order_ids / total_unique_orders if total_unique_orders > 0 else 0.0,
            'fetched_rate': fetched_order_ids / total_unique_orders if total_unique_orders > 0 else 0.0,
            'acceptance_rate': accepted_order_ids / total_unique_orders if total_unique_orders > 0 else 0.0,
            'orders_rejection_rate': rejected_order_ids / total_unique_orders if total_unique_orders > 0 else 0.0,
            'waybills_rejection_rate': waybills_rejected / total_waybills if total_waybills > 0 else 0.0,
            'not_assigned_rate': not_assigned_order_ids / total_unique_orders if total_unique_orders > 0 else 0.0,
            'simulated_delivery_rate': simulated_delivery_count / delivered_order_ids if delivered_order_ids > 0 else 0.0,
            'lost_rate': lost_count / total_unique_orders if total_unique_orders > 0 else 0.0,
            'lateness_rate': late_count / delivered_order_ids if delivered_order_ids > 0 else 0.0,
            'on_time_rate': on_time_count / delivered_order_ids if delivered_order_ids > 0 else 0.0,
            'divergence_rate': divergence_count / total_waybills if total_waybills > 0 else 0.0,
            # Income
            'total_income_delta': total_income_delta,
            'avg_income_delta': avg_income_delta,
            'avg_delivery_duration_sec': avg_duration,
            'divergence_types': div_types
        }
    
    def get_summary(self) -> Dict:
        """Get comprehensive summary including KPIs and sample orders."""
        kpis = self.calculate_kpis()
        
        # Add unique order IDs tracking
        kpis['unique_waybills'] = len(self.orders)
        kpis['unique_order_ids'] = len(set(o.order_id for o in self.orders.values()))
        kpis['completed_waybills'] = len(self.completed_orders)
        kpis['completed_order_ids'] = len(self.completed_order_ids)
        
        return kpis
    
    def get_orders_for_courier(self, courier_id: int) -> List[OrderState]:
        """Get all orders (current or original) for a courier."""
        return [
            o for o in self.orders.values()
            if o.current_courier_id == courier_id or o.original_courier_id == courier_id
        ]
