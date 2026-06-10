"""
Router: Greedy location-based routing without capacity constraints.

Computes delivery sequences for courier batches using simple nearest-neighbor heuristics.
"""
from typing import List, Dict, Tuple
import numpy as np
from .scorer import haversine_km


class Order:
    """Single order in a batch."""
    def __init__(
        self,
        waybill_id: int,
        sender_lat: float,
        sender_lng: float,
        recipient_lat: float,
        recipient_lng: float,
        eta_seconds: int,
        dispatch_time: int
    ):
        self.waybill_id = waybill_id
        self.sender_lat = sender_lat
        self.sender_lng = sender_lng
        self.recipient_lat = recipient_lat
        self.recipient_lng = recipient_lng
        self.eta_seconds = eta_seconds
        self.dispatch_time = dispatch_time
        self.picked_up = False
        self.delivered = False


class Router:
    """
    Greedy location-based routing.
    
    Assumptions:
    - Infinite capacity (no hard limit on concurrent orders)
    - Pickup sequence: nearest restaurant from courier's current location
    - Delivery sequence: nearest customer among picked-up orders
    - No travel time modeling (use straight-line distance as proxy)
    - Realized capacity = max concurrent orders in a batch
    """
    
    def __init__(self):
        pass
    
    def route_batch(
        self,
        courier_start_lat: float,
        courier_start_lng: float,
        orders: List[Order]
    ) -> Dict[str, any]:
        """
        Compute greedy pickup/delivery sequence.
        
        Args:
            courier_start_lat, courier_start_lng: courier's starting position
            orders: list of Order objects
        
        Returns:
            dict with:
              - pickup_sequence: List[waybill_id]
              - delivery_sequence: List[waybill_id]
              - realized_capacity: int (max concurrent orders)
              - total_distance_km: float
              - estimated_completion_time: int (dispatch_time + travel estimate)
        """
        if not orders:
            return {
                'pickup_sequence': [],
                'delivery_sequence': [],
                'realized_capacity': 0,
                'total_distance_km': 0.0,
                'estimated_completion_time': 0
            }
        
        # Greedy pickup: nearest restaurant first
        pickup_seq = []
        current_lat, current_lng = courier_start_lat, courier_start_lng
        remaining = [o for o in orders]
        total_dist = 0.0
        
        while remaining:
            # Find nearest restaurant
            nearest = min(remaining, key=lambda o: haversine_km(current_lat, current_lng, o.sender_lat, o.sender_lng))
            dist = haversine_km(current_lat, current_lng, nearest.sender_lat, nearest.sender_lng)
            total_dist += dist
            pickup_seq.append(nearest.waybill_id)
            nearest.picked_up = True
            current_lat, current_lng = nearest.sender_lat, nearest.sender_lng
            remaining.remove(nearest)
        
        # Greedy delivery: nearest customer among picked-up orders
        delivery_seq = []
        picked_up_orders = [o for o in orders if o.picked_up]
        max_concurrent = len(picked_up_orders)
        
        while picked_up_orders:
            # Find nearest customer
            nearest = min(picked_up_orders, key=lambda o: haversine_km(current_lat, current_lng, o.recipient_lat, o.recipient_lng))
            dist = haversine_km(current_lat, current_lng, nearest.recipient_lat, nearest.recipient_lng)
            total_dist += dist
            delivery_seq.append(nearest.waybill_id)
            nearest.delivered = True
            current_lat, current_lng = nearest.recipient_lat, nearest.recipient_lng
            picked_up_orders.remove(nearest)
        
        # Use actual ETA from order data instead of approximation
        # eta_seconds now contains real historical ETA (arrive_time - grab_time)
        first_dispatch = min(o.dispatch_time for o in orders)
        # For single order, use its actual ETA; for batch, use max ETA
        max_eta = max(o.eta_seconds for o in orders)
        completion_time = first_dispatch + max_eta
        
        return {
            'pickup_sequence': pickup_seq,
            'delivery_sequence': delivery_seq,
            'realized_capacity': len(orders),  # all orders in batch
            'total_distance_km': total_dist,
            'estimated_completion_time': completion_time
        }
