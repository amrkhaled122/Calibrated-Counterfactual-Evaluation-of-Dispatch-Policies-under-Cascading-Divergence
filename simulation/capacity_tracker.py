"""
Capacity Tracker: Monitor realized capacity without enforcement.

Tracks concurrent orders per courier and logs capacity overshoot events.
"""
from typing import Dict, List
from dataclasses import dataclass, field


@dataclass
class CapacityEvent:
    """Single capacity observation."""
    courier_id: int
    timestamp: int
    realized_capacity: int
    original_capacity: int  # capacity_at_dispatch from historical data
    overshoot: int  # realized - original


class CapacityTracker:
    """
    Track realized capacity vs historical capacity_at_dispatch.
    
    Assumptions:
    - No enforcement (infinite capacity mode)
    - Log overshoot when realized > original
    - Track capacity distribution for analysis
    """
    
    def __init__(self):
        self.events: List[CapacityEvent] = []
        self.current_capacity: Dict[int, int] = {}  # courier_id -> current load
        
    def observe_capacity(
        self,
        courier_id: int,
        timestamp: int,
        realized: int,
        original: int
    ):
        """
        Log a capacity observation.
        
        Args:
            courier_id: courier ID
            timestamp: dispatch time
            realized: actual concurrent orders
            original: historical capacity_at_dispatch
        """
        overshoot = realized - original
        event = CapacityEvent(
            courier_id=courier_id,
            timestamp=timestamp,
            realized_capacity=realized,
            original_capacity=original,
            overshoot=overshoot
        )
        self.events.append(event)
        self.current_capacity[courier_id] = realized
    
    def get_stats(self) -> Dict:
        """Return summary statistics."""
        if not self.events:
            return {
                'total_observations': 0,
                'overshoot_count': 0,
                'overshoot_rate': 0.0,
                'avg_realized': 0.0,
                'avg_original': 0.0,
                'max_realized': 0,
                'max_overshoot': 0
            }
        
        overshoots = [e for e in self.events if e.overshoot > 0]
        realized = [e.realized_capacity for e in self.events]
        original = [e.original_capacity for e in self.events]
        overshoot_vals = [e.overshoot for e in self.events]
        
        return {
            'total_observations': len(self.events),
            'overshoot_count': len(overshoots),
            'overshoot_rate': len(overshoots) / len(self.events),
            'avg_realized': sum(realized) / len(realized),
            'avg_original': sum(original) / len(original),
            'max_realized': max(realized),
            'max_overshoot': max(overshoot_vals) if overshoot_vals else 0
        }
