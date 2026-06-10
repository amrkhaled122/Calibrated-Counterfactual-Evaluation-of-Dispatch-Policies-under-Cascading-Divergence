"""
Dispatcher: Chronological log replay of historical offers.

Loads offers_observations.parquet and emits events in dispatch-time order.
Each event represents an offer presented to a courier.
"""
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import polars as pl
from tqdm import tqdm


@dataclass
class OfferEvent:
    """Single offer event from historical log."""
    courier_id: int
    waybill_id: int
    order_id: int  # Added to track unique orders
    local_day: str
    actual_dispatch_time: int
    historical_decision: int  # 1=keep, 0=prune
    features: Dict[str, float]  # observation vector for agent
    order_data: Dict[str, Any]  # full row for simulation (coords, ETA, etc.)
    # Cycle information
    dispatch_cycle_id: str = ""  # String like 'day1_cycle_1'
    dispatch_start_time: int = 0
    dispatch_end_time: int = 0
    offer_index_in_cycle: int = 0


class Dispatcher:
    """
    Replays historical offers in chronological order.
    
    Assumptions:
    - offers_observations.parquet contains all needed fields
    - sorted by (courier_id, local_day, actual_dispatch_time)
    - historical_decision is 'is_assigned_courier_accepted' column
    """
    
    def __init__(self, parquet_path: str, manifest: Dict[str, Any], max_hours: Optional[int] = None,
                 start_hour: int = 0):
        """
        Args:
            parquet_path: path to offers_observations.parquet
            manifest: manifest.json dict with feature_order
            max_hours: if set, only load events within N hours from start_hour
            start_hour: skip events before this hour (for train/eval split)
        """
        self.parquet_path = parquet_path
        self.feature_order = manifest['feature_order']
        self.max_hours = max_hours
        self.start_hour = start_hour
        self.df = None
        self.events: List[OfferEvent] = []
        
    def load(self):
        """Load and prepare events."""
        print(f"Loading offers from {self.parquet_path}...")
        self.df = pl.read_parquet(self.parquet_path)
        
        # Sort chronologically
        sort_cols = [c for c in ['actual_dispatch_time', 'courier_id', 'local_day'] if c in self.df.columns]
        if sort_cols:
            self.df = self.df.sort(sort_cols)
        
        # Filter by time window if specified
        if 'actual_dispatch_time' in self.df.columns:
            min_time = self.df['actual_dispatch_time'].min()
            
            # Apply start_hour offset
            if self.start_hour > 0:
                start_time = min_time + (self.start_hour * 3600)
                self.df = self.df.filter(pl.col('actual_dispatch_time') >= start_time)
                print(f"Skipped first {self.start_hour} hours, starting from hour {self.start_hour}")
                min_time = start_time  # Update min_time for max_hours calculation
            
            # Apply max_hours limit
            if self.max_hours is not None:
                max_time = min_time + (self.max_hours * 3600)
                self.df = self.df.filter(pl.col('actual_dispatch_time') <= max_time)
                print(f"Filtered to {self.max_hours} hours (hours {self.start_hour}-{self.start_hour + self.max_hours}): {self.df.height} rows")
        
        # Build event list
        self.events = []
        for row in tqdm(self.df.iter_rows(named=True), total=self.df.height, desc="Preparing events"):
            # Extract features for agent
            features = {}
            for f in self.feature_order:
                val = row.get(f)
                features[f] = float(val) if val is not None else 0.0
            # Clamp negative ETAs (can occur from data issues)
            if 'eta_seconds_current' in features and features['eta_seconds_current'] < 0:
                features['eta_seconds_current'] = 0.0
            
            # Filter out prebooked orders
            if int(row.get('is_prebook', 0)) == 1:
                continue

            # Historical decision
            hist_dec = int(row.get('is_assigned_courier_accepted', 0))
            
            event = OfferEvent(
                courier_id=int(row.get('courier_id', -1)),
                waybill_id=int(row.get('waybill_id', -1)),
                order_id=int(row.get('order_id', -1)),  # Read order_id
                local_day=str(row.get('local_day', '')),
                actual_dispatch_time=int(row.get('actual_dispatch_time', 0)),
                historical_decision=hist_dec,
                features=features,
                order_data=dict(row),  # full row for simulation access
                # Cycle information from parquet (dispatch_cycle_id is a string like 'day1_cycle_1')
                dispatch_cycle_id=str(row.get('dispatch_cycle_id', '')),
                dispatch_start_time=int(row.get('dispatch_start_time', 0)),
                dispatch_end_time=int(row.get('dispatch_end_time', 0)),
                offer_index_in_cycle=int(row.get('offer_index_in_cycle', 0))
            )
            self.events.append(event)
        
        # Calculate unique counts
        unique_couriers = len(set(e.courier_id for e in self.events))
        unique_orders = len(set(e.order_id for e in self.events))
        
        print(f"Loaded {len(self.events)} offer events")
        print(f"  Unique couriers: {unique_couriers}")
        print(f"  Unique orders: {unique_orders}")
        
    def get_events(self) -> List[OfferEvent]:
        """Return all events in chronological order."""
        return self.events
    
    def get_stats(self) -> dict:
        """Return statistics about loaded events."""
        if not self.events:
            return {'total_events': 0, 'unique_couriers': 0, 'unique_orders': 0}
        
        return {
            'total_events': len(self.events),
            'unique_couriers': len(set(e.courier_id for e in self.events)),
            'unique_orders': len(set(e.order_id for e in self.events))
        }
