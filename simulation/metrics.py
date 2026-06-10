"""
Metrics Logger: Track divergences, income, late deliveries, acceptance rates.
"""
from dataclasses import dataclass, field
from typing import List, Dict
import json


@dataclass
class DivergenceEvent:
    """Single agent-system divergence."""
    waybill_id: int
    courier_id_original: int
    courier_id_assigned: int  # who actually got it after re-scoring
    agent_decision: int  # 1=keep, 0=prune
    historical_decision: int
    timestamp: int
    divergence_type: str  # 'agent_prune_hist_keep', 'agent_keep_hist_prune'


@dataclass
class IncomeEvent:
    """Income delta per order."""
    waybill_id: int
    courier_id: int
    historical_income: float
    simulated_income: float
    delta: float
    timestamp: int


@dataclass
class SystemMetrics:
    """
    System-wide metrics for the simulation run.
    In baseline mode: reflects historical system performance.
    In agent mode: reflects agent-driven system performance.
    """
    total_orders: int = 0
    total_accepts: int = 0
    total_rejects: int = 0
    total_late_deliveries: int = 0
    total_on_time_deliveries: int = 0
    total_lost_orders: int = 0  # Orders that couldn't be assigned (agent mode only)
    courier_capacity_sum: Dict[int, int] = None
    courier_cycle_count: Dict[int, int] = None
    
    def __post_init__(self):
        if self.courier_capacity_sum is None:
            self.courier_capacity_sum = {}
        if self.courier_cycle_count is None:
            self.courier_cycle_count = {}


class MetricsLogger:
    """
    Collect simulation metrics.
    
    Tracks system-wide performance metrics that represent the ENTIRE system
    (all orders, all decisions, all outcomes) for the current simulation run.
    
    In baseline mode: metrics reflect historical system (pure replay).
    In agent mode: metrics reflect agent-driven system (with agent decisions).
    """
    
    def __init__(self):
        self.divergences: List[DivergenceEvent] = []
        self.income_events: List[IncomeEvent] = []
        self.total_events = 0
        self.aligned_events = 0
        self.system = SystemMetrics()  # The ENTIRE system's performance this run
        
    def log_aligned(self):
        """Log an aligned decision (no divergence)."""
        self.total_events += 1
        self.aligned_events += 1
    
    def log_divergence(
        self,
        waybill_id: int,
        courier_id_original: int,
        courier_id_assigned: int,
        agent_decision: int,
        historical_decision: int,
        timestamp: int
    ):
        """Log a divergence event."""
        self.total_events += 1
        
        if agent_decision == 0 and historical_decision == 1:
            div_type = 'agent_prune_hist_keep'
        elif agent_decision == 1 and historical_decision == 0:
            div_type = 'agent_keep_hist_prune'
        else:
            div_type = 'unknown'
        
        event = DivergenceEvent(
            waybill_id=waybill_id,
            courier_id_original=courier_id_original,
            courier_id_assigned=courier_id_assigned,
            agent_decision=agent_decision,
            historical_decision=historical_decision,
            timestamp=timestamp,
            divergence_type=div_type
        )
        self.divergences.append(event)
    
    def log_income(
        self,
        waybill_id: int,
        courier_id: int,
        historical_income: float,
        simulated_income: float,
        timestamp: int
    ):
        """Log income delta."""
        delta = simulated_income - historical_income
        event = IncomeEvent(
            waybill_id=waybill_id,
            courier_id=courier_id,
            historical_income=historical_income,
            simulated_income=simulated_income,
            delta=delta,
            timestamp=timestamp
        )
        self.income_events.append(event)
    
    def log_system_order(self, accepted: bool, late: bool, courier_id: int, capacity: int, lost: bool = False):
        """
        Log system-wide order metrics.
        Records the actual outcome in THIS simulation run (baseline or agent mode).
        """
        self.system.total_orders += 1
        
        if lost:
            self.system.total_lost_orders += 1
            # Lost orders count as rejections in the final tally
            self.system.total_rejects += 1
            return
        
        if accepted:
            self.system.total_accepts += 1
            if late:
                self.system.total_late_deliveries += 1
            else:
                self.system.total_on_time_deliveries += 1
        else:
            self.system.total_rejects += 1
        
        # Track capacity per courier
        if courier_id not in self.system.courier_capacity_sum:
            self.system.courier_capacity_sum[courier_id] = 0
            self.system.courier_cycle_count[courier_id] = 0
        self.system.courier_capacity_sum[courier_id] += capacity
        self.system.courier_cycle_count[courier_id] += 1
    
    def get_summary(self) -> Dict:
        """Return summary statistics for the entire system in this simulation run."""
        divergence_count = len(self.divergences)
        divergence_rate = divergence_count / self.total_events if self.total_events > 0 else 0.0
        
        # Income stats (only meaningful in agent mode when we simulate)
        income_deltas = [e.delta for e in self.income_events]
        total_income_delta = sum(income_deltas) if income_deltas else 0.0
        avg_income_delta = total_income_delta / len(income_deltas) if income_deltas else 0.0
        
        # Divergence type breakdown
        div_types = {}
        for d in self.divergences:
            div_types[d.divergence_type] = div_types.get(d.divergence_type, 0) + 1
        
        # System metrics
        acceptance_rate = self.system.total_accepts / self.system.total_orders if self.system.total_orders > 0 else 0.0
        rejection_rate = self.system.total_rejects / self.system.total_orders if self.system.total_orders > 0 else 0.0
        lost_order_rate = self.system.total_lost_orders / self.system.total_orders if self.system.total_orders > 0 else 0.0
        
        delivered_orders = self.system.total_late_deliveries + self.system.total_on_time_deliveries
        lateness_rate = self.system.total_late_deliveries / delivered_orders if delivered_orders > 0 else 0.0
        on_time_rate = self.system.total_on_time_deliveries / delivered_orders if delivered_orders > 0 else 0.0
        
        # Average capacity per courier (with safe division)
        avg_capacities = {}
        for cid in self.system.courier_capacity_sum:
            total_cap = self.system.courier_capacity_sum.get(cid, 0)
            cycle_count = self.system.courier_cycle_count.get(cid, 0)
            avg_capacities[cid] = total_cap / cycle_count if cycle_count > 0 else 0.0
        
        overall_avg_capacity = sum(avg_capacities.values()) / len(avg_capacities) if avg_capacities else 0.0
        
        result = {
            'total_events': self.total_events,
            'aligned_events': self.aligned_events,
            'divergence_count': divergence_count,
            'divergence_rate': divergence_rate,
            'divergence_types': div_types,
            'total_income_delta': total_income_delta,
            'avg_income_delta_per_order': avg_income_delta,
            'income_events_count': len(self.income_events),
            'system_metrics': {
                'total_orders': self.system.total_orders,
                'acceptance_rate': acceptance_rate,
                'rejection_rate': rejection_rate,
                'lateness_rate': lateness_rate,
                'lost_order_rate': lost_order_rate,
                'total_accepts': self.system.total_accepts,
                'total_rejects': self.system.total_rejects,
                'total_late_deliveries': self.system.total_late_deliveries,
                'total_on_time_deliveries': self.system.total_on_time_deliveries,
                'total_lost_orders': self.system.total_lost_orders,
                'overall_avg_capacity_per_courier': overall_avg_capacity,
                'courier_count': len(avg_capacities)
            }
        }
        
        return result
    
    def save_report(self, output_path: str):
        """Save full report to JSON."""
        summary = self.get_summary()
        
        # Add detailed divergence list
        summary['divergences'] = [
            {
                'waybill_id': d.waybill_id,
                'courier_original': d.courier_id_original,
                'courier_assigned': d.courier_id_assigned,
                'agent_decision': d.agent_decision,
                'historical_decision': d.historical_decision,
                'timestamp': d.timestamp,
                'type': d.divergence_type
            }
            for d in self.divergences[:100]  # first 100 for brevity
        ]
        
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Saved metrics report to {output_path}")
