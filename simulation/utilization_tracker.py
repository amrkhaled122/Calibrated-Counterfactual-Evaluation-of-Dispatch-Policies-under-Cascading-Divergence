"""
Courier Utilization Tracker: Track idle vs active time for each courier block.

Calculates utilization rate = active_time / total_block_time
where active_time is the time spent actively working (from grab_time to arrive_time).

Timeline for each order:
- actual_dispatch_time: When platform dispatches/offers the order
- grab_time: When courier ACCEPTS the order (activity starts)
- fetch_time: When courier arrives at sender and picks up
- arrive_time: When courier delivers to recipient (activity ends)

Key insight: A courier can deliver multiple orders simultaneously (batch delivery),
so we need to track actual time spans, not sum of individual delivery durations.
Active time = grab_time → arrive_time (includes going to sender + delivery)
"""
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import json


@dataclass 
class DeliverySpan:
    """A time span during which the courier is actively delivering."""
    start_time: int
    end_time: int
    waybill_ids: List[int] = field(default_factory=list)
    
    @property
    def duration(self) -> int:
        return max(0, self.end_time - self.start_time)


@dataclass
class CourierBlockUtilization:
    """Utilization tracking for a single courier block."""
    courier_id: int
    block_start: int
    block_end: int
    # Time tracking
    total_block_time: int = 0  # block_end - block_start
    # Order tracking
    orders_delivered: int = 0
    order_details: List[Dict] = field(default_factory=list)
    # Delivery spans (merged overlapping periods)
    delivery_spans: List[DeliverySpan] = field(default_factory=list)
    
    def __post_init__(self):
        self.total_block_time = self.block_end - self.block_start
    
    @property
    def active_time(self) -> int:
        """Calculate active time from merged delivery spans."""
        return sum(span.duration for span in self.delivery_spans)
    
    @property
    def idle_time(self) -> int:
        """Idle time = block time - active time (clamped to 0)."""
        return max(0, self.total_block_time - self.active_time)
    
    @property
    def utilization_rate(self) -> float:
        if self.total_block_time == 0:
            return 0.0
        return min(1.0, self.active_time / self.total_block_time)  # Cap at 100%
    
    def add_order(self, waybill_id: int, grab_time: int, arrive_time: int):
        """
        Record an order delivery within this block.
        
        Args:
            waybill_id: Order identifier
            grab_time: When courier accepted the order (activity starts)
            arrive_time: When courier delivered to recipient (activity ends)
        
        Merges overlapping delivery periods to avoid double-counting batch deliveries.
        """
        self.orders_delivered += 1
        self.order_details.append({
            'waybill_id': waybill_id,
            'grab_time': grab_time,
            'arrive_time': arrive_time,
            'duration': arrive_time - grab_time
        })
        
        # Clamp delivery times to within the block
        clamped_start = max(grab_time, self.block_start)
        clamped_end = min(arrive_time, self.block_end)
        
        if clamped_end <= clamped_start:
            # Delivery entirely outside block
            return
        
        # Add new span and merge overlapping spans
        new_span = DeliverySpan(
            start_time=clamped_start,
            end_time=clamped_end,
            waybill_ids=[waybill_id]
        )
        self._merge_span(new_span)
    
    def _merge_span(self, new_span: DeliverySpan):
        """Merge a new delivery span with existing spans to avoid overlap."""
        if not self.delivery_spans:
            self.delivery_spans.append(new_span)
            return
        
        # Find all spans that overlap with the new one
        overlapping = []
        non_overlapping = []
        
        for span in self.delivery_spans:
            # Two spans overlap if one starts before the other ends
            if span.start_time <= new_span.end_time and new_span.start_time <= span.end_time:
                overlapping.append(span)
            else:
                non_overlapping.append(span)
        
        if not overlapping:
            # No overlap, just add the new span
            self.delivery_spans.append(new_span)
        else:
            # Merge all overlapping spans into one
            all_starts = [new_span.start_time] + [s.start_time for s in overlapping]
            all_ends = [new_span.end_time] + [s.end_time for s in overlapping]
            all_waybills = new_span.waybill_ids.copy()
            for s in overlapping:
                all_waybills.extend(s.waybill_ids)
            
            merged = DeliverySpan(
                start_time=min(all_starts),
                end_time=max(all_ends),
                waybill_ids=all_waybills
            )
            self.delivery_spans = non_overlapping + [merged]
        
        # Sort spans by start time for easier debugging
        self.delivery_spans.sort(key=lambda s: s.start_time)


class UtilizationTracker:
    """
    Track courier utilization across all blocks.
    
    For each courier, tracks utilization within their working blocks.
    Active time is calculated by merging overlapping delivery spans.
    """
    
    def __init__(self):
        # courier_id -> list of CourierBlockUtilization
        self.courier_blocks: Dict[int, List[CourierBlockUtilization]] = {}
        # Track unique waybills globally to avoid double-counting orders spanning multiple blocks
        self.unique_waybills: set = set()
    
    def initialize_courier_blocks(self, courier_id: int, blocks: List[Tuple[int, int]]):
        """Initialize utilization tracking for a courier's blocks."""
        if courier_id not in self.courier_blocks:
            self.courier_blocks[courier_id] = []
        
        for start, end in blocks:
            block_util = CourierBlockUtilization(
                courier_id=courier_id,
                block_start=start,
                block_end=end
            )
            self.courier_blocks[courier_id].append(block_util)
    
    def record_delivery(self, courier_id: int, waybill_id: int, grab_time: int, arrive_time: int):
        """
        Record a delivery for utilization tracking.
        
        Args:
            courier_id: Courier performing the delivery
            waybill_id: Order identifier
            grab_time: When courier accepted the order (activity starts)
            arrive_time: When courier delivered to recipient (activity ends)
        
        IMPORTANT: A single delivery can span MULTIPLE blocks when the courier
        is working continuously. We add the delivery to ALL blocks it overlaps with,
        and the clamping in add_order() will handle attributing the correct time
        portion to each block.
        
        Example: If grab_time is in Block 2 but arrive_time is in Block 3,
        both Block 2 and Block 3 get credited for their portion of the delivery.
        """
        if courier_id not in self.courier_blocks:
            return
        
        # Track unique waybills globally
        self.unique_waybills.add(waybill_id)
        
        # Find ALL blocks that overlap with this delivery and add to each
        for block in self.courier_blocks[courier_id]:
            # Delivery overlaps with block if:
            # grab_time < block_end AND arrive_time > block_start
            if grab_time < block.block_end and arrive_time > block.block_start:
                block.add_order(waybill_id, grab_time, arrive_time)
    
    def get_courier_stats(self, courier_id: int) -> Dict:
        """Get utilization stats for a specific courier."""
        if courier_id not in self.courier_blocks:
            return {}
        
        blocks = self.courier_blocks[courier_id]
        total_block_time = sum(b.total_block_time for b in blocks)
        total_active_time = sum(b.active_time for b in blocks)
        total_orders = sum(b.orders_delivered for b in blocks)
        
        return {
            'courier_id': courier_id,
            'num_blocks': len(blocks),
            'total_block_time_sec': total_block_time,
            'total_active_time_sec': total_active_time,
            'total_idle_time_sec': max(0, total_block_time - total_active_time),
            'total_orders': total_orders,
            'overall_utilization': min(1.0, total_active_time / total_block_time) if total_block_time > 0 else 0,
            'blocks': [
                {
                    'block_start': b.block_start,
                    'block_end': b.block_end,
                    'total_time_sec': b.total_block_time,
                    'active_time_sec': b.active_time,
                    'idle_time_sec': b.idle_time,
                    'utilization': b.utilization_rate,
                    'orders_delivered': b.orders_delivered,
                    'delivery_spans': len(b.delivery_spans)
                }
                for b in blocks
            ]
        }
    
    def get_summary(self) -> Dict:
        """Get overall utilization summary across all couriers."""
        if not self.courier_blocks:
            return {
                'total_couriers': 0,
                'total_blocks': 0,
                'avg_utilization': 0,
                'total_block_time_hours': 0,
                'total_active_time_hours': 0,
                'total_idle_time_hours': 0
            }
        
        all_utilizations = []
        total_block_time = 0
        total_active_time = 0
        total_blocks = 0
        total_orders = 0
        
        for courier_id, blocks in self.courier_blocks.items():
            for block in blocks:
                total_blocks += 1
                total_block_time += block.total_block_time
                total_active_time += block.active_time
                total_orders += block.orders_delivered
                if block.total_block_time > 0:
                    all_utilizations.append(block.utilization_rate)
        
        # Calculate statistics
        avg_utilization = sum(all_utilizations) / len(all_utilizations) if all_utilizations else 0
        
        # Find under-utilized couriers (< 50% utilization)
        underutilized = []
        for courier_id, blocks in self.courier_blocks.items():
            courier_total_time = sum(b.total_block_time for b in blocks)
            courier_active_time = sum(b.active_time for b in blocks)
            if courier_total_time > 0:
                util = min(1.0, courier_active_time / courier_total_time)
                if util < 0.5:
                    underutilized.append({
                        'courier_id': courier_id,
                        'utilization': util,
                        'active_hours': courier_active_time / 3600,
                        'total_hours': courier_total_time / 3600
                    })
        
        # Clamp total active time to not exceed block time
        clamped_active_time = min(total_active_time, total_block_time)
        
        # Use unique waybills count for total orders (avoid double-counting cross-block deliveries)
        unique_orders_count = len(self.unique_waybills)
        
        return {
            'total_couriers': len(self.courier_blocks),
            'total_blocks': total_blocks,
            'total_orders_delivered': unique_orders_count,
            'total_block_order_credits': total_orders,  # Sum of per-block counts (includes cross-block double-counts)
            'avg_utilization': avg_utilization,
            'min_utilization': min(all_utilizations) if all_utilizations else 0,
            'max_utilization': max(all_utilizations) if all_utilizations else 0,
            'total_block_time_hours': total_block_time / 3600,
            'total_active_time_hours': clamped_active_time / 3600,
            'total_idle_time_hours': (total_block_time - clamped_active_time) / 3600,
            'overall_utilization': clamped_active_time / total_block_time if total_block_time > 0 else 0,
            'underutilized_couriers_count': len(underutilized),
            'underutilized_couriers': sorted(underutilized, key=lambda x: x['utilization'])[:10]
        }
    
    def save_report(self, filepath: str):
        """Save detailed utilization report to JSON."""
        report = {
            'summary': self.get_summary(),
            'courier_details': {
                str(cid): self.get_courier_stats(cid) 
                for cid in self.courier_blocks.keys()
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
    
    def count_deliveries_beyond_time(self, timestamp: int) -> Tuple[int, int]:
        """
        Count deliveries that complete after a given timestamp.
        
        Returns:
            Tuple of (delivery_count, courier_count) - number of deliveries
            and number of unique couriers with deliveries completing after timestamp.
        """
        delivery_count = 0
        couriers_with_pending = set()
        
        for courier_id, blocks in self.courier_blocks.items():
            for block in blocks:
                for detail in block.order_details:
                    if detail['arrive_time'] > timestamp:
                        delivery_count += 1
                        couriers_with_pending.add(courier_id)
        
        return delivery_count, len(couriers_with_pending)
    
    def extend_blocks_for_pending_deliveries(self, simulation_end_time: int, extended_end_time: int):
        """
        Extend the last block of each courier if they have deliveries that complete
        after the simulation's time window ends.
        
        This eliminates artificial idle time caused by the simulation ending while
        couriers still have orders in transit. Instead of counting time between
        the last delivery and block_end as idle (when there were no more orders
        to dispatch), we extend the block_end to the last delivery completion.
        
        Args:
            simulation_end_time: Original end time of the simulation's last cycle
            extended_end_time: Extended end time based on latest delivery completion
        """
        if extended_end_time <= simulation_end_time:
            return  # No extension needed
        
        for courier_id, blocks in self.courier_blocks.items():
            if not blocks:
                continue
            
            # Find the last block that overlaps with the simulation end
            last_block = None
            for block in blocks:
                if block.block_start <= simulation_end_time <= block.block_end:
                    last_block = block
                elif block.block_end >= simulation_end_time:
                    # Block ends after simulation end, it might have deliveries extending past
                    if last_block is None or block.block_start > last_block.block_start:
                        last_block = block
            
            if last_block is None:
                # Try the last block by time
                last_block = max(blocks, key=lambda b: b.block_end)
            
            # Check if any delivery in this block extends past the original block_end
            # but within the extended_end_time
            max_delivery_end = 0
            for detail in last_block.order_details:
                if detail['arrive_time'] > max_delivery_end:
                    max_delivery_end = detail['arrive_time']
            
            # If there's a delivery that extends past the original block_end,
            # we need to re-add it with extended boundaries so active time is counted correctly
            if max_delivery_end > last_block.block_end:
                # Extend block_end to include the delivery
                new_block_end = min(max_delivery_end, extended_end_time)
                if new_block_end > last_block.block_end:
                    # Update block_end and recalculate
                    last_block.block_end = new_block_end
                    last_block.total_block_time = last_block.block_end - last_block.block_start
                    
                    # Recalculate delivery spans with new boundary
                    # Clear and re-add all orders
                    old_details = last_block.order_details.copy()
                    last_block.order_details = []
                    last_block.delivery_spans = []
                    last_block.orders_delivered = 0
                    
                    for detail in old_details:
                        last_block.add_order(
                            detail['waybill_id'],
                            detail['grab_time'],
                            detail['arrive_time']
                        )
