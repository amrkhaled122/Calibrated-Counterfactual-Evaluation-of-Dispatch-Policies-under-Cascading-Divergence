"""
Income and distance utilities. Uses Asia/Shanghai timezone for hour extraction.
"""
import numpy as np
from datetime import datetime, timezone, timedelta
import pytz
from .logger_setup import get_logger

logger = get_logger("income_utils")

SH_TZ = pytz.timezone("Asia/Shanghai")


def haversine_km(lat1, lng1, lat2, lng2):
    lat1, lng1, lat2, lng2 = map(np.radians, [lat1, lng1, lat2, lng2])
    dlat = lat2 - lat1
    dlng = lng2 - lng1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlng / 2) ** 2
    a = np.where(np.isnan(a), 0.0, a)
    a = np.clip(a, 0.0, 1.0)
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371.0
    return c * r



def compute_income_row(sender_lat, sender_lng, recipient_lat, recipient_lng, arrive_time, estimate_arrived_time):
    """
    Compute estimated income for a single delivery order.
    
    Income model:
    - Base fee: 3.15 CNY
    - Distance fee: 0.2 CNY per 100m beyond 3km
    - Time bonus: 1.5 CNY (3-6 AM), 1.0 CNY (0-3 AM), 0 otherwise
    - Late penalty: 50% reduction if arrive_time > estimate_arrived_time
    
    Args:
        sender_lat, sender_lng: Restaurant location
        recipient_lat, recipient_lng: Customer location  
        arrive_time: Actual arrival timestamp (epoch seconds)
        estimate_arrived_time: Platform estimated arrival (epoch seconds)
    
    Returns:
        float: Estimated income in CNY
    """
    # Calculate distance
    try:
        d = haversine_km(sender_lat, sender_lng, recipient_lat, recipient_lng)
    except (TypeError, ValueError):
        d = 0.0
    
    # Base fee
    f_base = 3.15
    
    # Distance fee: 0.2 CNY per 100m beyond 3km
    f_dist = max(0.0, 0.2 * ((d - 3.0) / 0.1)) if d > 3 else 0.0
    
    # Time bonus based on arrival hour (Asia/Shanghai)
    f_time = 0.0
    if arrive_time is not None:
        try:
            arrive_ts = int(arrive_time)
            if arrive_ts > 0:
                at = datetime.fromtimestamp(arrive_ts, tz=timezone.utc).astimezone(SH_TZ)
                hour = at.hour
                if 0 <= hour < 3:
                    f_time = 1.0
                elif 3 <= hour < 6:
                    f_time = 1.5
        except (TypeError, ValueError, OSError):
            pass  # Keep f_time = 0.0
    
    # Check if late
    late = False
    if arrive_time is not None and estimate_arrived_time is not None:
        try:
            late = int(arrive_time) > int(estimate_arrived_time)
        except (TypeError, ValueError):
            pass
    
    # Calculate total with late penalty
    total = f_base + f_dist + f_time
    income = 0.5 * total if late else total
    return float(income)

