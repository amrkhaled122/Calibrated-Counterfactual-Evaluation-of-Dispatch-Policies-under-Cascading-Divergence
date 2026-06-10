
"""
Travel Time Model - Empirical lookup tables for simulation.

Usage:
    from travel_time_model import TravelTimeModel
    model = TravelTimeModel.load('data/travel_time_model.json')

    # Get pickup time (courier → restaurant)
    pickup_seconds = model.get_pickup_time(distance_km=1.5, hour=12)

    # Get delivery time (restaurant → customer)  
    delivery_seconds = model.get_delivery_time(distance_km=2.0, hour=19)

    # Get estimated fetch time accounting for meal prep
    fetch_time = model.get_fetch_time(
        courier_arrival_time=dispatch_time + pickup_seconds,
        estimate_meal_prepare_time=meal_prep_time
    )
"""
import json
import random

class TravelTimeModel:
    def __init__(self, model_data):
        self.pickup = model_data['pickup_time']
        self.delivery = model_data['delivery_time']
        self.meal_prep = model_data['meal_prep_wait']
        self.distance_bins = model_data['distance_bins']
        self.distance_labels = model_data['distance_labels']

    @classmethod
    def load(cls, path):
        with open(path, 'r') as f:
            return cls(json.load(f))

    def _get_distance_bucket(self, distance_km):
        """Map distance to bucket label."""
        for i, upper in enumerate(self.distance_bins[1:]):
            if distance_km < upper:
                return self.distance_labels[i]
        return self.distance_labels[-1]  # 10+ km

    def _get_time_block(self, hour):
        """Map hour to time block."""
        if 7 <= hour < 10:
            return 'morning_rush'
        elif 10 <= hour < 14:
            return 'lunch_rush'
        elif 14 <= hour < 17:
            return 'afternoon'
        elif 17 <= hour < 21:
            return 'dinner_rush'
        else:
            return 'off_peak'

    def get_pickup_time(self, distance_km, hour, use_median=True, add_noise=False):
        """
        Get estimated pickup time (courier → restaurant) in seconds.

        Args:
            distance_km: straight-line distance from courier to restaurant
            hour: hour of day (0-23)
            use_median: if True, return median; if False, return mean
            add_noise: if True, add random noise within p25-p75 range
        """
        bucket = self._get_distance_bucket(distance_km)
        time_block = self._get_time_block(hour)

        stats = self.pickup.get(bucket, {}).get(time_block)
        if stats is None:
            # Ultimate fallback: 5 min/km + 60s base
            return int(distance_km * 300 + 60)

        base = stats['median'] if use_median else stats['mean']

        if add_noise and 'p25' in stats and 'p75' in stats:
            noise = random.uniform(stats['p25'] - base, stats['p75'] - base)
            return max(60, int(base + noise))

        return int(base)

    def get_delivery_time(self, distance_km, hour, use_median=True, add_noise=False):
        """
        Get estimated delivery time (restaurant → customer) in seconds.
        """
        bucket = self._get_distance_bucket(distance_km)
        time_block = self._get_time_block(hour)

        stats = self.delivery.get(bucket, {}).get(time_block)
        if stats is None:
            return int(distance_km * 300 + 60)

        base = stats['median'] if use_median else stats['mean']

        if add_noise and 'p25' in stats and 'p75' in stats:
            noise = random.uniform(stats['p25'] - base, stats['p75'] - base)
            return max(60, int(base + noise))

        return int(base)

    def get_fetch_time(self, courier_arrival_at_restaurant, estimate_meal_prepare_time):
        """
        Get actual fetch time accounting for meal prep.

        If courier arrives before food is ready, they wait.
        If food is ready before courier, courier can pick up immediately.

        Returns: actual fetch time (unix timestamp)
        """
        return max(courier_arrival_at_restaurant, estimate_meal_prepare_time)

    def simulate_full_delivery(self, dispatch_time, courier_lat, courier_lng,
                                sender_lat, sender_lng, recipient_lat, recipient_lng,
                                estimate_meal_prepare_time, hour=None):
        """
        Simulate complete delivery timeline.

        Returns dict with:
            - grab_time: when courier accepted (= dispatch_time)
            - courier_arrival_at_restaurant: dispatch + pickup travel
            - fetch_time: max(courier_arrival, meal_prep)
            - arrive_time: fetch + delivery travel
            - total_time: arrive - dispatch
            - courier_waited: seconds courier waited for food (0 if none)
            - food_waited: seconds food waited for courier (0 if none)
        """
        from math import radians, sin, cos, sqrt, asin

        def haversine(lat1, lon1, lat2, lon2):
            R = 6371
            lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
            dlat, dlon = lat2-lat1, lon2-lon1
            a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
            return 2 * R * asin(sqrt(a))

        if hour is None:
            from datetime import datetime
            hour = datetime.fromtimestamp(dispatch_time).hour

        pickup_dist = haversine(courier_lat, courier_lng, sender_lat, sender_lng)
        delivery_dist = haversine(sender_lat, sender_lng, recipient_lat, recipient_lng)

        pickup_time = self.get_pickup_time(pickup_dist, hour, add_noise=True)
        delivery_time = self.get_delivery_time(delivery_dist, hour, add_noise=True)

        grab_time = dispatch_time
        courier_arrival = grab_time + pickup_time
        fetch_time = self.get_fetch_time(courier_arrival, estimate_meal_prepare_time)
        arrive_time = fetch_time + delivery_time

        return {
            'grab_time': grab_time,
            'courier_arrival_at_restaurant': courier_arrival,
            'fetch_time': fetch_time,
            'arrive_time': arrive_time,
            'total_time': arrive_time - dispatch_time,
            'courier_waited': max(0, estimate_meal_prepare_time - courier_arrival),
            'food_waited': max(0, courier_arrival - estimate_meal_prepare_time)
        }
