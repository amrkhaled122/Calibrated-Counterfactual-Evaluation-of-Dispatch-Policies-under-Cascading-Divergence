"""
Modular simulation environment for testing pruning agent against historical system.

Core Components:
- dispatcher: chronological log replay of offers
- scorer: re-match orders to next-best courier on divergence
- router: greedy location-based routing without capacity constraints
- order_tracker: manage order state throughout lifecycle
- capacity_tracker: track realized capacity (no enforcement)
- utilization_tracker: track courier utilization
- metrics: log divergences, capacity, income, late deliveries
- travel_time_model: empirical lookup tables for travel times
- baseline_agent: baseline agent that follows historical decisions
- simulator: main orchestrator

Analysis Subpackage:
- analysis/compare_results.py: comparison of consolidated result JSONs
- analysis/select_case_courier.py: choose a divergence case-study courier
- analysis/plot_divergence_case_study.py: render the case-study figure"""
