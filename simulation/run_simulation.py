"""
CLI entry point for running the simulation.

Usage:
  python simulation/run_simulation.py --agent_ckpt agents/outputs/bc_model/bc_model.pt --output_dir results
  python simulation/run_simulation.py --baseline  # For baseline mode
  python simulation/run_simulation.py --agent_type ddqn
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from simulation.simulator import Simulator


def main():
    parser = argparse.ArgumentParser(
        description="Run agent simulation against historical system"
    )

    # Mode selection
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Run in baseline mode (no agent intervention, pure historical replay)",
    )

    # Agent type selection
    parser.add_argument(
        "--agent_type",
        default="bc",
        choices=[
            "bc",
            "ddqn",
        ],
        help='Agent type: "bc" or "ddqn".',
    )

    # Agent paths
    parser.add_argument(
        "--agent_ckpt",
        default=None,
        help="Path to trained agent checkpoint (auto-selected based on agent_type if not provided)",
    )
    parser.add_argument(
        "--scaler",
        default=None,
        help="Path to scaler JSON (auto-selected based on agent_type if not provided)",
    )
    parser.add_argument(
        "--thresholds",
        default=str(
            Path("agents") / "outputs" / "bc_model" / "thresholds.json"
        ),
        help="Path to thresholds JSON (BC agent only)",
    )
    parser.add_argument(
        "--threshold_name",
        default="f1_opt",
        help="Threshold to use (default: f1_opt, BC agent only)",
    )

    # Data paths
    parser.add_argument(
        "--parquet",
        default=str(
            Path("data") / "features" / "offers_observations.parquet"
        ),
        help="Path to historical offers parquet",
    )
    parser.add_argument(
        "--manifest",
        default=str(Path("data") / "features" / "manifest.json"),
        help="Path to manifest JSON",
    )

    # Output
    parser.add_argument(
        "--output_dir",
        default=str(Path("results")),
        help="Directory to save simulation results",
    )

    # Time window
    parser.add_argument(
        "--hours",
        type=int,
        default=None,
        help="Only simulate the first N hours of data (e.g., --hours 24 for first day)",
    )
    parser.add_argument(
        "--start_hour",
        type=int,
        default=0,
        help="Start simulation at this hour offset (for train/eval splits, e.g., --start_hour 48)",
    )
    parser.add_argument(
        "--end_hour",
        type=int,
        default=None,
        help="End simulation at this hour (e.g., --end_hour 96). Takes precedence over --hours if both specified.",
    )

    # Verbose logging
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging to show detailed event processing, scorer decisions, and routing",
    )

    # Travel time model (for accurate simulated delivery times)
    parser.add_argument(
        "--travel_time_model",
        default=str(Path("data") / "travel_time_model.json"),
        help="Path to travel_time_model.json for empirical pickup/delivery time simulation. "
        "If not provided, falls back to historical ETA.",
    )

    # Scorer and simulation sensitivity controls
    parser.add_argument(
        "--scorer_mode",
        default="distance_only",
        choices=["distance_only", "composite"],
        help="Scorer mode for reassignment: distance_only or composite",
    )
    parser.add_argument(
        "--scorer_distance_weight",
        type=float,
        default=1.0,
        help="Composite scorer distance weight",
    )
    parser.add_argument(
        "--scorer_load_penalty_weight",
        type=float,
        default=0.35,
        help="Composite scorer load penalty weight",
    )
    parser.add_argument(
        "--scorer_late_penalty_weight",
        type=float,
        default=0.35,
        help="Composite scorer late penalty weight",
    )
    parser.add_argument(
        "--scorer_idle_bonus_weight",
        type=float,
        default=0.15,
        help="Composite scorer idle bonus weight",
    )
    parser.add_argument(
        "--max_courier_distance_km",
        type=float,
        default=3.0,
        help="Maximum reassignment candidate distance (km)",
    )
    parser.add_argument(
        "--block_buffer_seconds",
        type=int,
        default=60,
        help="Block boundary availability buffer (seconds)",
    )
    parser.add_argument(
        "--max_pending_per_cycle",
        type=int,
        default=50,
        help="Max pending orders processed per cycle",
    )
    parser.add_argument(
        "--pending_cycles_limit",
        type=int,
        default=5,
        help="Number of cycles before pending order is marked lost",
    )
    parser.add_argument(
        "--divergence_window_seconds",
        type=int,
        default=40 * 60,
        help="Historical invalidation window after divergence (seconds)",
    )
    parser.add_argument(
        "--travel_time_add_noise",
        action="store_true",
        help="Enable stochastic noise in travel time model",
    )
    parser.add_argument(
        "--no_travel_time_noise",
        action="store_true",
        help="Disable stochastic noise in travel time model",
    )

    args = parser.parse_args()

    # Auto-select paths based on agent type if not provided
    if args.agent_ckpt is None:
        if args.agent_type == "bc":
            args.agent_ckpt = str(
                Path("agents") / "outputs" / "bc_model" / "bc_model.pt"
            )
        elif args.agent_type == "ddqn":
            args.agent_ckpt = str(
                Path("agents")
                / "outputs"
                / "ddqn_model_integrated"
                / "ddqn_model.pt"
            )

    if args.scaler is None:
        if args.agent_type == "bc":
            args.scaler = str(
                Path("agents") / "outputs" / "bc_model" / "scaler.json"
            )
        elif args.agent_type == "ddqn":
            args.scaler = str(
                Path("agents")
                / "outputs"
                / "ddqn_model_integrated"
                / "ddqn_scaler.json"
            )

    sim_agent_type = args.agent_type

    # Calculate max_hours from end_hour if specified
    max_hours = args.hours
    if args.end_hour is not None:
        max_hours = args.end_hour - args.start_hour

    # Create simulator
    travel_time_add_noise = True
    if args.travel_time_add_noise:
        travel_time_add_noise = True
    if args.no_travel_time_noise:
        travel_time_add_noise = False

    sim = Simulator(
        agent_ckpt=args.agent_ckpt,
        scaler_json=args.scaler,
        thresholds_json=args.thresholds,
        threshold_name=args.threshold_name,
        parquet_path=args.parquet,
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        baseline_mode=args.baseline,
        agent_type=sim_agent_type,
        max_hours=max_hours,
        verbose=args.verbose,
        travel_time_model_path=args.travel_time_model,
        start_hour=args.start_hour,
        scorer_mode=args.scorer_mode,
        scorer_distance_weight=args.scorer_distance_weight,
        scorer_load_penalty_weight=args.scorer_load_penalty_weight,
        scorer_late_penalty_weight=args.scorer_late_penalty_weight,
        scorer_idle_bonus_weight=args.scorer_idle_bonus_weight,
        max_courier_distance_km=args.max_courier_distance_km,
        block_buffer_seconds=args.block_buffer_seconds,
        max_pending_per_cycle=args.max_pending_per_cycle,
        pending_cycles_limit=args.pending_cycles_limit,
        divergence_window_seconds=args.divergence_window_seconds,
        travel_time_add_noise=travel_time_add_noise,
    )

    # Run simulation
    sim.setup()
    sim.run()


if __name__ == "__main__":
    main()
