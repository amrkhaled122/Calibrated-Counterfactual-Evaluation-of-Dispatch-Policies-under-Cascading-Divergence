"""
Run all simulation agents sequentially and save results.

This script runs:
  1. Baseline model (pure historical replay)
  2. BC agent (Behavioral Cloning)
  3. DDQN agent

Each agent's results are saved in the output directory with appropriate suffixes.

Usage:
  python simulation/run_all_simulations.py
  python simulation/run_all_simulations.py --hours 24  # First 24 hours only
  python simulation/run_all_simulations.py --verbose   # Enable detailed logging
"""
import argparse
import sys
import time
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from simulation.simulator import Simulator


def format_duration(seconds: float) -> str:
    """Format duration in human-readable format."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins}m {secs}s"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        return f"{hours}h {mins}m"


def run_simulation(
    name: str,
    baseline_mode: bool,
    agent_type: str,
    output_dir: str,
    parquet_path: str,
    manifest_path: str,
    travel_time_model_path: str,
    max_hours: int = None,
    verbose: bool = False,
) -> dict:
    """Run a single simulation and return timing info."""
    
    print(f"\n{'='*60}")
    print(f"  Running: {name}")
    print(f"{'='*60}")
    
    # Auto-select paths based on agent type
    if baseline_mode:
        # Baseline doesn't need agent checkpoint
        agent_ckpt = str(Path('agents') / 'outputs' / 'bc_model' / 'bc_model.pt')
        scaler_json = str(Path('agents') / 'outputs' / 'bc_model' / 'scaler.json')
    elif agent_type == 'bc':
        agent_ckpt = str(Path('agents') / 'outputs' / 'bc_model' / 'bc_model.pt')
        scaler_json = str(Path('agents') / 'outputs' / 'bc_model' / 'scaler.json')
    else:
        agent_ckpt = str(Path('agents') / 'outputs' / 'ddqn_model_integrated' / 'ddqn_model.pt')
        scaler_json = str(Path('agents') / 'outputs' / 'ddqn_model_integrated' / 'ddqn_scaler.json')
    
    thresholds_json = str(Path('agents') / 'outputs' / 'bc_model' / 'thresholds.json')
    
    start_time = time.time()
    
    try:
        sim = Simulator(
            agent_ckpt=agent_ckpt,
            scaler_json=scaler_json,
            thresholds_json=thresholds_json,
            threshold_name='f1_opt',
            parquet_path=parquet_path,
            manifest_path=manifest_path,
            output_dir=output_dir,
            baseline_mode=baseline_mode,
            agent_type=agent_type,
            max_hours=max_hours,
            verbose=verbose,
            travel_time_model_path=travel_time_model_path
        )
        
        sim.setup()
        sim.run()
        
        elapsed = time.time() - start_time
        print(f"\n✓ {name} completed in {format_duration(elapsed)}")
        
        return {
            'name': name,
            'success': True,
            'elapsed_seconds': elapsed,
            'error': None
        }
        
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n✗ {name} FAILED after {format_duration(elapsed)}: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            'name': name,
            'success': False,
            'elapsed_seconds': elapsed,
            'error': str(e)
        }


def main():
    parser = argparse.ArgumentParser(description="Run all simulation agents sequentially")
    
    # Data paths
    parser.add_argument('--parquet', default=str(Path('data') / 'features' / 'offers_observations.parquet'),
                        help='Path to historical offers parquet')
    parser.add_argument('--manifest', default=str(Path('data') / 'features' / 'manifest.json'),
                        help='Path to manifest JSON')
    
    # Output
    parser.add_argument('--output_dir', default=str(Path('results')),
                        help='Directory to save simulation results')
    
    # Time window
    parser.add_argument('--hours', type=int, default=None,
                        help='Only simulate the first N hours of data (e.g., --hours 24 for first day)')
    
    # Verbose logging
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose logging')
    
    # Travel time model
    parser.add_argument('--travel_time_model', default=str(Path('data') / 'travel_time_model.json'),
                        help='Path to travel_time_model.json')
    
    # Select which agents to run
    parser.add_argument('--skip_baseline', action='store_true',
                        help='Skip baseline simulation')
    parser.add_argument('--skip_bc', action='store_true',
                        help='Skip BC agent simulation')
    parser.add_argument('--skip_ddqn', action='store_true',
                        help='Skip DDQN agent simulation')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("  ALL SIMULATIONS RUNNER")
    print("="*60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if args.hours:
        print(f"Time window: First {args.hours} hours")
    print(f"Output directory: {args.output_dir}")
    
    total_start = time.time()
    results = []
    
    # 1. Baseline simulation
    if not args.skip_baseline:
        result = run_simulation(
            name="Baseline (Historical Replay)",
            baseline_mode=True,
            agent_type='bc',  # Doesn't matter for baseline
            output_dir=args.output_dir,
            parquet_path=args.parquet,
            manifest_path=args.manifest,
            travel_time_model_path=args.travel_time_model,
            max_hours=args.hours,
            verbose=args.verbose,
        )
        results.append(result)
    
    # 2. BC agent simulation
    if not args.skip_bc:
        result = run_simulation(
            name="BC Agent (Behavioral Cloning)",
            baseline_mode=False,
            agent_type='bc',
            output_dir=args.output_dir,
            parquet_path=args.parquet,
            manifest_path=args.manifest,
            travel_time_model_path=args.travel_time_model,
            max_hours=args.hours,
            verbose=args.verbose,
        )
        results.append(result)
    
    # 3. DDQN agent simulation
    if not args.skip_ddqn:
        result = run_simulation(
            name="DDQN Agent",
            baseline_mode=False,
            agent_type='ddqn',
            output_dir=args.output_dir,
            parquet_path=args.parquet,
            manifest_path=args.manifest,
            travel_time_model_path=args.travel_time_model,
            max_hours=args.hours,
            verbose=args.verbose,
        )
        results.append(result)
    
    # Summary
    total_elapsed = time.time() - total_start
    
    print("\n" + "="*60)
    print("  SUMMARY")
    print("="*60)
    
    for r in results:
        status = "✓" if r['success'] else "✗"
        print(f"  {status} {r['name']}: {format_duration(r['elapsed_seconds'])}")
        if r['error']:
            print(f"      Error: {r['error']}")
    
    print(f"\nTotal time: {format_duration(total_elapsed)}")
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # List output files
    print(f"\nResults saved in: {args.output_dir}/")
    print("  Baseline: system_simulation_metrics_baseline.json")
    print("  BC Agent: system_simulation_metrics_bc_agent.json")
    print("  DDQN Agent: system_simulation_metrics_ddqn_agent.json")
    
    # Return exit code based on success
    failed = [r for r in results if not r['success']]
    if failed:
        print(f"\n⚠ {len(failed)} simulation(s) failed!")
        sys.exit(1)
    else:
        print(f"\n✓ All {len(results)} simulations completed successfully!")
        sys.exit(0)


if __name__ == '__main__':
    main()
