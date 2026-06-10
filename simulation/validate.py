"""
Quick validation script to test simulation components load correctly.

Run this before full simulation to catch import/config errors early.
"""
import json
from pathlib import Path
import sys

# Add parent directory to path so we can import simulation package
sys.path.insert(0, str(Path(__file__).parent.parent))

def validate_simulation():
    """Check all simulation components can be imported and basic setup works."""
    
    print("=== Simulation Validation ===\n")
    
    # Check imports
    print("1. Checking imports...")
    try:
        from simulation.dispatcher import Dispatcher
        from simulation.scorer import Scorer, CourierState
        from simulation.router import Router
        from simulation.capacity_tracker import CapacityTracker
        from simulation.metrics import MetricsLogger
        from simulation.simulator import Simulator
        print("   ✓ All simulation modules imported successfully")
    except ImportError as e:
        print(f"   ✗ Import error: {e}")
        return False
    
    # Check agent imports
    print("\n2. Checking agent imports...")
    try:
        from agents.agents.bc_agent import BCPruningAgent
        from agents.agents.ddqn_agent import DDQNAgent
        print("   ✓ BC and DDQN agents imported successfully")
    except ImportError as e:
        print(f"   ✗ Import error: {e}")
        return False
    
    # Check required files exist
    print("\n3. Checking required files...")
    required_files = [
        'agents/outputs/bc_model/bc_model.pt',
        'agents/outputs/bc_model/scaler.json',
        'agents/outputs/bc_model/thresholds.json',
        'agents/outputs/ddqn_model_integrated/ddqn_model.pt',
        'agents/outputs/ddqn_model_integrated/ddqn_scaler.json',
        'data/features/offers_observations.parquet',
        'data/features/manifest.json',
        'data/actual_eta_by_order.csv',
        'data/courier_working_blocks.csv',
        'data/travel_time_model.json',
    ]
    
    all_exist = True
    for fpath in required_files:
        exists = Path(fpath).exists()
        status = "✓" if exists else "✗"
        print(f"   {status} {fpath}")
        if not exists:
            all_exist = False
    
    if not all_exist:
        print("\n   Missing required files. Please run preprocessing and agent training first.")
        return False
    
    # Check manifest structure
    print("\n4. Validating manifest...")
    try:
        with open('data/features/manifest.json', 'r') as f:
            manifest = json.load(f)
        
        if 'feature_order' not in manifest:
            print("   ✗ Manifest missing 'feature_order'")
            return False
        
        feature_count = len(manifest['feature_order'])
        print(f"   ✓ Manifest valid with {feature_count} features")
        
        # Check for order_income_value
        if 'order_income_value' in manifest['feature_order']:
            print("   ✓ order_income_value present in feature_order")
        else:
            print("   ⚠ order_income_value not in feature_order (using fallback)")
    
    except Exception as e:
        print(f"   ✗ Manifest validation error: {e}")
        return False
    
    # Try loading agents (quick check)
    print("\n5. Testing agent loads...")
    try:
        bc_agent = BCPruningAgent(
            model_ckpt='agents/outputs/bc_model/bc_model.pt',
            scaler_json='agents/outputs/bc_model/scaler.json',
            thresholds_json='agents/outputs/bc_model/thresholds.json',
            threshold_name='f1_opt'
        )
        ddqn_agent = DDQNAgent(
            model_ckpt='agents/outputs/ddqn_model_integrated/ddqn_model.pt',
            scaler_json='agents/outputs/ddqn_model_integrated/ddqn_scaler.json',
        )
        print(f"   ✓ BC agent loaded with {len(bc_agent.feature_names)} features")
        print(f"   ✓ DDQN agent loaded with {len(ddqn_agent.feature_names)} features")
    except Exception as e:
        print(f"   ✗ Agent load error: {e}")
        return False
    
    print("\n=== Validation Complete ===")
    print("✓ All checks passed. Ready to run simulation.\n")
    print("Run:")
    print("  python -m simulation.run_simulation")
    
    return True


if __name__ == '__main__':
    success = validate_simulation()
    sys.exit(0 if success else 1)
