"""
DDQN Agent Action Logger & Feature Influence Analyzer.

Runs the DDQN simulation and logs detailed information for every misaligned action:
- Q-values for accept/reject
- Feature values and their contribution to the decision
- Feature influence analysis via gradient-based attribution and Q-value perturbation

Usage:
    python simulation/run_ddqn_action_analysis.py --start_hour 48 --end_hour 96
    python simulation/run_ddqn_action_analysis.py --start_hour 48 --end_hour 96 --output_dir results/ddqn_analysis
"""
import argparse
import json
import sys
import csv
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

from simulation.simulator import Simulator
from simulation.dispatcher import OfferEvent
from agents.agents.ddqn_agent import DDQNAgent


class DDQNActionAnalyzer:
    """
    Wraps DDQNAgent to provide detailed Q-value analysis and feature influence
    for every decision, especially misaligned ones.
    """

    def __init__(self, ddqn_agent: DDQNAgent):
        self.agent = ddqn_agent
        self.feature_names = ddqn_agent.feature_names[:]
        self.mean = ddqn_agent.mean.copy()
        self.std = ddqn_agent.std.copy()
        self.device = ddqn_agent.device

    def analyze_decision(self, obs: Dict[str, float]) -> Dict:
        """
        Analyze a single decision with full feature influence breakdown.

        Returns dict with:
          - action, confidence, q_accept, q_reject
          - raw_features: original feature values
          - normalized_features: after z-score normalization
          - gradient_attribution: gradient of Q(chosen_action) w.r.t. each input feature
          - perturbation_influence: change in Q-gap when each feature is zeroed out
        """
        # 1. Encode raw features
        raw = np.zeros(len(self.feature_names), dtype=np.float64)
        for i, f in enumerate(self.feature_names):
            v = obs.get(f, np.nan)
            try:
                v = float(v)
            except Exception:
                v = np.nan
            raw[i] = v
        raw = np.where(np.isnan(raw), self.mean, raw)

        # 2. Normalize
        normed = (raw - self.mean) / self.std

        # 3. Q-values (no-grad for speed)
        x = torch.tensor(normed, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            q_vals = self.agent.model(x)
        q_reject = q_vals[0].item()
        q_accept = q_vals[1].item()
        action = 1 if q_accept >= q_reject else 0
        probs = F.softmax(q_vals, dim=0)
        confidence = probs[1].item()

        # 4. Gradient attribution: dQ(action)/dX_i
        x_grad = torch.tensor(normed, dtype=torch.float32, device=self.device, requires_grad=True)
        self.agent.model.eval()
        q_vals_g = self.agent.model(x_grad)
        q_chosen = q_vals_g[action]
        q_chosen.backward()
        grad = x_grad.grad.detach().cpu().numpy()

        # Feature importance = |gradient * input_value| (integrated-gradient-like proxy)
        gradient_importance = np.abs(grad * normed)

        # 5. Perturbation influence: zero-out each feature and see Q-gap change
        q_gap_original = q_accept - q_reject
        perturbation_deltas = np.zeros(len(self.feature_names))
        for i in range(len(self.feature_names)):
            perturbed = normed.copy()
            perturbed[i] = 0.0  # zero-out (effectively mean of that feature)
            x_p = torch.tensor(perturbed, dtype=torch.float32, device=self.device)
            with torch.no_grad():
                q_p = self.agent.model(x_p)
            gap_p = q_p[1].item() - q_p[0].item()
            perturbation_deltas[i] = q_gap_original - gap_p  # positive = this feature pushed toward accept

        return {
            'action': action,
            'confidence': confidence,
            'q_accept': q_accept,
            'q_reject': q_reject,
            'q_gap': q_gap_original,
            'raw_features': {f: float(raw[i]) for i, f in enumerate(self.feature_names)},
            'normalized_features': {f: float(normed[i]) for i, f in enumerate(self.feature_names)},
            'gradient_attribution': {f: float(gradient_importance[i]) for i, f in enumerate(self.feature_names)},
            'gradient_raw': {f: float(grad[i]) for i, f in enumerate(self.feature_names)},
            'perturbation_influence': {f: float(perturbation_deltas[i]) for i, f in enumerate(self.feature_names)},
        }

    def get_top_influences(self, analysis: Dict, top_k: int = 5) -> List[Dict]:
        """Get top-K most influential features for a decision."""
        # Combine gradient and perturbation importance
        combined = {}
        for f in self.feature_names:
            grad_imp = abs(analysis['gradient_attribution'][f])
            pert_imp = abs(analysis['perturbation_influence'][f])
            combined[f] = {
                'feature': f,
                'gradient_importance': grad_imp,
                'perturbation_influence': analysis['perturbation_influence'][f],
                'raw_value': analysis['raw_features'][f],
                'normalized_value': analysis['normalized_features'][f],
                'gradient_direction': analysis['gradient_raw'][f],
                'combined_score': 0.5 * grad_imp + 0.5 * pert_imp
            }

        sorted_features = sorted(combined.values(), key=lambda x: x['combined_score'], reverse=True)
        return sorted_features[:top_k]


def run_ddqn_action_analysis(
    start_hour: int = 48,
    end_hour: int = 96,
    output_dir: str = 'results/ddqn_analysis',
    parquet_path: str = None,
    manifest_path: str = None,
    agent_ckpt: str = None,
    scaler_json: str = None,
    travel_time_model_path: str = None,
):
    """Run DDQN simulation with detailed action logging."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if parquet_path is None:
        parquet_path = str(Path('data') / 'features' / 'offers_observations.parquet')
    if manifest_path is None:
        manifest_path = str(Path('data') / 'features' / 'manifest.json')
    if agent_ckpt is None:
        agent_ckpt = str(Path('agents') / 'outputs' / 'ddqn_model_integrated' / 'ddqn_model.pt')
    if scaler_json is None:
        scaler_json = str(Path('agents') / 'outputs' / 'ddqn_model_integrated' / 'ddqn_scaler.json')
    if travel_time_model_path is None:
        travel_time_model_path = str(Path('data') / 'travel_time_model.json')

    max_hours = end_hour - start_hour

    print("=" * 70)
    print("DDQN AGENT ACTION ANALYSIS")
    print("=" * 70)
    print(f"Time window: hour {start_hour} to hour {end_hour} ({max_hours}h)")
    print(f"Agent checkpoint: {agent_ckpt}")
    print(f"Output: {output_dir}")
    print("=" * 70)

    # Load DDQN agent directly for analysis
    ddqn_agent = DDQNAgent(model_ckpt=agent_ckpt, scaler_json=scaler_json)
    analyzer = DDQNActionAnalyzer(ddqn_agent)

    # Create simulator
    sim = Simulator(
        agent_ckpt=agent_ckpt,
        scaler_json=scaler_json,
        thresholds_json=str(Path('agents') / 'outputs' / 'bc_model' / 'thresholds.json'),
        threshold_name='f1_opt',
        parquet_path=parquet_path,
        manifest_path=manifest_path,
        output_dir=output_dir,
        baseline_mode=False,
        agent_type='ddqn',
        max_hours=max_hours,
        start_hour=start_hour,
        verbose=False,
        travel_time_model_path=travel_time_model_path,
    )
    sim.setup()

    events = sim.dispatcher.get_events()
    sim._build_courier_dispatch_schedule(events)
    cycles = sim._build_cycles(events)

    print(f"Total cycles: {len(cycles)}, Total events: {len(events)}")

    # Storage for misaligned actions
    misaligned_actions = []
    all_gradient_attributions = defaultdict(list)       # feature -> list of |grad*x|
    all_perturbation_influences = defaultdict(list)  # feature -> list of delta
    action_counts = {'aligned_accept': 0, 'aligned_reject': 0, 'agent_reject_hist_accept': 0, 'agent_accept_hist_reject': 0}
    total_events_processed = 0

    # Also collect per-feature importance across ALL decisions (not just misaligned)
    global_gradient_importance = defaultdict(list)

    from tqdm import tqdm
    pbar = tqdm(total=len(events), desc="Analyzing DDQN actions")

    for cycle_id, cycle_info in enumerate(cycles):
        sim.current_cycle_id = cycle_id
        sim.current_cycle_start_time = cycle_info['start_time']
        sim.current_cycle_end_time = cycle_info['end_time']
        sim.scorer.invalidate_cache()
        sim._update_diverged_couriers_for_time(cycle_info['start_time'])

        if sim.pending_reassignments:
            sim._process_pending_reassignments(cycle_info)

        for event in cycle_info['events']:
            pbar.update(1)

            # Skip already-handled orders
            if event.waybill_id in sim.order_tracker.completed_orders or \
               event.waybill_id in sim.order_tracker.active_orders or \
               event.order_id in sim.order_tracker.completed_order_ids:
                continue

            sim._update_diverged_couriers_for_time(event.actual_dispatch_time)
            courier_state = sim.courier_states.get(event.courier_id)
            if courier_state:
                courier_state.record_offer(event.waybill_id)

            # Skip invalidated orders
            if courier_state and courier_state.is_order_invalidated(event.waybill_id):
                sim._handle_invalidated_order(event)
                continue

            hist_decision = event.historical_decision
            total_events_processed += 1

            # Run full analysis
            analysis = analyzer.analyze_decision(event.features)
            agent_action = analysis['action']

            # Collect global gradient importance for feature importance analysis
            for f in analyzer.feature_names:
                global_gradient_importance[f].append(analysis['gradient_attribution'][f])

            # Determine alignment
            is_misaligned = (agent_action != hist_decision)

            if agent_action == 1 and hist_decision == 1:
                action_counts['aligned_accept'] += 1
            elif agent_action == 0 and hist_decision == 0:
                action_counts['aligned_reject'] += 1
            elif agent_action == 0 and hist_decision == 1:
                action_counts['agent_reject_hist_accept'] += 1
            elif agent_action == 1 and hist_decision == 0:
                action_counts['agent_accept_hist_reject'] += 1

            if is_misaligned:
                top_influences = analyzer.get_top_influences(analysis, top_k=5)

                misaligned_entry = {
                    'event_index': total_events_processed,
                    'waybill_id': int(event.waybill_id),
                    'order_id': int(event.order_id),
                    'courier_id': int(event.courier_id),
                    'dispatch_time': int(event.actual_dispatch_time),
                    'dispatch_cycle_id': event.dispatch_cycle_id,
                    'timestamp_str': datetime.fromtimestamp(event.actual_dispatch_time).strftime('%Y-%m-%d %H:%M:%S'),
                    'agent_action': agent_action,
                    'historical_decision': hist_decision,
                    'misalignment_type': 'agent_reject_hist_accept' if agent_action == 0 else 'agent_accept_hist_reject',
                    'q_accept': round(analysis['q_accept'], 4),
                    'q_reject': round(analysis['q_reject'], 4),
                    'q_gap': round(analysis['q_gap'], 4),
                    'confidence': round(analysis['confidence'], 4),
                    'raw_features': {k: round(v, 4) for k, v in analysis['raw_features'].items()},
                    'top_5_influences': [
                        {
                            'feature': inf['feature'],
                            'raw_value': round(inf['raw_value'], 4),
                            'gradient_importance': round(inf['gradient_importance'], 4),
                            'perturbation_influence': round(inf['perturbation_influence'], 4),
                            'gradient_direction': round(inf['gradient_direction'], 4),
                        }
                        for inf in top_influences
                    ]
                }
                misaligned_actions.append(misaligned_entry)

                # Accumulate per-feature attribution for misaligned actions
                for f in analyzer.feature_names:
                    all_gradient_attributions[f].append(analysis['gradient_attribution'][f])
                    all_perturbation_influences[f].append(analysis['perturbation_influence'][f])

            # Process the event through the simulator (so state progresses correctly)
            sim._process_event(event)

    pbar.close()
    sim._finalize_pending_orders()

    # ========== Generate Reports ==========
    print(f"\nProcessed {total_events_processed} events")
    print(f"  Aligned accepts:           {action_counts['aligned_accept']}")
    print(f"  Aligned rejects:           {action_counts['aligned_reject']}")
    print(f"  Agent reject / Hist accept: {action_counts['agent_reject_hist_accept']}")
    print(f"  Agent accept / Hist reject: {action_counts['agent_accept_hist_reject']}")
    print(f"  Total misaligned:           {len(misaligned_actions)}")

    # 1. Save detailed misaligned actions log
    misaligned_path = output_path / 'misaligned_actions_detailed.json'
    with open(misaligned_path, 'w') as f:
        json.dump({
            'summary': {
                'time_window': f'hour {start_hour} to hour {end_hour}',
                'total_events': total_events_processed,
                'action_counts': action_counts,
                'total_misaligned': len(misaligned_actions),
                'misalignment_rate': len(misaligned_actions) / max(1, total_events_processed),
            },
            'misaligned_actions': misaligned_actions
        }, f, indent=2)
    print(f"\nDetailed misaligned actions saved to: {misaligned_path}")

    # 2. Feature importance summary (across ALL misaligned actions)
    feature_importance_summary = {}
    for f in analyzer.feature_names:
        grad_vals = all_gradient_attributions.get(f, [])
        pert_vals = all_perturbation_influences.get(f, [])
        global_vals = global_gradient_importance.get(f, [])

        feature_importance_summary[f] = {
            'mean_gradient_importance_misaligned': float(np.mean(grad_vals)) if grad_vals else 0.0,
            'std_gradient_importance_misaligned': float(np.std(grad_vals)) if grad_vals else 0.0,
            'mean_perturbation_influence_misaligned': float(np.mean(pert_vals)) if pert_vals else 0.0,
            'std_perturbation_influence_misaligned': float(np.std(pert_vals)) if pert_vals else 0.0,
            'mean_gradient_importance_global': float(np.mean(global_vals)) if global_vals else 0.0,
            'std_gradient_importance_global': float(np.std(global_vals)) if global_vals else 0.0,
            'times_in_top5_misaligned': sum(
                1 for entry in misaligned_actions
                if f in [inf['feature'] for inf in entry['top_5_influences']]
            ),
        }

    # Sort by global gradient importance
    sorted_features = sorted(
        feature_importance_summary.items(),
        key=lambda x: x[1]['mean_gradient_importance_global'],
        reverse=True
    )

    feature_importance_path = output_path / 'feature_importance_analysis.json'
    with open(feature_importance_path, 'w') as f:
        json.dump({
            'feature_ranking': [
                {'rank': i + 1, 'feature': name, **vals}
                for i, (name, vals) in enumerate(sorted_features)
            ],
            'total_events_analyzed': total_events_processed,
            'total_misaligned': len(misaligned_actions),
        }, f, indent=2)
    print(f"Feature importance analysis saved to: {feature_importance_path}")

    # 3. Aggregate misalignment patterns report
    reject_hist_accept = [a for a in misaligned_actions if a['misalignment_type'] == 'agent_reject_hist_accept']
    accept_hist_reject = [a for a in misaligned_actions if a['misalignment_type'] == 'agent_accept_hist_reject']

    def aggregate_features(entries):
        """Compute feature statistics for a set of misaligned entries."""
        if not entries:
            return {}
        stats = {}
        for f in analyzer.feature_names:
            vals = [e['raw_features'][f] for e in entries]
            stats[f] = {
                'mean': round(float(np.mean(vals)), 4),
                'std': round(float(np.std(vals)), 4),
                'min': round(float(np.min(vals)), 4),
                'max': round(float(np.max(vals)), 4),
                'median': round(float(np.median(vals)), 4),
            }
        return stats

    patterns_report = {
        'agent_reject_hist_accept': {
            'count': len(reject_hist_accept),
            'avg_q_accept': float(np.mean([a['q_accept'] for a in reject_hist_accept])) if reject_hist_accept else 0,
            'avg_q_reject': float(np.mean([a['q_reject'] for a in reject_hist_accept])) if reject_hist_accept else 0,
            'avg_q_gap': float(np.mean([a['q_gap'] for a in reject_hist_accept])) if reject_hist_accept else 0,
            'avg_confidence': float(np.mean([a['confidence'] for a in reject_hist_accept])) if reject_hist_accept else 0,
            'feature_stats': aggregate_features(reject_hist_accept),
            'top_influential_features': _count_top_features(reject_hist_accept),
        },
        'agent_accept_hist_reject': {
            'count': len(accept_hist_reject),
            'avg_q_accept': float(np.mean([a['q_accept'] for a in accept_hist_reject])) if accept_hist_reject else 0,
            'avg_q_reject': float(np.mean([a['q_reject'] for a in accept_hist_reject])) if accept_hist_reject else 0,
            'avg_q_gap': float(np.mean([a['q_gap'] for a in accept_hist_reject])) if accept_hist_reject else 0,
            'avg_confidence': float(np.mean([a['confidence'] for a in accept_hist_reject])) if accept_hist_reject else 0,
            'feature_stats': aggregate_features(accept_hist_reject),
            'top_influential_features': _count_top_features(accept_hist_reject),
        }
    }

    patterns_path = output_path / 'misalignment_patterns.json'
    with open(patterns_path, 'w') as f:
        json.dump(patterns_report, f, indent=2)
    print(f"Misalignment patterns saved to: {patterns_path}")

    # 4. CSV summary for quick exploration
    csv_path = output_path / 'misaligned_actions_summary.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['event_idx', 'waybill_id', 'courier_id', 'timestamp', 'type',
                  'q_accept', 'q_reject', 'q_gap', 'confidence',
                  'top1_feature', 'top1_importance', 'top2_feature', 'top2_importance',
                  'top3_feature', 'top3_importance']
        # Add raw feature columns
        header.extend(analyzer.feature_names)
        writer.writerow(header)

        for entry in misaligned_actions:
            tops = entry['top_5_influences']
            row = [
                entry['event_index'], entry['waybill_id'], entry['courier_id'],
                entry['timestamp_str'], entry['misalignment_type'],
                entry['q_accept'], entry['q_reject'], entry['q_gap'], entry['confidence'],
                tops[0]['feature'] if len(tops) > 0 else '',
                round(tops[0]['gradient_importance'], 4) if len(tops) > 0 else '',
                tops[1]['feature'] if len(tops) > 1 else '',
                round(tops[1]['gradient_importance'], 4) if len(tops) > 1 else '',
                tops[2]['feature'] if len(tops) > 2 else '',
                round(tops[2]['gradient_importance'], 4) if len(tops) > 2 else '',
            ]
            # Add raw feature values
            for feat in analyzer.feature_names:
                row.append(entry['raw_features'].get(feat, ''))
            writer.writerow(row)

    print(f"Summary CSV saved to: {csv_path}")

    # 5. Print feature importance ranking
    print("\n" + "=" * 70)
    print("FEATURE IMPORTANCE RANKING (Global Gradient Attribution)")
    print("=" * 70)
    print(f"{'Rank':<5} {'Feature':<30} {'Global Imp.':<14} {'Misaligned Imp.':<16} {'#Top5 Misal.'}")
    print("-" * 70)
    for i, (name, vals) in enumerate(sorted_features):
        print(f"{i+1:<5} {name:<30} {vals['mean_gradient_importance_global']:<14.4f} "
              f"{vals['mean_gradient_importance_misaligned']:<16.4f} {vals['times_in_top5_misaligned']}")
    print("=" * 70)

    # 6. Recommendations
    print("\n" + "=" * 70)
    print("FEATURE ANALYSIS RECOMMENDATIONS")
    print("=" * 70)

    # Features with low global importance (candidates for removal)
    threshold = np.percentile([v['mean_gradient_importance_global'] for _, v in sorted_features], 25)
    low_importance = [(n, v) for n, v in sorted_features if v['mean_gradient_importance_global'] < threshold]
    print(f"\nLow-importance features (below 25th percentile={threshold:.4f}):")
    for name, vals in low_importance:
        print(f"  - {name}: global_importance={vals['mean_gradient_importance_global']:.4f}")

    # Features that dominate misaligned decisions
    print(f"\nFeatures most driving misaligned decisions:")
    by_misaligned = sorted(sorted_features, key=lambda x: x[1]['times_in_top5_misaligned'], reverse=True)
    for name, vals in by_misaligned[:5]:
        print(f"  - {name}: appears in top-5 of {vals['times_in_top5_misaligned']} misaligned decisions")

    print("\nDone!")
    return {
        'action_counts': action_counts,
        'feature_importance': sorted_features,
        'misaligned_count': len(misaligned_actions),
    }


def _count_top_features(entries):
    """Count how often each feature appears in top 5 influences across entries."""
    counts = defaultdict(int)
    for entry in entries:
        for inf in entry['top_5_influences']:
            counts[inf['feature']] += 1
    return dict(sorted(counts.items(), key=lambda x: -x[1])[:10])


def main():
    parser = argparse.ArgumentParser(description='DDQN Agent Action Analysis with Feature Influence')
    parser.add_argument('--start_hour', type=int, default=48, help='Start hour offset')
    parser.add_argument('--end_hour', type=int, default=96, help='End hour offset')
    parser.add_argument('--output_dir', type=str, default='results/ddqn_analysis',
                        help='Output directory for analysis results')
    parser.add_argument('--agent_ckpt', type=str, default=None, help='DDQN model checkpoint')
    parser.add_argument('--scaler', type=str, default=None, help='DDQN scaler JSON')
    parser.add_argument('--parquet', type=str, default=None, help='Parquet path')
    parser.add_argument('--manifest', type=str, default=None, help='Manifest path')
    parser.add_argument('--travel_time_model', type=str, default=None, help='Travel time model path')

    args = parser.parse_args()

    run_ddqn_action_analysis(
        start_hour=args.start_hour,
        end_hour=args.end_hour,
        output_dir=args.output_dir,
        parquet_path=args.parquet,
        manifest_path=args.manifest,
        agent_ckpt=args.agent_ckpt,
        scaler_json=args.scaler,
        travel_time_model_path=args.travel_time_model,
    )


if __name__ == '__main__':
    main()
