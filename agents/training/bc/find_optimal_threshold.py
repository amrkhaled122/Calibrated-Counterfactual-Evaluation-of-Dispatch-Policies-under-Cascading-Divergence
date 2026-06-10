"""
Find Optimal Threshold for BC Agent

This script evaluates the BC agent at different thresholds to find the one
that minimizes divergence from historical decisions.

The goal is to find a threshold where:
- Agent accepts what history accepted (minimize false rejections)
- Agent rejects what history rejected (minimize false accepts)
- Overall divergence rate is minimized

Usage:
    python -m agents.training.bc.find_optimal_threshold
"""

import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from tqdm import tqdm

# Paths (adjust as needed)
BASE_DIR = Path(__file__).parent.parent.parent.parent
PARQUET_PATH = BASE_DIR / "data" / "features" / "offers_observations.parquet"
MODEL_CKPT = BASE_DIR / "agents" / "outputs" / "bc_model" / "bc_model.pt"
SCALER_JSON = BASE_DIR / "agents" / "outputs" / "bc_model" / "scaler.json"
THRESHOLDS_JSON = BASE_DIR / "agents" / "outputs" / "bc_model" / "thresholds.json"
MANIFEST_PATH = BASE_DIR / "data" / "features" / "manifest.json"


def load_data_and_model():
    """Load parquet data and BC model."""
    print("Loading data...")
    df = pd.read_parquet(PARQUET_PATH)
    print(f"  Loaded {len(df):,} rows")
    
    # Load manifest for feature names
    with open(MANIFEST_PATH, 'r') as f:
        manifest = json.load(f)
    feature_names = manifest['feature_order']
    print(f"  Using {len(feature_names)} features")
    
    # Load model
    print("Loading BC model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(MODEL_CKPT, map_location=device, weights_only=False)
    
    from agents.training.bc.model import MLPBC
    from agents.training.bc.dataset import load_scaler
    
    _, mean, std = load_scaler(str(SCALER_JSON))
    mean = mean.astype(np.float64)
    std = np.where(std == 0, 1.0, std).astype(np.float64)
    
    hidden = ckpt.get("hidden", [128, 128])
    dropout = ckpt.get("dropout", 0.1)
    model = MLPBC(input_dim=len(feature_names), hidden=hidden, dropout=dropout).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    
    return df, feature_names, model, mean, std, device


def get_predictions(df, feature_names, model, mean, std, device, batch_size=4096):
    """Get model predictions (probabilities) for all samples."""
    print("Getting model predictions...")
    
    # Extract features
    X = df[feature_names].values.astype(np.float64)
    
    # Handle NaN - replace with mean
    X = np.where(np.isfinite(X), X, np.nan)
    for i in range(X.shape[1]):
        mask = np.isnan(X[:, i])
        X[mask, i] = mean[i]
    
    # Normalize
    X = (X - mean) / std
    
    # Get predictions in batches
    probs = []
    with torch.no_grad():
        for start in tqdm(range(0, len(X), batch_size), desc="Predicting"):
            batch = X[start:start + batch_size]
            Xt = torch.tensor(batch, dtype=torch.float32, device=device)
            batch_probs = torch.sigmoid(model(Xt)).cpu().numpy()
            probs.append(batch_probs)
    
    probs = np.concatenate(probs).flatten()
    return probs


def evaluate_threshold(probs, y_true, threshold):
    """Evaluate a specific threshold."""
    y_pred = (probs >= threshold).astype(int)
    
    # Metrics
    acc = accuracy_score(y_true, y_pred)
    
    # Handle edge cases for precision/recall
    if y_pred.sum() == 0:
        prec = 0.0
        rec = 0.0
        f1 = 0.0
    else:
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # Divergence metrics
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    # Divergences:
    # - Case 1.1: Historical accept (1), Agent reject (0) = FN
    # - Case 1.2: Historical reject (0), Agent accept (1) = FP
    total_divergences = fn + fp
    divergence_rate = total_divergences / len(y_true)
    
    # Alignment rate = 1 - divergence_rate
    alignment_rate = 1 - divergence_rate
    
    return {
        'threshold': threshold,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'true_negatives': int(tn),
        'false_positives': int(fp),  # Case 1.2: reject→accept
        'false_negatives': int(fn),  # Case 1.1: accept→reject
        'true_positives': int(tp),
        'total_divergences': int(total_divergences),
        'divergence_rate': divergence_rate,
        'alignment_rate': alignment_rate,
        'case_1_1_count': int(fn),  # Historical accept, agent reject
        'case_1_2_count': int(fp),  # Historical reject, agent accept
    }


def find_optimal_thresholds(probs, y_true):
    """Find optimal thresholds for different objectives."""
    print("\nEvaluating thresholds...")
    
    # Test range of thresholds
    thresholds = np.arange(0.05, 0.96, 0.01)
    results = []
    
    for thr in tqdm(thresholds, desc="Testing thresholds"):
        result = evaluate_threshold(probs, y_true, thr)
        results.append(result)
    
    results_df = pd.DataFrame(results)
    
    # Find optimal thresholds for different objectives
    optimal = {}
    
    # 1. Minimum divergence (alignment with history)
    min_div_idx = results_df['divergence_rate'].idxmin()
    optimal['min_divergence'] = results_df.loc[min_div_idx, 'threshold']
    
    # 2. Maximum F1 score
    max_f1_idx = results_df['f1'].idxmax()
    optimal['max_f1'] = results_df.loc[max_f1_idx, 'threshold']
    
    # 3. Maximum accuracy
    max_acc_idx = results_df['accuracy'].idxmax()
    optimal['max_accuracy'] = results_df.loc[max_acc_idx, 'threshold']
    
    # 4. Balanced: minimize Case 1.1 (the expensive divergence)
    # Case 1.1 causes reassignment attempts
    min_case_1_1_idx = results_df['case_1_1_count'].idxmin()
    optimal['min_case_1_1'] = results_df.loc[min_case_1_1_idx, 'threshold']
    
    # 5. Youden's J (Sensitivity + Specificity - 1)
    # Specificity = TN / (TN + FP)
    results_df['specificity'] = results_df['true_negatives'] / (results_df['true_negatives'] + results_df['false_positives']).replace(0, 1)
    results_df['youden_j'] = results_df['recall'] + results_df['specificity'] - 1
    results_df['youden_j'] = results_df['youden_j'].fillna(0)
    max_youden_idx = results_df['youden_j'].idxmax()
    optimal['youden_j'] = results_df.loc[max_youden_idx, 'threshold']
    
    return results_df, optimal


def main():
    print("=" * 60)
    print("BC Agent Threshold Optimization")
    print("=" * 60)
    
    # Load data and model
    df, feature_names, model, mean, std, device = load_data_and_model()
    
    # Get historical decisions (labels)
    # The label column should indicate accept=1, reject=0
    label_col = None
    for col in ['is_accept', 'label', 'decision', 'accepted', 'keep', 'historical_decision']:
        if col in df.columns:
            label_col = col
            break
    
    if label_col is None:
        print("ERROR: Could not find label column!")
        print(f"  Available columns: {list(df.columns)[:20]}...")
        return
    
    y_true = df[label_col].values.astype(int)
    print(f"\nLabel distribution:")
    print(f"  Accept (1): {(y_true == 1).sum():,} ({(y_true == 1).mean():.1%})")
    print(f"  Reject (0): {(y_true == 0).sum():,} ({(y_true == 0).mean():.1%})")
    
    # Get predictions
    probs = get_predictions(df, feature_names, model, mean, std, device)
    
    # Find optimal thresholds
    results_df, optimal = find_optimal_thresholds(probs, y_true)
    
    # Print results
    print("\n" + "=" * 60)
    print("OPTIMAL THRESHOLDS")
    print("=" * 60)
    
    for name, thr in optimal.items():
        result = evaluate_threshold(probs, y_true, thr)
        print(f"\n{name.upper()} (threshold={thr:.2f}):")
        print(f"  Accuracy:       {result['accuracy']:.4f}")
        print(f"  Divergence:     {result['divergence_rate']:.4f} ({result['total_divergences']:,} events)")
        print(f"  Alignment:      {result['alignment_rate']:.4f}")
        print(f"  Case 1.1 (A→R): {result['case_1_1_count']:,}")
        print(f"  Case 1.2 (R→A): {result['case_1_2_count']:,}")
        print(f"  F1 Score:       {result['f1']:.4f}")
    
    # Show comparison table
    print("\n" + "=" * 60)
    print("THRESHOLD COMPARISON TABLE")
    print("=" * 60)
    print(f"{'Threshold':<12} {'Accuracy':<10} {'Divergence':<12} {'Case 1.1':<10} {'Case 1.2':<10} {'F1':<8}")
    print("-" * 62)
    
    for thr in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] + list(optimal.values()):
        thr = round(thr, 2)
        result = evaluate_threshold(probs, y_true, thr)
        print(f"{thr:<12.2f} {result['accuracy']:<10.4f} {result['divergence_rate']:<12.4f} "
              f"{result['case_1_1_count']:<10} {result['case_1_2_count']:<10} {result['f1']:<8.4f}")
    
    # Save optimal thresholds
    print("\n" + "=" * 60)
    print("SAVING OPTIMAL THRESHOLDS")
    print("=" * 60)
    
    # Update thresholds.json
    thresholds_to_save = {
        'default_0_5': 0.5,
        'min_divergence': float(optimal['min_divergence']),
        'max_f1': float(optimal['max_f1']),
        'max_accuracy': float(optimal['max_accuracy']),
        'min_case_1_1': float(optimal['min_case_1_1']),
        'youden_j': float(optimal['youden_j']),
        'f1_opt': float(optimal['max_f1']),  # Alias
    }
    
    with open(THRESHOLDS_JSON, 'w') as f:
        json.dump(thresholds_to_save, f, indent=2)
    print(f"Saved to: {THRESHOLDS_JSON}")
    print(json.dumps(thresholds_to_save, indent=2))
    
    # Save detailed results
    results_path = THRESHOLDS_JSON.parent / 'threshold_analysis.csv'
    results_df.to_csv(results_path, index=False)
    print(f"\nDetailed results saved to: {results_path}")
    
    # Recommendation
    print("\n" + "=" * 60)
    print("RECOMMENDATION")
    print("=" * 60)
    rec_thr = optimal['min_divergence']
    rec_result = evaluate_threshold(probs, y_true, rec_thr)
    print(f"Use threshold={rec_thr:.2f} for minimum divergence from history:")
    print(f"  - Alignment with history: {rec_result['alignment_rate']:.1%}")
    print(f"  - Expected divergences: ~{rec_result['divergence_rate']:.1%} of events")
    print(f"  - Case 1.1 (expensive): {rec_result['case_1_1_count']:,}")
    print(f"\nTo use this threshold in simulation:")
    print(f"  python -m simulation.run_simulation --agent_type bc --threshold_name min_divergence --hours 24")


if __name__ == "__main__":
    main()
