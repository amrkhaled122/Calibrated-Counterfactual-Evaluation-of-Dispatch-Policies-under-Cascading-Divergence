import argparse
import json
import os
from pathlib import Path
from typing import List

import numpy as np
import polars as pl
import torch

from .model import MLPBC
from .dataset import load_scaler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--parquet', required=True)
    parser.add_argument('--out_csv', default=str(Path('agents') / 'outputs' / 'bc_model' / 'preds.csv'))
    parser.add_argument('--model_ckpt', default=str(Path('agents') / 'outputs' / 'bc_model' / 'bc_model.pt'))
    parser.add_argument('--scaler', default=str(Path('agents') / 'outputs' / 'bc_model' / 'scaler.json'))
    parser.add_argument('--thresholds', default=str(Path('agents') / 'outputs' / 'bc_model' / 'thresholds.json'))
    parser.add_argument('--threshold_name', default='f1_opt')
    parser.add_argument('--label_col', default='is_assigned_courier_accepted')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    ckpt = torch.load(args.model_ckpt, map_location=device, weights_only=False)
    feature_names: List[str] = ckpt['feature_names']
    hidden = ckpt.get('hidden', [128, 128])
    dropout = ckpt.get('dropout', 0.1)

    _, mean, std = load_scaler(args.scaler)

    df = pl.read_parquet(args.parquet)
    X = df.select(feature_names).to_numpy()
    X = np.where(np.isfinite(X), X, np.nan)
    X = np.where(np.isnan(X), mean, X)
    std_safe = np.where(std == 0, 1.0, std)
    X = (X - mean) / std_safe
    X_t = torch.tensor(X, dtype=torch.float32, device=device)

    model = MLPBC(input_dim=len(feature_names), hidden=hidden, dropout=dropout).to(device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()

    with torch.no_grad():
        probs = torch.sigmoid(model(X_t)).detach().cpu().numpy()

    with open(args.thresholds, 'r') as f:
        thr = json.load(f)
    threshold = float(thr.get(args.threshold_name, 0.5))

    preds = (probs >= threshold).astype(int)

    out = df.select(["courier_id", "local_day", "actual_dispatch_time"]).to_pandas()
    if args.label_col in df.columns:
        out[args.label_col] = df.select(args.label_col).to_numpy().reshape(-1)
    out["prob_keep"] = probs
    out["pred_keep"] = preds
    if args.label_col in out.columns:
        correct = (out["pred_keep"] == out[args.label_col]).sum()
        total = len(out)
        accuracy = correct / max(1, total)
        print(f"Accuracy ({args.label_col}): {accuracy:.4f} ({correct}/{total})")

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print(f"Wrote predictions to {args.out_csv}")


if __name__ == '__main__':
    main()
