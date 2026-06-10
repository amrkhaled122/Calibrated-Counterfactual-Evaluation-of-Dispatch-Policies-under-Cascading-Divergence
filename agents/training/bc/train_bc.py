import argparse
import json
import os
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from .dataset import read_manifest, load_parquet, filter_rows, build_label, BCDataset, fit_standardizer, save_scaler, train_val_test_split_by_group, temporal_split
from .model import MLPBC
from .metrics import compute_basic_metrics, find_thresholds


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0.0
    n = 0
    for X, y in loader:
        optimizer.zero_grad(set_to_none=True)
        logits = model(X)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * X.shape[0]
        n += X.shape[0]
    return total_loss / max(1, n)


def eval_probs(model, loader):
    model.eval()
    ys = []
    ps = []
    with torch.no_grad():
        for X, y in loader:
            logits = model(X)
            prob = torch.sigmoid(logits)
            ys.append(y.detach().cpu().numpy())
            ps.append(prob.detach().cpu().numpy())
    y = np.concatenate(ys)
    p = np.concatenate(ps)
    return y, p


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--parquet', default=str(Path('data') / 'features' / 'offers_observations.parquet'))
    parser.add_argument('--manifest', default=str(Path('data') / 'features' / 'manifest.json'))
    parser.add_argument('--out_dir', default=str(Path('agents') / 'outputs' / 'bc_model'))
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--hidden', type=str, default='128,128')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--train_hours', type=int, default=0,
                        help='Temporal split: train on first N hours. 0 = random courier-day split (legacy).')
    parser.add_argument('--eval_hours', type=int, default=0,
                        help='Temporal split: evaluate on next N hours after train window.')
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    os.makedirs(args.out_dir, exist_ok=True)

    manifest = read_manifest(args.manifest)
    feature_names: List[str] = manifest['feature_order']

    df = load_parquet(args.parquet)
    df = filter_rows(df)
    df = build_label(df, label_col='is_assigned_courier_accepted')

    feature_names = [f for f in feature_names if f in df.columns]

    if args.train_hours > 0 and args.eval_hours > 0:
        print(f"Using TEMPORAL split: train 0-{args.train_hours}h, eval {args.train_hours}-{args.train_hours+args.eval_hours}h")
        splits = temporal_split(df, train_hours=args.train_hours, eval_hours=args.eval_hours, seed=args.seed)
    else:
        print("Using random courier-day group split (70/15/15)")
        splits = train_val_test_split_by_group(df)

    X_all = df.select(feature_names).to_numpy()
    X_train = X_all[splits.train_idx]
    mean, std = fit_standardizer(X_train)

    scaler_path = os.path.join(args.out_dir, 'scaler.json')
    save_scaler(scaler_path, feature_names, mean, std)

    ds_all = BCDataset(df, feature_names, label_col='is_assigned_courier_accepted', mean=mean, std=std, device=device)
    train_ds = Subset(ds_all, splits.train_idx)
    val_ds = Subset(ds_all, splits.val_idx)
    test_ds = Subset(ds_all, splits.test_idx)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    input_dim = len(feature_names)
    hidden = [int(x) for x in args.hidden.split(',') if x]
    model = MLPBC(input_dim=input_dim, hidden=hidden, dropout=args.dropout).to(device)

    y_train = df.select('is_assigned_courier_accepted').to_numpy().reshape(-1)[splits.train_idx]
    pos = float((y_train == 1).sum())
    neg = float((y_train == 0).sum())
    pos_weight = torch.tensor([neg / max(1.0, pos)], dtype=torch.float32, device=device)

    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_val_auc = -1.0
    best_path = os.path.join(args.out_dir, 'bc_model.pt')
    patience = 3
    no_improve = 0

    for epoch in range(1, args.epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        y_val, p_val = eval_probs(model, val_loader)
        m_val = compute_basic_metrics(y_val, p_val, threshold=0.5)
        auc = m_val.get('roc_auc', float('nan'))
        print(f"Epoch {epoch:03d} | train_loss={tr_loss:.4f} | val_auc={auc:.4f} | val_f1@0.5={m_val.get('f1', float('nan')):.4f}")
        if not np.isnan(auc) and auc > best_val_auc:
            best_val_auc = auc
            no_improve = 0
            torch.save({
                'state_dict': model.state_dict(),
                'feature_names': feature_names,
                'mean': mean,
                'std': std,
                'hidden': hidden,
                'dropout': args.dropout,
            }, best_path)
        else:
            no_improve += 1
            if no_improve >= patience:
                print("Early stopping: no improvement in val AUC")
                break

    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['state_dict'])

    y_val, p_val = eval_probs(model, val_loader)
    y_test, p_test = eval_probs(model, test_loader)

    thresholds = find_thresholds(y_val, p_val)
    with open(os.path.join(args.out_dir, 'thresholds.json'), 'w') as f:
        json.dump(thresholds, f, indent=2)

    metrics = { 'val': {}, 'test': {} }
    for name, thr in thresholds.items():
        metrics['val'][name] = compute_basic_metrics(y_val, p_val, threshold=thr)
        metrics['test'][name] = compute_basic_metrics(y_test, p_test, threshold=thr)

    with open(os.path.join(args.out_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    print("Saved:")
    print(f" - model: {best_path}")
    print(f" - scaler: {scaler_path}")
    print(f" - thresholds: {os.path.join(args.out_dir, 'thresholds.json')}")
    print(f" - metrics: {os.path.join(args.out_dir, 'metrics.json')}")


if __name__ == '__main__':
    main()
