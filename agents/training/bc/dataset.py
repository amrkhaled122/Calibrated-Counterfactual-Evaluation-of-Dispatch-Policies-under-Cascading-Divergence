import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset


@dataclass
class SplitIndices:
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray


class BCDataset(Dataset):
    def __init__(
        self,
        df: pl.DataFrame,
        feature_names: List[str],
        label_col: str,
        mean: Optional[np.ndarray] = None,
        std: Optional[np.ndarray] = None,
        fill_nan_with_mean: bool = True,
        device: Optional[torch.device] = None,
    ):
        self.feature_names = feature_names
        self.label_col = label_col
        self.device = device

        X = df.select(feature_names).to_numpy()
        y_raw = df.select(label_col).to_numpy().reshape(-1)
        y = np.array([0 if (v is None or (isinstance(v, float) and math.isnan(v))) else int(v) for v in y_raw], dtype=np.int64)

        X = np.where(np.isfinite(X), X, np.nan)
        if mean is not None and std is not None:
            if fill_nan_with_mean:
                X = np.where(np.isnan(X), mean, X)
            std_safe = np.where(std == 0, 1.0, std)
            X = (X - mean) / std_safe

        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

        if self.device is not None:
            self.X = self.X.to(self.device)
            self.y = self.y.to(self.device)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def read_manifest(manifest_path: str) -> Dict:
    with open(manifest_path, 'r') as f:
        return json.load(f)


def load_parquet(parquet_path: str) -> pl.DataFrame:
    return pl.read_parquet(parquet_path)


def filter_rows(df: pl.DataFrame) -> pl.DataFrame:
    required = [c for c in ["courier_id", "local_day", "actual_dispatch_time"] if c in df.columns]
    if required:
        for c in required:
            df = df.filter(~pl.col(c).is_null())
    if "eta_seconds_current" in df.columns:
        df = df.with_columns([
            pl.when(pl.col("eta_seconds_current") < 0).then(0).otherwise(pl.col("eta_seconds_current")).alias("eta_seconds_current")
        ])
    return df


def build_label(df: pl.DataFrame, label_col: str = "is_assigned_courier_accepted") -> pl.DataFrame:
    if label_col not in df.columns:
        df = df.with_columns([pl.lit(0).alias(label_col)])
    else:
        df = df.with_columns([
            pl.when(pl.col(label_col).is_null()).then(0).otherwise(pl.col(label_col)).alias(label_col)
        ])
    return df


def group_keys(df: pl.DataFrame) -> List[Tuple]:
    keys = []
    cid = df.select("courier_id").to_numpy().reshape(-1)
    day = df.select("local_day").to_numpy().reshape(-1)
    for i in range(df.height):
        keys.append((cid[i], day[i]))
    return keys


def train_val_test_split_by_group(
    df: pl.DataFrame,
    seed: int = 42,
    ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
) -> SplitIndices:
    keys = group_keys(df)
    uniq = list({k for k in keys})
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)

    n = len(uniq)
    n_train = int(ratios[0] * n)
    n_val = int(ratios[1] * n)
    train_groups = set(uniq[:n_train])
    val_groups = set(uniq[n_train:n_train + n_val])
    test_groups = set(uniq[n_train + n_val:])

    idx_train, idx_val, idx_test = [], [], []
    for i, k in enumerate(keys):
        if k in train_groups:
            idx_train.append(i)
        elif k in val_groups:
            idx_val.append(i)
        else:
            idx_test.append(i)

    return SplitIndices(
        train_idx=np.array(idx_train, dtype=np.int64),
        val_idx=np.array(idx_val, dtype=np.int64),
        test_idx=np.array(idx_test, dtype=np.int64),
    )


def temporal_split(
    df: pl.DataFrame,
    train_hours: int = 96,
    eval_hours: int = 96,
    val_ratio: float = 0.15,
    time_col: str = 'actual_dispatch_time',
    seed: int = 42,
) -> SplitIndices:
    """
    Temporal train/val/test split aligned with DDQN integrated training.

    - Train: first ``train_hours`` from dataset start
    - The eval window (next ``eval_hours``) is randomly split into val / test
      by courier-day groups so no courier-day leaks across val and test.
    """
    ts = df.select(time_col).to_numpy().reshape(-1).astype(np.float64)
    t0 = float(np.nanmin(ts))
    train_end = t0 + train_hours * 3600
    eval_end = train_end + eval_hours * 3600

    train_mask = ts < train_end
    eval_mask = (ts >= train_end) & (ts < eval_end)

    train_idx = np.where(train_mask)[0]
    eval_idx = np.where(eval_mask)[0]

    print(f"  Temporal split: train={len(train_idx):,} rows (0-{train_hours}h), "
          f"eval={len(eval_idx):,} rows ({train_hours}-{train_hours+eval_hours}h)")

    # Split eval into val/test by courier-day group
    if len(eval_idx) == 0:
        return SplitIndices(
            train_idx=train_idx.astype(np.int64),
            val_idx=np.array([], dtype=np.int64),
            test_idx=np.array([], dtype=np.int64),
        )

    cid = df.select('courier_id').to_numpy().reshape(-1)
    day = df.select('local_day').to_numpy().reshape(-1)
    eval_keys = [(cid[i], day[i]) for i in eval_idx]
    uniq = list({k for k in eval_keys})
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    n_val = max(1, int(val_ratio * len(uniq)))
    val_groups = set(uniq[:n_val])

    val_idx_list, test_idx_list = [], []
    for pos, global_i in enumerate(eval_idx):
        if eval_keys[pos] in val_groups:
            val_idx_list.append(global_i)
        else:
            test_idx_list.append(global_i)

    return SplitIndices(
        train_idx=train_idx.astype(np.int64),
        val_idx=np.array(val_idx_list, dtype=np.int64),
        test_idx=np.array(test_idx_list, dtype=np.int64),
    )


def fit_standardizer(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = np.nanmean(X, axis=0)
    std = np.nanstd(X, axis=0)
    std = np.where(std == 0, 1.0, std)
    return mean, std


def save_scaler(path: str, feature_names: List[str], mean: np.ndarray, std: np.ndarray):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "feature_names": feature_names,
        "mean": mean.tolist(),
        "std": std.tolist(),
    }
    with open(path, 'w') as f:
        json.dump(payload, f, indent=2)


def load_scaler(path: str) -> Tuple[List[str], np.ndarray, np.ndarray]:
    with open(path, 'r') as f:
        j = json.load(f)
    return j["feature_names"], np.array(j["mean"], dtype=np.float64), np.array(j["std"], dtype=np.float64)
