from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import json
import os

import numpy as np
import torch

from agents.training.bc.model import MLPBC, DeepMLPBC
from agents.training.bc.dataset import load_scaler


class BCPruningAgent:
    """
    Behavior Cloning pruning agent.

    - Loads trained MLP and scaler from disk
    - Uses manifest feature order for observation encoding
    - Produces action: Keep=1, Prune=0, along with probability of Keep
    """

    def __init__(
        self,
        model_ckpt: str,
        scaler_json: str,
        thresholds_json: Optional[str] = None,
        threshold_name: str = "f1_opt",
        device: Optional[torch.device] = None,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(model_ckpt, map_location=self.device, weights_only=False)
        self.feature_names: List[str] = ckpt["feature_names"]
        hidden = ckpt.get("hidden", [128, 128])
        dropout = ckpt.get("dropout", 0.1)
        
        # Detect model architecture from checkpoint keys
        model_type = ckpt.get("model_type", None)
        state_dict_keys = list(ckpt["state_dict"].keys())
        is_deep_model = any("input_proj" in k or "blocks" in k for k in state_dict_keys)
        if ckpt.get("use_coord_embedding", False) or model_type == "DeepMLPBCWithCoordEmbedding":
            raise ValueError(
                "This package does not include the removed coordinate-embedding "
                "experiment. Use the paper BC checkpoint trained with MLPBC/DeepMLPBC."
            )

        _, mean, std = load_scaler(scaler_json)
        self.mean = mean.astype(np.float64)
        self.std = np.where(std == 0, 1.0, std).astype(np.float64)

        # Choose model class based on architecture detection
        if is_deep_model or model_type == "DeepMLPBC":
            self.model = DeepMLPBC(input_dim=len(self.feature_names), hidden=hidden, dropout=dropout).to(self.device)
        else:
            self.model = MLPBC(input_dim=len(self.feature_names), hidden=hidden, dropout=dropout).to(self.device)
        
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()

        self.threshold = 0.5
        if thresholds_json and os.path.exists(thresholds_json):
            with open(thresholds_json, "r") as f:
                thr = json.load(f)
            self.threshold = float(thr.get(threshold_name, 0.5))

    def _encode_obs(self, obs: Dict[str, float]) -> torch.Tensor:
        """Encode a single observation dict to normalized tensor."""
        x = np.zeros((len(self.feature_names),), dtype=np.float64)
        for i, f in enumerate(self.feature_names):
            v = obs.get(f, np.nan)
            try:
                v = float(v)
            except Exception:
                v = np.nan
            x[i] = v
        x = np.where(np.isnan(x), self.mean, x)
        x = (x - self.mean) / self.std
        xt = torch.tensor(x, dtype=torch.float32, device=self.device)
        return xt

    @torch.no_grad()
    def act(self, obs: Dict[str, float]) -> Tuple[int, float]:
        """Return (action, prob_keep) for a single observation.
        action: 1=Keep, 0=Prune
        """
        xt = self._encode_obs(obs)
        logits = self.model(xt)
        prob = torch.sigmoid(logits).item()
        action = 1 if prob >= self.threshold else 0
        return action, float(prob)

    @torch.no_grad()
    def act_batch(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Batch inference when X is already aligned to feature_names and unnormalized.
        X: shape (N, D), columns in self.feature_names order
        Returns (actions, probs_keep)
        """
        X = np.where(np.isfinite(X), X, np.nan)
        X = np.where(np.isnan(X), self.mean, X)
        X = (X - self.mean) / self.std
        Xt = torch.tensor(X, dtype=torch.float32, device=self.device)
        probs = torch.sigmoid(self.model(Xt)).detach().cpu().numpy()
        actions = (probs >= self.threshold).astype(int)
        return actions, probs
