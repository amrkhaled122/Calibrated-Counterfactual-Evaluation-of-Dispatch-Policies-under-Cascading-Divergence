"""
DDQN agent for courier assignment decisions.

Unlike BC which predicts probability of acceptance, DDQN learns
Q-values representing expected cumulative reward for each action.
Action selection is via argmax over Q-values.
"""
from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import json
import os

import numpy as np
import torch
import torch.nn as nn


class QNet(nn.Module):
    """Q-Network architecture (must match training)."""
    def __init__(self, input_dim: int, hidden: List[int] = [256, 256, 128], n_actions: int = 2, dropout: float = 0.1, use_layer_norm: bool = True):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            if use_layer_norm:
                layers.append(nn.LayerNorm(h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        self.fe = nn.Sequential(*layers) if layers else nn.Identity()
        self.head = nn.Linear(prev, n_actions)

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        z = self.fe(x)
        return self.head(z).squeeze(0) if z.shape[0] == 1 else self.head(z)


class DDQNAgent:
    """
    DDQN agent for courier assignment.

    - Loads trained Q-network and scaler from disk
    - Uses argmax over Q-values for action selection
    - Actions: 0=Reject, 1=Accept (Keep)
    
    Key difference from BC:
    - BC: P(accept) via sigmoid, threshold-based decision
    - DDQN: Q(s,a) for each action, argmax-based decision
    """

    def __init__(
        self,
        model_ckpt: str,
        scaler_json: Optional[str] = None,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize DDQN agent.
        
        Args:
            model_ckpt: Path to ddqn_model.pt checkpoint
            scaler_json: Path to ddqn_scaler.json (optional, will use checkpoint values if not provided)
            device: PyTorch device (auto-selects CUDA if available)
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load checkpoint
        ckpt = torch.load(model_ckpt, map_location=self.device, weights_only=False)
        self.feature_names: List[str] = ckpt["feature_order"]
        hidden = ckpt.get("hidden", [128, 128])
        dropout = ckpt.get("dropout", 0.1)
        self.gamma = ckpt.get("gamma", 0.99)
        
        # Detect whether checkpoint was trained with LayerNorm
        # With LayerNorm: keys are fe.0 (Linear), fe.1 (LN), so fe.1.weight exists
        # Without LayerNorm: keys are fe.0 (Linear), fe.1 (ReLU), so fe.1.weight does NOT exist
        state_dict = ckpt["state_dict"]
        use_layer_norm = "fe.1.weight" in state_dict
        
        # Load scaler (from checkpoint or separate file)
        n_features = len(self.feature_names)
        scaler_loaded = False
        if scaler_json and os.path.exists(scaler_json):
            with open(scaler_json, "r") as f:
                scaler_data = json.load(f)
            _mean = np.array(scaler_data["mean"], dtype=np.float64)
            _std = np.array(scaler_data["std"], dtype=np.float64)
            if _mean.shape[0] == n_features:
                self.mean, self.std = _mean, _std
                scaler_loaded = True
            else:
                print(f"[DDQNAgent] WARNING: scaler file has {_mean.shape[0]} dims "
                      f"but model expects {n_features}; ignoring scaler file, "
                      f"using checkpoint mean/std instead.")
        if not scaler_loaded and "mean" in ckpt and "std" in ckpt:
            self.mean = np.array(ckpt["mean"], dtype=np.float64)
            self.std = np.array(ckpt["std"], dtype=np.float64)
            scaler_loaded = True
        if not scaler_loaded:
            # Fallback: no normalization
            self.mean = np.zeros(n_features, dtype=np.float64)
            self.std = np.ones(n_features, dtype=np.float64)
        
        self.std = np.where(self.std == 0, 1.0, self.std)
        
        self.model = QNet(
            input_dim=len(self.feature_names),
            hidden=hidden,
            n_actions=2,
            dropout=dropout,
            use_layer_norm=use_layer_norm,
        ).to(self.device)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()

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
        # Fill NaN with mean
        x = np.where(np.isnan(x), self.mean, x)
        # Normalize
        x = (x - self.mean) / self.std
        xt = torch.tensor(x, dtype=torch.float32, device=self.device)
        return xt

    @torch.no_grad()
    def act(self, obs: Dict[str, float], **kwargs) -> Tuple[int, float]:
        """
        Select action for a single observation.
        
        Returns same interface as BCPruningAgent for compatibility:
            Tuple of (action, confidence):
            - action: 1=Accept (Keep), 0=Reject (Prune)
            - confidence: Normalized score (softmax probability of chosen action)
        
        Note: kwargs allows passing historical_decision like BC agent (ignored by DDQN)
        """
        xt = self._encode_obs(obs)
        q_values = self.model(xt)  # [Q_reject, Q_accept]
        
        q_reject = q_values[0].item()
        q_accept = q_values[1].item()
        
        # Argmax: choose action with higher Q-value
        action = 1 if q_accept >= q_reject else 0
        
        # Compute softmax probability as "confidence" (for compatibility with BC interface)
        # This gives a probability-like score between 0 and 1
        import torch.nn.functional as F
        probs = F.softmax(q_values, dim=0)
        confidence = probs[1].item()  # Probability of accept action
        
        return action, confidence

    @torch.no_grad()
    def act_with_q_values(self, obs: Dict[str, float]) -> Tuple[int, float, float, float]:
        """
        Select action with full Q-value information.
        
        Returns:
            Tuple of (action, confidence, q_accept, q_reject):
            - action: 1=Accept (Keep), 0=Reject (Prune)
            - confidence: Softmax probability of accept
            - q_accept: Raw Q-value for accepting
            - q_reject: Raw Q-value for rejecting
        """
        xt = self._encode_obs(obs)
        q_values = self.model(xt)  # [Q_reject, Q_accept]
        
        q_reject = q_values[0].item()
        q_accept = q_values[1].item()
        
        action = 1 if q_accept >= q_reject else 0
        
        import torch.nn.functional as F
        probs = F.softmax(q_values, dim=0)
        confidence = probs[1].item()
        
        return action, confidence, q_accept, q_reject

    @torch.no_grad()
    def act_batch(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Batch inference when X is already aligned to feature_names and unnormalized.
        
        Args:
            X: shape (N, D), columns in self.feature_names order
            
        Returns:
            Tuple of (actions, q_accepts, q_rejects):
            - actions: array of 0/1 decisions
            - q_accepts: Q-values for accept action
            - q_rejects: Q-values for reject action
        """
        X = np.where(np.isfinite(X), X, np.nan)
        X = np.where(np.isnan(X), self.mean, X)
        X = (X - self.mean) / self.std
        Xt = torch.tensor(X, dtype=torch.float32, device=self.device)
        
        q_values = self.model(Xt).detach().cpu().numpy()  # (N, 2): [reject, accept]
        q_rejects = q_values[:, 0]
        q_accepts = q_values[:, 1]
        
        actions = (q_accepts >= q_rejects).astype(int)
        return actions, q_accepts, q_rejects
        
    def get_q_advantage(self, obs: Dict[str, float]) -> float:
        """
        Get the advantage of accepting over rejecting.
        
        Returns:
            Q(accept) - Q(reject): positive means accept is better
        """
        _, _, q_accept, q_reject = self.act_with_q_values(obs)
        return float(q_accept - q_reject)
