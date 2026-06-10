"""
DDQN Training Integrated with Full Simulator.

This script trains a DDQN agent by directly using the Simulator class,
NOT a simplified environment. The agent learns from its own decisions
while the full simulator handles:

- Time-based location interpolation (grab_time → fetch_time → arrive_time)
- Capacity tracking (increase on accept, decrease on delivery completion)
- Order reassignment when agent rejects historical accepts
- Divergence handling and 40-minute invalidation window
- Travel time model for realistic delivery simulation
- Utilization tracking

The DDQN agent replaces the BC/baseline agent in the simulator's agent slot,
but with exploration enabled during training.

Usage:
    python -m simulation.train_ddqn_integrated --episodes 10 --hours 24
"""
import argparse
import json
import os
import random
import copy
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# Import from simulation
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


# ============================================================================
# Q-Network Architecture
# ============================================================================

class QNet(nn.Module):
    """Q-Network for 2 actions: reject (0) and accept (1)."""
    def __init__(self, input_dim: int, hidden: List[int] = [256, 256, 128], n_actions: int = 2, dropout: float = 0.1):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
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


# ============================================================================
# Prioritized Replay Buffer
# ============================================================================

class PrioritizedReplayBuffer:
    """Experience replay with Bellman-error prioritization."""
    
    def __init__(self, capacity: int = 500000, alpha: float = 0.6, beta_start: float = 0.4):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta_start
        self.beta_increment = 0.0001
        
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.max_priority = 1.0
    
    def push(self, state: np.ndarray, action: int, reward: float,
             next_state: np.ndarray, done: bool):
        """Store transition with max priority."""
        transition = (state.copy(), action, reward, next_state.copy(), done)
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.position] = transition
        
        self.priorities[self.position] = self.max_priority
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int) -> Tuple:
        n = len(self.buffer)
        if n < batch_size:
            batch_size = n
        
        priorities = self.priorities[:n] ** self.alpha
        probs = priorities / priorities.sum()
        
        indices = np.random.choice(n, batch_size, p=probs, replace=False)
        
        weights = (n * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for idx in indices:
            s, a, r, ns, d = self.buffer[idx]
            states.append(s)
            actions.append(a)
            rewards.append(r)
            next_states.append(ns)
            dones.append(d)
        
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
            indices,
            np.array(weights, dtype=np.float32)
        )
    
    def update_priorities(self, indices: np.ndarray, bellman_errors: np.ndarray):
        for idx, bellman_error in zip(indices, bellman_errors):
            self.priorities[idx] = abs(bellman_error) + 1e-6
            self.max_priority = max(self.max_priority, abs(bellman_error) + 1e-6)
    
    def __len__(self):
        return len(self.buffer)


# ============================================================================
# Training DDQN Agent (with exploration)
# ============================================================================

class TrainingDDQNAgent:
    """
    DDQN Agent used during training with exploration.
    
    This agent:
    1. Uses epsilon-greedy exploration during training
    2. Collects (s, a, r, s') transitions
    3. Performs DDQN updates after each simulation step
    4. Is compatible with Simulator.agent interface (.act method)
    
    After training, the standard DDQNAgent class loads the saved checkpoint.
    """
    
    def __init__(
        self,
        feature_names: List[str],
        hidden: List[int] = [256, 256, 128],
        lr: float = 3e-4,
        gamma: float = 0.99,
        epsilon_start: float = 0.3,
        epsilon_end: float = 0.01,
        epsilon_decay_steps: int = 100000,
        target_update_freq: int = 1000,
        batch_size: int = 256,
        buffer_size: int = 500000,
        device: str = 'auto',
        use_double_dqn: bool = True,
    ):
        self.feature_names = feature_names
        self.n_features = len(feature_names)
        self.gamma = gamma
        self.batch_size = batch_size
        self.use_double_dqn = use_double_dqn
        
        # Exploration
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.explore = True  # Can be disabled for evaluation
        
        # Device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Networks
        self.q_net = QNet(self.n_features, hidden).to(self.device)
        self.target_net = QNet(self.n_features, hidden).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()
        
        # Optimizer with gradient clipping
        self.optimizer = torch.optim.AdamW(self.q_net.parameters(), lr=lr, weight_decay=1e-5)
        
        # Replay buffer
        self.buffer = PrioritizedReplayBuffer(buffer_size)
        
        # Tracking
        self.target_update_freq = target_update_freq
        self.steps = 0
        self.updates = 0
        self.hidden = hidden
        
        # Online normalization (Welford's algorithm)
        self.running_mean = np.zeros(self.n_features, dtype=np.float64)
        self.running_var = np.ones(self.n_features, dtype=np.float64)
        self.n_samples = 0
        
        # Metrics
        self.bellman_losses = []
        self.q_values_history = []
        self.rewards_history = []
        
        # Transition tracking for delayed rewards
        self.pending_transitions: Dict[int, Dict] = {}  # waybill_id -> {state, action, ...}
        self.last_state: Optional[np.ndarray] = None
        self.last_action: Optional[int] = None
        self.last_waybill_id: Optional[int] = None
    
    def _update_running_stats(self, obs: np.ndarray):
        """Update running mean/var using Welford's algorithm."""
        self.n_samples += 1
        delta = obs - self.running_mean
        self.running_mean += delta / self.n_samples
        delta2 = obs - self.running_mean
        self.running_var += delta * delta2
    
    def _get_std(self) -> np.ndarray:
        """Get standard deviation from running variance."""
        if self.n_samples < 2:
            return np.ones(self.n_features, dtype=np.float64)
        std = np.sqrt(self.running_var / (self.n_samples - 1))
        return np.where(std < 1e-8, 1.0, std)
    
    def _normalize(self, obs: np.ndarray) -> np.ndarray:
        """Normalize observation using running stats."""
        std = self._get_std()
        return (obs - self.running_mean) / std
    
    def _encode_features(self, features: Dict[str, float]) -> np.ndarray:
        """Convert feature dict to numpy array in correct order."""
        arr = np.zeros(self.n_features, dtype=np.float64)
        for i, f in enumerate(self.feature_names):
            v = features.get(f, 0.0)
            try:
                v = float(v) if v is not None and np.isfinite(float(v)) else 0.0
            except:
                v = 0.0
            arr[i] = v
        return arr
    
    def act(self, features: Dict[str, float], **kwargs) -> Tuple[int, float]:
        """
        Select action - compatible with Simulator.agent.act() interface.
        
        Returns:
            (action, confidence) where action is 0 (reject) or 1 (accept)
        """
        # Encode and normalize
        obs = self._encode_features(features)
        self._update_running_stats(obs)
        obs_norm = self._normalize(obs)
        
        # Store for transition
        self.last_state = obs_norm.copy()
        
        # Get Q-values
        with torch.no_grad():
            x = torch.tensor(obs_norm, dtype=torch.float32, device=self.device)
            q_vals = self.q_net(x)
            if q_vals.dim() == 0:
                q_vals = q_vals.unsqueeze(0)
            q_reject = q_vals[0].item()
            q_accept = q_vals[1].item()
        
        # Epsilon-greedy exploration
        if self.explore and random.random() < self.epsilon:
            action = random.randint(0, 1)
        else:
            action = 1 if q_accept >= q_reject else 0
        
        self.last_action = action
        
        # Confidence as softmax probability
        probs = F.softmax(q_vals, dim=0)
        confidence = probs[1].item()
        
        self.steps += 1
        self._decay_epsilon()
        
        return action, confidence
    
    def _decay_epsilon(self):
        """Linear epsilon decay."""
        progress = min(1.0, self.steps / self.epsilon_decay_steps)
        self.epsilon = self.epsilon_start + (self.epsilon_end - self.epsilon_start) * progress
    
    def store_transition(self, state: np.ndarray, action: int, reward: float,
                         next_state: np.ndarray, done: bool):
        """Store transition in replay buffer."""
        self.buffer.push(state, action, reward, next_state, done)
        self.rewards_history.append(reward)
    
    def update(self) -> Optional[float]:
        """Perform one DDQN update step with Double DQN."""
        if len(self.buffer) < self.batch_size:
            return None
        
        # Sample batch
        states, actions, rewards, next_states, dones, indices, weights = \
            self.buffer.sample(self.batch_size)
        
        # To tensors
        states_t = torch.tensor(states, device=self.device)
        actions_t = torch.tensor(actions, device=self.device)
        rewards_t = torch.tensor(rewards, device=self.device)
        next_states_t = torch.tensor(next_states, device=self.device)
        dones_t = torch.tensor(dones, device=self.device)
        weights_t = torch.tensor(weights, device=self.device)
        
        # Current Q values
        q_values = self.q_net(states_t)
        q_a = q_values.gather(1, actions_t.unsqueeze(1)).squeeze(1)
        
        # Bellman target with Double DQN
        with torch.no_grad():
            if self.use_double_dqn:
                next_actions = self.q_net(next_states_t).argmax(dim=1, keepdim=True)
                q_next = self.target_net(next_states_t).gather(1, next_actions).squeeze(1)
            else:
                q_next = self.target_net(next_states_t).max(dim=1)[0]
            
            bellman_target = rewards_t + (1 - dones_t) * self.gamma * q_next
        
        # Bellman errors for priority update
        bellman_errors = (q_a - bellman_target).detach().cpu().numpy()
        
        # Huber loss with importance sampling weights
        loss = F.smooth_l1_loss(q_a, bellman_target, reduction='none')
        loss = (loss * weights_t).mean()
        
        # Backprop
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.optimizer.step()
        
        # Update priorities
        self.buffer.update_priorities(indices, bellman_errors)
        
        self.updates += 1
        self.bellman_losses.append(loss.item())
        self.q_values_history.append(q_a.mean().item())
        
        # Update target network
        if self.updates % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
        
        return loss.item()
    
    def save(self, path: str):
        """Save checkpoint compatible with DDQNAgent."""
        std = self._get_std()
        
        checkpoint = {
            'state_dict': self.q_net.state_dict(),
            'feature_order': self.feature_names,
            'mean': self.running_mean.tolist(),
            'std': std.tolist(),
            'gamma': self.gamma,
            'hidden': self.hidden,
            'dropout': 0.1,
            'epsilon': self.epsilon,
            'steps': self.steps,
            'updates': self.updates,
        }
        torch.save(checkpoint, path)
    
    def save_full_checkpoint(self, path: str, epoch: int, all_stats: List[Dict], 
                              training_log: List[Dict] = None):
        """
        Save FULL checkpoint for resuming training.
        
        Includes:
        - Model weights (q_net and target_net)
        - Optimizer state
        - Replay buffer (compressed)
        - Running stats for normalization
        - Epsilon and training progress
        - Training history
        """
        std = self._get_std()
        
        # Serialize buffer efficiently (sample subset if too large)
        buffer_data = {
            'buffer': self.buffer.buffer[:len(self.buffer.buffer)],  # List of transitions
            'priorities': self.buffer.priorities[:len(self.buffer.buffer)].tolist(),
            'position': self.buffer.position,
            'max_priority': self.buffer.max_priority,
            'beta': self.buffer.beta,
        }
        
        checkpoint = {
            # Model state
            'q_net_state_dict': self.q_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            
            # Feature info
            'feature_order': self.feature_names,
            'hidden': self.hidden,
            
            # Normalization stats
            'running_mean': self.running_mean.tolist(),
            'running_var': self.running_var.tolist(),
            'n_samples': self.n_samples,
            
            # Training state
            'gamma': self.gamma,
            'epsilon': self.epsilon,
            'epsilon_start': self.epsilon_start,
            'epsilon_end': self.epsilon_end,
            'epsilon_decay_steps': self.epsilon_decay_steps,
            'steps': self.steps,
            'updates': self.updates,
            
            # Epoch info
            'epoch': epoch,
            'all_stats': all_stats,
            
            # Recent metrics (last 10000 for plotting)
            'bellman_losses': self.bellman_losses[-10000:],
            'q_values_history': self.q_values_history[-10000:],
            'rewards_history': self.rewards_history[-10000:],
            
            # Training log
            'training_log': training_log or [],
            
            # Buffer (for full resume)
            'buffer_data': buffer_data,
        }
        
        torch.save(checkpoint, path)
        print(f"  Full checkpoint saved: {path}")
    
    def load_full_checkpoint(self, path: str) -> Tuple[int, List[Dict], List[Dict]]:
        """
        Load FULL checkpoint to resume training.
        
        Returns:
            (start_epoch, all_stats, training_log)
        """
        # weights_only=False needed for numpy arrays in checkpoint (PyTorch 2.6+)
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        # Load model weights
        self.q_net.load_state_dict(checkpoint['q_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load normalization stats
        self.running_mean = np.array(checkpoint['running_mean'], dtype=np.float64)
        self.running_var = np.array(checkpoint['running_var'], dtype=np.float64)
        self.n_samples = checkpoint['n_samples']
        
        # Load training state
        self.epsilon = checkpoint['epsilon']
        self.epsilon_start = checkpoint.get('epsilon_start', 0.3)
        self.epsilon_end = checkpoint.get('epsilon_end', 0.01)
        self.epsilon_decay_steps = checkpoint.get('epsilon_decay_steps', 100000)
        self.steps = checkpoint['steps']
        self.updates = checkpoint['updates']
        
        # Load metrics history
        self.bellman_losses = checkpoint.get(
            'bellman_losses',
            checkpoint.get('td_losses', []),
        )
        self.q_values_history = checkpoint.get('q_values_history', [])
        self.rewards_history = checkpoint.get('rewards_history', [])
        
        # Load replay buffer
        buffer_data = checkpoint.get('buffer_data', {})
        if buffer_data:
            self.buffer.buffer = list(buffer_data.get('buffer', []))
            priorities = buffer_data.get('priorities', [])
            if priorities:
                self.buffer.priorities[:len(priorities)] = np.array(priorities, dtype=np.float32)
            self.buffer.position = buffer_data.get('position', 0)
            self.buffer.max_priority = buffer_data.get('max_priority', 1.0)
            self.buffer.beta = buffer_data.get('beta', 0.4)
        
        start_epoch = checkpoint.get('epoch', 0)
        all_stats = checkpoint.get('all_stats', [])
        training_log = checkpoint.get('training_log', [])
        
        print(f"  Resumed from epoch {start_epoch}, step {self.steps:,}, buffer size {len(self.buffer)}")
        
        return start_epoch, all_stats, training_log
    
    def get_metrics(self) -> Dict:
        """Get training metrics."""
        return {
            'steps': self.steps,
            'updates': self.updates,
            'buffer_size': len(self.buffer),
            'epsilon': self.epsilon,
            'mean_bellman_loss': np.mean(self.bellman_losses[-1000:]) if self.bellman_losses else 0.0,
            'mean_q_value': np.mean(self.q_values_history[-1000:]) if self.q_values_history else 0.0,
            'mean_reward': np.mean(self.rewards_history[-1000:]) if self.rewards_history else 0.0,
        }


# ============================================================================
# Reward Calculator
# ============================================================================

class RewardCalculator:
    """
    Computes rewards for DDQN learning based on simulation outcomes.
    
    The reward signal comes from the Simulator's actual outcomes:
    - Order income when accepting
    - On-time/late delivery status
    - Capacity utilization
    """
    
    def __init__(
        self,
        income_scale: float = 1.0,
        reject_penalty: float = -0.05,
        late_penalty: float = -0.3,
        on_time_bonus: float = 0.1,
        capacity_threshold: int = 6,
        overload_penalty: float = -0.2,
        lost_order_penalty: float = -0.5
    ):
        self.income_scale = income_scale
        self.reject_penalty = reject_penalty
        self.late_penalty = late_penalty
        self.on_time_bonus = on_time_bonus
        self.capacity_threshold = capacity_threshold
        self.overload_penalty = overload_penalty
        self.lost_order_penalty = lost_order_penalty
    
    def compute_accept_reward(
        self,
        income: float,
        is_late: bool,
        capacity: int
    ) -> float:
        """Reward for accepting an order."""
        reward = income * self.income_scale
        
        if is_late:
            reward += self.late_penalty
        else:
            reward += self.on_time_bonus
        
        if capacity >= self.capacity_threshold:
            reward += self.overload_penalty
        
        return reward
    
    def compute_reject_reward(
        self,
        income: float,
        capacity: int,
        order_lost: bool = False
    ) -> float:
        """Reward for rejecting an order."""
        if order_lost:
            return self.lost_order_penalty
        
        reward = self.reject_penalty
        
        # Less penalty if overloaded
        if capacity >= self.capacity_threshold:
            reward += 0.05
        
        # More penalty for high-value orders
        if income > 5.0:
            reward -= 0.05 * (income - 5.0) / 5.0
        
        return reward


class AggressiveAcceptReward(RewardCalculator):
    """
    Strongly penalizes rejections to maximize acceptance rate.
    
    Use this when:
    - Agent is rejecting too many orders
    - You want near-100% acceptance rate
    - Orders are valuable and should rarely be rejected
    """
    
    def __init__(self):
        super().__init__(
            income_scale=0.4,
            reject_penalty=-0.5,        # 10x stronger than baseline
            late_penalty=-0.2,          # Less harsh on late deliveries
            on_time_bonus=0.15,
            capacity_threshold=7,       # Higher threshold before penalty
            overload_penalty=-0.15,     # Smaller overload penalty
            lost_order_penalty=-1.0
        )
    
    def compute_reject_reward(self, income: float, capacity: int, order_lost: bool = False) -> float:
        """Strong penalty for rejecting, especially low-capacity couriers."""
        if order_lost:
            return self.lost_order_penalty
        
        # Base penalty is much higher
        reward = self.reject_penalty
        
        # Extra penalty for rejecting when capacity is low (room to accept)
        if capacity < 4:
            reward -= 0.2
        
        # Penalty proportional to order value
        reward -= income * 0.1
        
        # Only small relief if severely overloaded
        if capacity >= 8:
            reward += 0.1
        
        return reward


class OpportunityCostReward(RewardCalculator):
    """
    Penalty for rejection proportional to what could have been earned.
    
    Use this when:
    - You want the agent to learn order value
    - High-value orders should almost never be rejected
    - Low-value orders can be rejected if courier is busy
    """
    
    def __init__(self):
        super().__init__(
            income_scale=0.5,           # Full income value
            reject_penalty=-0.1,
            late_penalty=-0.25,
            on_time_bonus=0.15,
            capacity_threshold=6,
            overload_penalty=-0.2,
            lost_order_penalty=-0.8
        )
    
    def compute_reject_reward(self, income: float, capacity: int, order_lost: bool = False) -> float:
        """Penalty = opportunity cost (what you could have earned)."""
        if order_lost:
            return self.lost_order_penalty
        
        # Opportunity cost: scaled down version of what accept would give
        opportunity_cost = -income * 0.3
        
        # Capacity relief: if overloaded, reduce penalty
        if capacity >= self.capacity_threshold:
            opportunity_cost += 0.15
        
        return opportunity_cost


class CapacityBalancedReward(RewardCalculator):
    """
    Rewards efficient capacity utilization.
    
    Use this when:
    - You want optimal capacity (not too high, not too low)
    - Penalize both over-rejection and over-acceptance
    - Balance throughput with quality
    """
    
    def __init__(self):
        super().__init__(
            income_scale=0.35,
            reject_penalty=-0.25,
            late_penalty=-0.35,
            on_time_bonus=0.2,
            capacity_threshold=5,       # Optimal around 5
            overload_penalty=-0.4,      # Strong penalty for overload
            lost_order_penalty=-0.6
        )
    
    def compute_accept_reward(self, income: float, is_late: bool, capacity: int) -> float:
        """Bonus for accepting when capacity is low, penalty when high."""
        reward = income * self.income_scale
        
        if is_late:
            reward += self.late_penalty
        else:
            reward += self.on_time_bonus
        
        # Capacity-based adjustment
        if capacity >= self.capacity_threshold:
            reward += self.overload_penalty
        elif capacity <= 2:
            # Bonus for accepting when nearly empty
            reward += 0.15
        
        return reward
    
    def compute_reject_reward(self, income: float, capacity: int, order_lost: bool = False) -> float:
        """Strong penalty for rejecting when capacity is low."""
        if order_lost:
            return self.lost_order_penalty
        
        reward = self.reject_penalty
        
        # Extra penalty for rejecting when capacity is low
        if capacity <= 3:
            reward -= 0.15
        
        # Relief if overloaded
        if capacity >= self.capacity_threshold:
            reward += 0.2
        
        return reward


class BalancedReward(RewardCalculator):
    """
    Moderate approach with balanced incentives.
    
    Use this when:
    - You want a reasonable middle ground
    - Not too aggressive on accepts, not too lenient on rejects
    """
    
    def __init__(self):
        super().__init__(
            income_scale=0.8,
            reject_penalty=-0.3,        # Moderate penalty
            late_penalty=-0.25,
            on_time_bonus=0.15,
            capacity_threshold=6,
            overload_penalty=-0.2,
            lost_order_penalty=-0.6
        )


class IncomeOnlyReward(RewardCalculator):
    """
    Pure income-based reward. Simplest possible signal.
    
    Accept: Get the order income
    Reject: Zero (neutral) or small penalty
    
    Use this when:
    - You want the agent to learn pure income maximization
    - Testing if complex rewards are hurting learning
    """
    
    def __init__(self):
        super().__init__(
            income_scale=1.0,           # Full income value
            reject_penalty=0.0,         # Neutral reject
            late_penalty=0.0,           # Ignore delivery timing
            on_time_bonus=0.0,
            capacity_threshold=10,      # Effectively disabled
            overload_penalty=0.0,
            lost_order_penalty=-0.5
        )
    
    def compute_accept_reward(self, income: float, is_late: bool, capacity: int) -> float:
        """Just the income."""
        return income * self.income_scale
    
    def compute_reject_reward(self, income: float, capacity: int, order_lost: bool = False) -> float:
        """Zero for reject (income foregone is implicit)."""
        if order_lost:
            return self.lost_order_penalty
        return 0.0


class RegretBasedReward(RewardCalculator):
    """
    Symmetric reward: accepting is as good as rejecting is bad.
    
    This creates a strong gradient for learning order value.
    
    Use this when:
    - Agent needs clear signal on order importance
    - You want balanced exploration of both actions
    """
    
    def __init__(self):
        super().__init__(
            income_scale=0.5,
            reject_penalty=-0.1,
            late_penalty=-0.2,
            on_time_bonus=0.1,
            capacity_threshold=6,
            overload_penalty=-0.2,
            lost_order_penalty=-1.0
        )
    
    def compute_accept_reward(self, income: float, is_late: bool, capacity: int) -> float:
        """Positive reward based on income."""
        reward = income * self.income_scale
        if is_late:
            reward += self.late_penalty
        else:
            reward += self.on_time_bonus
        if capacity >= self.capacity_threshold:
            reward += self.overload_penalty
        return reward
    
    def compute_reject_reward(self, income: float, capacity: int, order_lost: bool = False) -> float:
        """Regret = negative of what you would have earned."""
        if order_lost:
            return self.lost_order_penalty
        
        # Symmetric: rejecting costs you proportional to order value
        regret = -income * self.income_scale
        
        # Relief if overloaded (rejecting was reasonable)
        if capacity >= self.capacity_threshold:
            regret += 0.2
        
        return regret


class UrgencyAwareReward(RewardCalculator):
    """
    Penalizes rejecting urgent/time-critical orders more heavily.
    
    Use this when:
    - On-time delivery rate is critical
    - Urgent orders should almost never be rejected
    """
    
    def __init__(self):
        super().__init__(
            income_scale=0.4,
            reject_penalty=-0.2,
            late_penalty=-0.4,          # Strong late penalty
            on_time_bonus=0.2,          # Good on-time bonus
            capacity_threshold=6,
            overload_penalty=-0.2,
            lost_order_penalty=-0.8
        )
    
    def compute_accept_reward(self, income: float, is_late: bool, capacity: int) -> float:
        """Standard accept reward with strong timing incentives."""
        reward = income * self.income_scale
        if is_late:
            reward += self.late_penalty
        else:
            reward += self.on_time_bonus
        if capacity >= self.capacity_threshold:
            reward += self.overload_penalty
        return reward
    
    def compute_reject_reward(self, income: float, capacity: int, order_lost: bool = False) -> float:
        """Extra penalty for rejecting - agent should accept more."""
        if order_lost:
            return self.lost_order_penalty
        
        reward = self.reject_penalty
        
        # Penalty proportional to value (don't reject valuable orders)
        reward -= income * 0.15
        
        # Only relief if severely overloaded
        if capacity >= 7:
            reward += 0.15
        
        return reward


class MaxAcceptReward(RewardCalculator):
    """
    Extreme: Always accept unless severely overloaded.
    
    Very strong accept incentive, minimal reject allowance.
    
    Use this when:
    - You want near-100% acceptance rate
    - Testing upper bound of acceptance
    """
    
    def __init__(self):
        super().__init__(
            income_scale=0.5,
            reject_penalty=-1.0,        # Very strong penalty
            late_penalty=-0.1,          # Mild late penalty
            on_time_bonus=0.05,
            capacity_threshold=8,       # High threshold
            overload_penalty=-0.1,
            lost_order_penalty=-2.0
        )
    
    def compute_accept_reward(self, income: float, is_late: bool, capacity: int) -> float:
        """Generous accept rewards."""
        reward = income * self.income_scale + 0.2  # Base bonus for accepting
        if is_late:
            reward += self.late_penalty
        else:
            reward += self.on_time_bonus
        if capacity >= self.capacity_threshold:
            reward += self.overload_penalty
        return reward
    
    def compute_reject_reward(self, income: float, capacity: int, order_lost: bool = False) -> float:
        """Severe reject penalty unless truly overloaded."""
        if order_lost:
            return self.lost_order_penalty
        
        # Only allow rejection if capacity >= 7
        if capacity >= 7:
            return -0.1  # Small penalty, acceptable to reject
        else:
            return self.reject_penalty  # -1.0, very bad


class UltraAggressiveReward(RewardCalculator):
    """
    ULTRA AGGRESSIVE: Minimize rejections, late deliveries, AND lost orders.
    
    This is the most aggressive reward function:
    - Massive reject penalty (-2.0)
    - Strong late delivery penalty (-0.5)
    - Huge lost order penalty (-3.0)
    - Big on-time bonus (+0.3)
    - Income-proportional reject penalty (don't reject valuable orders)
    - Capacity awareness (harsh penalty for rejecting when underutilized)
    
    Use this when:
    - You want absolute minimum rejections
    - Late deliveries and lost orders are equally unacceptable
    - Testing the extreme upper bound of acceptance + quality
    """
    
    def __init__(self):
        super().__init__(
            income_scale=0.6,           # Higher income weight
            reject_penalty=-2.0,        # MASSIVE reject penalty
            late_penalty=-0.5,          # Strong late penalty
            on_time_bonus=0.3,          # Big on-time bonus
            capacity_threshold=7,
            overload_penalty=-0.15,
            lost_order_penalty=-3.0     # HUGE lost order penalty
        )
    
    def compute_accept_reward(self, income: float, is_late: bool, capacity: int) -> float:
        """
        Generous accept rewards with strong timing incentives.
        Accept = good, but MUST deliver on time.
        """
        # Base reward: income + acceptance bonus
        reward = income * self.income_scale + 0.25  # Bonus for accepting
        
        # Strong timing incentives
        if is_late:
            reward += self.late_penalty  # -0.5 for late
            # Extra penalty proportional to how valuable the order was
            reward -= income * 0.1  # Wasted opportunity if late
        else:
            reward += self.on_time_bonus  # +0.3 for on-time
            # Extra bonus for on-time delivery of valuable orders
            reward += income * 0.05
        
        # Mild capacity penalty (still want to accept even when busy)
        if capacity >= self.capacity_threshold:
            reward += self.overload_penalty
        elif capacity <= 2:
            # Bonus for keeping courier active
            reward += 0.1
        
        return reward
    
    def compute_reject_reward(self, income: float, capacity: int, order_lost: bool = False) -> float:
        """
        BRUTAL reject penalties.
        Rejecting is almost never acceptable.
        """
        if order_lost:
            return self.lost_order_penalty  # -3.0, catastrophic
        
        # Base reject penalty
        reward = self.reject_penalty  # -2.0
        
        # Income-proportional penalty (don't reject valuable orders!)
        reward -= income * 0.3
        
        # Capacity-based penalty/relief
        if capacity <= 2:
            # Rejecting when nearly empty is TERRIBLE
            reward -= 0.5
        elif capacity <= 4:
            # Rejecting when underutilized is bad
            reward -= 0.25
        elif capacity >= 8:
            # Only acceptable to reject when severely overloaded
            reward += 1.0  # Reduces penalty to -1.0
        elif capacity >= 7:
            reward += 0.5  # Reduces penalty to -1.5
        
        return reward


class PaperReward(RewardCalculator):
    """
    EXACT PAPER REWARD: Matches Equation 1 from the paper.
    
    From the paper (Section 2, Equation 1):
        r(s, a) = order_income   if a = 1 (accept)
                = -0.3           if a = 0 (reject)
    
    This is the simplest possible reward function:
    - No late penalty
    - No on-time bonus  
    - No capacity considerations
    - No overload penalty
    - Pure income for accept, fixed penalty for reject
    
    Use this when:
    - Reproducing paper results exactly
    - Testing baseline behavior without reward shaping
    - Comparing against other reward variants
    """
    
    def __init__(self):
        super().__init__(
            income_scale=1.0,           # Full income value (no scaling)
            reject_penalty=-0.3,        # Paper's exact reject penalty
            late_penalty=0.0,           # No late penalty
            on_time_bonus=0.0,          # No on-time bonus
            capacity_threshold=100,     # Effectively disabled
            overload_penalty=0.0,       # No overload penalty
            lost_order_penalty=-0.3     # Same as reject penalty (order lost = rejected)
        )
    
    def compute_accept_reward(self, income: float, is_late: bool, capacity: int) -> float:
        """
        Paper equation: r(s, a=1) = order_income
        
        Pure income signal, no timing or capacity adjustments.
        """
        return income * self.income_scale
    
    def compute_reject_reward(self, income: float, capacity: int, order_lost: bool = False) -> float:
        """
        Paper equation: r(s, a=0) = -0.3
        
        Fixed penalty regardless of order value or capacity.
        """
        # Paper doesn't distinguish lost orders, but we use same penalty
        return self.reject_penalty  # -0.3


class DataDrivenReward(RewardCalculator):
    """
    DATA-DRIVEN REWARD: Calibrated mathematically from historical Meituan percentiles.
    
    Tuning Evidence:
    - Median Income: 3.15 CNY.
    - Reject Rates: ~13% naturally. Penalty set to roughly 40% of median income (-1.3).
    - Capacity Limits: 75% of accepted orders happen at ≤3 capacity, 95% at ≤5. Threshold set to 3.
    - Lateness Limits: 90th percentile miss ETA by +2 mins, 95th by +5 mins. Penalty set at ~10-15% of value.
    - Anomaly limits: Lost orders break all limits, scaled aggressively.
    """
    
    def __init__(self):
        super().__init__(
            income_scale=1.0,           # Full realistic CNY limits
            reject_penalty=-1.3,        # ~41% of median 3.15 income
            late_penalty=-0.4,          # ~12% of median income (for natural variance bounds)
            on_time_bonus=0.15,         # Small nudge reward for beating tight SLAs
            capacity_threshold=3,       # 75th percentile of observed concurrent load
            overload_penalty=-0.4,      # Escalating penalty for breaching human batched load standard (4+)
            lost_order_penalty=-4.0     # Severely out-scales rejection (-1.3)
        )
    
    def compute_accept_reward(self, income: float, is_late: bool, capacity: int) -> float:
        """Data-informed positive flow limits."""
        reward = income * self.income_scale
        
        # Applying empirical time adjustments
        if is_late:
            reward += self.late_penalty
            # Extra penalty for missing a highly valuable order
            if income > 4.5:
                reward -= 0.1
        else:
            reward += self.on_time_bonus
            
        # Overload check (capacity above historical 75th percentile of 3)
        if capacity > self.capacity_threshold:
            # Scaled overload penalty: hitting 95th percentile (5) hits exactly -0.8
            overages = capacity - self.capacity_threshold
            reward += (self.overload_penalty * overages)
            
        return reward
    
    def compute_reject_reward(self, income: float, capacity: int, order_lost: bool = False) -> float:
        """Data-informed negative flow limits."""
        if order_lost:
            return self.lost_order_penalty
        
        reward = self.reject_penalty # Base -1.3
        
        # Penalize rejecting valuable properties (top 5% incomes)
        if income > 4.15:
            reward -= (income - 4.15) * 0.2
            
        # Relief: If rejecting because they are at 90th percentile human capacity limit (4+)
        if capacity >= 4:
            reward += 0.5  # Makes rejecting only cost -0.8
            
        return reward


def get_reward_calculator(reward_type: str) -> RewardCalculator:
    """
    Factory function to get reward calculator by type.
    
    Available types:
    - 'baseline': Original reward function (weak reject penalty)
    - 'aggressive_accept': Strong reject penalties to maximize accepts
    - 'opportunity_cost': Penalty proportional to order value
    - 'capacity_balanced': Optimal capacity utilization focus
    - 'balanced': Moderate middle-ground approach
    - 'income_only': Pure income signal, no shaping
    - 'regret': Symmetric rewards (accept good = reject bad)
    - 'urgency': Extra penalty for rejecting urgent orders
    - 'max_accept': Extreme accept incentive (near-100% target)
    - 'ultra_aggressive': MAXIMUM penalties for reject/late/lost (most aggressive)
    - 'paper': EXACT paper reward (Equation 1): income if accept, -0.3 if reject
    - 'data_driven': Reward modeled off empirical quantiles of dataset behaviors.
    
    Args:
        reward_type: One of the available types
        
    Returns:
        RewardCalculator instance
    """
    calculators = {
        'baseline': RewardCalculator,
        'aggressive_accept': AggressiveAcceptReward,
        'opportunity_cost': OpportunityCostReward,
        'capacity_balanced': CapacityBalancedReward,
        'balanced': BalancedReward,
        'income_only': IncomeOnlyReward,
        'regret': RegretBasedReward,
        'urgency': UrgencyAwareReward,
        'max_accept': MaxAcceptReward,
        'ultra_aggressive': UltraAggressiveReward,
        'paper': PaperReward,
        'data_driven': DataDrivenReward,
    }
    
    if reward_type not in calculators:
        available = list(calculators.keys())
        raise ValueError(f"Unknown reward type: {reward_type}. Available: {available}")
    
    return calculators[reward_type]()


# ============================================================================
# Simulator Wrapper for DDQN Training
# ============================================================================

class DDQNTrainingSimulator:
    """
    Wraps the full Simulator to collect DDQN transitions during simulation.
    
    This class:
    1. Runs the full Simulator with a TrainingDDQNAgent
    2. Hooks into simulation events to collect (s, a, r, s') transitions
    3. Computes rewards based on actual simulation outcomes
    4. Triggers DDQN updates during simulation
    
    The Simulator runs with all its dynamics:
    - Location interpolation (grab_time → fetch_time → arrive_time)
    - Capacity tracking with increase and decrease
    - Order reassignment
    - Divergence handling
    
    Supports hour ranges for train/eval split:
    - start_hour: Skip events before this hour
    - end_hour: Stop after this hour (max_hours from start_hour)
    """
    
    def __init__(
        self,
        parquet_path: str,
        manifest_path: str,
        max_hours: Optional[int] = None,
        start_hour: int = 0,
        travel_time_model_path: Optional[str] = None,
        output_dir: str = 'results',
        verbose: bool = False,
        bandit_mode: bool = False,
    ):
        self.parquet_path = parquet_path
        self.manifest_path = manifest_path
        self.max_hours = max_hours
        self.start_hour = start_hour  # Skip first N hours (for eval split)
        self.travel_time_model_path = travel_time_model_path
        self.output_dir = output_dir
        self.verbose = verbose
        self.bandit_mode = bandit_mode  # If True: done=True for all transitions (contextual bandit)
        
        # Load manifest for feature names
        with open(manifest_path, 'r') as f:
            self.manifest = json.load(f)
        self.feature_names = self.manifest['feature_order']
        
        # Reward calculator
        self.reward_calc = RewardCalculator()
        
        # Transition collection
        self.transitions: List[Tuple] = []
        self.episode_rewards: List[float] = []
    
    def run_training_episode(
        self,
        agent: TrainingDDQNAgent,
        updates_per_event: int = 1
    ) -> Dict:
        """
        Run one full simulation episode with DDQN learning.
        
        Returns episode statistics.
        """
        from simulation.simulator import Simulator
        
        # Create simulator with DDQN agent
        # We use a wrapper that intercepts events to collect transitions
        simulator = self._create_instrumented_simulator(agent)
        
        # Run simulation
        simulator.setup()
        
        # Clear episode tracking
        self.transitions = []
        self.episode_rewards = []
        agent.pending_transitions.clear()
        
        # Run with transition collection
        self._run_with_transition_collection(simulator, agent, updates_per_event)
        
        # Episode stats
        stats = {
            'total_transitions': len(self.transitions),
            'total_reward': sum(self.episode_rewards),
            'mean_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0.0,
            'agent_metrics': agent.get_metrics(),
            'simulator_metrics': self._extract_simulator_metrics(simulator)
        }
        
        return stats
    
    def _create_instrumented_simulator(self, agent: TrainingDDQNAgent):
        """Create simulator with DDQN agent pre-injected."""
        from simulation.simulator import Simulator
        
        simulator = Simulator(
            agent_ckpt='',  # Not used, we inject agent directly
            scaler_json='',
            thresholds_json='',
            threshold_name='',
            parquet_path=self.parquet_path,
            manifest_path=self.manifest_path,
            output_dir=self.output_dir,
            baseline_mode=False,
            agent_type='ddqn',
            max_hours=self.max_hours,
            start_hour=self.start_hour,
            verbose=self.verbose,
            travel_time_model_path=self.travel_time_model_path
        )
        
        # Inject our training agent BEFORE setup() is called
        # This prevents setup() from trying to load from disk
        simulator.agent = agent
        
        return simulator
    
    def _build_courier_event_index(self, events, cycles):
        """
        Pre-build per-courier chronological event index for proper MDP next-state.

        Returns:
            courier_events: dict[int, list[tuple(cycle_idx, event_idx, event)]]
                Each courier's events in chronological order across all cycles.
            event_to_courier_pos: dict[(cycle_idx, event_idx), int]
                Maps (cycle_idx, event_idx_in_cycle) → position in that courier's list.
        """
        from collections import defaultdict
        courier_events = defaultdict(list)

        for cycle_idx, cycle_info in enumerate(cycles):
            for ev_idx, event in enumerate(cycle_info['events']):
                courier_events[event.courier_id].append((cycle_idx, ev_idx, event))

        # Already sorted because cycles are sorted and events within cycles are
        # sorted by offer_index_in_cycle (both chronological).

        # Build reverse index: (cycle_idx, local_ev_idx, courier_id) → position
        event_to_courier_pos = {}
        for cid, ev_list in courier_events.items():
            for pos, (cyc_idx, ev_idx, _) in enumerate(ev_list):
                event_to_courier_pos[(cyc_idx, ev_idx, cid)] = pos

        return dict(courier_events), event_to_courier_pos

    def _get_next_state_for_courier(
        self,
        agent,
        simulator,
        courier_id: int,
        courier_pos: int,
        courier_events: dict,
    ):
        """
        Get the next state for a courier, accounting for divergence.

        If the courier has diverged from the historical trajectory, rebuild
        features from live CourierState (fresh proxy location, capacity, etc.).
        Otherwise, use the parquet features from their next historical event.

        Returns:
            (next_state_norm, done)
        """
        ev_list = courier_events.get(courier_id, [])
        next_pos = courier_pos + 1

        if next_pos >= len(ev_list):
            # Courier has no more events → terminal state
            return None, True

        _, _, next_event = ev_list[next_pos]

        # Check if courier has diverged; if so, rebuild features from live state
        courier_state = simulator.courier_states.get(courier_id)
        is_diverged = (courier_state is not None and courier_state.is_diverged)

        if is_diverged:
            # Build fresh features: order info from the next event's parquet data,
            # but courier info (location, capacity, etc.) from live CourierState.
            fresh_features = simulator._build_features_for_courier(
                next_event, courier_id, next_event.actual_dispatch_time
            )
            next_state = agent._encode_features(fresh_features)
        else:
            next_state = agent._encode_features(next_event.features)

        next_state_norm = agent._normalize(next_state)
        return next_state_norm, False

    def _run_with_transition_collection(
        self,
        simulator,
        agent: TrainingDDQNAgent,
        updates_per_event: int
    ):
        """
        Run simulation while collecting (s, a, r, s') transitions.

        MDP framing (FIX for Issue #1 from audit):
        - next_state is the SAME courier's next offer event (proper per-courier MDP)
        - When a courier has diverged, next_state features are rebuilt from live
          CourierState (fresh location, capacity, late ratio, etc.)
        - done=True when the courier has no more events (shift over)

        Bandit mode (--bandit-mode flag):
        - done=True for ALL transitions; no bootstrapping.
        - Q(s,a) ≈ r(s,a); simplest correct baseline.

        Reward (FIX for Issue #3 from audit):
        - Reward is computed based on the AGENT'S action, not the system outcome.
        - If agent accepts → accept reward; if agent rejects → reject reward.
        """
        from datetime import datetime
        import time

        events = simulator.dispatcher.get_events()
        simulator._build_courier_dispatch_schedule(events)
        cycles = simulator._build_cycles(events)

        print(f"Running training episode: {len(cycles)} cycles, {len(events)} events")

        # ── Pre-build per-courier event index ──
        if not self.bandit_mode:
            courier_events, event_to_courier_pos = self._build_courier_event_index(
                events, cycles
            )
            mode_label = "per-courier MDP"
        else:
            courier_events, event_to_courier_pos = {}, {}
            mode_label = "bandit (done=True always)"
        print(f"  Transition mode: {mode_label}")

        pbar = tqdm(total=len(events), desc="Training episode")

        for cycle_id, cycle_info in enumerate(cycles):
            simulator.current_cycle_id = cycle_id
            simulator.current_cycle_start_time = cycle_info['start_time']
            simulator.current_cycle_end_time = cycle_info['end_time']

            # Update diverged couriers for this time
            simulator._update_diverged_couriers_for_time(cycle_info['start_time'])

            # Process pending reassignments (for learning from reassignment outcomes)
            if simulator.pending_reassignments:
                simulator._process_pending_reassignments(cycle_info)

            # Process events
            for ev_idx, event in enumerate(cycle_info['events']):
                # Skip already handled orders (by waybill_id or order_id)
                if event.waybill_id in simulator.order_tracker.completed_orders or \
                   event.waybill_id in simulator.order_tracker.active_orders or \
                   event.order_id in simulator.order_tracker.completed_order_ids or \
                   event.order_id in simulator.order_tracker.accepted_order_ids:
                    pbar.update(1)
                    continue

                # Capture state BEFORE decision
                state_before = agent._encode_features(event.features)
                agent._update_running_stats(state_before)
                state_norm = agent._normalize(state_before)

                # Get courier state for capacity
                courier_state = simulator.courier_states.get(event.courier_id)
                capacity_before = courier_state.capacity_observations if courier_state else 0

                # Process event (agent makes decision inside)
                simulator._process_event(event)

                # Get what action was taken
                action = agent.last_action

                # ── Compute reward based on AGENT'S action (Fix #3) ──
                income = float(event.order_data.get('order_income_value', 0.0))

                if action == 1:
                    # Agent chose ACCEPT — use accept reward
                    # Determine late status from the delivery outcome
                    waybill_id = event.waybill_id
                    order_state = simulator.order_tracker.orders.get(waybill_id)
                    is_late = False
                    if order_state and hasattr(order_state, 'is_late'):
                        is_late = order_state.is_late
                    # Cross-check from courier state if available
                    if courier_state and waybill_id in getattr(courier_state, 'delivered_waybills', set()):
                        is_late = waybill_id in getattr(courier_state, 'late_waybills', set())

                    reward = self.reward_calc.compute_accept_reward(
                        income=income,
                        is_late=is_late,
                        capacity=capacity_before
                    )
                else:
                    # Agent chose REJECT — always give reject reward
                    # order_lost only if the order couldn't be reassigned at all
                    waybill_id = event.waybill_id
                    order_lost = waybill_id in simulator.pending_reassignments
                    reward = self.reward_calc.compute_reject_reward(
                        income=income,
                        capacity=capacity_before,
                        order_lost=order_lost
                    )

                # ── Get next state (Fix #1: per-courier or bandit) ──
                if self.bandit_mode:
                    # Bandit: every transition is terminal
                    next_state_norm = state_norm.copy()
                    done = True
                else:
                    # Per-courier MDP: find this courier's next event
                    cpos_key = (cycle_id, ev_idx, event.courier_id)
                    courier_pos = event_to_courier_pos.get(cpos_key, -1)

                    if courier_pos >= 0:
                        ns, done = self._get_next_state_for_courier(
                            agent, simulator,
                            event.courier_id, courier_pos, courier_events,
                        )
                        if ns is not None:
                            next_state_norm = ns
                        else:
                            next_state_norm = state_norm.copy()
                            done = True
                    else:
                        # Fallback: treat as terminal
                        next_state_norm = state_norm.copy()
                        done = True

                # Store transition
                agent.store_transition(
                    state=state_norm,
                    action=action,
                    reward=reward,
                    next_state=next_state_norm,
                    done=done
                )
                self.episode_rewards.append(reward)

                # Perform DDQN updates
                for _ in range(updates_per_event):
                    agent.update()

                pbar.update(1)

                # Update progress
                if agent.steps % 500 == 0:
                    metrics = agent.get_metrics()
                    pbar.set_postfix({
                        'eps': f"{metrics['epsilon']:.3f}",
                        'loss': f"{metrics['mean_bellman_loss']:.4f}",
                        'Q': f"{metrics['mean_q_value']:.2f}",
                        'reward': f"{metrics['mean_reward']:.3f}"
                    })

        pbar.close()

        # Finalize pending orders
        simulator._finalize_pending_orders()
    
    def _extract_simulator_metrics(self, simulator) -> Dict:
        """Extract key metrics from simulator including utilization."""
        metrics = simulator.metrics
        system = metrics.system  # SystemMetrics dataclass
        
        # Get utilization summary from the utilization tracker
        utilization_summary = simulator.utilization_tracker.get_summary()
        
        # Order-tracker unique-order-id counts (ground truth for order-level metrics)
        ot = simulator.order_tracker
        unique_accepted = len(ot.accepted_order_ids)
        unique_lost = len(ot.not_assigned_order_ids)
        unique_delivered = len(ot.delivered_order_ids)
        unique_total = len(set(o.order_id for o in ot.orders.values()))
        
        return {
            # Waybill-event-level counts (from metrics.py)
            'total_orders': system.total_orders,
            'total_accepts': system.total_accepts,
            'total_rejects': system.total_rejects,
            'total_late_deliveries': system.total_late_deliveries,
            'total_on_time_deliveries': system.total_on_time_deliveries,
            'total_lost_orders': system.total_lost_orders,
            'divergences': len(metrics.divergences),
            'aligned': metrics.aligned_events,
            # Order-ID-level counts (from order_tracker)
            'unique_orders': unique_total,
            'unique_accepted': unique_accepted,
            'unique_lost': unique_lost,
            'unique_delivered': unique_delivered,
            # Utilization metrics
            'courier_utilization': utilization_summary,
        }


# ============================================================================
# Training Loop
# ============================================================================

def train_ddqn_integrated(
    parquet_path: str,
    manifest_path: str,
    out_dir: str,
    episodes: int = 10,
    train_hours: int = 48,
    eval_hours: int = 48,
    lr: float = 3e-4,
    gamma: float = 0.99,
    epsilon_start: float = 0.3,
    epsilon_end: float = 0.01,
    batch_size: int = 256,
    updates_per_event: int = 2,
    device: str = 'auto',
    save_every: int = 1,
    travel_time_model: Optional[str] = None,
    reward_type: str = 'balanced',
    resume_from: Optional[str] = None,
    bandit_mode: bool = False,
):
    """
    Train DDQN agent integrated with full simulator.
    
    Supports:
    - Train/eval split: train on first N hours, evaluate on next M hours
    - Resume from checkpoint if training is interrupted
    - CSV logging for loss/Q-value monitoring
    - Full checkpoint saving after each epoch
    - Per-courier MDP transitions (default) or contextual bandit mode
    
    Args:
        train_hours: Hours of data to use for training (default: 48)
        eval_hours: Hours of data to use for evaluation (default: 48, starts after train_hours)
        resume_from: Path to checkpoint to resume from (optional)
        reward_type: Reward function to use
    """
    import csv
    
    os.makedirs(out_dir, exist_ok=True)
    
    # Get reward calculator
    reward_calc = get_reward_calculator(reward_type)
    
    print("\n" + "="*70)
    print("DDQN TRAINING - INTEGRATED WITH FULL SIMULATOR")
    print("="*70)
    print(f"Episodes (epochs): {episodes}")
    print(f"Training hours: 0-{train_hours} (first {train_hours}h)")
    print(f"Evaluation hours: {train_hours}-{train_hours + eval_hours} (next {eval_hours}h)")
    print(f"Learning rate: {lr}")
    print(f"Gamma: {gamma}")
    print(f"Epsilon: {epsilon_start} → {epsilon_end}")
    print(f"Batch size: {batch_size}")
    print(f"Updates per event: {updates_per_event}")
    print(f"Reward type: {reward_type}")
    print(f"  - Reject penalty: {reward_calc.reject_penalty}")
    print(f"  - Income scale: {reward_calc.income_scale}")
    print(f"Bandit mode: {bandit_mode}")
    print(f"Resume from: {resume_from or 'None (fresh start)'}")
    print(f"Output: {out_dir}")
    print("="*70 + "\n")
    
    # Load manifest
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    feature_names = manifest['feature_order']
    
    print(f"[1/3] Loaded {len(feature_names)} features")
    
    # Create training agent
    print(f"[2/3] Initializing DDQN agent...")
    agent = TrainingDDQNAgent(
        feature_names=feature_names,
        lr=lr,
        gamma=gamma,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        batch_size=batch_size,
        device=device,
    )
    print(f"  Q-network on device: {agent.device}")
    # Resume from checkpoint if specified
    start_epoch = 0
    all_stats = []
    training_log = []
    
    if resume_from and os.path.exists(resume_from):
        print(f"\n[RESUME] Loading checkpoint: {resume_from}")
        start_epoch, all_stats, training_log = agent.load_full_checkpoint(resume_from)
        print(f"  Resuming from epoch {start_epoch + 1}")
    
    # Setup CSV logging
    csv_path = os.path.join(out_dir, 'training_log.csv')
    csv_exists = os.path.exists(csv_path) and resume_from
    csv_file = open(csv_path, 'a' if csv_exists else 'w', newline='')
    csv_writer = csv.writer(csv_file)
    if not csv_exists:
        csv_writer.writerow([
            'epoch', 'step', 'update', 'epsilon', 'mean_loss', 'mean_q_value', 
            'mean_reward', 'total_reward', 'acceptance_rate',
            'unique_accepted', 'unique_lost', 'unique_orders',
            'late_deliveries', 'lost_orders', 'timestamp'
        ])
        csv_file.flush()
    
    # Create TRAINING simulator (first N hours)
    print(f"[3/3] Creating training simulator (hours 0-{train_hours})...")
    training_sim = DDQNTrainingSimulator(
        parquet_path=parquet_path,
        manifest_path=manifest_path,
        max_hours=train_hours,
        start_hour=0,
        travel_time_model_path=travel_time_model,
        output_dir=out_dir,
        verbose=False,
        bandit_mode=bandit_mode
    )
    
    # Training loop
    print("\n" + "-"*70)
    print("Starting training...")
    print("-"*70)
    
    for ep in range(start_epoch, episodes):
        print(f"\n{'='*70}")
        print(f"EPOCH {ep + 1}/{episodes}")
        print(f"{'='*70}")
        
        # Run training episode
        stats = training_sim.run_training_episode(
            agent=agent,
            updates_per_event=updates_per_event
        )
        all_stats.append(stats)
        
        # Get metrics
        agent_metrics = stats['agent_metrics']
        sim_metrics = stats['simulator_metrics']
        # Order-level acceptance rate: unique accepted / (unique accepted + unique lost)
        unique_accepted = sim_metrics.get('unique_accepted', sim_metrics['total_accepts'])
        unique_lost = sim_metrics.get('unique_lost', sim_metrics['total_lost_orders'])
        unique_orders = sim_metrics.get('unique_orders', sim_metrics['total_orders'])
        unique_delivered = sim_metrics.get('unique_delivered', 0)
        order_acceptance_rate = unique_accepted / max(1, unique_accepted + unique_lost)
        
        # Log to CSV
        log_entry = {
            'epoch': ep + 1,
            'step': agent_metrics['steps'],
            'update': agent_metrics['updates'],
            'epsilon': agent_metrics['epsilon'],
            'mean_loss': agent_metrics['mean_bellman_loss'],
            'mean_q_value': agent_metrics['mean_q_value'],
            'mean_reward': stats['mean_reward'],
            'total_reward': stats['total_reward'],
            'acceptance_rate': order_acceptance_rate,
            'unique_accepted': unique_accepted,
            'unique_lost': unique_lost,
            'unique_orders': unique_orders,
            'late_deliveries': sim_metrics['total_late_deliveries'],
            'lost_orders': sim_metrics['total_lost_orders'],
            'timestamp': datetime.now().isoformat()
        }
        training_log.append(log_entry)
        
        csv_writer.writerow([
            log_entry['epoch'], log_entry['step'], log_entry['update'],
            f"{log_entry['epsilon']:.4f}", f"{log_entry['mean_loss']:.6f}",
            f"{log_entry['mean_q_value']:.4f}", f"{log_entry['mean_reward']:.4f}",
            f"{log_entry['total_reward']:.2f}", f"{log_entry['acceptance_rate']:.4f}",
            log_entry['unique_accepted'], log_entry['unique_lost'],
            log_entry['unique_orders'],
            log_entry['late_deliveries'], log_entry['lost_orders'],
            log_entry['timestamp']
        ])
        csv_file.flush()
        
        # Print epoch summary
        print(f"\nEpoch {ep + 1} Summary:")
        print(f"  Total reward: {stats['total_reward']:.2f}")
        print(f"  Mean reward: {stats['mean_reward']:.4f}")
        print(f"  Mean Bellman loss: {agent_metrics['mean_bellman_loss']:.6f}")
        print(f"  Mean Q-value: {agent_metrics['mean_q_value']:.4f}")
        print(f"  Agent steps: {agent_metrics['steps']:,}")
        print(f"  DDQN updates: {agent_metrics['updates']:,}")
        print(f"  Epsilon: {agent_metrics['epsilon']:.4f}")
        print(f"  Buffer size: {agent_metrics['buffer_size']:,}")
        print(f"  Orders accepted (unique): {unique_accepted}/{unique_accepted + unique_lost} ({order_acceptance_rate:.1%})")
        print(f"  Unique orders: {unique_orders}  |  Delivered: {unique_delivered}  |  Lost: {unique_lost}")
        print(f"  Late deliveries: {sim_metrics['total_late_deliveries']}")
        print(f"  Divergences: {sim_metrics['divergences']}")
        
        # Save FULL checkpoint (for resume)
        if (ep + 1) % save_every == 0:
            full_ckpt_path = os.path.join(out_dir, f'checkpoint_ep{ep + 1}.pt')
            agent.save_full_checkpoint(full_ckpt_path, ep + 1, all_stats, training_log)
            
            # Also save lightweight inference checkpoint
            ckpt_path = os.path.join(out_dir, f'ddqn_integrated_ep{ep + 1}.pt')
            agent.save(ckpt_path)
    
    csv_file.close()
    
    # Save final model
    print("\n" + "="*70)
    print("Training complete!")
    print("="*70)
    
    final_path = os.path.join(out_dir, 'ddqn_model.pt')
    agent.save(final_path)
    print(f"Final model saved: {final_path}")
    
    # Save scaler
    std = agent._get_std()
    scaler = {
        'mean': agent.running_mean.tolist(),
        'std': std.tolist(),
        'feature_order': agent.feature_names
    }
    scaler_path = os.path.join(out_dir, 'ddqn_scaler.json')
    with open(scaler_path, 'w') as f:
        json.dump(scaler, f, indent=2)
    print(f"Scaler saved: {scaler_path}")
    
    # Save training metrics
    training_metrics = {
        'episodes': episodes,
        'train_hours': train_hours,
        'eval_hours': eval_hours,
        'total_steps': agent.steps,
        'total_updates': agent.updates,
        'final_epsilon': agent.epsilon,
        'reward_type': reward_type,
        'episode_stats': [
            {
                'episode': i + 1,
                'total_reward': s['total_reward'],
                'mean_reward': s['mean_reward'],
                'transitions': s['total_transitions'],
                'unique_accepted': s['simulator_metrics'].get('unique_accepted', s['simulator_metrics']['total_accepts']),
                'unique_lost': s['simulator_metrics'].get('unique_lost', s['simulator_metrics']['total_lost_orders']),
                'unique_orders': s['simulator_metrics'].get('unique_orders', s['simulator_metrics']['total_orders']),
                'late_deliveries': s['simulator_metrics']['total_late_deliveries'],
            }
            for i, s in enumerate(all_stats)
        ]
    }
    metrics_path = os.path.join(out_dir, 'ddqn_integrated_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(training_metrics, f, indent=2)
    print(f"Metrics saved: {metrics_path}")
    print(f"Training log CSV: {csv_path}")
    
    # ========================================================================
    # EVALUATION PHASE
    # ========================================================================
    print("\n" + "="*70)
    print("EVALUATION PHASE")
    print(f"Running on held-out data: hours {train_hours}-{train_hours + eval_hours}")
    print("="*70)
    
    # Disable exploration for evaluation
    agent.explore = False
    
    # Create EVALUATION simulator (next M hours after training data)
    eval_sim = DDQNTrainingSimulator(
        parquet_path=parquet_path,
        manifest_path=manifest_path,
        max_hours=eval_hours,
        start_hour=train_hours,  # Start AFTER training data
        travel_time_model_path=travel_time_model,
        output_dir=out_dir,
        verbose=False,
        bandit_mode=bandit_mode
    )
    
    # Run evaluation (no DDQN updates)
    eval_stats = eval_sim.run_training_episode(
        agent=agent,
        updates_per_event=0  # No learning during evaluation
    )
    
    eval_metrics = eval_stats['simulator_metrics']
    eval_unique_accepted = eval_metrics.get('unique_accepted', eval_metrics['total_accepts'])
    eval_unique_lost = eval_metrics.get('unique_lost', eval_metrics['total_lost_orders'])
    eval_unique_orders = eval_metrics.get('unique_orders', eval_metrics['total_orders'])
    eval_unique_delivered = eval_metrics.get('unique_delivered', 0)
    eval_acceptance = eval_unique_accepted / max(1, eval_unique_accepted + eval_unique_lost)
    
    # Extract utilization metrics
    util_metrics = eval_metrics.get('courier_utilization', {})
    overall_utilization = util_metrics.get('overall_utilization', 0.0)
    
    print(f"\nEvaluation Results (hours {train_hours}-{train_hours + eval_hours}):")
    print(f"  Orders accepted (unique): {eval_unique_accepted}/{eval_unique_accepted + eval_unique_lost} ({eval_acceptance:.1%})")
    print(f"  Unique orders: {eval_unique_orders}  |  Delivered: {eval_unique_delivered}  |  Lost: {eval_unique_lost}")
    print(f"  Late deliveries: {eval_metrics['total_late_deliveries']}")
    print(f"  Mean reward: {eval_stats['mean_reward']:.4f}")
    print(f"  Total reward: {eval_stats['total_reward']:.2f}")
    print(f"  Courier utilization: {overall_utilization:.2%}")
    
    # Save evaluation results
    eval_results = {
        'train_hours': train_hours,
        'eval_hours': eval_hours,
        'eval_start_hour': train_hours,
        'unique_orders': eval_unique_orders,
        'unique_accepted': eval_unique_accepted,
        'unique_lost': eval_unique_lost,
        'unique_delivered': eval_unique_delivered,
        'acceptance_rate': eval_acceptance,
        'late_deliveries': eval_metrics['total_late_deliveries'],
        'mean_reward': eval_stats['mean_reward'],
        'total_reward': eval_stats['total_reward'],
        'courier_utilization': util_metrics,
    }
    eval_path = os.path.join(out_dir, 'evaluation_results.json')
    with open(eval_path, 'w') as f:
        json.dump(eval_results, f, indent=2)
    print(f"Evaluation results saved: {eval_path}")
    
    # Summary
    print("\n" + "-"*70)
    print("Training Summary:")
    print(f"  Total steps: {agent.steps:,}")
    print(f"  Total DDQN updates: {agent.updates:,}")
    print(f"  Final epsilon: {agent.epsilon:.4f}")
    print(f"  Mean training reward: {np.mean([s['total_reward'] for s in all_stats]):.2f}")
    print(f"  Evaluation acceptance: {eval_acceptance:.1%}")
    print("-"*70)
    
    return agent


def main():
    parser = argparse.ArgumentParser(description='DDQN Training Integrated with Full Simulator')
    parser.add_argument('--parquet', type=str,
                        default='data/features/offers_observations.parquet',
                        help='Path to parquet with offer observations')
    parser.add_argument('--manifest', type=str,
                        default='data/features/manifest.json',
                        help='Path to manifest.json')
    parser.add_argument('--out_dir', type=str,
                        default='agents/outputs/ddqn_model_integrated',
                        help='Output directory')
    parser.add_argument('--episodes', type=int, default=10,
                        help='Number of epochs (full simulation runs)')
    parser.add_argument('--train_hours', type=int, default=48,
                        help='Hours of data for training (default: 48)')
    parser.add_argument('--eval_hours', type=int, default=48,
                        help='Hours of data for evaluation (default: 48, starts after train_hours)')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor')
    parser.add_argument('--epsilon_start', type=float, default=0.3,
                        help='Initial exploration rate')
    parser.add_argument('--epsilon_end', type=float, default=0.01,
                        help='Final exploration rate')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size for DDQN updates')
    parser.add_argument('--updates_per_event', type=int, default=2,
                        help='DDQN updates per simulation event')
    parser.add_argument('--save_every', type=int, default=1,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device (auto, cpu, cuda)')
    parser.add_argument('--travel_time_model', type=str,
                        default='data/travel_time_model.json',
                        help='Path to travel time model')
    parser.add_argument('--reward_type', type=str, default='balanced',
                        choices=['baseline', 'aggressive_accept', 'opportunity_cost', 
                                 'capacity_balanced', 'balanced', 'income_only', 
                                 'regret', 'urgency', 'max_accept', 'ultra_aggressive', 'paper', 'data_driven'],
                        help='Reward function type to use (default: balanced). '
                             'Use "paper" for exact paper reward (Eq. 1): income if accept, -0.3 if reject')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume training from')
    parser.add_argument('--bandit_mode', action='store_true',
                        help='Contextual bandit mode: done=True for all transitions, no bootstrapping. '
                             'Simpler and correct baseline; Q(s,a) learns immediate reward only.')
    
    args = parser.parse_args()
    
    # Print reward function choice
    print(f"\n[Reward Function] Using: {args.reward_type}")
    reward_calc = get_reward_calculator(args.reward_type)
    print(f"  Reject penalty: {reward_calc.reject_penalty}")
    print(f"  Income scale: {reward_calc.income_scale}")
    print(f"  Late penalty: {reward_calc.late_penalty}")
    print(f"  On-time bonus: {reward_calc.on_time_bonus}")
    
    train_ddqn_integrated(
        parquet_path=args.parquet,
        manifest_path=args.manifest,
        out_dir=args.out_dir,
        episodes=args.episodes,
        train_hours=args.train_hours,
        eval_hours=args.eval_hours,
        lr=args.lr,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        batch_size=args.batch_size,
        updates_per_event=args.updates_per_event,
        device=args.device,
        save_every=args.save_every,
        travel_time_model=args.travel_time_model,
        reward_type=args.reward_type,
        resume_from=args.resume,
        bandit_mode=args.bandit_mode,
    )


if __name__ == '__main__':
    main()
