"""
Agents for courier assignment decisions.

Available agents:
- BCPruningAgent: Behavior Cloning agent using threshold-based pruning
- DDQNAgent: DDQN agent using argmax over Q-values
"""

from .bc_agent import BCPruningAgent
from .ddqn_agent import DDQNAgent

__all__ = [
    "BCPruningAgent",
    "DDQNAgent",
]
