"""
Baseline Agent: Always returns the historical decision.

Used for pure log replay to establish ground-truth metrics without agent intervention.
"""
from typing import Dict, Tuple


class BaselineAgent:
    """
    No-intervention agent that always replicates historical decisions.
    
    This enables running the simulation in "baseline mode" to compute
    metrics on the historical system without any pruning agent.
    
    Usage:
      agent = BaselineAgent()
      action, prob = agent.act(obs, historical_decision=1)
      # Returns: (1, 1.0)
    """
    
    def __init__(self):
        """Initialize baseline agent (no model loading needed)."""
        pass
    
    def act(self, obs: Dict[str, float], historical_decision: int = 0) -> Tuple[int, float]:
        """
        Return the historical decision without modification.
        
        Args:
            obs: observation dict (ignored in baseline mode)
            historical_decision: what actually happened in the logs
        
        Returns:
            (action, probability) tuple matching historical_decision
        """
        # Always return historical decision with 100% confidence
        return (historical_decision, 1.0)
    
    def act_batch(self, X, historical_decisions):
        """
        Batch version for consistency with BC agent interface.
        
        Args:
            X: observation matrix (ignored)
            historical_decisions: array of historical decisions
        
        Returns:
            (actions, probabilities) matching historical_decisions
        """
        import numpy as np
        actions = np.array(historical_decisions, dtype=int)
        probs = np.ones_like(actions, dtype=float)
        return actions, probs
