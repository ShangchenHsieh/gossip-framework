"""Metrics collection and analysis."""

import numpy as np
from typing import List, Dict, Any
from dataclasses import dataclass, field


@dataclass
class MetricsCollector:
    """Collects and stores metrics during simulation."""
    
    timestamps: List[int] = field(default_factory=list)
    states: List[np.ndarray] = field(default_factory=list)
    messages: List[int] = field(default_factory=list)
    errors: List[float] = field(default_factory=list)
    
    def record_state(self, timestamp: int, state: np.ndarray) -> None:
        """Record state vector at a timestamp."""
        self.timestamps.append(timestamp)
        self.states.append(state.copy())
        
    def record_messages(self, timestamp: int, num_messages: int) -> None:
        """Record number of messages sent."""
        self.messages.append(num_messages)
        
    def record_error(self, timestamp: int, error: float) -> None:
        """Record convergence error."""
        self.errors.append(error)
        
    def reset(self) -> None:
        """Reset all collected metrics."""
        self.timestamps.clear()
        self.states.clear()
        self.messages.clear()
        self.errors.clear()
        
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of collected metrics."""
        if not self.errors:
            return {}
            
        return {
            "total_rounds": len(self.timestamps),
            "total_messages": sum(self.messages),
            "avg_messages_per_round": np.mean(self.messages) if self.messages else 0,
            "final_error": self.errors[-1],
            "min_error": np.min(self.errors),
            "convergence_rounds": self._estimate_convergence_rounds(1e-6),
        }
        
    def _estimate_convergence_rounds(self, threshold: float) -> int:
        """Estimate rounds to convergence."""
        for i, error in enumerate(self.errors):
            if error < threshold:
                return i
        return len(self.errors)
