"""Base class for gossip algorithms."""

from abc import ABC, abstractmethod
from typing import Optional
from ..core.network import Network


class GossipAlgorithm(ABC):
    """
    Abstract base class for gossip algorithms.
    
    All algorithms should implement the round() method which executes
    one communication round of the algorithm.
    """
    
    def __init__(self, name: str):
        """
        Initialize the algorithm.
        
        Args:
            name: Descriptive name for the algorithm
        """
        self.name = name
        
    @abstractmethod
    def round(self, network: Network) -> int:
        """
        Execute one round of the gossip algorithm.
        
        Args:
            network: The network to operate on
            
        Returns:
            Number of messages sent in this round
        """
        pass
        
    def reset(self) -> None:
        """Reset algorithm state if needed."""
        pass
        
    def __repr__(self) -> str:
        return f"{self.name}"
