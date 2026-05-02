"""Node class representing a participant in the gossip network."""

import numpy as np
from typing import Set, Optional, Dict, Any


class Node:
    """
    Represents a node in the gossip network.
    
    Each node maintains:
    - An identifier
    - A current state value
    - A set of neighbors
    - Historical state values for convergence tracking
    """
    
    def __init__(self, node_id: int, initial_value: float, active: bool = True):
        """
        Initialize a node.
        
        Args:
            node_id: Unique identifier for the node
            initial_value: Initial state value (used for averaging)
            active: Whether the node is active in the network
        """
        self.node_id = node_id
        self.value = initial_value
        self.initial_value = initial_value
        self.active = active
        self.neighbors: Set[int] = set()
        
        # Tracking for visualization and analysis
        self.history = [initial_value]  # State history over time
        self.message_count = 0  # Total messages sent/received
        
    def add_neighbor(self, neighbor_id: int) -> None:
        """Add a neighbor to this node's contact list."""
        self.neighbors.add(neighbor_id)
        
    def remove_neighbor(self, neighbor_id: int) -> None:
        """Remove a neighbor from this node's contact list."""
        self.neighbors.discard(neighbor_id)
        
    def set_value(self, new_value: float) -> None:
        """Update the node's state value."""
        self.value = new_value
        self.history.append(new_value)
        
    def get_neighbors(self) -> Set[int]:
        """Get the set of neighbor IDs."""
        return self.neighbors.copy()
        
    def reset(self) -> None:
        """Reset the node to its initial state."""
        self.value = self.initial_value
        self.history = [self.initial_value]
        self.message_count = 0
        
    def __repr__(self) -> str:
        return f"Node(id={self.node_id}, value={self.value:.4f}, neighbors={len(self.neighbors)})"
