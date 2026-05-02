"""Implementations of various gossip protocols."""

import numpy as np
from typing import Optional, Set, Tuple
from .base import GossipAlgorithm
from ..core.network import Network


class PushGossip(GossipAlgorithm):
    """
    Push Gossip Algorithm.
    
    In each round, every node sends its current value to a random neighbor.
    Each receiver averages its value with the received value.
    """
    
    def __init__(self, seed: Optional[int] = None):
        super().__init__("Push Gossip")
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
            
    def round(self, network: Network) -> int:
        """Execute one round of push gossip."""
        messages_sent = 0
        new_values = {}
        
        # Each node selects a random neighbor to send to
        for node in network.get_all_nodes():
            if len(node.neighbors) > 0:
                # Send to random neighbor
                neighbor_id = np.random.choice(list(node.neighbors))
                neighbor = network.get_node(neighbor_id)
                
                # Receiver averages its value with sender's value
                if neighbor_id not in new_values:
                    new_values[neighbor_id] = neighbor.value
                new_values[neighbor_id] = (new_values[neighbor_id] + node.value) / 2
                
                messages_sent += 1
        
        # Update all values
        for node_id, new_value in new_values.items():
            network.get_node(node_id).set_value(new_value)
            
        return messages_sent


class PullGossip(GossipAlgorithm):
    """
    Pull Gossip Algorithm.
    
    In each round, every node selects a random neighbor and requests its value.
    The node then averages its current value with the received value.
    """
    
    def __init__(self, seed: Optional[int] = None):
        super().__init__("Pull Gossip")
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
            
    def round(self, network: Network) -> int:
        """Execute one round of pull gossip."""
        messages_sent = 0
        new_values = {}
        
        # Each node selects a random neighbor to pull from
        for node in network.get_all_nodes():
            if len(node.neighbors) > 0:
                # Pull from random neighbor
                neighbor_id = np.random.choice(list(node.neighbors))
                neighbor_value = network.get_node(neighbor_id).value
                
                # Average with neighbor's value
                new_value = (node.value + neighbor_value) / 2
                new_values[node.node_id] = new_value
                
                messages_sent += 1
        
        # Update all values
        for node_id, new_value in new_values.items():
            network.get_node(node_id).set_value(new_value)
            
        return messages_sent


class PushPullGossip(GossipAlgorithm):
    """
    Push-Pull Gossip Algorithm.
    
    Combines push and pull mechanisms for faster convergence.
    Nodes can both push and pull in the same round.
    """
    
    def __init__(self, push_probability: float = 0.5, seed: Optional[int] = None):
        super().__init__("Push-Pull Gossip")
        self.push_probability = push_probability
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
            
    def round(self, network: Network) -> int:
        """Execute one round of push-pull gossip."""
        messages_sent = 0
        new_values = {}
        
        for node in network.get_all_nodes():
            if len(node.neighbors) == 0:
                continue
                
            neighbor_id = np.random.choice(list(node.neighbors))
            neighbor = network.get_node(neighbor_id)
            
            if np.random.random() < self.push_probability:
                # Push operation
                if neighbor_id not in new_values:
                    new_values[neighbor_id] = neighbor.value
                new_values[neighbor_id] = (new_values[neighbor_id] + node.value) / 2
            else:
                # Pull operation
                new_value = (node.value + neighbor.value) / 2
                new_values[node.node_id] = new_value
                
            messages_sent += 1
        
        # Update all values
        for node_id, new_value in new_values.items():
            network.get_node(node_id).set_value(new_value)
            
        return messages_sent


class RandomAveraging(GossipAlgorithm):
    """
    Random Averaging Algorithm.
    
    In each round, nodes participate in pairwise averaging with randomly selected neighbors.
    Multiple interactions per round are possible.
    """
    
    def __init__(self, interactions_per_node: int = 1, seed: Optional[int] = None):
        super().__init__("Random Averaging")
        self.interactions_per_node = interactions_per_node
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
            
    def round(self, network: Network) -> int:
        """Execute one round of random averaging."""
        messages_sent = 0
        values_to_update = {node.node_id: node.value for node in network.get_all_nodes()}
        
        # Each node initiates interactions_per_node interactions
        for node in network.get_all_nodes():
            for _ in range(self.interactions_per_node):
                if len(node.neighbors) == 0:
                    continue
                    
                neighbor_id = np.random.choice(list(node.neighbors))
                neighbor_value = network.get_node(neighbor_id).value
                
                # Both nodes average
                avg = (values_to_update[node.node_id] + neighbor_value) / 2
                values_to_update[node.node_id] = avg
                
                messages_sent += 1
        
        # Update all values
        for node_id, new_value in values_to_update.items():
            network.get_node(node_id).set_value(new_value)
            
        return messages_sent
