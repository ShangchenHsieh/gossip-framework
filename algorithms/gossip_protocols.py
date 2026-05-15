"""Implementations of various gossip protocols."""

import numpy as np
from typing import Optional, Tuple
from .base import GossipAlgorithm
from ..core.network import Network


def _pairwise_average(network: Network, node_id: int, neighbor_id: int) -> None:
    """Average two endpoint values in place, preserving the global sum."""
    node = network.get_node(node_id)
    neighbor = network.get_node(neighbor_id)
    avg = (node.value + neighbor.value) / 2.0
    node.set_value(avg)
    neighbor.set_value(avg)


def _average_path(network: Network, path: list[int]) -> None:
    """Average all node values along a path in place."""
    if not path:
        return

    avg = float(np.mean([network.get_node(node_id).value for node_id in path]))
    for node_id in path:
        network.get_node(node_id).set_value(avg)


class PushGossip(GossipAlgorithm):
    """
    Push Gossip Algorithm.
    
    In each round, every node contacts a random neighbor and the pair averages.
    """
    
    def __init__(self, seed: Optional[int] = None):
        super().__init__("Push Gossip")
        self.seed = seed
        self.rng = np.random.default_rng(seed)
            
    def round(self, network: Network) -> int:
        """Execute one round of push gossip."""
        messages_sent = 0

        for node in network.get_all_nodes():
            if len(node.neighbors) > 0:
                neighbor_id = int(self.rng.choice(list(node.neighbors)))
                _pairwise_average(network, node.node_id, neighbor_id)
                messages_sent += 1

        return messages_sent


class PullGossip(GossipAlgorithm):
    """
    Pull Gossip Algorithm.
    
    In each round, every node selects a random neighbor and the pair averages.
    """
    
    def __init__(self, seed: Optional[int] = None):
        super().__init__("Pull Gossip")
        self.seed = seed
        self.rng = np.random.default_rng(seed)
            
    def round(self, network: Network) -> int:
        """Execute one round of pull gossip."""
        messages_sent = 0

        for node in network.get_all_nodes():
            if len(node.neighbors) > 0:
                neighbor_id = int(self.rng.choice(list(node.neighbors)))
                _pairwise_average(network, node.node_id, neighbor_id)
                messages_sent += 1

        return messages_sent


class RandomAveraging(GossipAlgorithm):
    """
    Randomized pairwise averaging algorithm.

    This implements the asynchronous clock-tick model analyzed by Boyd et al.:
    one active node is selected uniformly per tick, it contacts one neighbor
    according to a transition matrix, and both endpoints replace their values
    with their average.
    """

    def __init__(
        self,
        clock_ticks_per_round: int = 1,
        transition: str = "uniform",
        seed: Optional[int] = None,
        interactions_per_node: Optional[int] = None,
    ):
        super().__init__("Random Averaging")
        if interactions_per_node is not None:
            clock_ticks_per_round = interactions_per_node
        if clock_ticks_per_round < 1:
            raise ValueError("clock_ticks_per_round must be at least 1")
        self.clock_ticks_per_round = clock_ticks_per_round
        self.transition = transition
        self.seed = seed
        self._transition_cache = None
        self._transition_cache_key = None
        self.rng = np.random.default_rng(seed)

    def round(self, network: Network) -> int:
        """Execute asynchronous random averaging clock ticks."""
        messages_sent = 0
        cache_key = (id(network), self.transition)
        if self._transition_cache_key != cache_key:
            self._transition_cache = network.get_transition_matrix(self.transition)
            self._transition_cache_key = cache_key
        transition_matrix = self._transition_cache

        for _ in range(self.clock_ticks_per_round):
            active_id = int(self.rng.integers(network.num_nodes))
            probabilities = transition_matrix[active_id].copy()
            probabilities[active_id] = 0.0
            total = probabilities.sum()
            if total <= 0.0:
                continue

            probabilities /= total
            neighbor_id = int(self.rng.choice(network.num_nodes, p=probabilities))
            _pairwise_average(network, active_id, neighbor_id)
            messages_sent += 1

        return messages_sent


class PathAveraging(GossipAlgorithm):
    """
    Randomized geographic path averaging.

    Each clock tick selects one active node, samples a target point in the unit
    square, randomly greedy-routes through neighbors that are closer to the
    target, and averages every node visited on the resulting path.
    """

    def __init__(self, clock_ticks_per_round: int = 1, seed: Optional[int] = None):
        super().__init__("Path Averaging")
        if clock_ticks_per_round < 1:
            raise ValueError("clock_ticks_per_round must be at least 1")
        self.clock_ticks_per_round = clock_ticks_per_round
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def round(self, network: Network) -> int:
        """Execute asynchronous random greedy path averaging clock ticks."""
        messages_sent = 0

        for _ in range(self.clock_ticks_per_round):
            source_id = int(self.rng.integers(network.num_nodes))
            target = self.rng.uniform(0.0, 1.0, size=2)
            path = self._random_greedy_route(network, source_id, target)
            _average_path(network, path)
            messages_sent += 2 * max(0, len(path) - 1)

        return messages_sent

    def _random_greedy_route(self, network: Network, start_id: int, target: np.ndarray) -> list[int]:
        """Route through random unvisited neighbors that move closer to target."""
        path = [start_id]
        visited = {start_id}
        current_id = start_id

        while True:
            current_distance = network.distance_to_point(current_id, target)
            candidates = [
                neighbor_id
                for neighbor_id in sorted(network.get_node(current_id).neighbors)
                if neighbor_id not in visited
                and network.distance_to_point(neighbor_id, target) < current_distance
            ]
            if not candidates:
                return path

            current_id = int(self.rng.choice(candidates))
            visited.add(current_id)
            path.append(current_id)


class GeographicGossip(GossipAlgorithm):
    """
    Geographic gossip with greedy routing and Voronoi rejection sampling.

    Each clock tick selects one active node, samples a target point in the unit
    square, routes greedily through the network topology toward that point, and
    averages with the reached node if the rejection sampler accepts it.
    """

    def __init__(
        self,
        clock_ticks_per_round: int = 1,
        rejection_threshold: Optional[float] = None,
        threshold_factor: float = 1.0,
        area_samples: int = 4096,
        max_rejection_attempts: int = 32,
        seed: Optional[int] = None,
    ):
        super().__init__("Geographic Gossip")
        if clock_ticks_per_round < 1:
            raise ValueError("clock_ticks_per_round must be at least 1")
        if area_samples < 1:
            raise ValueError("area_samples must be at least 1")
        if max_rejection_attempts < 1:
            raise ValueError("max_rejection_attempts must be at least 1")
        self.clock_ticks_per_round = clock_ticks_per_round
        self.rejection_threshold = rejection_threshold
        self.threshold_factor = threshold_factor
        self.area_samples = area_samples
        self.max_rejection_attempts = max_rejection_attempts
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self._area_cache_key = None
        self._area_cache = None

    def round(self, network: Network) -> int:
        """Execute asynchronous geographic gossip clock ticks."""
        messages_sent = 0
        areas = self._voronoi_area_estimates(network)
        threshold = self.rejection_threshold
        if threshold is None:
            threshold = self.threshold_factor / network.num_nodes

        for _ in range(self.clock_ticks_per_round):
            source_id = int(self.rng.integers(network.num_nodes))
            for _attempt in range(self.max_rejection_attempts):
                target = self.rng.uniform(0.0, 1.0, size=2)
                destination_id, forward_hops = self._greedy_route(network, source_id, target)
                messages_sent += forward_hops

                area = max(float(areas[destination_id]), np.finfo(float).eps)
                accept_probability = min(threshold / area, 1.0)
                if self.rng.random() > accept_probability:
                    continue

                if destination_id != source_id:
                    _pairwise_average(network, source_id, destination_id)
                    _return_destination, return_hops = self._greedy_route(
                        network,
                        destination_id,
                        network.get_position(source_id),
                    )
                    messages_sent += return_hops
                break

        return messages_sent

    def _greedy_route(self, network: Network, start_id: int, target: np.ndarray) -> Tuple[int, int]:
        """Route to a local minimum of distance to target using topology edges."""
        current_id = start_id
        hops = 0
        visited = {current_id}

        while True:
            current_distance = network.distance_to_point(current_id, target)
            candidates = [
                neighbor_id
                for neighbor_id in network.get_node(current_id).neighbors
                if neighbor_id not in visited
                and network.distance_to_point(neighbor_id, target) < current_distance
            ]
            if not candidates:
                return current_id, hops

            current_id = min(candidates, key=lambda node_id: network.distance_to_point(node_id, target))
            visited.add(current_id)
            hops += 1

    def _voronoi_area_estimates(self, network: Network) -> np.ndarray:
        """Estimate unit-square Voronoi cell areas by Monte Carlo sampling."""
        cache_key = (
            id(network),
            network.num_nodes,
            self.area_samples,
            network.positions.shape,
            network.positions.tobytes(),
        )
        if self._area_cache_key == cache_key:
            return self._area_cache

        samples = self.rng.uniform(0.0, 1.0, size=(self.area_samples, 2))
        distances = np.linalg.norm(samples[:, None, :] - network.positions[None, :, :], axis=2)
        closest_nodes = np.argmin(distances, axis=1)
        counts = np.bincount(closest_nodes, minlength=network.num_nodes)
        areas = counts.astype(float) / float(self.area_samples)

        self._area_cache_key = cache_key
        self._area_cache = areas
        return areas
