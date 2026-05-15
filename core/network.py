"""Network topology management for gossip simulations."""

import numpy as np
from typing import List, Optional, Dict, Tuple
from .node import Node


class Network:
    """
    Manages the network topology and nodes.

    Supports various network topologies:
    - Complete graph (all-to-all)
    - Random graph
    - Ring
    - Lattice/Grid
    - Scale-free (Barabasi-Albert)
    """

    def __init__(
        self,
        num_nodes: int,
        initial_values: Optional[List[float]] = None,
        seed: Optional[int] = None,
        positions: Optional[List[Tuple[float, float]]] = None,
    ):
        """
        Initialize a network.

        Args:
            num_nodes: Number of nodes in the network
            initial_values: List of initial values for each node
            seed: Random seed for reproducibility
            positions: Optional (x, y) coordinates in the unit square
        """
        self.num_nodes = num_nodes
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        if seed is not None:
            np.random.seed(seed)

        if initial_values is None:
            initial_values = np.random.uniform(0, 1, num_nodes)

        self.nodes: Dict[int, Node] = {
            i: Node(i, float(initial_values[i]))
            for i in range(num_nodes)
        }
        if positions is None:
            positions = self.rng.uniform(0.0, 1.0, size=(num_nodes, 2))
        self.positions = np.asarray(positions, dtype=float)
        if self.positions.shape != (num_nodes, 2):
            raise ValueError("positions must have shape (num_nodes, 2)")

    def add_edge(self, node_a: int, node_b: int, bidirectional: bool = True) -> None:
        """Add an edge between two nodes."""
        if node_a in self.nodes and node_b in self.nodes:
            self.nodes[node_a].add_neighbor(node_b)
            if bidirectional:
                self.nodes[node_b].add_neighbor(node_a)

    def remove_edge(self, node_a: int, node_b: int, bidirectional: bool = True) -> None:
        """Remove an edge between two nodes."""
        if node_a in self.nodes and node_b in self.nodes:
            self.nodes[node_a].remove_neighbor(node_b)
            if bidirectional:
                self.nodes[node_b].remove_neighbor(node_a)

    def create_complete_graph(self) -> None:
        """Create a complete graph topology (all nodes connected)."""
        for i in range(self.num_nodes):
            for j in range(i + 1, self.num_nodes):
                self.add_edge(i, j, bidirectional=True)

    def create_random_geometric_graph(self, radius: Optional[float] = None) -> None:
        """
        Create a random geometric graph using node positions.

        Args:
            radius: Communication radius. Defaults to sqrt(log(n) / n).
        """
        if radius is None:
            radius = float(np.sqrt(np.log(max(self.num_nodes, 2)) / self.num_nodes))

        for i in range(self.num_nodes):
            for j in range(i + 1, self.num_nodes):
                if self.distance(i, j) <= radius:
                    self.add_edge(i, j, bidirectional=True)

    def create_random_graph(self, edge_probability: float) -> None:
        """
        Create a random graph (Erdos-Renyi model).

        Args:
            edge_probability: Probability of edge between any two nodes
        """
        for i in range(self.num_nodes):
            for j in range(i + 1, self.num_nodes):
                if np.random.random() < edge_probability:
                    self.add_edge(i, j, bidirectional=True)

    def create_ring_graph(self) -> None:
        """Create a ring topology."""
        angles = np.linspace(0.0, 2.0 * np.pi, self.num_nodes, endpoint=False)
        self.positions = np.column_stack((0.5 + 0.45 * np.cos(angles), 0.5 + 0.45 * np.sin(angles)))
        for i in range(self.num_nodes):
            self.add_edge(i, (i + 1) % self.num_nodes, bidirectional=True)

    def create_lattice_graph(self, grid_side: Optional[int] = None) -> None:
        """
        Create a 2D lattice/grid topology.

        Args:
            grid_side: Side length of grid (defaults to sqrt(num_nodes))
        """
        if grid_side is None:
            grid_side = int(np.sqrt(self.num_nodes))

        if grid_side * grid_side != self.num_nodes:
            raise ValueError(f"num_nodes ({self.num_nodes}) must be a perfect square for lattice")

        coords = []
        denom = max(grid_side - 1, 1)
        for i in range(self.num_nodes):
            x, y = i // grid_side, i % grid_side
            coords.append((x / denom, y / denom))
        self.positions = np.asarray(coords, dtype=float)

        for i in range(self.num_nodes):
            x, y = i // grid_side, i % grid_side

            neighbors = []
            if x > 0:
                neighbors.append((x - 1) * grid_side + y)
            if x < grid_side - 1:
                neighbors.append((x + 1) * grid_side + y)
            if y > 0:
                neighbors.append(x * grid_side + (y - 1))
            if y < grid_side - 1:
                neighbors.append(x * grid_side + (y + 1))

            for neighbor in neighbors:
                self.add_edge(i, neighbor, bidirectional=True)

    def create_scale_free_graph(self, m: int = 2) -> None:
        """
        Create a scale-free network using the Barabasi-Albert model.

        Args:
            m: Number of edges per new node
        """
        for i in range(min(m + 1, self.num_nodes)):
            for j in range(i + 1, min(m + 1, self.num_nodes)):
                self.add_edge(i, j, bidirectional=True)

        for i in range(m + 1, self.num_nodes):
            degrees = [len(self.nodes[j].neighbors) for j in range(i)]
            if sum(degrees) == 0:
                targets = np.random.choice(i, m, replace=False)
            else:
                probs = np.array(degrees) / sum(degrees)
                targets = np.random.choice(i, m, p=probs, replace=False)

            for target in targets:
                self.add_edge(i, target, bidirectional=True)

    def get_node(self, node_id: int) -> Node:
        """Get a node by ID."""
        return self.nodes[node_id]

    def get_all_nodes(self) -> List[Node]:
        """Get all nodes in the network."""
        return list(self.nodes.values())

    def get_average_value(self) -> float:
        """Get the average value across all nodes."""
        return np.mean([node.value for node in self.nodes.values()])

    def get_state_vector(self) -> np.ndarray:
        """Get the current state vector of all nodes."""
        return np.array([self.nodes[i].value for i in range(self.num_nodes)])

    def get_position(self, node_id: int) -> np.ndarray:
        """Get the geographic position of a node."""
        return self.positions[node_id].copy()

    def set_position(self, node_id: int, position: Tuple[float, float]) -> None:
        """Set the geographic position of a node."""
        self.positions[node_id] = np.asarray(position, dtype=float)

    def distance_to_point(self, node_id: int, point: np.ndarray) -> float:
        """Return Euclidean distance from a node to a point."""
        return float(np.linalg.norm(self.positions[node_id] - point))

    def distance(self, node_a: int, node_b: int) -> float:
        """Return Euclidean distance between two nodes."""
        return float(np.linalg.norm(self.positions[node_a] - self.positions[node_b]))

    def reset_nodes(self) -> None:
        """Reset all nodes to their initial states."""
        for node in self.nodes.values():
            node.reset()

    def __repr__(self) -> str:
        avg_degree = np.mean([len(node.neighbors) for node in self.nodes.values()])
        return f"Network(nodes={self.num_nodes}, avg_degree={avg_degree:.2f})"

    # ------------------------- Spectral utilities -------------------------
    def get_transition_matrix(self, transition: str = "uniform") -> np.ndarray:
        """
        Construct the node-contact transition matrix P for random averaging.

        Supported transition rules:
        - "uniform": each node chooses uniformly among its neighbors.
        - "metropolis": symmetric Metropolis-Hastings weights.
        - "max_degree": symmetric maximum-degree random walk weights.
        """
        n = self.num_nodes
        P = np.zeros((n, n), dtype=float)
        degrees = [len(self.nodes[i].neighbors) for i in range(n)]

        if transition == "uniform":
            for i in range(n):
                if degrees[i] == 0:
                    P[i, i] = 1.0
                    continue
                weight = 1.0 / degrees[i]
                for j in self.nodes[i].neighbors:
                    P[i, j] = weight
            return P

        if transition == "metropolis":
            for i in range(n):
                for j in self.nodes[i].neighbors:
                    if i != j:
                        P[i, j] = 1.0 / (1 + max(degrees[i], degrees[j]))
            row_sums = P.sum(axis=1)
            for i in range(n):
                P[i, i] = 1.0 - row_sums[i]
            return P

        if transition == "max_degree":
            max_degree = max(degrees) if degrees else 0
            if max_degree == 0:
                np.fill_diagonal(P, 1.0)
                return P
            for i in range(n):
                for j in self.nodes[i].neighbors:
                    if i != j:
                        P[i, j] = 1.0 / max_degree
            row_sums = P.sum(axis=1)
            for i in range(n):
                P[i, i] = 1.0 - row_sums[i]
            return P

        raise ValueError(f"Unknown transition rule: {transition}")

    def get_expected_gossip_matrix(self, transition: str = "uniform") -> np.ndarray:
        """
        Return the expected one-clock-tick random averaging update matrix.

        If active node i is selected with probability 1/n and contacts j with
        probability P[i, j], the pairwise update matrix is
        I - 0.5 * (e_i - e_j)(e_i - e_j)^T. This method returns its expectation.
        """
        n = self.num_nodes
        P = self.get_transition_matrix(transition)
        W = np.eye(n, dtype=float)

        for i in range(n):
            for j in range(n):
                if i == j or P[i, j] == 0.0:
                    continue
                weight = P[i, j] / (2.0 * n)
                W[i, i] -= weight
                W[j, j] -= weight
                W[i, j] += weight
                W[j, i] += weight

        return W

    def get_averaging_matrix(self, transition: str = "uniform") -> np.ndarray:
        """Return the expected one-clock-tick random averaging matrix."""
        return self.get_expected_gossip_matrix(transition)

    def get_eigen_spectrum(self, transition: str = "uniform") -> np.ndarray:
        """Return sorted eigenvalues of the expected gossip matrix."""
        W = self.get_expected_gossip_matrix(transition)
        vals = np.linalg.eigvalsh(W)
        return np.array(sorted(vals, reverse=True))

    def second_largest_eigenvalue(self, transition: str = "uniform") -> float:
        """Return the convergence eigenvalue of the expected gossip matrix."""
        vals = self.get_eigen_spectrum(transition)
        if vals.size == 0:
            return 0.0
        nontrivial = vals[np.abs(vals - 1.0) > 1e-10]
        if nontrivial.size == 0:
            return 0.0
        return float(np.max(np.abs(nontrivial)))

    def spectral_gap(self, transition: str = "uniform") -> float:
        """Return the gossip spectral gap 1 - lambda."""
        lam2 = self.second_largest_eigenvalue(transition)
        return 1.0 - lam2
