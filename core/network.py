"""Network topology management for gossip simulations."""

import numpy as np
from typing import List, Optional, Tuple, Dict
from .node import Node


class Network:
    """
    Manages the network topology and nodes.
    
    Supports various network topologies:
    - Complete graph (all-to-all)
    - Random graph
    - Ring
    - Lattice/Grid
    - Scale-free (Barabási-Albert)
    """
    
    def __init__(self, num_nodes: int, initial_values: Optional[List[float]] = None, seed: Optional[int] = None):
        """
        Initialize a network.
        
        Args:
            num_nodes: Number of nodes in the network
            initial_values: List of initial values for each node
            seed: Random seed for reproducibility
        """
        self.num_nodes = num_nodes
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
            
        # Initialize nodes
        if initial_values is None:
            # Default: random values in [0, 1]
            initial_values = np.random.uniform(0, 1, num_nodes)
        
        self.nodes: Dict[int, Node] = {
            i: Node(i, float(initial_values[i]))
            for i in range(num_nodes)
        }
        
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
                
    def create_random_graph(self, edge_probability: float) -> None:
        """
        Create a random graph (Erdős–Rényi model).
        
        Args:
            edge_probability: Probability of edge between any two nodes
        """
        for i in range(self.num_nodes):
            for j in range(i + 1, self.num_nodes):
                if np.random.random() < edge_probability:
                    self.add_edge(i, j, bidirectional=True)
                    
    def create_ring_graph(self) -> None:
        """Create a ring topology (each node connected to k nearest neighbors)."""
        for i in range(self.num_nodes):
            # Connect to next 2 nodes in ring
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
            
        for i in range(self.num_nodes):
            x, y = i // grid_side, i % grid_side
            
            # Connect to neighbors
            neighbors = []
            if x > 0:
                neighbors.append((x - 1) * grid_side + y)  # up
            if x < grid_side - 1:
                neighbors.append((x + 1) * grid_side + y)  # down
            if y > 0:
                neighbors.append(x * grid_side + (y - 1))  # left
            if y < grid_side - 1:
                neighbors.append(x * grid_side + (y + 1))  # right
                
            for neighbor in neighbors:
                self.add_edge(i, neighbor, bidirectional=True)
                
    def create_scale_free_graph(self, m: int = 2) -> None:
        """
        Create a scale-free network using Barabási-Albert model.
        
        Args:
            m: Number of edges per new node
        """
        # Start with a complete graph of m+1 nodes
        for i in range(min(m + 1, self.num_nodes)):
            for j in range(i + 1, min(m + 1, self.num_nodes)):
                self.add_edge(i, j, bidirectional=True)
                
        # Add remaining nodes with preferential attachment
        for i in range(m + 1, self.num_nodes):
            # Calculate degree of existing nodes
            degrees = [len(self.nodes[j].neighbors) for j in range(i)]
            if sum(degrees) == 0:
                targets = np.random.choice(i, m, replace=False)
            else:
                # Preferential attachment: probability proportional to degree
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
        
    def reset_nodes(self) -> None:
        """Reset all nodes to their initial states."""
        for node in self.nodes.values():
            node.reset()
            
    def __repr__(self) -> str:
        avg_degree = np.mean([len(node.neighbors) for node in self.nodes.values()])
        return f"Network(nodes={self.num_nodes}, avg_degree={avg_degree:.2f})"

    # ------------------------- Spectral utilities -------------------------
    def get_averaging_matrix(self) -> np.ndarray:
        """
        Construct a (symmetric) averaging matrix W using Metropolis-Hastings weights.

        For i != j and (i,j) an edge:
            w_ij = 1 / (1 + max(deg(i), deg(j)))
        Diagonal entries are set so rows sum to 1 (stochastic matrix).

        Returns:
            W (num_nodes x num_nodes) numpy array
        """
        n = self.num_nodes
        W = np.zeros((n, n), dtype=float)

        degrees = [len(self.nodes[i].neighbors) for i in range(n)]

        for i in range(n):
            for j in self.nodes[i].neighbors:
                if i == j:
                    continue
                deg_i = degrees[i]
                deg_j = degrees[j]
                W[i, j] = 1.0 / (1 + max(deg_i, deg_j))

        # Set diagonal entries to make rows sum to 1
        row_sums = W.sum(axis=1)
        for i in range(n):
            W[i, i] = 1.0 - row_sums[i]

        return W

    def get_eigen_spectrum(self) -> np.ndarray:
        """Return sorted eigenvalues (descending) of the averaging matrix W."""
        W = self.get_averaging_matrix()
        vals = np.linalg.eigvals(W)
        # Sort by real part descending
        vals_sorted = np.array(sorted(vals, key=lambda x: -np.real(x)))
        return vals_sorted

    def second_largest_eigenvalue(self) -> float:
        """Return the second-largest (in magnitude of real part) eigenvalue of W.

        This function returns the eigenvalue λ2 used in gossip convergence bounds.
        """
        vals = self.get_eigen_spectrum()
        if vals.size == 0:
            return 0.0
        # Largest eigenvalue should be 1 (for stochastic W). Return next one.
        if vals.size >= 2:
            return float(np.real(vals[1]))
        return float(np.real(vals[0]))

    def spectral_gap(self) -> float:
        """Return the spectral gap 1 - λ2."""
        lam2 = self.second_largest_eigenvalue()
        return 1.0 - lam2
