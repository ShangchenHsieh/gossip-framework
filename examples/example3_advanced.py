"""
Example 3: Advanced Usage and Custom Algorithms
Demonstrates defining custom gossip algorithms and advanced simulations.
"""

import numpy as np
import sys
sys.path.insert(0, '/Users/sean/Classes/CS_262')

from gossip_framework import (
    Network, Simulator, GossipAlgorithm,
    MetricsCollector, GossipVisualizer
)


class WeightedGossip(GossipAlgorithm):
    """
    Custom Weighted Gossip Algorithm.
    
    Each node sends to a random neighbor with weighted probability
    based on the degree of neighbors.
    """
    
    def __init__(self, weight_type: str = "degree", seed: int = None):
        super().__init__("Weighted Gossip")
        self.weight_type = weight_type
        self.seed = seed
        if seed:
            np.random.seed(seed)
    
    def round(self, network):
        messages_sent = 0
        new_values = {}
        
        for node in network.get_all_nodes():
            if len(node.neighbors) == 0:
                continue
            
            neighbors = list(node.neighbors)
            
            # Weight neighbors by degree
            if self.weight_type == "degree":
                degrees = [len(network.get_node(n).neighbors) for n in neighbors]
                weights = np.array(degrees, dtype=float)
                weights /= weights.sum()
                neighbor = np.random.choice(neighbors, p=weights)
            else:
                neighbor = np.random.choice(neighbors)
            
            neighbor_node = network.get_node(neighbor)
            if neighbor not in new_values:
                new_values[neighbor] = neighbor_node.value
            new_values[neighbor] = (new_values[neighbor] + node.value) / 2
            messages_sent += 1
        
        for node_id, value in new_values.items():
            network.get_node(node_id).set_value(value)
        
        return messages_sent


class AdaptiveGossip(GossipAlgorithm):
    """
    Adaptive Gossip Algorithm.
    
    Nodes adapt their behavior based on local convergence state.
    Nodes with high variance communicate more frequently.
    """
    
    def __init__(self, seed: int = None):
        super().__init__("Adaptive Gossip")
        self.seed = seed
        self.node_activity = {}
        if seed:
            np.random.seed(seed)
    
    def round(self, network):
        messages_sent = 0
        new_values = {}
        
        # Initialize or update node activity
        for node in network.get_all_nodes():
            if node.node_id not in self.node_activity:
                self.node_activity[node.node_id] = 1.0
        
        for node in network.get_all_nodes():
            if len(node.neighbors) == 0:
                continue
            
            # Probability of communication based on activity
            activity = self.node_activity[node.node_id]
            if np.random.random() > activity:
                continue
            
            neighbor_id = np.random.choice(list(node.neighbors))
            neighbor = network.get_node(neighbor_id)
            
            if neighbor_id not in new_values:
                new_values[neighbor_id] = neighbor.value
            new_values[neighbor_id] = (new_values[neighbor_id] + node.value) / 2
            
            messages_sent += 1
        
        # Update activity based on changes
        for node in network.get_all_nodes():
            if node.node_id in new_values:
                # Node that just communicated reduces activity
                self.node_activity[node.node_id] *= 0.9
            else:
                # Idle nodes increase activity
                self.node_activity[node.node_id] = min(1.0, self.node_activity[node.node_id] * 1.1)
        
        for node_id, value in new_values.items():
            network.get_node(node_id).set_value(value)
        
        return messages_sent


def example_custom_algorithm():
    """Example using custom algorithms."""
    print("=" * 60)
    print("Example 3a: Custom Algorithm - Weighted Gossip")
    print("=" * 60)
    
    # Create network with non-uniform degree distribution
    num_nodes = 50
    network = Network(num_nodes, initial_values=np.random.uniform(0, 1, num_nodes), seed=42)
    network.create_scale_free_graph(m=2)  # Scale-free network
    
    print(f"Created scale-free network: {network}")
    
    # Run weighted gossip
    algorithm = WeightedGossip(weight_type="degree", seed=42)
    metrics = MetricsCollector()
    simulator = Simulator(network, algorithm, metrics, verbose=True)
    
    result = simulator.run(num_rounds=200, convergence_threshold=1e-6)
    
    print(f"Convergence rounds: {result['total_rounds']}")
    print(f"Final error: {result['final_error']:.2e}")
    
    return metrics, result


def example_dynamic_network():
    """Example with dynamic network (nodes joining/leaving)."""
    print("\n" + "=" * 60)
    print("Example 3b: Dynamic Network (Node Churn)")
    print("=" * 60)
    
    num_nodes = 50
    network = Network(num_nodes, initial_values=np.random.uniform(0, 1, num_nodes), seed=42)
    network.create_random_graph(0.5)
    
    algorithm = PushGossip(seed=42)
    metrics = MetricsCollector()
    simulator = Simulator(network, algorithm, metrics, verbose=False)
    
    # Run with node failures
    print("Running simulation with random node failures every 20 rounds...")
    
    for round_num in range(200):
        if round_num > 0 and round_num % 20 == 0:
            # Remove 2 random nodes
            nodes_to_remove = np.random.choice(num_nodes, 2, replace=False)
            for node_id in nodes_to_remove:
                network.get_node(node_id).active = False
                print(f"  Round {round_num}: Nodes {nodes_to_remove} deactivated")
        
        state_vector = network.get_state_vector()
        metrics.record_state(round_num, state_vector)
        
        messages = algorithm.round(network)
        metrics.record_messages(round_num, messages)
        
        # Only consider active nodes for error
        active_values = [network.get_node(i).value for i in range(num_nodes) 
                        if network.get_node(i).active]
        if active_values:
            avg_error = np.max(np.abs(np.array(active_values) - np.mean(active_values)))
            metrics.record_error(round_num, avg_error)
    
    print(f"Final error with failures: {metrics.errors[-1]:.2e}")
    
    return metrics


def example_convergence_benchmark():
    """Benchmark different algorithms systematically."""
    print("\n" + "=" * 60)
    print("Example 3c: Algorithm Benchmark")
    print("=" * 60)
    
    algorithms = [
        ("Weighted Gossip", WeightedGossip(seed=42)),
        ("Adaptive Gossip", AdaptiveGossip(seed=42)),
    ]
    
    # Additional imports
    from gossip_framework import PushGossip, PullGossip, GeographicGossip
    algorithms.extend([
        ("Push", PushGossip(seed=42)),
        ("Pull", PullGossip(seed=42)),
        ("Geographic", GeographicGossip(seed=42)),
    ])
    
    results = {}
    
    for algo_name, algo in algorithms:
        print(f"\nBenchmarking {algo_name}...")
        
        # Run on different network sizes
        convergence_times = []
        for size in [20, 50, 100]:
            network = Network(size, initial_values=np.random.uniform(0, 1, size), seed=42)
            network.create_complete_graph()
            
            metrics = MetricsCollector()
            simulator = Simulator(network, algo, metrics, verbose=False)
            
            result = simulator.run(num_rounds=500, convergence_threshold=1e-6)
            convergence_times.append(result['total_rounds'])
            
            print(f"  Size {size}: {result['total_rounds']} rounds")
        
        results[algo_name] = convergence_times
    
    return results


def example_visualization_gallery():
    """Create a gallery of visualizations."""
    print("\n" + "=" * 60)
    print("Example 3d: Visualization Gallery")
    print("=" * 60)
    
    # Create sample results
    num_nodes = 50
    network = Network(num_nodes, initial_values=np.random.uniform(0, 1, num_nodes), seed=42)
    network.create_complete_graph()
    
    from gossip_framework import PushGossip
    algorithm = PushGossip(seed=42)
    metrics = MetricsCollector()
    simulator = Simulator(network, algorithm, metrics, verbose=False)
    result = simulator.run(num_rounds=100)
    
    visualizer = GossipVisualizer()
    
    # Create multiple visualizations
    print("Creating visualizations...")
    
    # 1. Convergence plot
    fig1 = visualizer.plot_convergence(metrics)
    fig1.savefig("/tmp/viz_convergence.png", dpi=150)
    print("  - Convergence plot saved")
    
    # 2. Message complexity
    fig2 = visualizer.plot_message_complexity(metrics)
    fig2.savefig("/tmp/viz_messages.png", dpi=150)
    print("  - Message complexity saved")
    
    # 3. Node trajectories
    fig3 = visualizer.plot_node_trajectories(metrics, num_nodes_to_plot=10)
    fig3.savefig("/tmp/viz_trajectories.png", dpi=150)
    print("  - Node trajectories saved")
    
    # 4. Heatmap
    fig4 = visualizer.plot_heatmap_convergence(metrics)
    fig4.savefig("/tmp/viz_heatmap.png", dpi=150)
    print("  - State heatmap saved")
    
    print("\nAll visualizations saved to /tmp/")


if __name__ == "__main__":
    # Import PushGossip for benchmark
    from gossip_framework import PushGossip
    
    # Run examples
    metrics_custom, result_custom = example_custom_algorithm()
    metrics_dynamic = example_dynamic_network()
    benchmark_results = example_convergence_benchmark()
    example_visualization_gallery()
    
    print("\n" + "=" * 60)
    print("All advanced examples completed successfully!")
    print("=" * 60)
