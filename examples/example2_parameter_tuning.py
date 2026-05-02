"""
Example 2: Parameter Tuning and Sensitivity Analysis
Demonstrates how to tune parameters and analyze their impact on convergence.
"""

import numpy as np
import sys
sys.path.insert(0, '/Users/sean/Classes/CS_262')

from gossip_framework import (
    Network, Simulator, PushGossip, PushPullGossip,
    MetricsCollector, GossipVisualizer, SimulationConfig
)


def example_network_size_sensitivity():
    """Analyze impact of network size on convergence."""
    print("=" * 60)
    print("Example 2a: Network Size Sensitivity")
    print("=" * 60)
    
    network_sizes = [10, 25, 50, 100, 200]
    results = []
    
    for size in network_sizes:
        print(f"Simulating network with {size} nodes...")
        
        network = Network(size, initial_values=np.random.uniform(0, 1, size), seed=42)
        network.create_complete_graph()
        
        algorithm = PushGossip(seed=42)
        metrics = MetricsCollector()
        simulator = Simulator(network, algorithm, metrics, verbose=False)
        
        result = simulator.run(num_rounds=500, convergence_threshold=1e-6)
        results.append({
            "network_size": size,
            "convergence_rounds": result['total_rounds'],
            "final_error": result['final_error'],
            "total_messages": sum(metrics.messages),
        })
    
    # Print results
    print("\nResults:")
    print(f"{'Network Size':<15} {'Convergence Rounds':<20} {'Total Messages':<20}")
    print("-" * 55)
    for r in results:
        print(f"{r['network_size']:<15} {r['convergence_rounds']:<20} {r['total_messages']:<20}")
    
    return network_sizes, results


def example_edge_probability_sensitivity():
    """Analyze impact of edge probability on convergence (for random networks)."""
    print("\n" + "=" * 60)
    print("Example 2b: Edge Probability Sensitivity (Random Networks)")
    print("=" * 60)
    
    edge_probs = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
    num_nodes = 50
    results = []
    
    for prob in edge_probs:
        print(f"Simulating random network with edge probability {prob}...")
        
        network = Network(num_nodes, initial_values=np.random.uniform(0, 1, num_nodes), seed=42)
        
        if prob < 1.0:
            network.create_random_graph(prob)
        else:
            network.create_complete_graph()
        
        # Check connectivity
        avg_degree = np.mean([len(node.neighbors) for node in network.get_all_nodes()])
        
        algorithm = PushGossip(seed=42)
        metrics = MetricsCollector()
        simulator = Simulator(network, algorithm, metrics, verbose=False)
        
        result = simulator.run(num_rounds=500, convergence_threshold=1e-6)
        results.append({
            "edge_probability": prob,
            "avg_degree": avg_degree,
            "convergence_rounds": result['total_rounds'],
            "final_error": result['final_error'],
        })
    
    # Print results
    print("\nResults:")
    print(f"{'Edge Prob':<12} {'Avg Degree':<15} {'Conv. Rounds':<15}")
    print("-" * 42)
    for r in results:
        print(f"{r['edge_probability']:<12.1f} {r['avg_degree']:<15.2f} {r['convergence_rounds']:<15}")
    
    return edge_probs, results


def example_push_pull_ratio_sensitivity():
    """Analyze impact of push-pull ratio on convergence."""
    print("\n" + "=" * 60)
    print("Example 2c: Push-Pull Ratio Sensitivity")
    print("=" * 60)
    
    push_probs = [0.0, 0.25, 0.5, 0.75, 1.0]
    num_nodes = 50
    results = []
    
    for push_prob in push_probs:
        print(f"Simulating with push probability {push_prob}...")
        
        network = Network(num_nodes, initial_values=np.random.uniform(0, 1, num_nodes), seed=42)
        network.create_complete_graph()
        
        algorithm = PushPullGossip(push_probability=push_prob, seed=42)
        metrics = MetricsCollector()
        simulator = Simulator(network, algorithm, metrics, verbose=False)
        
        result = simulator.run(num_rounds=500, convergence_threshold=1e-6)
        results.append({
            "push_probability": push_prob,
            "convergence_rounds": result['total_rounds'],
            "final_error": result['final_error'],
            "total_messages": sum(metrics.messages),
        })
    
    # Print results
    print("\nResults:")
    print(f"{'Push Prob':<12} {'Conv. Rounds':<15} {'Avg Error':<15}")
    print("-" * 42)
    for r in results:
        print(f"{r['push_probability']:<12.2f} {r['convergence_rounds']:<15} {r['final_error']:<15.2e}")
    
    return push_probs, results


def example_multiple_trials():
    """Run multiple trials and analyze variability."""
    print("\n" + "=" * 60)
    print("Example 2d: Multiple Trials Analysis")
    print("=" * 60)
    
    num_trials = 10
    num_nodes = 50
    
    network = Network(num_nodes, initial_values=np.random.uniform(0, 1, num_nodes), seed=42)
    network.create_complete_graph()
    
    algorithm = PushGossip()  # Different seed each time
    metrics = MetricsCollector()
    simulator = Simulator(network, algorithm, metrics, verbose=False)
    
    print(f"Running {num_trials} trials...")
    aggregated_results = simulator.run_multiple_trials(
        num_trials=num_trials,
        num_rounds=500,
        convergence_threshold=1e-6,
    )
    
    print("\nResults Summary:")
    print(f"Average convergence rounds: {aggregated_results['total_rounds']:.1f}")
    print(f"Mean final error: {np.mean([r['final_error'] for r in aggregated_results['all_results']]):.2e}")
    print(f"Min final error: {np.min([r['final_error'] for r in aggregated_results['all_results']]):.2e}")
    print(f"Max final error: {np.max([r['final_error'] for r in aggregated_results['all_results']]):.2e}")
    
    return aggregated_results


if __name__ == "__main__":
    # Run sensitivity analyses
    sizes, size_results = example_network_size_sensitivity()
    edge_probs, edge_results = example_edge_probability_sensitivity()
    push_probs, push_pull_results = example_push_pull_ratio_sensitivity()
    multi_trial_results = example_multiple_trials()
    
    # Create visualizations
    visualizer = GossipVisualizer()
    
    # Plot network size sensitivity
    size_conv = [r['convergence_rounds'] for r in size_results]
    fig1 = visualizer.plot_parameter_sensitivity(
        "Network Size",
        sizes,
        [{"convergence_rounds": c} for c in size_conv],
        metric_key="convergence_rounds",
        title="Impact of Network Size on Convergence"
    )
    fig1.savefig("/tmp/sensitivity_network_size.png", dpi=150)
    
    # Plot edge probability sensitivity
    edge_conv = [r['convergence_rounds'] for r in edge_results]
    fig2 = visualizer.plot_parameter_sensitivity(
        "Edge Probability",
        edge_probs,
        [{"convergence_rounds": c} for c in edge_conv],
        metric_key="convergence_rounds",
        title="Impact of Edge Probability on Convergence"
    )
    fig2.savefig("/tmp/sensitivity_edge_prob.png", dpi=150)
    
    print("\nVisualizations saved to /tmp/")
    print("Examples completed successfully!")
