"""
Example 1: Basic Gossip Algorithm Simulation
Demonstrates running a simple push gossip simulation and plotting results.
"""

import numpy as np
import sys
sys.path.insert(0, '/Users/sean/Classes/CS_262')

from gossip_framework import (
    Network, Simulator, PushGossip, PullGossip, 
    MetricsCollector, GossipVisualizer, SimulationConfig
)


def example_basic_simulation():
    """Run a basic single algorithm simulation."""
    print("=" * 60)
    print("Example 1: Basic Push Gossip Simulation")
    print("=" * 60)
    
    # Create network
    num_nodes = 50
    initial_values = np.random.uniform(0, 1, num_nodes)
    network = Network(num_nodes, initial_values=initial_values, seed=42)
    network.create_complete_graph()
    
    print(f"Network: {network}")
    print(f"Initial average value: {network.get_average_value():.6f}")
    
    # Create algorithm and simulator
    algorithm = PushGossip(seed=42)
    metrics = MetricsCollector()
    simulator = Simulator(network, algorithm, metrics, verbose=True)
    
    # Run simulation
    results = simulator.run(
        num_rounds=100,
        convergence_threshold=1e-6,
        early_stopping=True
    )
    
    # Print results
    print(f"\nSimulation Results:")
    print(f"  Rounds to convergence: {results['total_rounds']}")
    print(f"  Final error: {results['final_error']:.2e}")
    print(f"  Target average: {results['target_average']:.6f}")
    print(f"  Final average: {np.mean(results['final_state']):.6f}")
    
    return network, metrics, results


def example_algorithm_comparison():
    """Compare different gossip algorithms on the same network."""
    print("\n" + "=" * 60)
    print("Example 2: Algorithm Comparison")
    print("=" * 60)
    
    algorithms = {
        'Push': PushGossip(seed=42),
        'Pull': PullGossip(seed=42),
    }
    
    results_dict = {}
    
    for algo_name, algo in algorithms.items():
        print(f"\nRunning {algo_name}...")
        
        # Create fresh network for each algorithm
        num_nodes = 50
        initial_values = np.random.uniform(0, 1, num_nodes)
        network = Network(num_nodes, initial_values=initial_values, seed=42)
        network.create_complete_graph()
        
        metrics = MetricsCollector()
        simulator = Simulator(network, algo, metrics, verbose=False)
        
        result = simulator.run(num_rounds=100, convergence_threshold=1e-6)
        results_dict[algo_name] = metrics
        
        print(f"  {algo_name}: {result['total_rounds']} rounds to convergence")
    
    return results_dict


def example_network_topology_impact():
    """Study impact of network topology on convergence."""
    print("\n" + "=" * 60)
    print("Example 3: Network Topology Impact")
    print("=" * 60)
    
    topologies = {
        'Complete': lambda n: Network(n, np.random.uniform(0, 1, n), seed=42),
        'Random (p=0.3)': lambda n: Network(n, np.random.uniform(0, 1, n), seed=42),
        'Ring': lambda n: Network(n, np.random.uniform(0, 1, n), seed=42),
        'Lattice': lambda n: Network(int(np.sqrt(n))**2, np.random.uniform(0, 1, int(np.sqrt(n))**2), seed=42),
    }
    
    results_dict = {}
    num_nodes = 100
    
    # Complete
    net = topologies['Complete'](num_nodes)
    net.create_complete_graph()
    results_dict['Complete'] = _run_simulation(net, num_nodes)
    print(f"Complete: {results_dict['Complete']['total_rounds']} rounds")
    
    # Random
    net = topologies['Random (p=0.3)'](num_nodes)
    net.create_random_graph(0.3)
    results_dict['Random (p=0.3)'] = _run_simulation(net, num_nodes)
    print(f"Random: {results_dict['Random (p=0.3)']['total_rounds']} rounds")
    
    # Ring
    net = topologies['Ring'](num_nodes)
    net.create_ring_graph()
    results_dict['Ring'] = _run_simulation(net, num_nodes)
    print(f"Ring: {results_dict['Ring']['total_rounds']} rounds")
    
    # Lattice
    sqrt_n = int(np.sqrt(num_nodes))
    net = topologies['Lattice'](sqrt_n * sqrt_n)
    net.create_lattice_graph(sqrt_n)
    results_dict['Lattice'] = _run_simulation(net, sqrt_n * sqrt_n)
    print(f"Lattice: {results_dict['Lattice']['total_rounds']} rounds")
    
    return results_dict


def _run_simulation(network, num_nodes):
    """Helper to run a simulation."""
    algorithm = PushGossip(seed=42)
    metrics = MetricsCollector()
    simulator = Simulator(network, algorithm, metrics, verbose=False)
    return simulator.run(num_rounds=1000, convergence_threshold=1e-6)


if __name__ == "__main__":
    # Run examples
    network, metrics, results = example_basic_simulation()
    
    # Visualize basic simulation
    visualizer = GossipVisualizer()
    fig1 = visualizer.plot_convergence(metrics, title="Push Gossip Convergence (Complete Graph)")
    fig1.savefig("/tmp/example_convergence.png", dpi=150)
    
    # Compare algorithms
    algo_results = example_algorithm_comparison()
    fig2 = visualizer.plot_multiple_convergence(algo_results, title="Algorithm Comparison")
    fig2.savefig("/tmp/example_algo_comparison.png", dpi=150)
    
    # Topology impact
    topo_results = example_network_topology_impact()
    print("\nSaved visualizations to /tmp/")
    
    print("\nExamples completed successfully!")
