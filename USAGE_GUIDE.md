"""
FRAMEWORK STRUCTURE AND USAGE GUIDE

This document provides an overview of the Gossip Algorithm Simulation Framework
and how to use it for your CS 262 final project.
"""

# ==============================================================================
# FRAMEWORK OVERVIEW
# ==============================================================================

The framework is structured in three main layers:

┌─────────────────────────────────────────────────────────────────────┐
│                         VISUALIZATION LAYER                          │
│              (Seaborn/Matplotlib plotting utilities)                 │
└─────────────────────────────────────────────────────────────────────┘
                                    ▲
                                    │
┌─────────────────────────────────────────────────────────────────────┐
│                      SIMULATION LAYER                                │
│        (Simulator orchestrates algorithms on networks)               │
└─────────────────────────────────────────────────────────────────────┘
                                    ▲
                    ┌───────────────┴───────────────┐
                    │                               │
         ┌──────────▼───────────┐       ┌──────────▼──────────┐
         │   ALGORITHM LAYER    │       │  NETWORK LAYER      │
         │  (Gossip protocols)  │       │  (Topologies)       │
         │ - Push               │       │ - Complete          │
         │ - Pull               │       │ - Random            │
         │ - Push-Pull          │       │ - Ring              │
         │ - Random Averaging   │       │ - Lattice           │
         └──────────────────────┘       │ - Scale-free        │
                                        └─────────────────────┘


# ==============================================================================
# QUICK START (5 MINUTES)
# ==============================================================================

1. Install dependencies:
   $ pip install numpy matplotlib seaborn

2. Navigate to framework directory:
   $ cd /Users/sean/Classes/CS_262/gossip_framework

3. Run quick start:
   $ python quickstart.py

   This will:
   - Create a 50-node network
   - Run Push Gossip simulation
   - Generate 5 visualization plots
   - Compare two algorithms
   - Analyze network size sensitivity


# ==============================================================================
# KEY MODULES AND CLASSES
# ==============================================================================

1. CORE SIMULATION (core/)
   ├── Network(num_nodes, initial_values, seed)
   │   ├── .create_complete_graph()
   │   ├── .create_random_graph(edge_probability)
   │   ├── .create_ring_graph()
   │   ├── .create_lattice_graph()
   │   ├── .create_scale_free_graph()
   │   ├── .add_edge(node_a, node_b)
   │   └── .get_average_value()
   │
   ├── Node(node_id, initial_value)
   │   ├── .value (current state)
   │   ├── .neighbors (set of neighbor IDs)
   │   └── .history (state over time)
   │
   └── Simulator(network, algorithm, metrics, verbose)
       ├── .run(num_rounds, convergence_threshold, early_stopping)
       └── .run_multiple_trials(num_trials, num_rounds)

2. ALGORITHMS (algorithms/)
   ├── GossipAlgorithm (base class)
   ├── PushGossip
   ├── PullGossip
   ├── PushPullGossip(push_probability)
   └── RandomAveraging(interactions_per_node)

3. METRICS (metrics/)
   └── MetricsCollector()
       ├── .record_state(timestamp, state)
       ├── .record_messages(timestamp, num_messages)
       ├── .record_error(timestamp, error)
       └── .get_summary()

4. VISUALIZATION (visualization/)
   └── GossipVisualizer()
       ├── .plot_convergence(metrics)
       ├── .plot_multiple_convergence(results_dict)
       ├── .plot_message_complexity(metrics)
       ├── .plot_node_trajectories(metrics)
       ├── .plot_heatmap_convergence(metrics)
       └── .plot_parameter_sensitivity(param_name, values, results)


# ==============================================================================
# COMMON WORKFLOWS
# ==============================================================================

WORKFLOW 1: Basic Simulation
────────────────────────────

import numpy as np
from gossip_framework import Network, Simulator, PushGossip, MetricsCollector

# Setup
network = Network(50, initial_values=np.random.uniform(0, 1, 50))
network.create_complete_graph()

# Run
algorithm = PushGossip(seed=42)
metrics = MetricsCollector()
simulator = Simulator(network, algorithm, metrics)
results = simulator.run(num_rounds=100, convergence_threshold=1e-6)

# Analyze
print(f"Converged in {results['total_rounds']} rounds")
print(f"Final error: {results['final_error']:.2e}")


WORKFLOW 2: Compare Algorithms
───────────────────────────────

from gossip_framework import PullGossip, PushPullGossip

algorithms = {
    'Push': PushGossip(),
    'Pull': PullGossip(),
    'Push-Pull': PushPullGossip(),
}

results_dict = {}
for name, algo in algorithms.items():
    net = Network(50, seed=42)
    net.create_complete_graph()
    sim = Simulator(net, algo)
    result = sim.run(num_rounds=100)
    results_dict[name] = sim.metrics

visualizer = GossipVisualizer()
visualizer.plot_multiple_convergence(results_dict)


WORKFLOW 3: Parameter Sensitivity Analysis
────────────────────────────────────────────

# Analyze impact of network size
sizes = [10, 25, 50, 100, 200]
results = []

for size in sizes:
    network = Network(size, seed=42)
    network.create_complete_graph()
    simulator = Simulator(network, PushGossip(), verbose=False)
    result = simulator.run(num_rounds=500)
    results.append({
        "size": size,
        "convergence_rounds": result['total_rounds']
    })

visualizer.plot_parameter_sensitivity(
    "Network Size",
    sizes,
    results,
    metric_key="convergence_rounds"
)


WORKFLOW 4: Multiple Trials with Variability
──────────────────────────────────────────────

network = Network(50)
network.create_complete_graph()

simulator = Simulator(network, PushGossip())

# Run 10 independent trials
aggregated = simulator.run_multiple_trials(
    num_trials=10,
    num_rounds=100
)

print(f"Mean convergence: {np.mean(aggregated['error_mean'])}")
print(f"Std dev: {np.mean(aggregated['error_std'])}")


WORKFLOW 5: Custom Algorithm
──────────────────────────────

from gossip_framework import GossipAlgorithm

class MyAlgorithm(GossipAlgorithm):
    def __init__(self):
        super().__init__("My Algorithm")
    
    def round(self, network):
        messages = 0
        # Your algorithm implementation
        for node in network.get_all_nodes():
            if node.neighbors:
                neighbor = np.random.choice(list(node.neighbors))
                # Perform gossip step
                messages += 1
        return messages

# Use like any other algorithm
algorithm = MyAlgorithm()
simulator = Simulator(network, algorithm)
results = simulator.run(num_rounds=100)


# ==============================================================================
# NETWORK TOPOLOGIES REFERENCE
# ==============================================================================

Complete Graph (All-to-all):
    network.create_complete_graph()
    • Fastest convergence: O(log n)
    • Highest message complexity
    • Best for upper bounds analysis

Random Graph (Erdős–Rényi):
    network.create_random_graph(edge_probability=0.3)
    • Tunable connectivity
    • Realistic sparse networks
    • Parameter: edge probability (0-1)

Ring Topology:
    network.create_ring_graph()
    • Each node connected to 2 neighbors
    • Slower convergence (bottleneck)
    • Circular structure

Lattice/Grid (2D):
    network.create_lattice_graph(grid_side=10)
    • 2D grid structure
    • num_nodes must be perfect square
    • Good for spatial applications

Scale-Free (Barabási-Albert):
    network.create_scale_free_graph(m=2)
    • Power-law degree distribution
    • Realistic for many real networks
    • Parameter m: edges per new node


# ==============================================================================
# VISUALIZATION OPTIONS
# ==============================================================================

1. plot_convergence(metrics)
   → Single algorithm convergence vs. rounds (log scale)
   → Shows error trajectory

2. plot_multiple_convergence(results_dict)
   → Compare multiple algorithms
   → Error vs. rounds for each algorithm

3. plot_message_complexity(metrics)
   → Messages per round (bar chart)
   → Cumulative messages (line chart)

4. plot_node_trajectories(metrics, num_nodes_to_plot=5)
   → Individual node value evolution
   → Useful for analyzing node synchronization

5. plot_heatmap_convergence(metrics)
   → Heatmap: nodes × rounds
   → Visual representation of convergence

6. plot_parameter_sensitivity(param_name, values, results, metric_key)
   → Study impact of parameters
   → Examples: network size, edge probability, algorithm parameters


# ==============================================================================
# ANALYSIS METRICS
# ==============================================================================

Error:
    max_error = max(|node_value - target_average|)
    • Tracked per round
    • Usually plotted on log scale
    • Convergence threshold: when error < threshold

Messages:
    total_messages = sum of messages per round
    • Communication complexity
    • Important for bandwidth-constrained scenarios

Convergence Time:
    rounds_to_convergence = first round where error < threshold
    • Efficiency metric
    • Compared across algorithms and topologies

Message Efficiency:
    messages_per_convergence = total_messages / convergence_rounds
    • Trade-off between speed and communication


# ==============================================================================
# TIPS FOR YOUR PROJECT
# ==============================================================================

1. For Experimental Validation:
   ✓ Use random graphs with varying edge probabilities
   ✓ Compare Push, Pull, and Push-Pull
   ✓ Analyze network size scaling
   ✓ Plot with error bars (run multiple trials)

2. For Theoretical Analysis:
   ✓ Complete graphs for upper bounds
   ✓ Ring/Lattice for lower bounds
   ✓ Scale-free for realistic scenarios
   ✓ Verify against Boyd et al. Theorem 5 predictions

3. For Visualizations:
   ✓ Save at high DPI (300) for publications
   ✓ Use log scale for error plots
   ✓ Include error bars/bands for multiple trials
   ✓ Label axes clearly with units

4. For Custom Experiments:
   ✓ Subclass GossipAlgorithm for new protocols
   ✓ Modify Network for custom topologies
   ✓ Extend MetricsCollector for custom metrics
   ✓ Create new Visualizer methods as needed


# ==============================================================================
# FILE LOCATIONS SUMMARY
# ==============================================================================

/Users/sean/Classes/CS_262/gossip_framework/
├── README.md                          ← Full documentation
├── quickstart.py                      ← 5-minute test
├── requirements.txt                   ← Dependencies
├── config.py                          ← Configuration
├── core/
│   ├── node.py                        ← Node class
│   ├── network.py                     ← Network topologies
│   └── simulator.py                   ← Main simulation
├── algorithms/
│   ├── base.py                        ← Base algorithm class
│   └── gossip_protocols.py            ← 4 algorithm implementations
├── metrics/
│   └── __init__.py                    ← MetricsCollector
├── visualization/
│   └── __init__.py                    ← GossipVisualizer (6 plot types)
└── examples/
    ├── example1_basic.py              ← Basic usage tutorial
    ├── example2_parameter_tuning.py   ← Parameter analysis
    └── example3_advanced.py           ← Custom algorithms & advanced


# ==============================================================================
# GETTING HELP
# ==============================================================================

1. Start with README.md for comprehensive documentation
2. Run quickstart.py to verify setup
3. Check examples/ for concrete usage patterns
4. Read docstrings in source files for details
5. Refer to this guide for quick reference

Questions about your paper's algorithms?
→ Check algorithms/gossip_protocols.py for implementation
→ Refer to comments in the code
→ See References section in README.md

"""

if __name__ == "__main__":
    print(__doc__)
