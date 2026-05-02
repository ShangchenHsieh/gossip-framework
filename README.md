# Gossip Algorithm Simulation Framework

A flexible, modular Python framework for simulating randomized gossip and consensus algorithms with support for various network topologies, parameters, and visualization using Seaborn and Matplotlib.

## Overview

This framework enables researchers and practitioners to:

1. **Simulate gossip algorithms** - Implement and test various gossip-based consensus protocols
2. **Configure networks** - Support for multiple network topologies (complete, random, ring, lattice, scale-free)
3. **Tune parameters** - Easily experiment with different algorithm parameters and settings
4. **Analyze performance** - Track convergence, communication complexity, and other metrics
5. **Visualize results** - Generate publication-quality plots with Seaborn and Matplotlib

## Architecture

```
gossip_framework/
├── core/                 # Core simulation components
│   ├── node.py          # Node class
│   ├── network.py       # Network topology management
│   └── simulator.py      # Main simulation engine
├── algorithms/          # Algorithm implementations
│   ├── base.py          # Abstract base class
│   └── gossip_protocols.py  # Push, Pull, Push-Pull algorithms
├── metrics/             # Metrics collection
│   └── __init__.py      # MetricsCollector class
├── visualization/       # Plotting and visualization
│   └── __init__.py      # GossipVisualizer class
├── config.py           # Configuration management
└── examples/           # Example simulations
    ├── example1_basic.py              # Basic usage
    ├── example2_parameter_tuning.py  # Parameter sensitivity
    └── example3_advanced.py          # Custom algorithms
```

## Installation

1. Ensure you have Python 3.7+ installed
2. Install dependencies:
   ```bash
   pip install numpy matplotlib seaborn
   ```

## Quick Start

### Basic Simulation

```python
from gossip_framework import Network, Simulator, PushGossip, MetricsCollector
import numpy as np

# Create network with 50 nodes
network = Network(50, initial_values=np.random.uniform(0, 1, 50))
network.create_complete_graph()

# Create algorithm and simulator
algorithm = PushGossip(seed=42)
metrics = MetricsCollector()
simulator = Simulator(network, algorithm, metrics)

# Run simulation
results = simulator.run(num_rounds=100, convergence_threshold=1e-6)

print(f"Converged in {results['total_rounds']} rounds")
```

### Visualizing Results

```python
from gossip_framework import GossipVisualizer

visualizer = GossipVisualizer()

# Plot convergence
fig = visualizer.plot_convergence(metrics)
fig.savefig("convergence.png")

# Plot node trajectories
fig = visualizer.plot_node_trajectories(metrics)
fig.savefig("trajectories.png")

# Plot message complexity
fig = visualizer.plot_message_complexity(metrics)
fig.savefig("messages.png")
```

### Comparing Algorithms

```python
from gossip_framework import PushGossip, PullGossip, PushPullGossip

algorithms = {
    'Push': PushGossip(),
    'Pull': PullGossip(),
    'Push-Pull': PushPullGossip(),
}

results_dict = {}
for name, algo in algorithms.items():
    network = Network(50, seed=42)
    network.create_complete_graph()
    
    simulator = Simulator(network, algo)
    result = simulator.run(num_rounds=100)
    results_dict[name] = simulator.metrics

# Compare convergence
visualizer.plot_multiple_convergence(results_dict)
```

## Core Components

### Network

The `Network` class manages the network topology:

```python
network = Network(num_nodes=50, initial_values=initial_vals)

# Create different topologies
network.create_complete_graph()        # All-to-all
network.create_random_graph(0.3)       # Random with edge probability
network.create_ring_graph()            # Ring topology
network.create_lattice_graph(7)        # 2D lattice
network.create_scale_free_graph(m=2)   # Scale-free (Barabási-Albert)

# Manually add/remove edges
network.add_edge(0, 1, bidirectional=True)
network.remove_edge(0, 1, bidirectional=True)
```

### Algorithms

Available algorithms:

- **PushGossip**: Each node sends to a random neighbor
- **PullGossip**: Each node pulls from a random neighbor
- **PushPullGossip**: Combines push and pull operations
- **RandomAveraging**: Supports multiple interactions per round

Create custom algorithms by subclassing `GossipAlgorithm`:

```python
from gossip_framework import GossipAlgorithm

class MyCustomAlgorithm(GossipAlgorithm):
    def __init__(self):
        super().__init__("My Algorithm")
    
    def round(self, network):
        messages_sent = 0
        # Implement your algorithm here
        return messages_sent
```

### Simulator

The `Simulator` runs the simulations:

```python
simulator = Simulator(network, algorithm, metrics)

# Single run
results = simulator.run(
    num_rounds=100,
    convergence_threshold=1e-6,
    early_stopping=True,
    verbose=True
)

# Multiple trials
aggregated = simulator.run_multiple_trials(
    num_trials=10,
    num_rounds=100
)
```

### Visualization

The `GossipVisualizer` class provides various plots:

```python
visualizer = GossipVisualizer()

# Convergence analysis
visualizer.plot_convergence(metrics)
visualizer.plot_multiple_convergence(results_dict)

# Communication analysis
visualizer.plot_message_complexity(metrics)

# Parameter sensitivity
visualizer.plot_parameter_sensitivity(
    "Network Size",
    [10, 20, 50, 100],
    results_list,
    metric_key="convergence_rounds"
)

# Node trajectories
visualizer.plot_node_trajectories(metrics)

# State heatmap
visualizer.plot_heatmap_convergence(metrics)
```

## Configuration

Use `SimulationConfig` to manage simulation parameters:

```python
from gossip_framework import SimulationConfig

config = SimulationConfig(
    num_nodes=100,
    network_type="random",
    edge_probability=0.3,
    algorithm="push_gossip",
    num_rounds=200,
    num_trials=5,
)

# Save to file
config.to_json("config.json")

# Load from file
config = SimulationConfig.from_json("config.json")
```

## Examples

The framework includes three comprehensive examples:

### Example 1: Basic Usage
Run basic simulations and visualize results
```bash
python examples/example1_basic.py
```

### Example 2: Parameter Tuning
Analyze sensitivity to parameters like network size, edge probability, and algorithm settings
```bash
python examples/example2_parameter_tuning.py
```

### Example 3: Advanced Usage
Implement custom algorithms, simulate dynamic networks, and create visualization galleries
```bash
python examples/example3_advanced.py
```

## Key Metrics

The framework tracks:

- **Convergence error**: Maximum deviation from target average value
- **Communication complexity**: Total messages sent per round
- **Convergence time**: Rounds to reach target accuracy
- **Node trajectories**: Individual node value evolution
- **Network statistics**: Degree distribution, connectivity metrics

## Citation and References

This framework was developed for CS 262 (Randomized Algorithms) based on:

- Boyd et al., "Randomized Gossip Algorithms" (IEEE/ACM Trans. Netw., 2006)
- Kempe et al., "Gossip-based computation of aggregate information" (FOCS, 2003)
- Dimakis et al., "Geographic gossip: efficient aggregation for sensor networks" (IPSN, 2006)

## Performance Considerations

- **Complete graphs**: O(log n) convergence time
- **Random graphs**: Depends on edge probability
- **Ring/Lattice**: Slower convergence (O(n²) in worst case)
- **Scale-free**: Varies with network parameters

## Extending the Framework

To add a new algorithm:

1. Create a class inheriting from `GossipAlgorithm`
2. Implement the `round(network)` method
3. Return the number of messages sent

```python
class MyAlgorithm(GossipAlgorithm):
    def round(self, network):
        # Your algorithm logic
        return messages_count
```

## Tips and Best Practices

1. **Use seeds** for reproducibility:
   ```python
   network = Network(50, seed=42)
   algorithm = PushGossip(seed=42)
   ```

2. **Enable verbose mode** to monitor progress:
   ```python
   simulator = Simulator(network, algorithm, verbose=True)
   ```

3. **Run multiple trials** to assess variability:
   ```python
   results = simulator.run_multiple_trials(num_trials=10, num_rounds=100)
   ```

4. **Save figures at high DPI** for publications:
   ```python
   fig.savefig("plot.png", dpi=300, bbox_inches='tight')
   ```

5. **Use early stopping** to save computation:
   ```python
   simulator.run(num_rounds=1000, convergence_threshold=1e-8, early_stopping=True)
   ```

## License

This framework is provided for educational and research purposes.

## Contact

For questions or issues, please refer to the documentation in the docstrings or the example files.
