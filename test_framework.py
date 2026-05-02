#!/usr/bin/env python3
"""
Framework Test Suite
Validates that the framework is working correctly.
"""

import sys
import traceback
from pathlib import Path

# Add framework to path (ensure parent dir is on sys.path so top-level
# package imports like `import gossip_framework` work when running the
# test file from inside the package directory)
framework_path = Path(__file__).parent.parent
sys.path.insert(0, str(framework_path.resolve()))


def test_imports():
    """Test that all modules can be imported."""
    print("[TEST 1] Testing imports...")
    try:
        from gossip_framework import (
            Network, Node, Simulator,
            PushGossip, PullGossip, PushPullGossip, RandomAveraging,
            MetricsCollector, GossipVisualizer, SimulationConfig
        )
        print("✓ All imports successful")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        traceback.print_exc()
        return False


def test_network_creation():
    """Test network creation and topology methods."""
    print("\n[TEST 2] Testing network creation...")
    try:
        from gossip_framework import Network
        import numpy as np
        
        # Test various topologies
        topologies = [
            ("Complete", lambda n: n.create_complete_graph()),
            ("Random", lambda n: n.create_random_graph(0.5)),
            ("Ring", lambda n: n.create_ring_graph()),
            ("Scale-free", lambda n: n.create_scale_free_graph(m=2)),
        ]
        
        for topo_name, topo_func in topologies:
            n = Network(16, np.random.uniform(0, 1, 16), seed=42)
            topo_func(n)
            print(f"  ✓ {topo_name} topology")
        
        return True
    except Exception as e:
        print(f"✗ Network creation failed: {e}")
        traceback.print_exc()
        return False


def test_algorithms():
    """Test algorithm implementations."""
    print("\n[TEST 3] Testing algorithms...")
    try:
        from gossip_framework import (
            Network, PushGossip, PullGossip, 
            PushPullGossip, RandomAveraging
        )
        import numpy as np
        
        network = Network(20, np.random.uniform(0, 1, 20), seed=42)
        network.create_complete_graph()
        
        algorithms = [
            ("Push", PushGossip()),
            ("Pull", PullGossip()),
            ("Push-Pull", PushPullGossip()),
            ("Random Averaging", RandomAveraging()),
        ]
        
        for algo_name, algo in algorithms:
            messages = algo.round(network)
            assert messages >= 0, f"Invalid message count: {messages}"
            print(f"  ✓ {algo_name} algorithm")
        
        return True
    except Exception as e:
        print(f"✗ Algorithm test failed: {e}")
        traceback.print_exc()
        return False


def test_simulation():
    """Test simulator execution."""
    print("\n[TEST 4] Testing simulator...")
    try:
        from gossip_framework import (
            Network, Simulator, PushGossip, MetricsCollector
        )
        import numpy as np
        
        network = Network(30, np.random.uniform(0, 1, 30), seed=42)
        network.create_complete_graph()
        
        algorithm = PushGossip(seed=42)
        metrics = MetricsCollector()
        simulator = Simulator(network, algorithm, metrics, verbose=False)
        
        # Test single run
        results = simulator.run(num_rounds=50, convergence_threshold=1e-6)
        assert "total_rounds" in results
        assert "final_error" in results
        print("  ✓ Single run simulation")
        
        # Test multiple trials
        network.reset_nodes()
        aggregated = simulator.run_multiple_trials(num_trials=3, num_rounds=30)
        assert "num_trials" in aggregated
        print("  ✓ Multiple trials simulation")
        
        return True
    except Exception as e:
        print(f"✗ Simulator test failed: {e}")
        traceback.print_exc()
        return False


def test_visualization():
    """Test visualization creation."""
    print("\n[TEST 5] Testing visualization...")
    try:
        from gossip_framework import (
            Network, Simulator, PushGossip, 
            MetricsCollector, GossipVisualizer
        )
        import numpy as np
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        
        # Create sample data
        network = Network(20, np.random.uniform(0, 1, 20), seed=42)
        network.create_complete_graph()
        
        simulator = Simulator(network, PushGossip(seed=42), MetricsCollector())
        simulator.run(num_rounds=50)
        metrics = simulator.metrics
        
        visualizer = GossipVisualizer()
        
        # Test various plots
        plots = [
            ("Convergence", lambda: visualizer.plot_convergence(metrics)),
            ("Message Complexity", lambda: visualizer.plot_message_complexity(metrics)),
            ("Node Trajectories", lambda: visualizer.plot_node_trajectories(metrics)),
            ("Heatmap", lambda: visualizer.plot_heatmap_convergence(metrics)),
        ]
        
        for plot_name, plot_func in plots:
            fig = plot_func()
            assert fig is not None
            print(f"  ✓ {plot_name} plot")
        
        return True
    except Exception as e:
        print(f"✗ Visualization test failed: {e}")
        traceback.print_exc()
        return False


def test_configuration():
    """Test configuration management."""
    print("\n[TEST 6] Testing configuration...")
    try:
        from gossip_framework import SimulationConfig
        import tempfile
        import os
        
        config = SimulationConfig(
            num_nodes=50,
            network_type="random",
            num_rounds=100
        )
        
        # Test to_dict
        config_dict = config.to_dict()
        assert isinstance(config_dict, dict)
        print("  ✓ Configuration to_dict()")
        
        # Test JSON save/load
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_file = f.name
        
        try:
            config.to_json(temp_file)
            loaded_config = SimulationConfig.from_json(temp_file)
            assert loaded_config.num_nodes == 50
            print("  ✓ Configuration JSON save/load")
        finally:
            os.unlink(temp_file)
        
        return True
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("GOSSIP FRAMEWORK - TEST SUITE")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_network_creation,
        test_algorithms,
        test_simulation,
        test_visualization,
        test_configuration,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"✗ Test crashed: {e}")
            results.append(False)
    
    # Summary
    print("\n" + "=" * 60)
    print(f"RESULTS: {sum(results)}/{len(results)} tests passed")
    print("=" * 60)
    
    if all(results):
        print("✓ All tests passed! Framework is ready to use.")
        return 0
    else:
        print("✗ Some tests failed. Check output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
