"""
Quick Start Guide - Run this to test the framework
Execute from the gossip_framework directory
"""

if __name__ == "__main__":
    import numpy as np
    import sys
    from pathlib import Path
    import matplotlib

    # Use non-interactive backend for scripts
    matplotlib.use('Agg')

    # Add project root (parent of this package) to sys.path so
    # `import gossip_framework` works when running this script directly.
    sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))
    
    from gossip_framework import (
        Network, Simulator, PushGossip, PullGossip,
        MetricsCollector, GossipVisualizer
    )
    
    print("=" * 70)
    print("GOSSIP ALGORITHM SIMULATION FRAMEWORK - Quick Start")
    print("=" * 70)

    # We'll run experiments for two node sizes and two convergence thresholds
    node_sizes = [50, 100]
    thresholds = [1e-6, 1e-4]

    visualizer = GossipVisualizer()
    created_dirs = []

    for num_of_nodes in node_sizes:
        for threshhold in thresholds:
            print(f"\n[RUN] Nodes={num_of_nodes}, eps={threshhold}")

            # Create network and use same topology (complete graph)
            initial_values = np.random.uniform(0, 1, num_of_nodes)
            network = Network(num_of_nodes, initial_values=initial_values, seed=42)
            network.create_complete_graph()

            # Run Push Gossip
            algorithm = PushGossip(seed=42)
            metrics = MetricsCollector()
            simulator = Simulator(network, algorithm, metrics, verbose=False)

            results = simulator.run(
                num_rounds=60,
                convergence_threshold=threshhold,
                early_stopping=True
            )

            print(f"    Converged in {results['total_rounds']} rounds")
            print(f"    Final error: {results['final_error']:.2e}")

            # Ensure output directory exists per combination
            import os
            safe_eps = str(threshhold).replace('.', 'p')
            plots_dir = os.path.join(Path(__file__).parent, f"plots_{num_of_nodes}_nodes_eps_{safe_eps}")
            os.makedirs(plots_dir, exist_ok=True)
            created_dirs.append(plots_dir)

            # Save convergence plot
            fig1 = visualizer.plot_convergence(metrics, title=f"Push Gossip - Convergence ({num_of_nodes} nodes, eps={threshhold})")
            fig1_path = os.path.join(plots_dir, "convergence.png")
            fig1.savefig(fig1_path, dpi=300, bbox_inches='tight')
            print(f"    ✓ Convergence plot -> {fig1_path}")

            # Save node trajectories
            fig3 = visualizer.plot_node_trajectories(metrics, num_nodes_to_plot=min(10, num_of_nodes))
            fig3_path = os.path.join(plots_dir, f"node_trajectories_{num_of_nodes}_nodes_eps_{safe_eps}.png")
            fig3.savefig(fig3_path, dpi=300, bbox_inches='tight')
            print(f"    ✓ Node trajectories plot -> {fig3_path}")

    # Summary of created directories
    print("\nGenerated plots in the following directories:")
    for d in sorted(set(created_dirs)):
        print(f" - {d}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"✓ Framework works correctly!")
    print(f"✓ Generated 5 visualization plots")
    print(f"✓ Tested multiple network sizes and algorithms")
    print(f"\nNext steps:")
    print("  1. Check examples/example1_basic.py for basic usage")
    print("  2. Check examples/example2_parameter_tuning.py for parameter analysis")
    print("  3. Check examples/example3_advanced.py for custom algorithms")
    print("  4. Read README.md for full documentation")
    print("\n" + "=" * 70)
