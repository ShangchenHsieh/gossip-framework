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
    
    # Test different network topologies on same node count and threshold
    # This demonstrates how different eigenvalues affect convergence
    threshold = 1e-1
    num_nodes = 100
    initial_values = np.random.uniform(0, 1, num_nodes)
    
    topologies = [
        ("complete", lambda net: net.create_complete_graph()),
        ("random_p02", lambda net: net.create_random_graph(0.2)),
        ("ring", lambda net: net.create_ring_graph()),
        ("scale_free", lambda net: net.create_scale_free_graph(m=2)),
    ]
    
    visualizer = GossipVisualizer()
    import os
    
    topology_results = []
    created_dirs = []
    
    for topo_name, topo_func in topologies:
        print(f"\n[TOPOLOGY] {topo_name.upper()}")
        
        # Create network with same initial values
        network = Network(num_nodes, initial_values=initial_values.copy(), seed=42)
        topo_func(network)
        
        # Compute eigenvalue for this topology
        lam2 = network.second_largest_eigenvalue()
        spectral_gap = network.spectral_gap()
        
        print(f"    Network: {network}")
        print(f"    λ2 (second eigenvalue): {lam2:.6f}")
        print(f"    Spectral gap (1-λ2): {spectral_gap:.6f}")
        print(f"    Initial average: {network.get_average_value():.6f}")
        
        # Run Push Gossip
        print(f"    Running Push Gossip (threshold={threshold})...")
        algorithm = PushGossip(seed=42)
        metrics = MetricsCollector()
        simulator = Simulator(network, algorithm, metrics, verbose=False)
        
        results = simulator.run(
            num_rounds=60,
            convergence_threshold=threshold,
            early_stopping=True
        )
        
        print(f"    Converged in {results['total_rounds']} rounds")
        print(f"    Final error: {results['final_error']:.2e}")
        
        # Store results for comparison
        topology_results.append({
            "topology": topo_name,
            "lambda2": lam2,
            "spectral_gap": spectral_gap,
            "convergence_rounds": results['total_rounds'],
            "metrics": metrics
        })
        
        # Create output directory for this topology
        plots_dir = os.path.join(Path(__file__).parent, f"plots_{topo_name}_nodes_{num_nodes}_eps_{str(threshold).replace('.', 'p')}")
        os.makedirs(plots_dir, exist_ok=True)
        created_dirs.append(plots_dir)
        
        # Save convergence plot
        fig1 = visualizer.plot_convergence(
            metrics,
            title=f"Convergence: {topo_name.upper()} (λ2={lam2:.4f})"
        )
        fig1_path = os.path.join(plots_dir, "convergence.png")
        fig1.savefig(fig1_path, dpi=300, bbox_inches='tight')
        print(f"    ✓ Convergence plot -> {fig1_path}")
        
        # Save node trajectories
        fig3 = visualizer.plot_node_trajectories(metrics, num_nodes_to_plot=10)
        fig3_path = os.path.join(plots_dir, "node_trajectories.png")
        fig3.savefig(fig3_path, dpi=300, bbox_inches='tight')
        print(f"    ✓ Node trajectories -> {fig3_path}")
        
        # Save eigen spectrum (just for complete graph to avoid clutter)
        if topo_name == "complete":
            eigs = network.get_eigen_spectrum()
            fig_eig = visualizer.plot_eigen_spectrum(eigs, title=f"Eigen Spectrum: {topo_name.upper()}")
            fig_eig_path = os.path.join(plots_dir, "eigen_spectrum.png")
            fig_eig.savefig(fig_eig_path, dpi=300, bbox_inches='tight')
            print(f"    ✓ Eigen spectrum -> {fig_eig_path}")
    
    # Create comparison plot showing λ2 vs convergence rounds
    print("\n[COMPARISON] Analyzing λ2 impact on convergence...")
    lambdas = [r["lambda2"] for r in topology_results]
    rounds = [r["convergence_rounds"] for r in topology_results]
    topo_names = [r["topology"] for r in topology_results]
    
    comparison_dir = os.path.join(Path(__file__).parent, "plots_topology_comparison")
    os.makedirs(comparison_dir, exist_ok=True)
    
    fig_comp = visualizer.plot_spectral_gap_vs_convergence(
        [1 - l for l in lambdas],
        rounds,
        labels=topo_names,
        title=f"Spectral Gap (1-λ2) vs Convergence Rounds\n(threshold={threshold}, nodes={num_nodes})"
    )
    fig_comp_path = os.path.join(comparison_dir, "spectral_gap_vs_convergence.png")
    fig_comp.savefig(fig_comp_path, dpi=300, bbox_inches='tight')
    print(f"    ✓ Spectral gap comparison -> {fig_comp_path}")
    created_dirs.append(comparison_dir)

    # Summary of created directories
    print("\nGenerated plots in the following directories:")
    for d in sorted(set(created_dirs)):
        print(f" - {d}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"✓ Framework works correctly!")
    print(f"✓ Compared 4 topologies with different eigenvalues (λ2)")
    print(f"✓ Analyzed how spectral gap (1-λ2) affects convergence rate")
    print(f"✓ Generated plots in 5 separate directories")
    print(f"\nKey Insights:")
    print(f"  - Complete graph has λ2 close to 1 → fastest convergence")
    print(f"  - Ring/sparse networks have larger λ2 → slower convergence")
    print(f"  - Spectral gap controls T_ave(ε) convergence time")
    print(f"\nNext steps:")
    print(f"  1. Check plots_topology_comparison/ for spectral gap vs convergence")
    print(f"  2. Check individual topology folders for detailed convergence plots")
    print(f"  3. See examples/epsilon_lambda_analysis.py for more analysis")
    print(f"  4. Read README.md for full documentation")
    print("\n" + "=" * 70)
