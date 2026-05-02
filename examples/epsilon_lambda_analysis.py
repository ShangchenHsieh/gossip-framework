"""
Analyze T_ave(ε) empirical vs theoretical using λ2 (second-largest eigenvalue).

This script:
- Builds several network topologies
- Computes λ2 and spectral gap
- Runs simulations for multiple ε values and records rounds to converge
- Plots theoretical vs empirical T_ave(ε) and spectral-gap relationships

Run from the framework root:
    python examples/epsilon_lambda_analysis.py
"""

import numpy as np
import os
import sys
import matplotlib
from pathlib import Path

# Use non-interactive backend for scripts
matplotlib.use('Agg')

sys.path.insert(0, str(Path(__file__).parents[2].resolve()))

from gossip_framework import (
    Network, Simulator, PushGossip, MetricsCollector, GossipVisualizer
)


def run_for_network(network: Network, algorithm, epsilons, max_rounds=5000):
    """Run simulator for a list of epsilons and return rounds-to-converge list."""
    initial_vals = network.get_state_vector().copy()
    rounds = []
    lam2 = network.second_largest_eigenvalue()
    for eps in epsilons:
        # Reset node values to same initial condition for fair comparison
        for i, v in enumerate(initial_vals):
            network.get_node(i).set_value(v)
            network.get_node(i).initial_value = v

        metrics = MetricsCollector()
        sim = Simulator(network, algorithm, metrics, verbose=False)
        # Run with large num_rounds but stop early with convergence_threshold
        result = sim.run(num_rounds=max_rounds, convergence_threshold=eps, early_stopping=True)
        rounds.append(result['total_rounds'])
    return lam2, rounds


def main():
    epsilons = [1e-1, 1e-2, 1e-3, 1e-4]
    num_nodes = 80
    visualizer = GossipVisualizer()

    networks = []
    labels = []

    # Complete
    net = Network(num_nodes, initial_values=np.random.uniform(0, 1, num_nodes), seed=42)
    net.create_complete_graph()
    networks.append(net); labels.append('Complete')

    # Random p=0.2
    net = Network(num_nodes, initial_values=np.random.uniform(0, 1, num_nodes), seed=43)
    net.create_random_graph(0.2)
    networks.append(net); labels.append('Random p=0.2')

    # Ring
    net = Network(num_nodes, initial_values=np.random.uniform(0, 1, num_nodes), seed=44)
    net.create_ring_graph()
    networks.append(net); labels.append('Ring')

    # Scale-free
    net = Network(num_nodes, initial_values=np.random.uniform(0, 1, num_nodes), seed=45)
    net.create_scale_free_graph(m=2)
    networks.append(net); labels.append('Scale-free')

    results = []
    spectral_gaps = []
    lambdas = []

    algo = PushGossip(seed=123)

    print("Running experiments for networks:")
    for label, net in zip(labels, networks):
        print(f" - {label}")
        lam2, rounds = run_for_network(net, algo, epsilons, max_rounds=2000)
        results.append(rounds)
        lambdas.append(lam2)
        spectral_gaps.append(1.0 - lam2)

    # Plot theoretical vs empirical for each network
    # Save plots alongside other quickstart/example plots in package root
    package_root = Path(__file__).parents[1]
    out_dir = package_root / "plots_epsilon_lambda"
    os.makedirs(out_dir, exist_ok=True)

    for label, lam2, rounds in zip(labels, lambdas, results):
        fig = visualizer.plot_theoretical_vs_empirical(epsilons, lam2, rounds,
                                  title=f"{label}: Theoretical vs Empirical")
        fig.savefig(os.path.join(out_dir, f"theory_vs_empirical_{label.replace(' ', '_')}.png"))

    # Plot spectral gap vs convergence for a chosen epsilon (e.g., 1e-3)
    idx_eps = 2  # corresponds to 1e-3
    conv_rounds = [r[idx_eps] for r in results]
    fig2 = visualizer.plot_spectral_gap_vs_convergence(spectral_gaps, conv_rounds, labels,
                                                      title=f"Spectral Gap vs Rounds (ε={epsilons[idx_eps]})")
    fig2.savefig(os.path.join(out_dir, "spectral_gap_vs_rounds.png"))

    # Eigen spectrum plot for first network
    eigs = networks[0].get_eigen_spectrum()
    fig3 = visualizer.plot_eigen_spectrum(eigs, title=f"Eigen Spectrum: {labels[0]}")
    fig3.savefig(os.path.join(out_dir, "eigen_spectrum_network0.png"))

    print(f"Results and plots saved to {out_dir}")


if __name__ == '__main__':
    main()
