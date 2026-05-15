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
import argparse
import os
import sys
import importlib.util
import matplotlib
from pathlib import Path

# Use non-interactive backend for scripts
matplotlib.use('Agg')

package_dir = Path(__file__).parents[1]
sys.path.insert(0, str(package_dir.parent.resolve()))
if package_dir.name != "gossip_framework" and "gossip_framework" not in sys.modules:
    spec = importlib.util.spec_from_file_location(
        "gossip_framework",
        package_dir / "__init__.py",
        submodule_search_locations=[str(package_dir.resolve())],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["gossip_framework"] = module
    spec.loader.exec_module(module)

from gossip_framework import (
    Network, Simulator, MetricsCollector, GossipVisualizer, RandomAveraging
)


DEFAULT_MAX_ROUNDS = 300_000
DEFAULT_EPSILONS = [1e-1, 5e-2, 1e-2, 5e-3, 1e-3]


def run_for_network(network: Network, algorithm, epsilons, max_rounds=DEFAULT_MAX_ROUNDS):
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate epsilon/lambda plots for the random averaging algorithm."
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=DEFAULT_MAX_ROUNDS,
        help="Maximum simulator rounds per epsilon value.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    epsilons = DEFAULT_EPSILONS
    num_nodes = 100
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

    algo = RandomAveraging(seed=122)

    print(f"Running random averaging experiments with max_rounds={args.max_rounds}:")
    for label, net in zip(labels, networks):
        print(f" - {label}")
        lam2, rounds = run_for_network(net, algo, epsilons, max_rounds=args.max_rounds)
        print(f"   lambda2={lam2:.6f}, rounds={rounds}")
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

    # Plot spectral gap vs convergence for a chosen epsilon.
    idx_eps = 0
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
