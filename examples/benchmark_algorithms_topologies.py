"""
Run all gossip algorithms across multiple topology instances.

The script creates multiple seeded instances for each topology, runs each
algorithm on each instance, writes per-run records to CSV, and prints averaged
results grouped by topology and algorithm.

Run from the framework root:
    python examples/benchmark_algorithms_topologies.py
"""

import argparse
import csv
import importlib.util
import statistics
import sys
from pathlib import Path

import numpy as np


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

from gossip_framework import (  # noqa: E402
    MetricsCollector,
    Network,
    PullGossip,
    PushGossip,
    GeographicGossip,
    RandomAveraging,
    PathAveraging,
    Simulator,
)


TOPOLOGIES = ("complete", "random_p02", "random_geometric", "ring", "scale_free")


def build_network(topology: str, num_nodes: int, initial_values: np.ndarray, seed: int) -> Network:
    """Build one topology instance with fixed initial values."""
    network = Network(num_nodes, initial_values=initial_values, seed=seed)

    if topology == "complete":
        network.create_complete_graph()
    elif topology == "random_p02":
        network.create_random_graph(0.2)
    elif topology == "random_geometric":
        network.create_random_geometric_graph()
    elif topology == "ring":
        network.create_ring_graph()
    elif topology == "scale_free":
        network.create_scale_free_graph(m=2)
    else:
        raise ValueError(f"Unknown topology: {topology}")

    return network


def make_algorithms(seed: int, random_ticks_per_round: int):
    """Create fresh algorithm instances for a single run."""
    return {
        "push": PushGossip(seed=seed + 11),
        "pull": PullGossip(seed=seed + 23),
        "geographic": GeographicGossip(seed=seed + 37),
        "random_avg": RandomAveraging(clock_ticks_per_round=random_ticks_per_round, seed=seed + 41),
        "path_avg": PathAveraging(seed=seed + 53),
    }


def summarize(records):
    """Aggregate per-run records by topology and algorithm."""
    grouped = {}
    for record in records:
        key = (record["topology"], record["algorithm"])
        grouped.setdefault(key, []).append(record)

    summaries = []
    for (topology, algorithm), rows in sorted(grouped.items()):
        rounds = [row["total_rounds"] for row in rows]
        messages = [row["total_messages"] for row in rows]
        errors = [row["final_error"] for row in rows]
        converged = [row["converged"] for row in rows]

        summaries.append({
            "topology": topology,
            "algorithm": algorithm,
            "runs": len(rows),
            "converged": sum(converged),
            "avg_rounds": statistics.mean(rounds),
            "std_rounds": statistics.pstdev(rounds),
            "avg_messages": statistics.mean(messages),
            "std_messages": statistics.pstdev(messages),
            "avg_final_error": statistics.mean(errors),
            "median_final_error": statistics.median(errors),
        })

    return summaries


def print_summary(summaries, threshold: float):
    """Print a compact table of averaged results."""
    print()
    print(f"Averaged results (convergence threshold={threshold:g})")
    print("-" * 126)
    print(
        f"{'Topology':<14} {'Algorithm':<12} {'Conv':>7} "
        f"{'Avg rounds':>12} {'Std rounds':>12} "
        f"{'Avg msgs':>14} {'Std msgs':>14} {'Avg final err':>15}"
    )
    print("-" * 126)

    for row in summaries:
        conv = f"{row['converged']}/{row['runs']}"
        print(
            f"{row['topology']:<14} {row['algorithm']:<12} {conv:>7} "
            f"{row['avg_rounds']:>12.1f} {row['std_rounds']:>12.1f} "
            f"{row['avg_messages']:>14.1f} {row['std_messages']:>14.1f} "
            f"{row['avg_final_error']:>15.4g}"
        )
    print("-" * 126)


def write_csv(path: Path, records):
    """Write per-run records to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "topology",
        "algorithm",
        "instance",
        "seed",
        "num_nodes",
        "avg_degree",
        "lambda2",
        "spectral_gap",
        "total_rounds",
        "total_messages",
        "avg_messages_per_round",
        "final_error",
        "target_average",
        "final_average",
        "converged",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def run_benchmark(args):
    """Run the full topology x algorithm benchmark."""
    records = []
    rng = np.random.default_rng(args.seed)
    random_ticks_per_round = args.random_ticks_per_round or args.num_nodes
    algorithms_per_run = len(make_algorithms(args.seed, random_ticks_per_round))

    print(
        f"Running {len(TOPOLOGIES)} topologies x {algorithms_per_run} algorithms x "
        f"{args.instances} instances ({len(TOPOLOGIES) * algorithms_per_run * args.instances} runs)"
    )
    print(f"RandomAveraging clock ticks per simulator round: {random_ticks_per_round}")

    for topology in TOPOLOGIES:
        print(f"\nTopology: {topology}")
        for instance in range(args.instances):
            topology_seed = args.seed + 10_000 * TOPOLOGIES.index(topology) + instance
            initial_values = rng.uniform(0.0, 1.0, args.num_nodes)
            base_network = build_network(topology, args.num_nodes, initial_values, topology_seed)
            avg_degree = float(np.mean([len(node.neighbors) for node in base_network.get_all_nodes()]))
            lambda2 = base_network.second_largest_eigenvalue()
            spectral_gap = 1.0 - lambda2

            for algorithm_name, algorithm in make_algorithms(
                args.seed + 100_000 * instance + 1_000 * TOPOLOGIES.index(topology),
                random_ticks_per_round,
            ).items():
                network = build_network(topology, args.num_nodes, initial_values.copy(), topology_seed)
                metrics = MetricsCollector()
                simulator = Simulator(network, algorithm, metrics, verbose=False)
                result = simulator.run(
                    num_rounds=args.max_rounds,
                    convergence_threshold=args.threshold,
                    early_stopping=True,
                )
                total_messages = int(sum(metrics.messages))
                final_average = float(np.mean(result["final_state"]))
                converged = bool(result["final_error"] < args.threshold)

                records.append({
                    "topology": topology,
                    "algorithm": algorithm_name,
                    "instance": instance,
                    "seed": topology_seed,
                    "num_nodes": args.num_nodes,
                    "avg_degree": avg_degree,
                    "lambda2": lambda2,
                    "spectral_gap": spectral_gap,
                    "total_rounds": int(result["total_rounds"]),
                    "total_messages": total_messages,
                    "avg_messages_per_round": float(np.mean(metrics.messages)) if metrics.messages else 0.0,
                    "final_error": float(result["final_error"]),
                    "target_average": float(result["target_average"]),
                    "final_average": final_average,
                    "converged": converged,
                })

            if (instance + 1) % max(1, args.instances // 5) == 0:
                print(f"  completed {instance + 1}/{args.instances} instances")

    write_csv(args.output, records)
    summaries = summarize(records)
    print_summary(summaries, args.threshold)
    print(f"\nPer-run log written to: {args.output}")


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark gossip algorithms across topology instances.")
    parser.add_argument("--instances", type=int, default=30, help="Topology instances per topology.")
    parser.add_argument("--num-nodes", type=int, default=100, help="Number of nodes per network.")
    parser.add_argument("--max-rounds", type=int, default=2000, help="Maximum simulator rounds per run.")
    parser.add_argument("--threshold", type=float, default=1e-2, help="Early-stopping convergence threshold.")
    parser.add_argument("--seed", type=int, default=2026, help="Base random seed.")
    parser.add_argument(
        "--random-ticks-per-round",
        type=int,
        default=None,
        help="Clock ticks per RandomAveraging simulator round. Defaults to num_nodes.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results") / "topology_algorithm_benchmark.csv",
        help="CSV path for per-run records.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_benchmark(parse_args())
