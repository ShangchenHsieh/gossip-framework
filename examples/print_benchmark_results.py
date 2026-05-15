"""
Print averaged benchmark results from the CSV log.

Run from the framework root:
    python examples/print_benchmark_results.py
"""

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path


TOPOLOGY_ORDER = ["complete", "random_p02", "random_geometric", "ring", "scale_free"]
ALGORITHM_ORDER = ["push", "pull", "geographic", "random_avg", "path_avg"]


def load_rows(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["topology"], row["algorithm"])].append(row)

    summaries = []
    for topology in TOPOLOGY_ORDER:
        for algorithm in ALGORITHM_ORDER:
            records = groups.get((topology, algorithm), [])
            if not records:
                continue

            rounds = [float(record["total_rounds"]) for record in records]
            messages = [float(record["total_messages"]) for record in records]
            final_errors = [float(record["final_error"]) for record in records]
            converged = sum(record["converged"] == "True" for record in records)

            summaries.append({
                "topology": topology,
                "algorithm": algorithm,
                "runs": len(records),
                "converged": converged,
                "avg_rounds": statistics.mean(rounds),
                "std_rounds": statistics.pstdev(rounds),
                "avg_messages": statistics.mean(messages),
                "std_messages": statistics.pstdev(messages),
                "avg_final_error": statistics.mean(final_errors),
            })

    return summaries


def print_table(summaries):
    print(
        f"{'Topology':<14} {'Algorithm':<12} {'Conv':>7} "
        f"{'Avg rounds':>12} {'Std rounds':>12} "
        f"{'Avg msgs':>14} {'Std msgs':>14} {'Avg final err':>15}"
    )
    print("-" * 126)
    for row in summaries:
        convergence = f"{row['converged']}/{row['runs']}"
        print(
            f"{row['topology']:<14} {row['algorithm']:<12} {convergence:>7} "
            f"{row['avg_rounds']:>12.1f} {row['std_rounds']:>12.1f} "
            f"{row['avg_messages']:>14.1f} {row['std_messages']:>14.1f} "
            f"{row['avg_final_error']:>15.4g}"
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Print averaged benchmark results.")
    parser.add_argument(
        "csv_path",
        nargs="?",
        type=Path,
        default=Path("results") / "topology_algorithm_benchmark.csv",
        help="Path to the benchmark CSV log.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print_table(summarize(load_rows(args.csv_path)))
