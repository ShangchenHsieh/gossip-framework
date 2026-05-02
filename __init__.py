"""
Gossip Algorithm Simulation Framework
A flexible framework for simulating randomized gossip and consensus algorithms
with support for various network topologies, parameters, and visualization.
"""

__version__ = "0.1.0"

from .core import Network, Node, Simulator
from .algorithms import GossipAlgorithm, PushGossip, PullGossip, PushPullGossip, RandomAveraging
from .config import SimulationConfig
from .metrics import MetricsCollector
from .visualization import GossipVisualizer

__all__ = [
    "Network",
    "Node",
    "Simulator",
    "GossipAlgorithm",
    "PushGossip",
    "PullGossip",
    "PushPullGossip",
    "RandomAveraging",
    "SimulationConfig",
    "MetricsCollector",
    "GossipVisualizer",
]
