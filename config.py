"""Configuration management for simulations."""

from dataclasses import dataclass, asdict, field
from typing import Optional, List, Dict, Any
import json


@dataclass
class SimulationConfig:
    """Configuration for gossip algorithm simulations."""
    
    # Network configuration
    num_nodes: int = 100
    network_type: str = "complete"  # complete, random, ring, lattice, scale_free
    edge_probability: Optional[float] = None  # for random networks
    grid_side: Optional[int] = None  # for lattice networks
    
    # Initial values
    initial_value_type: str = "uniform"  # uniform, normal, mixed
    initial_min: float = 0.0
    initial_max: float = 1.0
    seed: Optional[int] = None
    
    # Algorithm configuration
    algorithm: str = "push"  # push, pull, geographic, random_averaging
    algorithm_params: Dict[str, Any] = field(default_factory=dict)
    
    # Simulation parameters
    num_rounds: int = 100
    convergence_threshold: Optional[float] = None
    num_trials: int = 1
    verbose: bool = False
    
    # Output
    save_results: bool = False
    output_dir: str = "./results"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
        
    def to_json(self, filepath: str) -> None:
        """Save configuration to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
            
    @classmethod
    def from_json(cls, filepath: str) -> 'SimulationConfig':
        """Load configuration from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls(**data)
        
    def __repr__(self) -> str:
        return f"""
SimulationConfig:
  Network: {self.num_nodes} nodes ({self.network_type})
  Algorithm: {self.algorithm}
  Rounds: {self.num_rounds}
  Trials: {self.num_trials}
        """
