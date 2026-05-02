"""Main simulation engine for gossip algorithms."""

import numpy as np
from typing import Optional, List, Dict, Any, Callable
from .network import Network
from ..metrics import MetricsCollector


class Simulator:
    """
    Main simulation engine for running gossip algorithm simulations.
    
    Orchestrates:
    - Algorithm execution
    - Metrics collection
    - Round-by-round simulation
    """
    
    def __init__(self, network: Network, algorithm, metrics: Optional[MetricsCollector] = None, verbose: bool = False):
        """
        Initialize the simulator.
        
        Args:
            network: Network object
            algorithm: GossipAlgorithm instance
            metrics: MetricsCollector instance (created if None)
            verbose: Whether to print progress
        """
        self.network = network
        self.algorithm = algorithm
        self.verbose = verbose
        self.metrics = metrics or MetricsCollector()
        self.current_round = 0
        
    def run(
        self,
        num_rounds: int,
        convergence_threshold: Optional[float] = None,
        max_iterations: Optional[int] = None,
        early_stopping: bool = True,
    ) -> Dict[str, Any]:
        """
        Run the simulation for a specified number of rounds.
        
        Args:
            num_rounds: Maximum number of rounds to simulate
            convergence_threshold: Stop if error below this threshold
            max_iterations: Alternative stopping condition
            early_stopping: Whether to enable early stopping criteria
            
        Returns:
            Dictionary with simulation results and metrics
        """
        self.network.reset_nodes()
        self.metrics.reset()
        self.current_round = 0
        
        target_average = self.network.get_average_value()
        
        for round_num in range(num_rounds):
            self.current_round = round_num
            
            # Collect pre-round metrics
            state_vector = self.network.get_state_vector()
            self.metrics.record_state(round_num, state_vector)
            
            # Execute one round of the algorithm
            messages_sent = self.algorithm.round(self.network)
            self.metrics.record_messages(round_num, messages_sent)
            
            # Calculate convergence metrics
            avg_error = np.max(np.abs(self.network.get_state_vector() - target_average))
            self.metrics.record_error(round_num, avg_error)
            
            if self.verbose and round_num % max(1, num_rounds // 10) == 0:
                print(f"Round {round_num}: avg_error={avg_error:.6f}, messages={messages_sent}")
            
            # Early stopping criteria
            if early_stopping:
                if convergence_threshold is not None and avg_error < convergence_threshold:
                    if self.verbose:
                        print(f"Converged at round {round_num} with error {avg_error:.6f}")
                    break
                    
                if max_iterations is not None and round_num >= max_iterations:
                    break
        
        results = {
            "total_rounds": self.current_round + 1,
            "target_average": target_average,
            "final_error": self.metrics.errors[-1] if self.metrics.errors else float('inf'),
            "final_state": self.network.get_state_vector(),
            "metrics": self.metrics,
        }
        
        return results
        
    def run_multiple_trials(
        self,
        num_trials: int,
        num_rounds: int,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run multiple independent trials and aggregate results.
        
        Args:
            num_trials: Number of independent trials
            num_rounds: Rounds per trial
            **kwargs: Additional arguments to pass to run()
            
        Returns:
            Aggregated results across trials
        """
        all_results = []
        
        for trial in range(num_trials):
            if self.verbose:
                print(f"\n--- Trial {trial + 1}/{num_trials} ---")
            
            result = self.run(num_rounds, **kwargs)
            all_results.append(result)
        
        # Aggregate results
        errors_per_round = []
        for trial_result in all_results:
            errors_per_round.append(trial_result["metrics"].errors)
        
        # Pad to same length
        max_len = max(len(e) for e in errors_per_round)
        errors_padded = []
        for e in errors_per_round:
            e_padded = e + [e[-1]] * (max_len - len(e))
            errors_padded.append(e_padded)
        
        errors_array = np.array(errors_padded)
        
        aggregated = {
            "num_trials": num_trials,
            "total_rounds": np.mean([r["total_rounds"] for r in all_results]),
            "all_results": all_results,
            "error_mean": np.mean(errors_array, axis=0),
            "error_std": np.std(errors_array, axis=0),
            "error_min": np.min(errors_array, axis=0),
            "error_max": np.max(errors_array, axis=0),
        }
        
        return aggregated
