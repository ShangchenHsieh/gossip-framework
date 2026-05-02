"""Visualization utilities for gossip algorithm simulations."""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Any
from ..metrics import MetricsCollector


class GossipVisualizer:
    """
    Visualization utilities for gossip algorithm results.
    
    Supports:
    - Convergence plots
    - Communication complexity analysis
    - Parameter sensitivity analysis
    - Node state evolution
    - Network topology visualization
    """
    
    def __init__(self, style: str = "seaborn-v0_8-darkgrid"):
        """
        Initialize the visualizer.
        
        Args:
            style: Matplotlib style to use
        """
        try:
            sns.set_style("darkgrid")
            sns.set_palette("husl")
        except:
            pass
        self.style = style
        
    def plot_convergence(
        self,
        metrics: MetricsCollector,
        title: str = "Convergence Over Time",
        ylabel: str = "Max Error (log scale)",
        figsize: tuple = (10, 6),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot convergence error over rounds.
        
        Args:
            metrics: MetricsCollector instance
            title: Plot title
            ylabel: Y-axis label
            figsize: Figure size
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.semilogy(metrics.timestamps, metrics.errors, 'b-', linewidth=2, label='Error')
        ax.set_xlabel('Round', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
        
    def plot_multiple_convergence(
        self,
        results_dict: Dict[str, MetricsCollector],
        title: str = "Algorithm Comparison: Convergence",
        figsize: tuple = (12, 7),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Compare convergence across multiple algorithms.
        
        Args:
            results_dict: Dictionary of algorithm names to MetricsCollector
            title: Plot title
            figsize: Figure size
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        for algo_name, metrics in results_dict.items():
            ax.semilogy(metrics.timestamps, metrics.errors, marker='o', 
                       label=algo_name, linewidth=2, markersize=4, alpha=0.8)
        
        ax.set_xlabel('Round', fontsize=12)
        ax.set_ylabel('Max Error (log scale)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=11, loc='best')
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
        
    def plot_message_complexity(
        self,
        metrics: MetricsCollector,
        title: str = "Communication Complexity",
        figsize: tuple = (10, 6),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot message count over rounds.
        
        Args:
            metrics: MetricsCollector instance
            title: Plot title
            figsize: Figure size
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Messages per round
        ax1.bar(metrics.timestamps, metrics.messages, color='steelblue', alpha=0.7)
        ax1.set_xlabel('Round', fontsize=11)
        ax1.set_ylabel('Messages per Round', fontsize=11)
        ax1.set_title('Messages per Round', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Cumulative messages
        cumsum = np.cumsum(metrics.messages)
        ax2.plot(metrics.timestamps, cumsum, 'g-', linewidth=2, marker='o', markersize=4)
        ax2.set_xlabel('Round', fontsize=11)
        ax2.set_ylabel('Cumulative Messages', fontsize=11)
        ax2.set_title('Cumulative Communication', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.00)
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
        
    def plot_parameter_sensitivity(
        self,
        param_name: str,
        param_values: List[Any],
        results: List[Dict[str, Any]],
        metric_key: str = "final_error",
        title: Optional[str] = None,
        figsize: tuple = (10, 6),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot sensitivity to parameter variations.
        
        Args:
            param_name: Parameter name
            param_values: Values of the parameter
            results: Results for each parameter value
            metric_key: Metric to plot (e.g., 'final_error', 'convergence_rounds')
            title: Plot title
            figsize: Figure size
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Extract metric values
        metric_values = [r[metric_key] for r in results]
        
        ax.plot(param_values, metric_values, 'o-', linewidth=2, markersize=8, color='darkorange')
        ax.set_xlabel(param_name, fontsize=12)
        ax.set_ylabel(metric_key.replace('_', ' ').title(), fontsize=12)
        
        if title is None:
            title = f"Sensitivity to {param_name}"
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
        
    def plot_node_trajectories(
        self,
        metrics: MetricsCollector,
        num_nodes_to_plot: int = 5,
        title: str = "Node Value Trajectories",
        figsize: tuple = (12, 6),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot individual node value trajectories.
        
        Args:
            metrics: MetricsCollector instance
            num_nodes_to_plot: Number of nodes to plot
            title: Plot title
            figsize: Figure size
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        if len(metrics.states) == 0:
            print("No state data available")
            return fig
            
        num_nodes = metrics.states[0].shape[0]
        num_to_plot = min(num_nodes_to_plot, num_nodes)
        
        # Plot selected nodes
        for node_id in range(num_to_plot):
            node_trajectory = [state[node_id] for state in metrics.states]
            ax.plot(metrics.timestamps, node_trajectory, marker='o', 
                   label=f'Node {node_id}', linewidth=2, markersize=4, alpha=0.8)
        
        ax.set_xlabel('Round', fontsize=12)
        ax.set_ylabel('Node Value', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
        
    def plot_heatmap_convergence(
        self,
        metrics: MetricsCollector,
        num_nodes: Optional[int] = None,
        title: str = "Node State Heatmap (Over Time)",
        figsize: tuple = (14, 6),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot heatmap of node states evolving over time.
        
        Args:
            metrics: MetricsCollector instance
            num_nodes: Number of nodes (auto-detect if None)
            title: Plot title
            figsize: Figure size
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        if len(metrics.states) == 0:
            print("No state data available")
            return None
            
        # Create heatmap data (nodes x rounds)
        heatmap_data = np.array(metrics.states).T
        
        fig, ax = plt.subplots(figsize=figsize)
        
        im = sns.heatmap(heatmap_data, cmap='RdYlGn', center=np.mean(heatmap_data),
                        ax=ax, cbar_kws={'label': 'Node Value'})
        
        ax.set_xlabel('Round', fontsize=12)
        ax.set_ylabel('Node ID', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
        
    @staticmethod
    def show_plots() -> None:
        """Display all plots."""
        plt.show()

    def plot_eigen_spectrum(
        self,
        eigenvalues: List[complex],
        title: str = "Eigenvalue Spectrum",
        figsize: tuple = (8, 5),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot the eigenvalue spectrum (real parts) of the averaging matrix W."""
        real_vals = [np.real(v) for v in eigenvalues]
        imag_vals = [np.imag(v) for v in eigenvalues]

        fig, ax = plt.subplots(figsize=figsize)
        ax.scatter(real_vals, imag_vals, s=50, c='teal', alpha=0.8)
        ax.axvline(1.0, color='gray', linestyle='--', linewidth=1)
        ax.set_xlabel('Real Part', fontsize=12)
        ax.set_ylabel('Imaginary Part', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig

    def plot_spectral_gap_vs_convergence(
        self,
        spectral_gaps: List[float],
        convergence_rounds: List[float],
        labels: Optional[List[str]] = None,
        title: str = "Spectral Gap vs Empirical Convergence",
        xlabel: str = "Spectral Gap (1 - λ2)",
        ylabel: str = "Rounds to Convergence",
        figsize: tuple = (8, 6),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Scatter plot of spectral gap vs empirical convergence rounds."""
        fig, ax = plt.subplots(figsize=figsize)
        ax.scatter(spectral_gaps, convergence_rounds, s=80, c='crimson', alpha=0.8)
        for i, sg in enumerate(spectral_gaps):
            label = labels[i] if labels else str(i)
            ax.text(sg, convergence_rounds[i], f" {label}", fontsize=9)

        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig

    def plot_theoretical_vs_empirical(
        self,
        epsilons: List[float],
        lambda2: float,
        empirical_rounds: List[float],
        title: str = "Theoretical vs Empirical T_ave(ε)",
        figsize: tuple = (9, 6),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot predicted T(ε) = log(1/ε) / log(1/λ) vs empirical rounds.
        """
        eps = np.array(epsilons)
        empirical = np.array(empirical_rounds)

        # Avoid λ >=1 or λ<=0
        lam = max(min(lambda2, 0.9999), 1e-6)
        predicted = np.log(1.0 / eps) / np.log(1.0 / lam)

        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(eps, empirical, 'o-', label='Empirical', linewidth=2)
        ax.plot(eps, predicted, 's--', label='Predicted', linewidth=2)

        ax.set_xscale('log')
        ax.set_xlabel('ε (log scale)', fontsize=12)
        ax.set_ylabel('Rounds to Convergence', fontsize=12)
        ax.set_title(title + f" (λ2={lambda2:.4f})", fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, which='both', alpha=0.3)

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig
