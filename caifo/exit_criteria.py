"""
Exit criteria for CAIFO optimisation loop.
"""
import numpy as np
from typing import Dict, List, Any, Optional
import matplotlib.pyplot as plt

class ExitCriteria:
    """
    Provides exit criteria for the CAIFO optimisation loop.
    """
    
    def __init__(self, config):
        """
        Initialise the exit criteria manager.
        
        Args:
            config: Configuration object
        """
        self.config = config.caifo
        self.patience = self.config.patience
        self.min_delta = self.config.min_delta
        
        # Initialise tracking variables
        self.best_f1 = 0.0
        self.best_iteration = 0
        self.patience_counter = 0
        self.iteration_history = []
        self.per_class_history = {}
        self.early_stopping_triggered = False
        self.converged = False
    
    def update(self, iteration: int, metrics: Dict[str, Any], 
             struggling_class: str, per_class_metrics: Dict[str, Dict[str, float]]) -> bool:
        """
        Update exit criteria with the latest iteration results.
        
        Args:
            iteration: Current iteration number
            metrics: Overall metrics dictionary
            struggling_class: Current struggling class
            per_class_metrics: Per-class metrics dictionary
            
        Returns:
            True if optimisation should continue, False if it should stop
        """
        # Extract overall F1 score
        current_f1 = metrics.get('f1_weighted', 0.0)
        
        # Track iteration history
        self.iteration_history.append({
            'iteration': iteration,
            'f1_weighted': current_f1,
            'recall_weighted': metrics.get('recall_weighted', 0.0),
            'struggling_class': struggling_class
        })
        
        # Track per-class metrics
        for class_name, class_metrics in per_class_metrics.items():
            if class_name not in self.per_class_history:
                self.per_class_history[class_name] = []
            
            self.per_class_history[class_name].append({
                'iteration': iteration,
                'recall': class_metrics.get('recall', 0.0),
                'f1': class_metrics.get('f1', 0.0)
            })
        
        # Calculate improvement over best so far
        improvement = current_f1 - self.best_f1
        
        # Update best metrics if improvement is sufficient
        if improvement > self.min_delta:
            self.best_f1 = current_f1
            self.best_iteration = iteration
            self.patience_counter = 0
            return True  # Continue optimisation
        else:
            # No significant improvement
            self.patience_counter += 1
            
            # Check for patience exhaustion
            if self.patience_counter >= self.patience:
                self.early_stopping_triggered = True
                return False  # Stop optimisation
            
            # Check for convergence
            if self._check_convergence():
                self.converged = True
                return False  # Stop optimisation
            
            return True  # Continue optimisation
    
    def _check_convergence(self) -> bool:
        """
        Check if optimisation has converged based on recent history.
        
        Returns:
            True if converged, False otherwise
        """
        # Need at least 5 iterations to check convergence
        if len(self.iteration_history) < 5:
            return False
        
        # Get last 5 iterations
        recent_history = self.iteration_history[-5:]
        
        # Calculate mean and standard deviation of F1 scores
        f1_scores = [h['f1_weighted'] for h in recent_history]
        mean_f1 = np.mean(f1_scores)
        std_f1 = np.std(f1_scores)
        
        # If standard deviation is very small, we've converged
        return std_f1 < 0.001 and mean_f1 > 0.0
    
    def get_optimization_status(self) -> Dict[str, Any]:
        """
        Get current status of the optimisation process.
        
        Returns:
            Dictionary with optimisation status
        """
        return {
            'best_f1': self.best_f1,
            'best_iteration': self.best_iteration,
            'patience_counter': self.patience_counter,
            'patience_limit': self.patience,
            'current_min_delta': self.min_delta,
            'early_stopping_triggered': self.early_stopping_triggered,
            'converged': self.converged,
            'iterations_run': len(self.iteration_history)
        }
    
    def get_class_progress(self, class_name: str) -> List[Dict[str, Any]]:
        """
        Get progress history for a specific class.
        
        Args:
            class_name: Name of the class
            
        Returns:
            List of metrics for each iteration
        """
        return self.per_class_history.get(class_name, [])
    
    def plot_optimization_progress(self, output_path: Optional[str] = None) -> plt.Figure:
        """
        Plot optimisation progress over iterations.
        
        Args:
            output_path: Optional path to save the plot
            
        Returns:
            Matplotlib figure
        """
        if not self.iteration_history:
            return None
        
        plt.figure(figsize=(12, 8))
        
        # Extract data for plotting
        iterations = [h['iteration'] for h in self.iteration_history]
        f1_scores = [h['f1_weighted'] for h in self.iteration_history]
        recall_scores = [h['recall_weighted'] for h in self.iteration_history]
        
        # Plot overall metrics
        plt.subplot(2, 1, 1)
        plt.plot(iterations, f1_scores, 'b-o', label='F1 Weighted')
        plt.plot(iterations, recall_scores, 'g-^', label='Recall Weighted')
        
        # Mark best iteration
        if self.best_iteration > 0:
            plt.axvline(x=self.best_iteration, color='r', linestyle='--', 
                      label=f'Best Iteration ({self.best_iteration})')
        
        plt.title('CAIFO Optimisation Progress')
        plt.xlabel('Iteration')
        plt.ylabel('Score')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot per-class recall for a few classes
        plt.subplot(2, 1, 2)
        
        # Select up to 5 classes to plot
        classes_to_plot = list(self.per_class_history.keys())[:5]
        
        for class_name in classes_to_plot:
            history = self.per_class_history[class_name]
            if history:
                class_iterations = [h['iteration'] for h in history]
                class_recalls = [h['recall'] for h in history]
                plt.plot(class_iterations, class_recalls, 'o-', label=f'Class {class_name} Recall')
        
        plt.title('Per-Class Recall Progress')
        plt.xlabel('Iteration')
        plt.ylabel('Recall')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save if output path provided
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        
        return plt.gcf()