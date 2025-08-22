"""
Utility functions for HAR-CAIFO.
"""
import os
import time
import logging
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from sklearn.preprocessing import LabelEncoder

def setup_logger(name: str, log_dir: str = 'logs', level=logging.INFO) -> logging.Logger:
    """
    Set up a logger with file and console handlers.
    
    Args:
        name: Logger name
        log_dir: Directory to save log files
        level: Logging level
        
    Returns:
        Configured logger
    """
    # Create logs directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear existing handlers
    if logger.handlers:
        logger.handlers = []
    
    # Create file handler
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_file = os.path.join(log_dir, f"{name}_{timestamp}.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def encode_labels(*label_sets):
    """
    Convert string labels to numeric for model training.
    
    Args:
        *label_sets: One or more arrays of labels to encode
        
    Returns:
        Tuple of (encoded labels, label encoder, class mapping)
    """
    # Create label encoder
    encoder = LabelEncoder()
    
    # Combine all label sets for fitting
    all_labels = np.concatenate(label_sets)
    
    # Fit on all labels to ensure consistent encoding
    encoder.fit(all_labels)
    
    # Transform each label set
    encoded_sets = tuple(encoder.transform(label_set) for label_set in label_sets)
    
    # Create class mapping for reference
    class_mapping = {i: class_name for i, class_name in enumerate(encoder.classes_)}
    
    return encoded_sets, encoder, class_mapping

def compute_diversity_score(y_pred, n_classes):
    """
    Calculate a diversity score based on prediction distribution.
    
    Args:
        y_pred: Predicted class labels
        n_classes: Number of classes
        
    Returns:
        Diversity score (0-1, higher is more diverse)
    """
    # Count predictions per class
    counts = np.zeros(n_classes)
    for pred in y_pred:
        if pred < n_classes:  # Safety check
            counts[pred] += 1
    
    # Calculate normalised entropy
    total = len(y_pred)
    probs = counts / total
    probs = probs[probs > 0]  # Remove zeros for entropy calculation
    entropy = -np.sum(probs * np.log2(probs + 1e-10))
    max_entropy = np.log2(min(n_classes, len(np.unique(y_pred))))  # Maximum possible entropy
    
    # Return normalised entropy (0-1 scale, higher is more diverse)
    return entropy / max_entropy if max_entropy > 0 else 0

def format_time(seconds):
    """
    Format time in seconds to a human-readable string.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.1f} seconds"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f} minutes"
    else:
        hours = seconds / 3600
        minutes = (seconds % 3600) / 60
        return f"{int(hours)}h {int(minutes)}m"

def debug_weight_changes(previous_weights, new_weights, tag="weights"):
    """
    Debug function to track weight changes between iterations.
    
    Args:
        previous_weights: Previous weights
        new_weights: New weights
        tag: Type of weights for display
        
    Returns:
        True if significant changes, False otherwise
    """
    if previous_weights is None or new_weights is None:
        print(f"Cannot compare {tag}, one is None")
        return False
    
    # Convert to dictionaries if they're not already
    if not isinstance(previous_weights, dict):
        previous_dict = {i: w for i, w in enumerate(previous_weights)}
        new_dict = {i: w for i, w in enumerate(new_weights)}
    else:
        previous_dict = previous_weights
        new_dict = new_weights
    
    # Count changes
    changes = 0
    total = 0
    max_diff = 0.0
    
    # Find common keys
    common_keys = set(previous_dict.keys()).intersection(set(new_dict.keys()))
    
    for k in common_keys:
        total += 1
        prev_val = previous_dict[k]
        new_val = new_dict[k]
        diff = abs(new_val - prev_val)
        if diff > 0.0001:  # Threshold for considering a change
            changes += 1
            max_diff = max(max_diff, diff)
            if changes <= 5:  # Show at most 5 examples
                print(f"  {k}: {prev_val:.4f} → {new_val:.4f} (diff: {diff:.4f})")
    
    # Calculate percentage changed
    pct_changed = 100 * changes / max(1, total)
    
    print(f"{tag} changes: {changes}/{total} ({pct_changed:.1f}%) - Max diff: {max_diff:.4f}")
    
    # Check if anything changed significantly
    return pct_changed > 0.1  # Consider changed if more than 0.1% of weights changed

class CAIFOLogger:
    """Simple logger for CAIFO optimisation."""
    
    def __init__(self, config):
        """
        Initialise the logger.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.log_dir = config.output_dir
        self.history = []
        
        # Create logs directory if it doesn't exist
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Set up logging
        self.logger = setup_logger('caifo', self.log_dir)
    
    def log_event(self, message: str):
        """
        Log an event message.
        
        Args:
            message: Message to log
        """
        self.logger.info(message)
        print(message)  # Print to console as well
    
    def log_error(self, error: Exception, context: str = None):
        """
        Log an error with context.
        
        Args:
            error: Exception to log
            context: Optional context for the error
        """
        if context:
            self.logger.error(f"{context}: {str(error)}")
            print(f"ERROR: {context}: {str(error)}")
        else:
            self.logger.error(str(error))
            print(f"ERROR: {str(error)}")
    
    def log_iteration(self, iteration: int, metrics: Dict[str, Any]):
        """
        Log metrics for an iteration.
        
        Args:
            iteration: Iteration number
            metrics: Dictionary of metrics
        """
        self.history.append(metrics)
        
        # Log key metrics
        self.logger.info(f"Iteration {iteration}:")
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self.logger.info(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
            else:
                self.logger.info(f"  {key}: {value}")
    
    def save_history(self):
        """Save optimisation history to a file."""
        import json
        
        history_path = os.path.join(self.log_dir, "optimisation_history.json")
        
        try:
            # Convert any non-serialisable values to strings
            safe_history = []
            for entry in self.history:
                safe_entry = {}
                for key, value in entry.items():
                    if isinstance(value, (int, float, str, bool, list, dict, type(None))):
                        safe_entry[key] = value
                    else:
                        safe_entry[key] = str(value)
                safe_history.append(safe_entry)
            
            with open(history_path, 'w') as f:
                json.dump(safe_history, f, indent=2)
            
            self.logger.info(f"Saved optimisation history to {history_path}")
        except Exception as e:
            self.logger.error(f"Error saving history: {str(e)}")