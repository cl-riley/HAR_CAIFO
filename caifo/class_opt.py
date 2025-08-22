"""
Class weight optimisation for improving struggling classes in CAIFO.
"""
import numpy as np
from typing import Dict, Any, List, Tuple, Optional

class ClassWeightOptimizer:
    """
    Optimise class weights to improve performance on struggling classes
    while maintaining overall model performance.
    """
    
    def __init__(self, config):
        """
        Initialise the class weight optimiser.
        
        Args:
            config: Configuration object containing CAIFO parameters
        """
        self.config = config.caifo
        self.min_weight = self.config.class_weight_range[0]
        self.max_weight = self.config.class_weight_range[1]
        
    def optimize_weights_balanced(self, current_weights: Dict[str, float], 
                                struggling_class: str, struggling_recall: float,
                                overall_f1: float) -> Dict[str, float]:
        """
        Balanced class weight optimisation using configured weight ranges.
        Prevents single-class dominance while boosting struggling classes.
        
        Args:
            current_weights: Current class weights
            struggling_class: Identified struggling class
            struggling_recall: Current recall of the struggling class (negative for overpredicted classes)
            overall_f1: Current F1 score
            
        Returns:
            Updated class weights
        """
        # Create a copy of current weights
        new_weights = current_weights.copy() if current_weights else {}
        
        # Ensure all classes have a weight entry
        classes = list(new_weights.keys())
        for cls in classes:
            if cls not in new_weights:
                new_weights[cls] = 1.0
        
        # SEVERE MONOPOLY: If a class is taking over predictions completely
        if struggling_recall < -5.0:  # Special value passed from identify_struggling_class
            # Drastically reduce the dominant class weight
            reduction_factor = 0.2  # Reduce to 20% of current weight
            current_weight = new_weights.get(struggling_class, 1.0)
            new_weights[struggling_class] = max(current_weight * reduction_factor, self.min_weight)
            
            # Significantly boost ALL other classes
            other_classes = [c for c in new_weights if c != struggling_class]
            for c in other_classes:
                # Strong boost to all other classes
                boost_factor = 2.0  # Double their weights
                new_weights[c] = min(new_weights.get(c, 1.0) * boost_factor, self.max_weight)
            
            return new_weights
        
        # Handle overpredicted classes (negative recall signals overprediction)
        elif struggling_recall < 0:
            # Moderate reduction for overpredicted class
            reduction_factor = 0.6  # Reduce to 60% - less drastic
            
            # Get current weight, default to 1.0 if not found
            current_weight = new_weights.get(struggling_class, 1.0)
            
            # Apply reduction with minimum weight floor
            new_weights[struggling_class] = max(current_weight * reduction_factor, self.min_weight)
            
            # Boost other classes moderately
            other_classes = [c for c in new_weights if c != struggling_class]
            for c in other_classes:
                # Small boost to others
                boost_factor = 1.1  # 10% increase
                new_weights[c] = min(new_weights.get(c, 1.0) * boost_factor, self.max_weight)
        
        # For zero-recall classes, apply appropriate boosting
        elif struggling_recall == 0.0:
            boost_factor = 2.0  # Moderate boost for zero recall
            
            # Get current weight
            current_weight = new_weights.get(struggling_class, 1.0)
            new_weights[struggling_class] = min(current_weight * boost_factor, self.max_weight)
            
            # Reduce other weights slightly
            other_classes = [c for c in new_weights if c != struggling_class]
            for c in other_classes:
                # Gentle reduction
                reduction = 0.9  # 10% reduction
                new_weights[c] = max(new_weights[c] * reduction, self.min_weight)
        
        # For low-recall classes
        elif struggling_recall < 0.3:
            boost_factor = 1.5  # Moderate boost for low recall
            
            # Get current weight
            current_weight = new_weights.get(struggling_class, 1.0)
            new_weights[struggling_class] = min(current_weight * boost_factor, self.max_weight)
            
            # Reduce other weights minimally
            other_classes = [c for c in new_weights if c != struggling_class]
            for c in other_classes:
                # Very gentle reduction
                reduction = 0.95  # 5% reduction
                new_weights[c] = max(new_weights[c] * reduction, self.min_weight)
        
        # For classes with decent recall, just small adjustments
        else:
            boost_factor = 1.2  # Small boost
            
            # Get current weight
            current_weight = new_weights.get(struggling_class, 1.0)
            new_weights[struggling_class] = min(current_weight * boost_factor, self.max_weight)
            
            # Don't reduce other weights much at all
            other_classes = [c for c in new_weights if c != struggling_class]
            for c in other_classes:
                # Minimal reduction
                reduction = 0.98  # 2% reduction
                new_weights[c] = max(new_weights[c] * reduction, self.min_weight)
        
        # Normalise weights to prevent extreme ratios
        max_weight = max(new_weights.values())
        min_weight = min(new_weights.values())
        
        # If the ratio between max and min is too high, limit it
        max_allowed_ratio = 4.0  # Don't allow weights to differ by more than 4x
        
        if max_weight / min_weight > max_allowed_ratio:
            # Scale down the highest weights
            for cls in new_weights:
                if new_weights[cls] > min_weight * max_allowed_ratio:
                    new_weights[cls] = min_weight * max_allowed_ratio
        
        # Ensure weights are within configured range
        for cls in new_weights:
            new_weights[cls] = min(max(new_weights[cls], self.min_weight), self.max_weight)
        
        return new_weights

    def optimize_weights_aggressive(self, current_weights: Dict[str, float], 
                                struggling_class: str, struggling_recall: float) -> Dict[str, float]:
        """
        More aggressive weight optimisation for zero-recall classes.
        Maintains balance to prevent single-class dominance.
        
        Args:
            current_weights: Current class weights
            struggling_class: Identified struggling class
            struggling_recall: Current recall of the struggling class
            
        Returns:
            Updated class weights
        """
        # Create a copy of current weights
        new_weights = current_weights.copy() if current_weights else {}
        
        # For zero-recall classes, use more aggressive factors but avoid extremes
        if struggling_recall == 0.0:
            boost_factor = 3.0  # Stronger boost but not extreme
        elif struggling_recall < 0.1:
            boost_factor = 2.5
        elif struggling_recall < 0.3:
            boost_factor = 2.0
        else:
            boost_factor = 1.5
        
        # Apply boost to struggling class
        current_weight = new_weights.get(struggling_class, 1.0)
        new_weights[struggling_class] = min(current_weight * boost_factor, self.max_weight)
        
        # Reduce weights for other classes, but not too extremely
        other_classes = [c for c in new_weights if c != struggling_class]
        for c in other_classes:
            # Moderate reduction
            reduction_factor = 0.8  # Reduce to 80%
            new_weights[c] = max(new_weights[c] * reduction_factor, self.min_weight)
        
        # Check for and handle potential dominant classes
        max_weight = max(new_weights.values())
        min_weight = min(new_weights.values())
        
        # If weights are too imbalanced, adjust them
        if max_weight / min_weight > 5.0:
            # Scale down the highest weights
            for cls in new_weights:
                if new_weights[cls] > min_weight * 5.0:
                    new_weights[cls] = min_weight * 5.0
        
        # Ensure weights are within configured range
        for cls in new_weights:
            new_weights[cls] = min(max(new_weights[cls], self.min_weight), self.max_weight)
        
        return new_weights
    
    def format_sklearn_weights(self, weights: Dict[str, float], 
                            class_names: np.ndarray) -> Dict[Any, float]:
        """
        Format weights for use with sklearn models.
        
        Args:
            weights: Raw class weights
            class_names: Array of class names/values
            
        Returns:
            Formatted weights for sklearn models
        """
        # Create mapping from class name to class value
        formatted_weights = {}
        
        # Add weights for both string and original class values
        for class_name in class_names:
            # String version of class name
            class_str = str(class_name)
            
            # Use the weight if available, otherwise default to 1.0
            if class_str in weights:
                formatted_weights[class_name] = weights[class_str]  # Use actual class value as key
            elif class_name in weights:
                formatted_weights[class_name] = weights[class_name]
            else:
                # Default weight
                formatted_weights[class_name] = 1.0
        
        # Safety check: limit weight range to prevent extreme imbalance
        max_weight = max(formatted_weights.values())
        min_weight = min(formatted_weights.values())
        
        # Only apply safety scaling if range is extremely out of proportion
        max_allowed_ratio = 10.0  # Maximum allowed ratio between highest and lowest weight
        
        if max_weight / min_weight > max_allowed_ratio:
            # Scale weights within configured range, preserving relative proportions
            for cls in formatted_weights:
                # Normalise to [0,1] then scale to configured range
                normalised = (formatted_weights[cls] - min_weight) / (max_weight - min_weight)
                formatted_weights[cls] = self.min_weight + normalised * (self.max_weight - self.min_weight)
        
        return formatted_weights