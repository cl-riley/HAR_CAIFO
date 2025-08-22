"""
Feature weight optimisation for struggling classes in CAIFO.
"""
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from sklearn.metrics import recall_score

class FeatureOptimizer:
    """
    Optimise feature weights for struggling classes using CAIFO approach.
    
    This class identifies which classes are performing poorly and enhances 
    features that are most discriminative for those classes.
    """
    
    def __init__(self, config):
        """
        Initialise the feature optimiser.
        
        Args:
            config: Configuration object containing CAIFO parameters
        """
        self.config = config.caifo
        self.feature_config = config.feature
        self.class_history = {}  # Track class performance over time
        
    def apply_zero_recall_boosting(self, X: np.ndarray, y: np.ndarray, 
                                struggling_class: str, feature_names: List[str]) -> np.ndarray:
        """
        Apply boosting for features that might help with zero-recall classes.
        Focuses on stable, generalisable features that work across users.
        
        Args:
            X: Feature matrix
            y: Target labels
            struggling_class: Class with zero recall
            feature_names: List of feature names
            
        Returns:
            Boosted feature matrix
        """
        # Convert class to int if it's a string
        struggling_class_int = int(struggling_class)
        
        # Identify samples belonging to the struggling class
        mask = (y == struggling_class_int)
        if not np.any(mask):
            return X
        
        # Get samples of the struggling class
        X_class = X[mask]
        
        # Calculate mean values for each feature for this class
        class_means = np.mean(X_class, axis=0)
        
        # Calculate global mean values
        global_means = np.mean(X, axis=0)
        
        # Calculate feature difference (how distinctive each feature is for this class)
        feature_diff = np.abs(class_means - global_means)
        
        # Calculate standard deviation WITHIN this class for each feature
        # Features with low within-class variability are more reliable indicators
        class_stds = np.std(X_class, axis=0)
        
        # Calculate coefficient of variation (CV = std/mean) for each feature
        # Lower CV means more consistent feature (stable across different users)
        epsilon = 1e-10  # Prevent division by zero
        class_cv = class_stds / (np.abs(class_means) + epsilon)
        
        # Calculate normalised feature stability score (1-CV, bounded to [0,1])
        # Higher value = more stable feature
        stability_scores = 1.0 - np.minimum(class_cv, 1.0)
        
        # Calculate global standard deviation for each feature
        global_std = np.std(X, axis=0)
        
        # Calculate Z-scores (how many standard deviations from global mean)
        z_scores = np.abs(class_means - global_means) / (global_std + epsilon)
        z_scores_normalised = np.minimum(z_scores / 3.0, 1.0)  # Normalise to [0,1], capping at 3 std
        
        # Combine distinctiveness (z-score) and stability (inverse CV)
        # to get features that are both distinctive AND stable across users
        combined_score = (0.7 * stability_scores) + (0.3 * z_scores_normalised)
        
        # Apply modest boosting with focus on stability
        boost_factor = 3.0
        feature_weights = 1.0 + (boost_factor * combined_score)
        
        # Cap maximum boost to prevent extreme values
        feature_weights = np.minimum(feature_weights, 4.0)  # Max 4x boost
        
        # Identify top N most stable AND distinctive features
        top_n = min(8, len(feature_names))  # Use only top 8 features
        top_indices = np.argsort(combined_score)[-top_n:]
        
        # Try to identify potentially dominant classes
        try:
            from sklearn.ensemble import RandomForestClassifier
            
            # Train a simple model to find dominant classes
            temp_model = RandomForestClassifier(
                n_estimators=50, 
                max_depth=4,
                min_samples_leaf=5,
                random_state=42
            )
            temp_model.fit(X, y)
            y_pred = temp_model.predict(X)
            
            # Find classes that might be dominating predictions
            class_pred_counts = {}
            class_true_counts = {}
            
            for cls in np.unique(y):
                class_pred_counts[cls] = np.sum(y_pred == cls)
                class_true_counts[cls] = np.sum(y == cls)
            
            dominant_classes = []
            for cls in np.unique(y):
                pred_ratio = class_pred_counts[cls] / len(y_pred)
                true_ratio = class_true_counts[cls] / len(y)
                
                # If class is predicted much more frequently than it occurs
                if pred_ratio > 0.4:  # Class takes >40% of predictions
                    dominant_classes.append(cls)
            
            # Suppress features important for dominant classes
            if dominant_classes and hasattr(temp_model, 'feature_importances_'):
                importances = temp_model.feature_importances_
                
                # Get top features for dominant classes
                dominant_feature_indices = []
                top_k = 5  # Only suppress a few key features
                
                # Get indices of top features
                top_importance_indices = np.argsort(importances)[-top_k:]
                dominant_feature_indices.extend(top_importance_indices)
                
                # Suppress these features to give other classes a chance
                for idx in dominant_feature_indices:
                    if idx < len(feature_weights) and idx not in top_indices:
                        feature_weights[idx] = 0.5  # Moderately reduce importance
        
        except Exception:
            # If dominant class analysis fails, continue without it
            pass
        
        # Apply weights to create boosted features
        X_boosted = X * feature_weights
        
        return X_boosted

    def identify_struggling_class(self, y_true: np.ndarray, y_pred: np.ndarray,
                                iteration: int = 0) -> Tuple[str, float]:
        """
        Identify the class with lowest performance that needs improvement.
        Uses a rotation strategy to ensure all classes get attention.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            iteration: Current CAIFO iteration (used for adaptive strategy)
            
        Returns:
            Tuple of (struggling_class, recall_or_penalty)
        """
        # Calculate per-class recall
        classes = np.unique(y_true)
        recalls = recall_score(y_true, y_pred, average=None, labels=classes)
        
        # Get prediction counts for each class
        class_pred_counts = {cls: np.sum(y_pred == cls) for cls in classes}
        class_true_counts = {cls: np.sum(y_true == cls) for cls in classes}
        total_samples = len(y_pred)
        
        # Check for monopoly (one class taking >80% of predictions)
        for cls in classes:
            pred_pct = class_pred_counts[cls] / total_samples * 100
            if pred_pct > 80:
                # Return with special negative recall value to trigger extreme rebalancing
                return str(cls), -10.0
        
        # Check for classes that are never predicted (but should be)
        zero_pred_classes = []
        for cls in classes:
            if class_pred_counts[cls] == 0 and class_true_counts[cls] > 0:
                # Class has instances but never predicted
                zero_pred_classes.append((cls, class_true_counts[cls]))
        
        # Force rotation through zero-prediction classes
        if zero_pred_classes:
            # Sort by true count (most frequent first)
            zero_pred_classes.sort(key=lambda x: x[1], reverse=True)
            
            # Get index based on iteration
            idx = iteration % len(zero_pred_classes)
            cls, _ = zero_pred_classes[idx]
            
            # Return with special zero-recall code to trigger stronger handling
            return str(cls), 0.0
        
        # Main strategy: Rotation through ALL classes to ensure each gets attention
        cls_index = iteration % len(classes)
        target_class = classes[cls_index]
        class_recall = recalls[cls_index]
        
        return str(target_class), class_recall
    
    def calculate_feature_weights(self, model, X: np.ndarray, y: np.ndarray, 
                                   feature_names: List[str], 
                                   struggling_class: str) -> Dict[str, float]:
        """
        Calculate optimised feature weights with enhanced exclusivity boosting.
        
        Args:
            model: Trained model
            X: Feature matrix
            y: Target labels
            feature_names: List of feature names
            struggling_class: Identified struggling class
            
        Returns:
            Dictionary of feature weights
        """
        # Initialise weights to 1.0
        feature_weights = {feature: 1.0 for feature in feature_names}
        
        try:
            # Import or create FeatureImportanceCalculator
            from features import FeatureImportanceCalculator
            importance_calculator = FeatureImportanceCalculator()
            
            # Calculate feature importance for all classes
            importance_dict = importance_calculator.calculate_per_class_importance(
                model, X, y, feature_names
            )
            
            # Check if this is an overpredicted class (negative recall case)
            struggling_class_int = int(struggling_class)
            is_overpredicted = False
            
            # Check for overprediction by looking at prediction distribution
            class_pred_counts = {}
            class_true_counts = {}
            for cls in np.unique(y):
                class_pred_counts[cls] = np.sum(model.predict(X) == cls)
                class_true_counts[cls] = np.sum(y == cls)
            
            # If the class appears more frequently in predictions than in truth
            if struggling_class_int in class_pred_counts and struggling_class_int in class_true_counts:
                pred_ratio = class_pred_counts[struggling_class_int] / len(y)
                true_ratio = class_true_counts[struggling_class_int] / len(y)
                
                if pred_ratio > 2 * true_ratio and pred_ratio > 0.15:
                    is_overpredicted = True
            
            # For overpredicted classes, REDUCE weights of its most important features
            if is_overpredicted:
                class_importance = importance_dict.get(struggling_class, {})
                
                if not class_importance and importance_dict:
                    # Look for any overpredicted class if struggling class not found
                    overpredicted_classes = []
                    for cls in importance_dict:
                        cls_int = int(cls)
                        if cls_int in class_pred_counts and cls_int in class_true_counts:
                            pred_ratio = class_pred_counts[cls_int] / len(y)
                            true_ratio = class_true_counts[cls_int] / len(y)
                            if pred_ratio > 2 * true_ratio and pred_ratio > 0.15:
                                overpredicted_classes.append((cls, pred_ratio / true_ratio))
                    
                    if overpredicted_classes:
                        # Sort by overprediction ratio and use the most overpredicted
                        overpredicted_classes.sort(key=lambda x: x[1], reverse=True)
                        overpredicted_class = overpredicted_classes[0][0]
                        class_importance = importance_dict[overpredicted_class]
                
                # Get top features for this class
                top_n = min(20, len(class_importance))  # Consider top 20 features
                top_features = sorted(
                    class_importance.items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )[:top_n]
                
                # Aggressive reduction for top discriminative features of overpredicted class
                reduction_factor = 0.2  # Reduce to 20% of original importance
                for feature, _ in top_features:
                    feature_weights[feature] = reduction_factor
                
                # Find other classes with low recall that need boosting
                low_recall_classes = []
                for cls in np.unique(y):
                    cls_str = str(cls)
                    if cls_str in importance_dict and cls != struggling_class_int:
                        # Calculate recall for this class
                        y_pred = model.predict(X)
                        cls_recall = np.sum((y_pred == cls) & (y == cls)) / max(1, np.sum(y == cls))
                        
                        if cls_recall < 0.3:  # Low recall threshold
                            low_recall_classes.append((cls_str, cls_recall))
                
                # If we have low recall classes, boost their features
                if low_recall_classes:
                    # Sort by recall (ascending)
                    low_recall_classes.sort(key=lambda x: x[1])
                    
                    # Take the worst performing class
                    boost_class, _ = low_recall_classes[0]
                    
                    if boost_class in importance_dict:
                        boost_importance = importance_dict[boost_class]
                        
                        # Get top features for this class
                        boost_top_n = min(10, len(boost_importance))
                        boost_top_features = sorted(
                            boost_importance.items(),
                            key=lambda x: x[1],
                            reverse=True
                        )[:boost_top_n]
                        
                        # Boost these features
                        boost_factor = 2.5  # Significant boost
                        for feature, _ in boost_top_features:
                            # Only boost if not already reduced for overpredicted class
                            if feature_weights[feature] > 0.5:
                                feature_weights[feature] = boost_factor
                
                return feature_weights
            
            # For regular struggling classes with low recall, boost their distinctive features
            
            # Check if importance dict has the struggling class
            if struggling_class not in importance_dict:
                return feature_weights
            
            class_importance = importance_dict[struggling_class]
            
            # Calculate exclusivity scores for features (how unique they are to this class)
            exclusivity_scores = {}
            
            for feature in feature_names:
                # Skip if feature not in importance dictionaries
                if feature not in class_importance:
                    continue
                
                # Get importance of this feature for the struggling class
                this_class_importance = class_importance.get(feature, 0.0)
                
                # Get max importance of this feature for any other class
                other_class_importance = 0.0
                for other_class, other_importances in importance_dict.items():
                    if other_class != struggling_class and feature in other_importances:
                        other_class_importance = max(other_class_importance, other_importances[feature])
                
                # Calculate exclusivity (how much more important for this class vs others)
                # Add small epsilon to prevent division by zero
                if other_class_importance > 0:
                    exclusivity = this_class_importance / other_class_importance
                else:
                    exclusivity = this_class_importance * 2.0  # Highly exclusive
                
                exclusivity_scores[feature] = exclusivity
            
            # Find top exclusive features (distinctive for this class)
            top_exclusive = sorted(
                exclusivity_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )[:min(10, len(exclusivity_scores))]
            
            # Apply boost based on exclusivity
            for feature, exclusivity in top_exclusive:
                # Only boost if the feature has some importance for this class
                if feature in class_importance and class_importance[feature] > 0.01:
                    # More moderate boost based on exclusivity
                    boost = min(2.5, 1.0 + (exclusivity * 1.5))
                    feature_weights[feature] = boost
            
            # Get top N features by raw importance 
            top_important = sorted(
                class_importance.items(),
                key=lambda x: x[1],
                reverse=True
            )[:min(10, len(class_importance))]
            
            # Apply standard boost to generally important features
            boost_factor = 2.0
            for feature, importance in top_important:
                # Only apply if not already boosted by exclusivity
                if feature_weights[feature] == 1.0 and importance > 0.01:
                    feature_weights[feature] = boost_factor
            
            # Additionally, REDUCE weights of features important for overpredicted classes
            overpredicted_classes = []
            for cls in importance_dict:
                cls_int = int(cls)
                if cls_int in class_pred_counts and cls_int in class_true_counts:
                    pred_ratio = class_pred_counts[cls_int] / len(y)
                    true_ratio = class_true_counts[cls_int] / len(y)
                    if pred_ratio > 2 * true_ratio and pred_ratio > 0.15:
                        overpredicted_classes.append(cls)
            
            # Reduce features important to overpredicted classes
            if overpredicted_classes:
                for overpredicted_class in overpredicted_classes:
                    if overpredicted_class in importance_dict:
                        over_class_importance = importance_dict[overpredicted_class]
                        
                        # Get top features for overpredicted class
                        top_over_features = sorted(
                            over_class_importance.items(), 
                            key=lambda x: x[1], 
                            reverse=True
                        )[:10]  # Top 10 features
                        
                        # Reduce weights for these features if not already boosted for struggling class
                        for feature, _ in top_over_features:
                            if feature_weights[feature] == 1.0:  # Not already modified
                                feature_weights[feature] = 0.5  # Moderate reduction
            
            return feature_weights
            
        except Exception as e:
            # If there's an error, return default weights
            return {feature: 1.0 for feature in feature_names}
    
    def apply_feature_weights(self, X: np.ndarray, feature_weights: Dict[str, float], 
                            feature_names: List[str]) -> np.ndarray:
        """
        Apply feature weights to the feature matrix.
        
        Args:
            X: Feature matrix
            feature_weights: Dictionary of feature weights
            feature_names: List of feature names
            
        Returns:
            Weighted feature matrix
        """
        # Create weight array in the same order as feature_names
        weights = np.ones(X.shape[1])  # Initialise with 1.0 (no change)
        
        # Apply weights only for features that exist in both X and feature_weights
        for i, name in enumerate(feature_names):
            if i < X.shape[1]:  # Ensure index is within X dimensions
                weights[i] = feature_weights.get(name, 1.0)
        
        # Apply weights to features
        X_weighted = X * weights
        
        # Apply a sanity check normalisation to prevent extreme scaling
        if np.max(weights) / np.min(weights) > 20.0:
            # Normalise X_weighted to have similar overall magnitude to X
            orig_norm = np.linalg.norm(X)
            weighted_norm = np.linalg.norm(X_weighted)
            if weighted_norm > 0:
                X_weighted = X_weighted * (orig_norm / weighted_norm)
        
        return X_weighted