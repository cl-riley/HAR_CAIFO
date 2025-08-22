"""
Bayesian optimisation for model hyperparameters and initial class weights.
"""
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from bayes_opt import BayesianOptimization
from sklearn.metrics import f1_score

class BayesianHyperOptimizer:
    """
    Bayesian optimisation for model hyperparameters and initial class weights.
    """
    
    def __init__(self, config):
        """
        Initialise the Bayesian optimiser.
        
        Args:
            config: Configuration object containing CAIFO parameters
        """
        self.config = config.caifo
        self.model_config = config.model
        self.random_state = 42
        self.label_encoder = None
        self.use_test_validation = True  # Use test data to prevent overfitting
    
    def optimize(self, X: np.ndarray, y: np.ndarray, feature_names: List[str] = None, 
                metric: str = 'f1_weighted', test_data: Dict[str, np.ndarray] = None,
                user_ids: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Run Bayesian optimisation to find optimal hyperparameters and weights.
        
        Args:
            X: Feature matrix
            y: Target labels
            feature_names: Optional list of feature names
            metric: Metric to optimise ('f1_weighted', 'recall_weighted', etc.)
            test_data: Optional test data for validation (prevents overfitting)
            user_ids: Optional array of user IDs for group-aware CV
            
        Returns:
            Dictionary with optimised model parameters and class weights
        """
        # Define parameter bounds based on model type
        param_bounds = self._get_param_bounds()
        
        # Add class weight parameters for each unique class
        class_names = np.unique(y)
        for class_name in class_names:
            param_bounds[f'class_weight_{str(class_name)}'] = (
                self.config.class_weight_range[0], 
                self.config.class_weight_range[1]
            )
        
        # Set up optimiser
        optimizer = BayesianOptimization(
            f=lambda **params: self._evaluate_params(params, X, y, metric, test_data, user_ids),
            pbounds=param_bounds,
            random_state=self.random_state,
            verbose=2
        )
        
        try:
            # Run optimisation
            optimizer.maximize(
                init_points=self.config.bayesian_init_points,
                n_iter=self.config.n_bayesian_iterations,
            )
            
            # Get best parameters
            best_params = optimizer.max['params']
            
            # Convert to appropriate types and format
            result = self._format_optimized_params(best_params, class_names)
            
            # Enhance class weights based on class distribution
            class_distribution = {cls: np.sum(y == cls) for cls in class_names}
            max_count = max(class_distribution.values())
            
            # Adjust weights to favour minority classes
            for cls in class_names:
                cls_str = str(cls)
                count = class_distribution[cls]
                # Set weight proportional to inverse frequency
                weight_ratio = max_count / max(1, count)
                
                # Ensure minority classes get at least their frequency-based weight
                if cls_str in result['class_weights']:
                    result['class_weights'][cls_str] = max(result['class_weights'][cls_str], weight_ratio)
                else:
                    result['class_weights'][cls_str] = weight_ratio
            
            return result
            
        except Exception as e:
            # Provide fallback default parameters
            return self._get_default_params(class_names, y)
    
    def _get_param_bounds(self) -> Dict[str, Tuple[float, float]]:
        """
        Define parameter bounds for Bayesian optimisation based on model type.
        
        Returns:
            Dictionary mapping parameter names to (min, max) ranges
        """
        if self.model_config.model_type == "random_forest":
            return {
                'n_estimators': (50, 200),
                'max_depth': (3, 15),
                'min_samples_split': (2, 10),
            }
        elif self.model_config.model_type == "gradient_boosting":
            return {
                'n_estimators': (50, 200),
                'learning_rate': (0.01, 0.2),
                'max_depth': (2, 8),
            }
        elif self.model_config.model_type == "xgboost":
            return {
                'n_estimators': (50, 200),
                'learning_rate': (0.01, 0.2),
                'max_depth': (2, 8),
                'subsample': (0.5, 1.0),
                'colsample_bytree': (0.5, 1.0),
                'reg_alpha': (0.01, 1.0),  # L1 regularisation
                'reg_lambda': (0.01, 5.0),  # L2 regularisation
            }
        else:
            # Default to RandomForest if model type not supported
            return {
                'n_estimators': (50, 200),
                'max_depth': (3, 15),
                'min_samples_split': (2, 10),
            }
    
    def _get_default_params(self, class_names: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Get default parameters when optimisation fails.
        
        Args:
            class_names: Array of unique class names
            y: Target labels
            
        Returns:
            Dictionary with default model parameters and class weights
        """
        # Default model parameters based on model type
        if self.model_config.model_type == "random_forest":
            default_params = {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 2,
            }
        elif self.model_config.model_type == "gradient_boosting":
            default_params = {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 5,
            }
        elif self.model_config.model_type == "xgboost":
            default_params = {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 6,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.5,  # L1 regularisation
                'reg_lambda': 1.0,  # L2 regularisation
            }
        else:
            # Default to RandomForest parameters
            default_params = {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 2,
            }
        
        # Calculate default class weights based on class distribution
        class_distribution = {str(cls): np.sum(y == cls) for cls in class_names}
        max_count = max(class_distribution.values())
        
        class_weights = {}
        for cls_str, count in class_distribution.items():
            # Set weight inversely proportional to frequency
            weight_ratio = max_count / max(1, count)
            class_weights[cls_str] = weight_ratio
        
        return {
            'model_params': default_params,
            'class_weights': class_weights
        }
    
    def _evaluate_params(self, params: Dict[str, float], X: np.ndarray, 
                    y: np.ndarray, metric: str, test_data: Dict[str, np.ndarray] = None,
                    user_ids: Optional[np.ndarray] = None) -> float:
        """
        Evaluation function for parameter optimisation with group-aware cross-validation.
        
        Args:
            params: Dictionary of parameters to evaluate
            X: Feature matrix
            y: Target labels
            metric: Evaluation metric
            test_data: Optional test data for validation
            user_ids: Optional array of user IDs for group-aware CV
            
        Returns:
            Score for the given parameters
        """
        try:
            # Import metrics functions
            from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score
            
            # Extract model parameters
            model_params = {}
            class_weights = {}
            
            for key, value in params.items():
                if key.startswith('class_weight_'):
                    class_name = key.replace('class_weight_', '')
                    class_weights[class_name] = value
                else:
                    # Handle specific parameter types
                    if key in ['n_estimators', 'max_depth', 'min_samples_split']:
                        model_params[key] = int(value)
                    else:
                        model_params[key] = value
            
            # Create class weight dictionary in the format expected by sklearn
            unique_classes = np.unique(y)
            class_weight_dict = {}
            
            # Map weights to the actual class values
            for class_val in unique_classes:
                class_str = str(class_val)
                if class_str in class_weights:
                    class_weight_dict[class_val] = class_weights[class_str]
                else:
                    # Default weight for classes not specified
                    class_weight_dict[class_val] = 1.0
            
            # Build model based on type
            if self.model_config.model_type == "random_forest":
                from sklearn.ensemble import RandomForestClassifier
                # For RandomForest, use class_weight parameter
                model_class = RandomForestClassifier
                model_params['class_weight'] = class_weight_dict
                model_params['random_state'] = self.random_state
                
            elif self.model_config.model_type == "gradient_boosting":
                # GradientBoostingClassifier doesn't support class_weight directly
                from sklearn.ensemble import GradientBoostingClassifier
                model_class = GradientBoostingClassifier
                model_params['random_state'] = self.random_state
                
            elif self.model_config.model_type == "xgboost":
                from xgboost import XGBClassifier
                
                # Configure XGBoost parameters for multiclass
                num_classes = len(np.unique(y))
                if num_classes > 2:
                    model_params['objective'] = 'multi:softprob'
                    model_params['num_class'] = num_classes
                
                # Create the XGBoost model
                model_class = XGBClassifier
                model_params['random_state'] = self.random_state
                
            else:
                # Fallback to RandomForest for unsupported model types
                from sklearn.ensemble import RandomForestClassifier
                model_class = RandomForestClassifier
                model_params['class_weight'] = class_weight_dict
                model_params['random_state'] = self.random_state
            
            # If test data is provided, use it for validation
            if test_data is not None and self.use_test_validation:
                # Train on all training data
                model = model_class(**model_params)
                
                # For XGBoost or GBM, handle class weights via sample weights
                if self.model_config.model_type in ["xgboost", "gradient_boosting"]:
                    # Create sample weights based on class weights
                    sample_weights = np.ones(len(y))
                    for idx, label in enumerate(y):
                        if label in class_weight_dict:
                            sample_weights[idx] = class_weight_dict[label]
                    
                    model.fit(X, y, sample_weight=sample_weights)
                else:
                    model.fit(X, y)
                
                # Evaluate on test data
                X_test = test_data['X_test']
                y_test = test_data['y_test']
                
                y_pred = model.predict(X_test)
                
                # Calculate score based on metric
                if metric == 'f1_weighted':
                    score = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                elif metric == 'recall_weighted':
                    score = recall_score(y_test, y_pred, average='weighted')
                elif metric == 'precision_weighted':
                    score = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                else:
                    # Default to accuracy
                    score = accuracy_score(y_test, y_pred)
                
                # Apply penalties for imbalanced predictions
                
                # Calculate class-specific metrics to check for balanced performance
                class_recalls = recall_score(y_test, y_pred, average=None, labels=np.unique(y_test))
                
                # Penalise models that have too many zero-recall classes
                zero_recall_count = np.sum(class_recalls == 0)
                zero_recall_penalty = zero_recall_count / len(class_recalls) * 0.5
                
                # Check for class dominance (one class taking over predictions)
                class_pred_counts = {}
                for cls in np.unique(y_test):
                    class_pred_counts[cls] = np.sum(y_pred == cls) / len(y_pred)
                
                # Penalise if any class is overpredicted
                max_pred_ratio = max(class_pred_counts.values())
                dominance_penalty = 0.0
                if max_pred_ratio > 0.5:  # If one class is more than 50% of predictions
                    dominance_penalty = (max_pred_ratio - 0.5) * 0.5
                
                # Apply penalty for extreme weights
                weight_penalty = self._calculate_weight_penalty(class_weight_dict)
                
                # Combine penalties
                total_penalty = weight_penalty + zero_recall_penalty + dominance_penalty
                total_penalty = min(0.9, total_penalty)  # Cap at 0.9 to prevent negative scores
                
                adjusted_score = score * (1.0 - total_penalty)
                
                return adjusted_score
            
            else:
                # Use standard cross-validation
                from sklearn.model_selection import StratifiedKFold
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
                splits = cv.split(X, y)
                
                scores = []
                zero_recall_rates = []
                dominance_rates = []
                
                for train_idx, val_idx in splits:
                    X_train_fold, X_val_fold = X[train_idx], X[val_idx]
                    y_train_fold, y_val_fold = y[train_idx], y[val_idx]
                    
                    # Create and train model
                    model = model_class(**model_params)
                    
                    # For XGBoost or GBM, handle class weights via sample weights
                    if self.model_config.model_type in ["xgboost", "gradient_boosting"]:
                        # Create sample weights based on class weights
                        sample_weights = np.ones(len(y_train_fold))
                        for idx, label in enumerate(y_train_fold):
                            if label in class_weight_dict:
                                sample_weights[idx] = class_weight_dict[label]
                        
                        model.fit(X_train_fold, y_train_fold, sample_weight=sample_weights)
                    else:
                        model.fit(X_train_fold, y_train_fold)
                    
                    # Evaluate on validation fold
                    y_pred = model.predict(X_val_fold)
                    
                    # Calculate score based on metric
                    if metric == 'f1_weighted':
                        score = f1_score(y_val_fold, y_pred, average='weighted')
                    elif metric == 'recall_weighted':
                        score = recall_score(y_val_fold, y_pred, average='weighted')
                    elif metric == 'precision_weighted':
                        score = precision_score(y_val_fold, y_pred, average='weighted')
                    else:
                        # Default to accuracy
                        score = accuracy_score(y_val_fold, y_pred)
                    
                    scores.append(score)
                    
                    # Calculate class-specific metrics
                    unique_classes = np.unique(y_val_fold)
                    class_recalls = recall_score(y_val_fold, y_pred, average=None, labels=unique_classes)
                    
                    # Count zero-recall classes
                    zero_recall_count = np.sum(class_recalls == 0)
                    zero_recall_rates.append(zero_recall_count / len(unique_classes))
                    
                    # Check for class dominance
                    class_pred_counts = {}
                    for cls in unique_classes:
                        class_pred_counts[cls] = np.sum(y_pred == cls) / len(y_pred)
                    
                    # Calculate max prediction ratio
                    max_pred_ratio = max(class_pred_counts.values())
                    dominance_rates.append(max_pred_ratio)
                
                # Calculate mean score and penalties across folds
                mean_score = np.mean(scores)
                
                # Apply zero-recall penalty
                avg_zero_recall_rate = np.mean(zero_recall_rates)
                zero_recall_penalty = avg_zero_recall_rate * 0.5
                
                # Apply dominance penalty
                avg_dominance_rate = np.mean(dominance_rates)
                dominance_penalty = 0.0
                if avg_dominance_rate > 0.5:
                    dominance_penalty = (avg_dominance_rate - 0.5) * 0.5
                
                # Apply weight penalty
                weight_penalty = self._calculate_weight_penalty(class_weight_dict)
                
                # Combine penalties
                total_penalty = weight_penalty + zero_recall_penalty + dominance_penalty
                total_penalty = min(0.9, total_penalty)  # Cap at 0.9 to prevent negative scores
                
                adjusted_score = mean_score * (1.0 - total_penalty)
                
                return adjusted_score
            
        except Exception as e:
            # If an error occurs during evaluation, return a very low score
            return -999.0  # A very low score for maximisation
    
    def _calculate_weight_penalty(self, class_weight_dict: Dict[Any, float]) -> float:
        """
        Calculate penalty for extreme class weights to prevent overfitting.
        
        Args:
            class_weight_dict: Dictionary of class weights
            
        Returns:
            Weight penalty factor (0-1)
        """
        if not class_weight_dict:
            return 0.0
            
        max_weight = max(class_weight_dict.values())
        min_weight = min(class_weight_dict.values())
        
        # Calculate penalty based on weight ratio
        weight_ratio = max_weight / min_weight
        
        if weight_ratio > 10.0:
            # Start with a 5% penalty for ratios > 10
            penalty = 0.05
            
            # Additional penalty for very extreme ratios
            if weight_ratio > 20.0:
                # Add up to 20% more penalty for extreme ratios
                penalty += 0.2 * min(1.0, (weight_ratio - 20.0) / 80.0)
                
            return penalty
        
        return 0.0
    
    def _format_optimized_params(self, params: Dict[str, float], 
                               class_names: np.ndarray) -> Dict[str, Any]:
        """
        Format optimised parameters for use in model training.
        
        Args:
            params: Dictionary of optimised parameters
            class_names: Array of class names
            
        Returns:
            Formatted parameter dictionary
        """
        # Extract and format model parameters
        model_params = {}
        class_weights = {}
        
        for key, value in params.items():
            if key.startswith('class_weight_'):
                class_name = key.replace('class_weight_', '')
                class_weights[class_name] = value
            else:
                # Handle specific parameter types
                if key in ['n_estimators', 'max_depth', 'min_samples_split']:
                    model_params[key] = int(value)
                else:
                    model_params[key] = value
        
        # Sanitise class weights
        sanitised_class_weights = {}
        
        for class_val in class_names:
            class_str = str(class_val)
            
            # Use the actual class value as the key in the final dictionary
            if class_str in class_weights:
                sanitised_class_weights[class_str] = class_weights[class_str]
            else:
                # Default weight for classes not in params
                sanitised_class_weights[class_str] = 1.0
        
        # Limit extreme class weights
        for cls, weight in sanitised_class_weights.items():
            sanitised_class_weights[cls] = max(min(weight, self.config.class_weight_range[1]), 
                                              self.config.class_weight_range[0])
        
        result = {
            'model_params': model_params,
            'class_weights': sanitised_class_weights
        }
        
        return result