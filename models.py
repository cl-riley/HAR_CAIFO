"""
Model creation and management for HAR-CAIFO.
"""
import numpy as np
import os
import joblib
from typing import Dict, Any, Optional, Union, List
from sklearn.base import BaseEstimator

class BaseModel:
    """
    Base model for HAR with CAIFO optimization.
    """
    
    def __init__(self, config):
        """
        Initialize the base model.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.model_config = config.model
        self.model = None
        self.label_encoder = None
        self.class_mapping = None
        self.performance_history = []
    
    def create_model(self, class_weight: Optional[Dict[Any, float]] = None) -> BaseEstimator:
        """
        Create a new model instance with proper class weight handling.
        
        Args:
            class_weight: Optional class weights dictionary
            
        Returns:
            Initialized model instance
        """
        model_type = self.model_config.model_type
        model_params = self.model_config.model_params.copy() if hasattr(self.model_config, 'model_params') and self.model_config.model_params else {}
        
        # Set random_state if not already in params
        if 'random_state' not in model_params:
            model_params['random_state'] = 42
        
        # Fix: Remove class_weight from model_params if we're providing custom weights
        if class_weight is not None and 'class_weight' in model_params:
            print(f"Removing 'class_weight: {model_params['class_weight']}' from model_params to use custom weights")
            del model_params['class_weight']
        
        # Log what we're doing with class weights
        if class_weight is not None:
            print(f"Creating model with custom class weights: {class_weight}")
        else:
            print("Creating model without custom class weights")
        
        # Model creation based on type
        if model_type == "random_forest":
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(class_weight=class_weight, **model_params)
            
        elif model_type == "gradient_boosting":
            from sklearn.ensemble import GradientBoostingClassifier
            # GradientBoostingClassifier doesn't support class_weight directly
            # We'll handle this differently via sample weights in the train method
            return GradientBoostingClassifier(**model_params)
            
        elif model_type == "xgboost":
            try:
                from xgboost import XGBClassifier
                
                # Create the model (class weights handled differently for XGBoost)
                return XGBClassifier(**model_params)
            except ImportError:
                print("XGBoost not available. Falling back to RandomForest.")
                from sklearn.ensemble import RandomForestClassifier
                return RandomForestClassifier(class_weight=class_weight, **model_params)
                
        else:
            # Default to RandomForest for unsupported model types
            print(f"WARNING: Unsupported model type {model_type}, using RandomForest")
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(class_weight=class_weight, random_state=42)
    
    def _safeguard_class_weights(self, class_weight: Dict[Any, float]) -> Dict[Any, float]:
        """
        Apply safeguards to class weights.
        
        Args:
            class_weight: Raw class weights
            
        Returns:
            Safeguarded class weights
        """
        if not class_weight:
            return class_weight
        
        safeguarded = {}
        
        # Set reasonable limits
        max_weight_cap = 10.0
        min_weight_floor = 0.1
        
        # Check if any weight is extremely high
        max_weight = max(class_weight.values())
        min_weight = min(class_weight.values())
        weight_ratio = max_weight / min_weight
        
        # Different handling based on weight ratio
        if max_weight > max_weight_cap and weight_ratio <= 100.0:
            print(f"WARNING: High class weight detected ({max_weight:.1f}). Applying scaling.")
            
            # Scale down all weights proportionally if any exceeds the maximum
            scaling_factor = max_weight_cap / max_weight
            for cls, weight in class_weight.items():
                safeguarded[cls] = weight * scaling_factor
                
            # Ensure no weight falls below minimum
            for cls in safeguarded:
                if safeguarded[cls] < min_weight_floor:
                    safeguarded[cls] = min_weight_floor
                    
        else:
            # Copy weights to safeguarded dict when within acceptable range
            safeguarded = class_weight.copy()
            
            # Apply minimum weight floor
            for cls in safeguarded:
                if safeguarded[cls] < min_weight_floor:
                    safeguarded[cls] = min_weight_floor
        
        return safeguarded
    
    def train(self, X: np.ndarray, y: np.ndarray, 
            class_weight: Optional[Dict[Any, float]] = None) -> BaseEstimator:
        """
        Train the model with proper class weight handling.
        
        Args:
            X: Feature matrix
            y: Target labels
            class_weight: Optional class weights dictionary
            
        Returns:
            Trained model instance
        """
        try:
            # Apply safeguards to class weights
            if class_weight:
                # Convert string keys to integers for sklearn compatibility
                # This is the critical fix - sklearn requires numeric class labels as keys
                safeguarded_weights = {}
                for key, value in class_weight.items():
                    # Try to convert key to int if it's a string representation of a number
                    try:
                        numeric_key = int(key)
                        safeguarded_weights[numeric_key] = value
                    except (ValueError, TypeError):
                        # If conversion fails, keep the original key (but this might not work with sklearn)
                        safeguarded_weights[key] = value
                
                # Add additional safeguards for extreme weights
                max_weight = max(safeguarded_weights.values())
                min_weight = min(safeguarded_weights.values())
                
                # Apply cap if weights are too extreme
                if max_weight / min_weight > 50:
                    print(f"WARNING: Extreme weight ratio detected ({max_weight/min_weight:.1f}). Capping weights.")
                    for k in safeguarded_weights:
                        if safeguarded_weights[k] > 10:
                            safeguarded_weights[k] = 10
                
                class_weight = safeguarded_weights
                print(f"Using class weights: {class_weight}")
            
            # Create model with safeguarded class weights
            self.model = self.create_model(class_weight)
            
            # Handle different model types appropriately
            if self.model_config.model_type == "xgboost":
                self._train_xgboost_model(X, y, class_weight)
            elif self.model_config.model_type == "gradient_boosting":
                # For GBM, use sample weights since class_weight is not directly supported
                sample_weights = np.ones(len(y))
                if class_weight:
                    for idx, label in enumerate(y):
                        if label in class_weight:
                            sample_weights[idx] = class_weight[label]
                self.model.fit(X, y, sample_weight=sample_weights)
            else:
                # For other models, train normally
                self.model.fit(X, y)
            
            return self.model
        
        except Exception as e:
            print(f"Error in model training: {str(e)}")
            print("Falling back to RandomForest with balanced weights")
            
            # Fallback to RandomForest with balanced weights
            from sklearn.ensemble import RandomForestClassifier
            self.model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
            self.model.fit(X, y)
            return self.model
    
    def _train_xgboost_model(self, X: np.ndarray, y: np.ndarray, class_weight: Optional[Dict[Any, float]]) -> None:
        """
        Train XGBoost model with appropriate handling of class weights.
        
        Args:
            X: Feature matrix
            y: Target labels
            class_weight: Optional class weights dictionary
        """
        # Get number of classes
        num_classes = len(np.unique(y))
        
        # Configure for multiclass if needed
        if num_classes > 2:
            if hasattr(self.model, 'set_params'):
                self.model.set_params(
                    objective='multi:softprob',
                    num_class=num_classes
                )
        
        # For multiclass XGBoost, use sample weights
        if num_classes > 2 and class_weight:
            # Create sample weights based on class weights
            sample_weights = np.ones(len(y))
            for idx, label in enumerate(y):
                if label in class_weight:
                    sample_weights[idx] = class_weight[label]
                elif str(label) in class_weight:
                    sample_weights[idx] = class_weight[str(label)]
            
            # Train with sample weights
            self.model.fit(X, y, sample_weight=sample_weights)
        else:
            # Train normally
            self.model.fit(X, y)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate the model with performance monitoring.
        
        Args:
            X: Feature matrix
            y: Target labels
            
        Returns:
            Dictionary of evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # Make predictions
        y_pred = self.predict(X)
        
        # Calculate metrics
        from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix, accuracy_score
        
        accuracy = accuracy_score(y, y_pred)
        classes = np.unique(y)
        f1 = f1_score(y, y_pred, average='weighted')
        recall = recall_score(y, y_pred, average='weighted')
        precision = precision_score(y, y_pred, average='weighted')
        
        # Calculate per-class metrics
        per_class_recall = recall_score(y, y_pred, average=None, labels=classes)
        per_class_f1 = f1_score(y, y_pred, average=None, labels=classes)
        per_class_precision = precision_score(y, y_pred, average=None, labels=classes)
        
        # Create per-class metrics dictionary
        per_class_metrics = {}
        for i, class_idx in enumerate(classes):
            # Use class index as dictionary key
            class_name = str(class_idx)
                
            per_class_metrics[class_name] = {
                'recall': per_class_recall[i],
                'precision': per_class_precision[i],
                'f1': per_class_f1[i],
                'support': np.sum(y == class_idx)
            }
        
        # Calculate confusion matrix
        cm = confusion_matrix(y, y_pred, labels=classes)
        
        # Compute result dictionary
        result = {
            'accuracy': accuracy,
            'f1_weighted': f1,
            'recall_weighted': recall,
            'precision_weighted': precision,
            'per_class_metrics': per_class_metrics,
            'confusion_matrix': cm,
            'classes': classes
        }
        
        # Record performance metrics
        self.performance_history.append({
            'f1_weighted': f1,
            'recall_weighted': recall
        })
        
        return result
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the model.
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of predicted labels
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
            
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of class probabilities
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
            
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            # If model doesn't support probability estimation,
            # create pseudo-probabilities from hard predictions
            y_pred = self.predict(X)
            
            # Get unique classes
            if hasattr(self.model, 'classes_'):
                classes = self.model.classes_
            else:
                classes = np.unique(y_pred)
                
            # Create one-hot encoded probabilities
            n_samples = X.shape[0]
            n_classes = len(classes)
            probas = np.zeros((n_samples, n_classes))
            
            for i, pred in enumerate(y_pred):
                class_idx = np.where(classes == pred)[0][0]
                probas[i, class_idx] = 1.0
                
            return probas
    
    def save(self, file_path: str) -> None:
        """
        Save the model to disk.
        
        Args:
            file_path: Path to save the model to
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Save model, label encoder, and class mapping
        save_dict = {
            'model': self.model,
            'label_encoder': self.label_encoder,
            'class_mapping': self.class_mapping,
            'model_type': self.model_config.model_type,
            'performance_history': self.performance_history
        }
        
        # Save to disk
        joblib.dump(save_dict, file_path)
    
    def load(self, file_path: str) -> None:
        """
        Load model from disk.
        
        Args:
            file_path: Path to load the model from
        """
        if not os.path.exists(file_path):
            raise ValueError(f"Model file not found: {file_path}")
        
        # Load from disk
        save_dict = joblib.load(file_path)
        
        # Extract model, label encoder, and class mapping
        self.model = save_dict['model']
        self.label_encoder = save_dict.get('label_encoder')
        self.class_mapping = save_dict.get('class_mapping')
        self.model_config.model_type = save_dict.get('model_type', 'random_forest')
        self.performance_history = save_dict.get('performance_history', [])