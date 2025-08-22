"""
Feature extraction and importance calculation for HAR-CAIFO.
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union
from scipy import stats, signal, fft
from sklearn.feature_selection import mutual_info_classif, SelectKBest

class FeatureExtractor:
    """Extract features from sensor data for HAR."""
    
    def __init__(self, config):
        """
        Initialize the feature extractor with configuration.
        
        Args:
            config: FeatureConfig object containing feature parameters
        """
        self.config = config
        self.feature_names = []
    
    def extract_features(self, windowed_data: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        """
        Extract features from windowed sensor data.
        
        Args:
            windowed_data: Numpy array of shape (n_windows, window_size, n_features)
            
        Returns:
            Tuple of (features array, feature names)
        """
        if len(windowed_data) == 0:
            return np.array([]), []
            
        n_windows, window_size, n_channels = windowed_data.shape
        all_features = []
        self.feature_names = []
        
        # Process each window
        for window_idx in range(n_windows):
            window_features = []
            
            # Process each channel in the window
            for channel_idx in range(n_channels):
                channel_data = windowed_data[window_idx, :, channel_idx]
                
                # Extract time domain features
                if self.config.time_domain_features:
                    time_features = self._extract_time_domain_features(channel_data)
                    window_features.extend(time_features)
                    
                    # Add feature names on first iteration
                    if window_idx == 0:
                        self.feature_names.extend([
                            f"time_{name}_ch{channel_idx}" for name in [
                                "mean", "std", "min", "max", "median", "rms", 
                                "zero_crossings", "mean_abs_diff", "peak_to_peak"
                            ]
                        ])
                
                # Extract frequency domain features
                if self.config.frequency_domain_features:
                    freq_features = self._extract_frequency_domain_features(channel_data)
                    window_features.extend(freq_features)
                    
                    # Add feature names on first iteration
                    if window_idx == 0:
                        self.feature_names.extend([
                            f"freq_{name}_ch{channel_idx}" for name in [
                                "mean", "std", "energy", "dom_freq", "dom_freq_mag"
                            ]
                        ])
                
                # Extract statistical features
                if self.config.statistical_features:
                    stat_features = self._extract_statistical_features(channel_data)
                    window_features.extend(stat_features)
                    
                    # Add feature names on first iteration
                    if window_idx == 0:
                        self.feature_names.extend([
                            f"stat_{name}_ch{channel_idx}" for name in [
                                "correlation", "autocorrelation"
                            ]
                        ])
            
            all_features.append(window_features)
        
        # Convert to numpy array
        features_array = np.array(all_features)
        
        return features_array, self.feature_names
    
    def _extract_time_domain_features(self, data: np.ndarray) -> List[float]:
        """
        Extract time domain features from a single channel.
        
        Args:
            data: 1D array of sensor data
            
        Returns:
            List of time domain features
        """
        # Basic statistical features
        mean = np.mean(data)
        std = np.std(data)
        min_val = np.min(data)
        max_val = np.max(data)
        median = np.median(data)
        
        # Root mean square
        rms = np.sqrt(np.mean(np.square(data)))
        
        # Zero crossings
        zero_crossings = np.sum(np.diff(np.signbit(data)))
        
        # Mean absolute difference
        mean_abs_diff = np.mean(np.abs(np.diff(data)))
        
        # Peak to peak amplitude
        peak_to_peak = max_val - min_val
        
        return [
            mean, std, min_val, max_val, median, rms, 
            zero_crossings, mean_abs_diff, peak_to_peak
        ]
    
    def _extract_frequency_domain_features(self, data: np.ndarray) -> List[float]:
        """
        Extract frequency domain features from a single channel.
        
        Args:
            data: 1D array of sensor data
            
        Returns:
            List of frequency domain features
        """
        # Compute FFT
        fft_values = np.abs(fft.rfft(data))
        fft_freq = fft.rfftfreq(len(data))
        
        # Basic spectral features
        mean = np.mean(fft_values)
        std = np.std(fft_values)
        
        # Energy
        energy = np.sum(np.square(fft_values)) / len(fft_values)
        
        # Dominant frequency and its magnitude
        dom_freq_idx = np.argmax(fft_values)
        dom_freq = fft_freq[dom_freq_idx]
        dom_freq_mag = fft_values[dom_freq_idx]
        
        return [mean, std, energy, dom_freq, dom_freq_mag]
    
    def _extract_statistical_features(self, data: np.ndarray) -> List[float]:
        """
        Extract additional statistical features from a single channel.
        
        Args:
            data: 1D array of sensor data
            
        Returns:
            List of statistical features
        """
        # Calculate auto-correlation
        if len(data) > 1:
            correlation = np.mean(data[:-1] * data[1:])
            autocorr = np.corrcoef(data[:-1], data[1:])[0, 1]
        else:
            correlation = 0
            autocorr = 0
        
        return [correlation, autocorr]
    
    def select_features(self, X: np.ndarray, y: np.ndarray, feature_names: List[str],
                      fixed_indices: Optional[List[int]] = None) -> Tuple[np.ndarray, List[str]]:
        """
        Select most relevant features using the configured method or use fixed indices.
        
        Args:
            X: Feature matrix
            y: Target labels
            feature_names: List of feature names
            fixed_indices: Optional list of pre-selected feature indices to use
            
        Returns:
            Tuple of (selected features array, selected feature names)
        """
        if X.shape[1] == 0:
            return X, feature_names
            
        # If fixed indices are provided, use them instead of selecting new features
        if fixed_indices is not None:
            # Make sure indices are within bounds
            valid_indices = [i for i in fixed_indices if i < X.shape[1]]
            if len(valid_indices) > 0:
                X_selected = X[:, valid_indices]
                selected_feature_names = [feature_names[i] for i in valid_indices if i < len(feature_names)]
                return X_selected, selected_feature_names
            else:
                return X, feature_names
                
        n_features = min(self.config.n_features_to_select, X.shape[1])
        
        try:
            if self.config.feature_selection_method == "permutation":
                from sklearn.inspection import permutation_importance
                from sklearn.ensemble import RandomForestClassifier
                
                # Train a simple model
                model = RandomForestClassifier(n_estimators=50, random_state=42)
                model.fit(X, y)
                
                # Compute permutation importance
                result = permutation_importance(
                    model, X, y, n_repeats=5, random_state=42, n_jobs=-1
                )
                indices = np.argsort(result.importances_mean)[::-1][:n_features]
                
            elif self.config.feature_selection_method == "mutual_info":
                selector = SelectKBest(mutual_info_classif, k=n_features)
                X_selected = selector.fit_transform(X, y)
                indices = selector.get_support(indices=True)
                
            else:
                # Fallback to using first n_features if method is unknown
                print(f"Unknown feature selection method: {self.config.feature_selection_method}")
                print(f"Using first {n_features} features instead.")
                indices = np.arange(min(n_features, len(feature_names)))
        
        except Exception as e:
            print(f"Error in feature selection: {str(e)}")
            print(f"Using first {n_features} features as fallback.")
            indices = np.arange(min(n_features, len(feature_names)))
        
        # Ensure indices are within bounds of feature_names
        valid_indices = [i for i in indices if i < len(feature_names)]
        
        # Return selected features and their names
        if len(valid_indices) > 0:
            X_selected = X[:, valid_indices]
            selected_feature_names = [feature_names[i] for i in valid_indices]
            return X_selected, selected_feature_names
        else:
            return X, feature_names


class FeatureImportanceCalculator:
    """Calculate feature importance per class."""
    
    def __init__(self):
        """Initialize the feature importance calculator."""
        pass
    
    def calculate_per_class_importance(self, model, X: np.ndarray, y: np.ndarray, 
                                     feature_names: List[str]) -> Dict[str, Dict[str, float]]:
        """
        Calculate feature importance for each class.
        
        Args:
            model: Trained model with predict method
            X: Feature matrix
            y: Target labels
            feature_names: List of feature names
            
        Returns:
            Dictionary mapping class labels to feature importance dictionaries
        """
        class_names = np.unique(y)
        result = {}

        # Try different methods of importance calculation
        try:
            # First check if model has built-in feature importance
            if hasattr(model, 'feature_importances_'):
                # For tree-based models that have built-in feature importance
                importances = model.feature_importances_
                
                # Create initial importance dict for all classes
                for class_name in class_names:
                    importance_dict = {
                        feature_names[i]: importances[i] if i < len(importances) else 0
                        for i in range(min(len(feature_names), len(importances)))
                    }
                    
                    # Sort by importance (descending)
                    importance_dict = dict(sorted(
                        importance_dict.items(), 
                        key=lambda x: x[1], 
                        reverse=True
                    ))
                    
                    result[str(class_name)] = importance_dict
                
                return result
                
        except Exception as e:
            print(f"Error calculating built-in importance: {str(e)}")
        
        # Fallback to a simpler correlation-based method
        print("Using correlation-based feature importance calculation")
        
        # For each class, calculate a simple measure of feature importance
        for class_name in class_names:
            # Convert to binary classification problem (one-vs-rest)
            y_binary = (y == class_name).astype(int)
            
            # Calculate importance based on correlation with the target
            importance_dict = {}
            
            for i, feature_name in enumerate(feature_names):
                if i < X.shape[1]:
                    # Extract this feature's column
                    feature_col = X[:, i]
                    
                    # Calculate correlation
                    try:
                        correlation = np.abs(np.corrcoef(feature_col, y_binary)[0, 1])
                        if np.isnan(correlation):
                            correlation = 0.0
                    except:
                        correlation = 0.0
                        
                    importance_dict[feature_name] = correlation
            
            # Sort by importance (descending)
            importance_dict = dict(sorted(
                importance_dict.items(), 
                key=lambda x: x[1], 
                reverse=True
            ))
            
            result[str(class_name)] = importance_dict
        
        return result

    def identify_class_specific_features(self, importances: Dict[str, Dict[str, float]],
                                       top_n: int = 10) -> Dict[str, List[str]]:
        """
        Identify the most important features for each class.
        
        Args:
            importances: Dictionary of per-class feature importances
            top_n: Number of top features to select per class
            
        Returns:
            Dictionary mapping class names to lists of their most important features
        """
        class_features = {}
        
        for class_name, class_importances in importances.items():
            # Get top N features for this class
            top_features = list(class_importances.keys())[:min(top_n, len(class_importances))]
            class_features[class_name] = top_features
            
        return class_features