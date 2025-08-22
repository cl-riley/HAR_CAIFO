"""
Data loading and preprocessing for HAR-CAIFO with strict user-based three-way data splitting.
"""
import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from sklearn.model_selection import train_test_split
from scipy import signal

class SensorDataLoader:
    """Load and preprocess sensor data for Human Activity Recognition with user-based data splitting."""
    
    def __init__(self, config):
        """
        Initialize the SensorDataLoader with configuration.
        
        Args:
            config: DataConfig object containing data parameters
        """
        self.config = config
        self.window_size = config.window_size
        self.window_step = config.window_step
        self.random_state = config.random_state
        self.enforce_user_separation = getattr(config, 'enforce_user_separation', True)
    
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load data from file with automatic format detection.
        
        Args:
            file_path: Path to the data file
            
        Returns:
            DataFrame containing the loaded data
        
        Raises:
            FileNotFoundError: If the data file doesn't exist
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        # Infer file format from extension
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.parquet'):
            df = pd.read_parquet(file_path)
        else:
            # Default to CSV
            df = pd.read_csv(file_path)
        
        print(f"Loaded {len(df)} samples from {file_path}")
        
        return df
    
    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess sensor data by removing NaN values, applying low-pass filter, and normalizing.
        
        Args:
            data: Raw sensor data DataFrame
            
        Returns:
            Preprocessed DataFrame
        """
        # Handle missing values
        data = data.copy()
        # Forward fill then backward fill to handle missing values
        data = data.ffill().bfill()
        
        # Apply low-pass filter to accelerometer and gyroscope columns
        sensor_cols = [col for col in data.columns if any(sensor in col.lower() for sensor in ['acc', 'gyro'])]
        
        for col in sensor_cols:
            # Butterworth low-pass filter
            b, a = signal.butter(3, 0.1, 'low')
            data[col] = signal.filtfilt(b, a, data[col])
        
        # Normalize sensor data columns (Z-score normalization)
        for col in sensor_cols:
            mean_val = data[col].mean()
            std_val = data[col].std()
            # Prevent division by zero
            if std_val > 0:
                data[col] = (data[col] - mean_val) / std_val
            
        return data
    
    def segment_data(self, data: Union[pd.DataFrame, np.ndarray], 
                    labels: Optional[Union[pd.Series, np.ndarray]] = None,
                    user_ids: Optional[Union[pd.Series, np.ndarray]] = None) -> Tuple[List[np.ndarray], Optional[List], Optional[List]]:
        """
        Segment data into windows for feature extraction.
        
        Args:
            data: DataFrame or array containing sensor data
            labels: Optional Series or array containing activity labels
            user_ids: Optional Series or array containing user IDs
            
        Returns:
            Tuple of (windowed_data_list, windowed_labels_list, windowed_user_ids_list)
        """
        # Check if data is a DataFrame or a NumPy array
        is_dataframe = isinstance(data, pd.DataFrame)
        
        # If no labels provided, use default windowing
        if labels is None:
            n_samples = len(data)
            n_windows = (n_samples - self.window_size) // self.window_step + 1
            
            if n_windows <= 0:
                return [], None, None
            
            windows = []
            for i in range(0, n_samples - self.window_size + 1, self.window_step):
                if is_dataframe:
                    window = data.iloc[i:i+self.window_size].values
                else:
                    window = data[i:i+self.window_size]
                windows.append(window)
            
            return windows, None, None
        
        # Regular activity-based segmentation with user tracking
        windows = []
        window_labels = []
        window_users = [] if user_ids is not None else None
        
        # Group by user and activity to prevent data leakage
        if user_ids is not None:
            # Group by user and activity
            unique_users = np.unique(user_ids)
            
            for user in unique_users:
                user_mask = (user_ids == user)
                user_data = data[user_mask] if is_dataframe else data[user_mask]
                user_labels = labels[user_mask]
                
                # Group by activity within user
                unique_activities = np.unique(user_labels)
                
                for activity in unique_activities:
                    activity_mask = (user_labels == activity)
                    activity_data = user_data[activity_mask] if is_dataframe else user_data[activity_mask]
                    
                    # Skip if not enough samples for a window
                    if len(activity_data) < self.window_size:
                        continue
                    
                    # Create windows
                    for i in range(0, len(activity_data) - self.window_size + 1, self.window_step):
                        if is_dataframe:
                            window = activity_data.iloc[i:i+self.window_size].values
                        else:
                            window = activity_data[i:i+self.window_size]
                        windows.append(window)
                        window_labels.append(activity)
                        window_users.append(user)
        else:
            # Group by activity only if no user IDs
            unique_activities = np.unique(labels)
            for activity in unique_activities:
                activity_mask = (labels == activity)
                activity_data = data[activity_mask] if is_dataframe else data[activity_mask]
                
                # Skip if not enough samples for a window
                if len(activity_data) < self.window_size:
                    continue
                
                # Create windows
                for i in range(0, len(activity_data) - self.window_size + 1, self.window_step):
                    if is_dataframe:
                        window = activity_data.iloc[i:i+self.window_size].values
                    else:
                        window = activity_data[i:i+self.window_size]
                    windows.append(window)
                    window_labels.append(activity)
        
        return windows, window_labels, window_users
    
    def prepare_data(self, train_file: str = None, val_file: str = None, test_file: str = None) -> Dict[str, Union[np.ndarray, pd.DataFrame]]:
        """
        Prepare data for model training, validation, and testing with strict user-based split.
        
        Args:
            train_file: Path to training data file (optional)
            val_file: Path to validation data file (optional)
            test_file: Path to test data file (optional)
            
        Returns:
            Dictionary containing train/validation/test data and labels
        """
        # Set default file paths if not provided
        if train_file is None:
            train_file = os.path.join(self.config.data_dir, self.config.train_data)
        
        # Initialize result dictionary
        result = {}
        
        # Load training data
        train_data = self.load_data(train_file)
        
        # Identify label column
        label_cols = [col for col in train_data.columns if col.lower() in ['activity', 'label', 'class', 'target']]
        if not label_cols:
            raise ValueError("Could not identify label column in the data")
        
        label_col = label_cols[0]
        
        # Get feature columns (all except label)
        feature_cols = [col for col in train_data.columns if col != label_col]
        
        # Check if user_id column exists for user-based splitting
        has_user_id = 'user_id' in train_data.columns
        user_id_col = 'user_id' if has_user_id else None
        
        # Handle different data loading scenarios
        # Scenario 1: Three separate files (train, val, test)
        if val_file is not None and test_file is not None:
            print("Using separate files for train, validation, and test")
            
            # Load validation and test data
            val_data = self.load_data(val_file)
            test_data = self.load_data(test_file)
            
            # Extract features and labels
            X_train = train_data[feature_cols]
            y_train = train_data[label_col]
            X_val = val_data[feature_cols]
            y_val = val_data[label_col]
            X_test = test_data[feature_cols]
            y_test = test_data[label_col]
            
            # Extract user IDs if available
            user_ids_train = train_data['user_id'] if has_user_id else None
            user_ids_val = val_data['user_id'] if has_user_id and 'user_id' in val_data.columns else None
            user_ids_test = test_data['user_id'] if has_user_id and 'user_id' in test_data.columns else None
            
            # Keep user_id separate from features if it exists
            if has_user_id:
                X_train = X_train.drop(columns=['user_id'])
                if 'user_id' in X_val.columns: X_val = X_val.drop(columns=['user_id'])
                if 'user_id' in X_test.columns: X_test = X_test.drop(columns=['user_id'])
                if 'user_id' in feature_cols: feature_cols.remove('user_id')
        
        # Scenario 2: One file with validation file but no test file
        elif val_file is not None and test_file is None:
            print("Using separate train and validation files, creating test set from validation")
            
            # Load validation data
            val_data = self.load_data(val_file)
            
            # Extract features and labels from train
            X_train = train_data[feature_cols]
            y_train = train_data[label_col]
            
            # Extract validation features and labels
            X_val_full = val_data[feature_cols]
            y_val_full = val_data[label_col]
            
            # Extract user IDs if available
            user_ids_train = train_data['user_id'] if has_user_id else None
            user_ids_val_full = val_data['user_id'] if has_user_id and 'user_id' in val_data.columns else None
            
            # Keep user_id separate from features if it exists
            if has_user_id:
                X_train = X_train.drop(columns=['user_id'])
                if 'user_id' in X_val_full.columns: X_val_full = X_val_full.drop(columns=['user_id'])
                if 'user_id' in feature_cols: feature_cols.remove('user_id')
            
            # Split validation into validation and test
            if has_user_id and user_ids_val_full is not None and self.enforce_user_separation:
                # User-based split for validation/test
                unique_val_users = user_ids_val_full.unique()
                np.random.seed(self.random_state)
                np.random.shuffle(unique_val_users)
                
                # Split users for validation and test
                n_test_users = max(1, len(unique_val_users) // 2)  # Use half for test
                val_users = unique_val_users[:-n_test_users]
                test_users = unique_val_users[-n_test_users:]
                
                # Split data based on users
                val_mask = user_ids_val_full.isin(val_users)
                test_mask = user_ids_val_full.isin(test_users)
                
                X_val = X_val_full[val_mask]
                y_val = y_val_full[val_mask]
                user_ids_val = user_ids_val_full[val_mask]
                
                X_test = X_val_full[test_mask]
                y_test = y_val_full[test_mask]
                user_ids_test = user_ids_val_full[test_mask]
                
                print(f"Split validation file into validation ({len(val_users)} users) and test ({len(test_users)} users)")
            else:
                # Regular stratified split if no user_id
                X_val, X_test, y_val, y_test = train_test_split(
                    X_val_full, y_val_full, 
                    test_size=0.5,  # Split evenly
                    random_state=self.random_state,
                    stratify=y_val_full if len(y_val_full.unique()) > 1 else None
                )
                user_ids_val = None
                user_ids_test = None
                
                print("No user_id found or user separation disabled, using stratified split for validation/test")
                
        # Scenario 3: One train file and one test file, but no validation file
        elif test_file is not None and val_file is None:
            print("Using train file for train+validation and separate test file")
            
            # Load test data
            test_data = self.load_data(test_file)
            
            # Extract features and labels from train
            X_train_full = train_data[feature_cols]
            y_train_full = train_data[label_col]
            
            # Extract user IDs if available
            user_ids_full = train_data['user_id'] if has_user_id else None
            
            # Keep user_id separate from features if it exists
            if has_user_id:
                X_train_full = X_train_full.drop(columns=['user_id'])
                if 'user_id' in feature_cols: feature_cols.remove('user_id')
            
            # Preprocess train data
            X_train_full = self.preprocess_data(X_train_full)
            
            # Extract test features and labels
            X_test = test_data[feature_cols]
            y_test = test_data[label_col]
            user_ids_test = test_data['user_id'] if has_user_id and 'user_id' in test_data.columns else None
            
            # Keep user_id separate from test features if it exists
            if has_user_id and 'user_id' in X_test.columns:
                X_test = X_test.drop(columns=['user_id'])
            
            # Split train into train+validation
            if has_user_id and user_ids_full is not None and self.enforce_user_separation:
                # User-based split for train/validation
                unique_users = user_ids_full.unique()
                np.random.seed(self.random_state)
                np.random.shuffle(unique_users)
                
                # Split users for train and validation
                val_size = getattr(self.config, 'validation_split', 0.15)
                n_val_users = max(1, int(len(unique_users) * val_size))
                train_users = unique_users[:-n_val_users]
                val_users = unique_users[-n_val_users:]
                
                # Split data based on users
                train_mask = user_ids_full.isin(train_users)
                val_mask = user_ids_full.isin(val_users)
                
                X_train = X_train_full[train_mask]
                y_train = y_train_full[train_mask]
                user_ids_train = user_ids_full[train_mask]
                
                X_val = X_train_full[val_mask]
                y_val = y_train_full[val_mask]
                user_ids_val = user_ids_full[val_mask]
                
                print(f"Split training data into train ({len(train_users)} users) and validation ({len(val_users)} users)")
            else:
                # Regular stratified split if no user_id
                print("No user_id found or user separation disabled, using stratified split for train/validation")
                X_train, X_val, y_train, y_val = train_test_split(
                    X_train_full, y_train_full, 
                    test_size=self.config.validation_split,
                    random_state=self.random_state,
                    stratify=y_train_full if len(y_train_full.unique()) > 1 else None
                )
                user_ids_train = None
                user_ids_val = None
                
        # Scenario 4: Only one file available (must split into three sets)
        else:
            print("Using a single file for train/validation/test")
            
            # Extract features and labels
            X_all = train_data[feature_cols]
            y_all = train_data[label_col]
            
            # Extract user IDs if available
            user_ids_all = train_data['user_id'] if has_user_id else None
            
            # Keep user_id separate from features if it exists
            if has_user_id:
                X_all = X_all.drop(columns=['user_id'])
                if 'user_id' in feature_cols: feature_cols.remove('user_id')
            
            # Preprocess all data
            X_all = self.preprocess_data(X_all)
            
            # If we have user IDs and user separation is enforced, do three-way user-based split
            if has_user_id and user_ids_all is not None and self.enforce_user_separation:
                print("Performing user-based three-way split (train/validation/test)")
                
                # Get unique users
                unique_users = user_ids_all.unique()
                
                # Shuffle users
                np.random.seed(self.random_state)
                np.random.shuffle(unique_users)
                
                # Split users for train, validation, and test
                test_size = getattr(self.config, 'test_split', 0.15)
                val_size = getattr(self.config, 'validation_split', 0.15)
                
                n_users = len(unique_users)
                n_test_users = max(1, int(n_users * test_size))
                n_val_users = max(1, int(n_users * val_size))
                n_train_users = n_users - n_test_users - n_val_users
                
                train_users = unique_users[:n_train_users]
                val_users = unique_users[n_train_users:n_train_users+n_val_users]
                test_users = unique_users[n_train_users+n_val_users:]
                
                # Split data based on users
                train_mask = user_ids_all.isin(train_users)
                val_mask = user_ids_all.isin(val_users)
                test_mask = user_ids_all.isin(test_users)
                
                X_train = X_all[train_mask]
                y_train = y_all[train_mask]
                user_ids_train = user_ids_all[train_mask]
                
                X_val = X_all[val_mask]
                y_val = y_all[val_mask]
                user_ids_val = user_ids_all[val_mask]
                
                X_test = X_all[test_mask]
                y_test = y_all[test_mask]
                user_ids_test = user_ids_all[test_mask]
                
                print(f"Train set: {len(X_train)} samples from {len(train_users)} users")
                print(f"Validation set: {len(X_val)} samples from {len(val_users)} users")
                print(f"Test set: {len(X_test)} samples from {len(test_users)} users")
                
                # Verify no overlap between sets
                train_val_overlap = set(train_users).intersection(set(val_users))
                train_test_overlap = set(train_users).intersection(set(test_users))
                val_test_overlap = set(val_users).intersection(set(test_users))
                
                if train_val_overlap or train_test_overlap or val_test_overlap:
                    print("WARNING: User overlap detected between splits!")
                else:
                    print("✓ No user overlap between train, validation, and test sets")
            else:
                # Regular stratified split without user IDs
                print("No user_id found or user separation disabled, using stratified split for train/validation/test")
                
                # First split into train+val and test
                X_trainval, X_test, y_trainval, y_test = train_test_split(
                    X_all, y_all, 
                    test_size=getattr(self.config, 'test_split', 0.15),
                    random_state=self.random_state,
                    stratify=y_all if len(y_all.unique()) > 1 else None
                )
                
                # Then split train+val into train and validation
                test_fraction = getattr(self.config, 'test_split', 0.15)
                val_fraction = getattr(self.config, 'validation_split', 0.15)
                # Adjust validation fraction based on remaining data
                adjusted_val_fraction = val_fraction / (1 - test_fraction)
                
                X_train, X_val, y_train, y_val = train_test_split(
                    X_trainval, y_trainval, 
                    test_size=adjusted_val_fraction,
                    random_state=self.random_state,
                    stratify=y_trainval if len(y_trainval.unique()) > 1 else None
                )
                
                user_ids_train = None
                user_ids_val = None
                user_ids_test = None
        
        # Preprocess data (if not already done)
        if not isinstance(X_train, pd.DataFrame) or not ('_preprocessed' in X_train.columns.tolist()):
            X_train = self.preprocess_data(X_train)
            X_val = self.preprocess_data(X_val)
            X_test = self.preprocess_data(X_test)
        
        # Create windows for feature extraction
        X_train_windows, y_train_windows, user_train_windows = self.segment_data(X_train, y_train, user_ids_train)
        X_val_windows, y_val_windows, user_val_windows = self.segment_data(X_val, y_val, user_ids_val)
        X_test_windows, y_test_windows, user_test_windows = self.segment_data(X_test, y_test, user_ids_test)
        
        # Ensure we have enough data in each set
        if len(X_train_windows) == 0:
            raise ValueError("No windows created for training set. Adjust window size or data split.")
        if len(X_val_windows) == 0:
            raise ValueError("No windows created for validation set. Adjust window size or data split.")
        if len(X_test_windows) == 0:
            raise ValueError("No windows created for test set. Adjust window size or data split.")
            
        print(f"Created {len(X_train_windows)} training windows, {len(X_val_windows)} validation windows, and {len(X_test_windows)} test windows")
            
        # Convert to numpy arrays
        X_train_arr = np.array(X_train_windows)
        y_train_arr = np.array(y_train_windows) if y_train_windows is not None else None
        X_val_arr = np.array(X_val_windows)
        y_val_arr = np.array(y_val_windows) if y_val_windows is not None else None
        X_test_arr = np.array(X_test_windows)
        y_test_arr = np.array(y_test_windows) if y_test_windows is not None else None
        
        # Create result dictionary
        result = {
            'X_train': X_train_arr,
            'y_train': y_train_arr,
            'X_val': X_val_arr,
            'y_val': y_val_arr,
            'X_test': X_test_arr,
            'y_test': y_test_arr,
            'feature_names': feature_cols
        }
        
        # Add user IDs if available
        if has_user_id:
            result['user_ids_train'] = np.array(user_train_windows) if user_train_windows else None
            result['user_ids_val'] = np.array(user_val_windows) if user_val_windows else None
            result['user_ids_test'] = np.array(user_test_windows) if user_test_windows else None
        
        # Add basic data distribution info
        if y_train_arr is not None:
            result['train_class_counts'] = {str(c): np.sum(y_train_arr == c) for c in np.unique(y_train_arr)}
        if y_val_arr is not None:
            result['val_class_counts'] = {str(c): np.sum(y_val_arr == c) for c in np.unique(y_val_arr)}
        if y_test_arr is not None:
            result['test_class_counts'] = {str(c): np.sum(y_test_arr == c) for c in np.unique(y_test_arr)}
            
        # Check if all classes exist in all splits
        if y_train_arr is not None and y_val_arr is not None and y_test_arr is not None:
            train_classes = set(np.unique(y_train_arr))
            val_classes = set(np.unique(y_val_arr))
            test_classes = set(np.unique(y_test_arr))
            
            all_classes = train_classes.union(val_classes, test_classes)
            
            missing_in_train = all_classes - train_classes
            missing_in_val = all_classes - val_classes
            missing_in_test = all_classes - test_classes
            
            if missing_in_train:
                print(f"WARNING: Classes missing in training set: {missing_in_train}")
            if missing_in_val:
                print(f"WARNING: Classes missing in validation set: {missing_in_val}")
            if missing_in_test:
                print(f"WARNING: Classes missing in test set: {missing_in_test}")
        
        return result