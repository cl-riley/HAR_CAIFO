"""
Simplified configuration management for HAR-CAIFO.
"""
import os
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional

@dataclass
class DataConfig:
    """Configuration for data loading and processing."""
    data_dir: str = "data/"
    train_data: str = "train.csv"
    test_data: str = "test.csv"
    validation_data: str = ""  # Empty string means there's no dedicated validation file
    validation_split: float = 0.15  # 15% of data used for validation if no validation file
    test_split: float = 0.15  # 15% of data used for testing if no test file
    window_size: int = 128
    window_step: int = 64
    random_state: int = 42
    enforce_user_separation: bool = True

@dataclass
class FeatureConfig:
    """Configuration for feature engineering."""
    time_domain_features: bool = True
    frequency_domain_features: bool = True
    statistical_features: bool = True
    feature_selection_method: str = "permutation"
    n_features_to_select: int = 50

@dataclass
class CAIFOConfig:
    """Configuration for the CAIFO algorithm."""
    # Bayesian Optimization
    n_bayesian_iterations: int = 20
    bayesian_init_points: int = 5
    
    # CAIFO Core
    max_iterations: int = 15
    min_improvement_threshold: float = 0.01
    feature_boost_factor: float = 2.0
    class_weight_range: Tuple[float, float] = (0.1, 10.0)
    
    # Exit criteria
    patience: int = 3
    min_delta: float = 0.001

@dataclass
class ModelConfig:
    """Configuration for model selection and training."""
    model_type: str = "xgboost"  # Options: "random_forest", "gradient_boosting", "xgboost"
    save_dir: str = "models/"
    
    # Initialize model_params based on model_type
    model_params: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize default model parameters if not provided."""
        if not self.model_params:
            if self.model_type == "random_forest":
                self.model_params = {
                    "n_estimators": 100,
                    "max_depth": None,
                    "random_state": 42
                }
            elif self.model_type == "gradient_boosting":
                self.model_params = {
                    "n_estimators": 100,
                    "learning_rate": 0.1,
                    "max_depth": 3,
                    "random_state": 42
                }
            elif self.model_type == "xgboost":
                self.model_params = {
                    "n_estimators": 100,
                    "max_depth": 6,
                    "learning_rate": 0.1,
                    "random_state": 42
                }
            else:
                # Default to RandomForest
                self.model_type = "random_forest"
                self.model_params = {
                    "n_estimators": 100,
                    "max_depth": None,
                    "random_state": 42
                }

@dataclass
class Config:
    """Main configuration class for HAR-CAIFO."""
    data: DataConfig = field(default_factory=DataConfig)
    feature: FeatureConfig = field(default_factory=FeatureConfig)
    caifo: CAIFOConfig = field(default_factory=CAIFOConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    
    # General settings
    output_dir: str = "output/"
    verbose: bool = True
    
    def __post_init__(self):
        """Create necessary directories."""
        os.makedirs(self.data.data_dir, exist_ok=True)
        os.makedirs(self.model.save_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for saving."""
        return {
            "data": self.data.__dict__,
            "feature": self.feature.__dict__,
            "caifo": self.caifo.__dict__,
            "model": {
                "model_type": self.model.model_type,
                "model_params": self.model.model_params,
                "save_dir": self.model.save_dir
            },
            "output_dir": self.output_dir,
            "verbose": self.verbose
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """Create config from dictionary."""
        data_config = DataConfig(**config_dict.get("data", {}))
        feature_config = FeatureConfig(**config_dict.get("feature", {}))
        caifo_config = CAIFOConfig(**config_dict.get("caifo", {}))
        
        model_dict = config_dict.get("model", {})
        model_config = ModelConfig(
            model_type=model_dict.get("model_type", "xgboost"),
            model_params=model_dict.get("model_params", {}),
            save_dir=model_dict.get("save_dir", "models/")
        )
        
        return cls(
            data=data_config,
            feature=feature_config,
            caifo=caifo_config,
            model=model_config,
            output_dir=config_dict.get("output_dir", "output/"),
            verbose=config_dict.get("verbose", True)
        )
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'Config':
        """Load configuration from YAML file."""
        import yaml
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)
    
    def to_yaml(self, yaml_path: str) -> None:
        """Save configuration to YAML file."""
        import yaml
        with open(yaml_path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)