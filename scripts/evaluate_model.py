#!/usr/bin/env python
"""
Evaluate a trained HAR-CAIFO model on validation and test data.
"""
import os
import numpy as np
import argparse
import joblib
import yaml
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, recall_score

# Add parent directory to path to allow imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import HAR-CAIFO modules
from config import Config
from data import SensorDataLoader
from features import FeatureExtractor
from models import BaseModel
from evaluation import CAIFOEvaluator
from visualization import HARVisualizer

def evaluate_model(model_path, config_path=None, output_dir=None, verbose=True):
    """
    Evaluate model on validation and test data with detailed diagnostics.
    
    Args:
        model_path: Path to the model or model directory
        config_path: Optional path to configuration file
        output_dir: Optional directory for saving evaluation results
        verbose: Whether to print detailed output
    
    Returns:
        Dictionary with evaluation results
    """
    # Load configuration
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        config = Config.from_dict(config_dict)
    else:
        config = Config()
    
    # Set output directory
    if output_dir:
        config.output_dir = output_dir
    
    # Create output directory if it doesn't exist
    os.makedirs(config.output_dir, exist_ok=True)
    
    if verbose:
        print(f"Loading model from {model_path}...")
    
    # Check if this is a directory containing multiple models
    if os.path.isdir(model_path):
        base_model_path = os.path.join(model_path, "base_model.joblib")
        bayesian_model_path = os.path.join(model_path, "bayesian_model.joblib")
        caifo_model_path = os.path.join(model_path, "caifo_model.joblib")
    else:
        # Single model file
        caifo_model_path = model_path
        base_model_path = None
        bayesian_model_path = None
    
    # Load the models
    models = {}
    
    # Load CAIFO model
    caifo_model = BaseModel(config)
    caifo_model.load(caifo_model_path)
    models['caifo'] = caifo_model
    
    if verbose:
        print("CAIFO model loaded successfully")
    
    # Load base model if available
    if base_model_path and os.path.exists(base_model_path):
        base_model = BaseModel(config)
        base_model.load(base_model_path)
        models['base'] = base_model
        
        if verbose:
            print("Base model loaded successfully")
    
    # Load Bayesian model if available
    if bayesian_model_path and os.path.exists(bayesian_model_path):
        bayesian_model = BaseModel(config)
        bayesian_model.load(bayesian_model_path)
        models['bayesian'] = bayesian_model
        
        if verbose:
            print("Bayesian-optimised model loaded successfully")
    
    # Load feature names and label encoder
    if os.path.isdir(model_path):
        metadata_dir = model_path
    else:
        metadata_dir = os.path.dirname(model_path)
    
    feature_names_path = os.path.join(metadata_dir, "feature_names.joblib")
    label_encoder_path = os.path.join(metadata_dir, "label_encoder.joblib")
    class_mapping_path = os.path.join(metadata_dir, "class_mapping.joblib")
    
    selected_feature_names = joblib.load(feature_names_path)
    label_encoder = joblib.load(label_encoder_path)
    class_mapping = joblib.load(class_mapping_path)
    
    if verbose:
        print(f"Loaded metadata with {len(selected_feature_names)} features")
        print(f"Class mapping: {class_mapping}")
    
    # Load data
    if verbose:
        print("Loading data...")
        
    data_loader = SensorDataLoader(config.data)
    data_dict = data_loader.prepare_data()
    
    # Get validation data
    X_val = data_dict['X_val']
    y_val_orig = data_dict['y_val']
    
    # Get test data if available
    has_test_data = 'X_test' in data_dict and 'y_test' in data_dict
    if has_test_data:
        X_test = data_dict['X_test']
        y_test_orig = data_dict['y_test']
    else:
        if verbose:
            print("No test data found. Evaluation will only use validation data.")
    
    # Transform labels to numeric
    y_val = label_encoder.transform(y_val_orig)
    if has_test_data:
        y_test = label_encoder.transform(y_test_orig)
    
    # Extract features
    if verbose:
        print("Extracting features...")
        
    feature_extractor = FeatureExtractor(config.feature)
    X_val_features, _ = feature_extractor.extract_features(X_val)
    if has_test_data:
        X_test_features, _ = feature_extractor.extract_features(X_test)
    
    # Apply feature selection
    if verbose:
        print("Selecting features...")
        
    X_val_selected, _ = feature_extractor.select_features(X_val_features, y_val, selected_feature_names)
    if has_test_data:
        X_test_selected, _ = feature_extractor.select_features(X_test_features, y_test, selected_feature_names)
    
    # Initialize evaluation components
    evaluator = CAIFOEvaluator(config)
    visualizer = HARVisualizer(config)
    
    # Create validation evaluation directory
    val_evaluation_dir = os.path.join(config.output_dir, "validation_evaluation")
    os.makedirs(val_evaluation_dir, exist_ok=True)
    
    # Create test evaluation directory if needed
    if has_test_data:
        test_evaluation_dir = os.path.join(config.output_dir, "test_evaluation")
        os.makedirs(test_evaluation_dir, exist_ok=True)
    
    # Evaluate models
    results = {}
    
    if verbose:
        print("\n=== VALIDATION SET EVALUATION ===")
    
    # Run appropriate evaluation based on available models
    if len(models) >= 3:
        # Comprehensive comparison of all models on validation set
        val_comparison = evaluator.compare_all_stages(
            models['base'],
            models['bayesian'],
            models['caifo'],
            X_val_selected, y_val
        )
        
        # Generate validation report
        val_report = evaluator.generate_report(
            val_comparison,
            output_dir=val_evaluation_dir
        )
        
        # Visualize validation results
        visualizer.create_evaluation_dashboard(
            val_comparison['base_metrics'],
            val_comparison['bayesian_metrics'],
            val_comparison['caifo_metrics'],
            class_mapping,
            dataset_name="Validation",
            output_dir=val_evaluation_dir
        )
        
        # Print validation summary
        if verbose:
            print("\nValidation Set Performance Summary:")
            print(f"Base Model F1: {val_comparison['base_metrics']['f1_weighted']:.4f}")
            print(f"Bayesian Model F1: {val_comparison['bayesian_metrics']['f1_weighted']:.4f}")
            print(f"CAIFO Model F1: {val_comparison['caifo_metrics']['f1_weighted']:.4f}")
            
            # Print per-class improvements
            print("\nPer-Class Recall Improvements (CAIFO vs Base):")
            for class_name in val_comparison['caifo_per_class']:
                base_recall = val_comparison['base_metrics']['per_class_metrics'][class_name]['recall']
                caifo_recall = val_comparison['caifo_metrics']['per_class_metrics'][class_name]['recall']
                improvement = caifo_recall - base_recall
                
                # Get original class name from mapping if available
                original_name = None
                for key, val in class_mapping.items():
                    if str(val) == class_name or val == class_name:
                        original_name = key
                        break
                
                display_name = f"{class_name} ({original_name})" if original_name else class_name
                print(f"Class {display_name}: {base_recall:.4f} → {caifo_recall:.4f} ({improvement:+.4f})")
        
        # Store validation results
        results['validation'] = val_comparison
        
        # If test data is available, perform test evaluation
        if has_test_data:
            if verbose:
                print("\n=== TEST SET EVALUATION ===")
            
            # Comprehensive comparison of all models on test set
            test_comparison = evaluator.compare_all_stages(
                models['base'],
                models['bayesian'],
                models['caifo'],
                X_test_selected, y_test
            )
            
            # Generate test report
            test_report = evaluator.generate_report(
                test_comparison,
                output_dir=test_evaluation_dir
            )
            
            # Visualize test results
            visualizer.create_evaluation_dashboard(
                test_comparison['base_metrics'],
                test_comparison['bayesian_metrics'],
                test_comparison['caifo_metrics'],
                class_mapping,
                dataset_name="Test",
                output_dir=test_evaluation_dir
            )
            
            # Print test summary
            if verbose:
                print("\nTest Set Performance Summary:")
                print(f"Base Model F1: {test_comparison['base_metrics']['f1_weighted']:.4f}")
                print(f"Bayesian Model F1: {test_comparison['bayesian_metrics']['f1_weighted']:.4f}")
                print(f"CAIFO Model F1: {test_comparison['caifo_metrics']['f1_weighted']:.4f}")
                
                # Print per-class improvements
                print("\nPer-Class Recall Improvements (CAIFO vs Base):")
                for class_name in test_comparison['caifo_per_class']:
                    base_recall = test_comparison['base_metrics']['per_class_metrics'][class_name]['recall']
                    caifo_recall = test_comparison['caifo_metrics']['per_class_metrics'][class_name]['recall']
                    improvement = caifo_recall - base_recall
                    
                    # Get original class name from mapping if available
                    original_name = None
                    for key, val in class_mapping.items():
                        if str(val) == class_name or val == class_name:
                            original_name = key
                            break
                    
                    display_name = f"{class_name} ({original_name})" if original_name else class_name
                    print(f"Class {display_name}: {base_recall:.4f} → {caifo_recall:.4f} ({improvement:+.4f})")
            
            # Store test results
            results['test'] = test_comparison
    else:
        # Just evaluate the CAIFO model on validation set
        val_results = evaluator.evaluate_model(
            models['caifo'], X_val_selected, y_val, selected_feature_names
        )
        
        # Generate validation report
        val_report = evaluator.generate_report(
            val_results,
            output_dir=val_evaluation_dir
        )
        
        # Plot confusion matrix
        evaluator.plot_confusion_matrix(
            val_results['confusion_matrix'],
            val_results['classes'],
            os.path.join(val_evaluation_dir, "caifo_validation_confusion_matrix.png")
        )
        
        # Print validation summary
        if verbose:
            print("\nCAIFO Model Validation Performance:")
            print(f"Accuracy: {val_results['accuracy']:.4f}")
            print(f"F1 (weighted): {val_results['f1_weighted']:.4f}")
            print(f"Recall (weighted): {val_results['recall_weighted']:.4f}")
            print(f"Precision (weighted): {val_results['precision_weighted']:.4f}")
        
        # Store validation results
        results['validation'] = val_results
        
        # If test data is available, perform test evaluation
        if has_test_data:
            if verbose:
                print("\n=== TEST SET EVALUATION ===")
            
            # Evaluate CAIFO model on test set
            test_results = evaluator.evaluate_model(
                models['caifo'], X_test_selected, y_test, selected_feature_names
            )
            
            # Generate test report
            test_report = evaluator.generate_report(
                test_results,
                output_dir=test_evaluation_dir
            )
            
            # Plot confusion matrix
            evaluator.plot_confusion_matrix(
                test_results['confusion_matrix'],
                test_results['classes'],
                os.path.join(test_evaluation_dir, "caifo_test_confusion_matrix.png")
            )
            
            # Print test summary
            if verbose:
                print("\nCAIFO Model Test Performance:")
                print(f"Accuracy: {test_results['accuracy']:.4f}")
                print(f"F1 (weighted): {test_results['f1_weighted']:.4f}")
                print(f"Recall (weighted): {test_results['recall_weighted']:.4f}")
                print(f"Precision (weighted): {test_results['precision_weighted']:.4f}")
            
            # Store test results
            results['test'] = test_results
    
    if verbose:
        print(f"\nEvaluation results saved to {config.output_dir}")
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate HAR-CAIFO model")
    parser.add_argument("--model", type=str, required=True, help="Path to the model or model directory")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--output", type=str, help="Directory to save evaluation results")
    parser.add_argument("--quiet", action="store_true", help="Suppress detailed output")
    args = parser.parse_args()
    
    # Run evaluation
    evaluate_model(args.model, args.config, args.output, not args.quiet)