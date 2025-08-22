#!/usr/bin/env python
"""
Train a HAR model with CAIFO optimisation.
"""
import os
import numpy as np
import argparse
import joblib
import yaml
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, recall_score, precision_score

# Add parent directory to path to allow imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import HAR-CAIFO modules
from config import Config
from data import SensorDataLoader
from features import FeatureExtractor, FeatureImportanceCalculator
from caifo.bayesian_opt import BayesianHyperOptimizer
from caifo.feature_opt import FeatureOptimizer
from caifo.class_opt import ClassWeightOptimizer
from caifo.exit_criteria import ExitCriteria
from models import BaseModel
from evaluation import CAIFOEvaluator
from visualization import HARVisualizer
from utils import CAIFOLogger, encode_labels

def run_caifo_optimisation(config, X_train, y_train, X_val, y_val, initial_model, 
                         initial_class_weights, selected_feature_names, class_mapping,
                         label_encoder, logger, max_iterations=None, model_name="CAIFO"):
    """
    Run CAIFO optimisation starting from a given model and initial class weights.
    
    Args:
        config: Configuration object
        X_train, y_train: Training data
        X_val, y_val: Validation data
        initial_model: Model to start optimisation from
        initial_class_weights: Initial class weights
        selected_feature_names: List of selected feature names
        class_mapping: Mapping from class indices to names
        label_encoder: Label encoder for classes
        logger: Logger instance
        max_iterations: Optional maximum iterations (overrides config)
        model_name: Name prefix for the output model
    
    Returns:
        Tuple of (optimised model, history, exit criteria)
    """
    # Initialise CAIFO components
    feature_opt = FeatureOptimizer(config)
    class_weight_opt = ClassWeightOptimizer(config)
    exit_criteria = ExitCriteria(config)
    visualiser = HARVisualizer(config)
    
    # Set maximum iterations
    if max_iterations is None:
        max_iterations = config.caifo.max_iterations
    
    # Initialise tracking variables
    current_model = initial_model
    current_X_train = X_train.copy()
    current_X_val = X_val.copy()
    current_class_weights = initial_class_weights.copy() if initial_class_weights else {}
    
    # Store previous state
    previous_struggling_class = None
    oscillation_count = 0
    max_oscillations = 3
    
    # Iteration history
    history = []
    
    # CAIFO optimisation process with progress bar
    iterations = range(1, max_iterations + 1)
    progress_bar = tqdm(iterations, desc=f"{model_name} Iterations")
    
    for iteration in progress_bar:
        try:
            # Make predictions with current model
            y_val_pred = current_model.predict(current_X_val)
            
            # Identify struggling class
            struggling_class_idx, struggling_recall = feature_opt.identify_struggling_class(
                y_val, y_val_pred, iteration=iteration
            )
            
            # Get class name for display
            struggling_class_name = class_mapping.get(int(struggling_class_idx), struggling_class_idx)
            logger.log_event(f"Target class: {struggling_class_name} (recall: {struggling_recall:.4f})")
            
            # Check for oscillation
            if struggling_class_idx == previous_struggling_class:
                oscillation_count += 1
                if oscillation_count >= max_oscillations:
                    # Get class distribution from predictions
                    classes = np.unique(y_val)
                    class_counts = {cls: np.sum(y_val_pred == cls) for cls in classes}
                    
                    # Find classes with zero or very few predictions
                    zero_pred_classes = [cls for cls in classes if class_counts[cls] < 10]
                    
                    if zero_pred_classes:
                        # Pick a zero-prediction class to focus on
                        struggling_class_idx = str(zero_pred_classes[0])
                        struggling_recall = 0.0  # Force zero-recall handling
                        logger.log_event(f"Breaking oscillation by focusing on zero-prediction class {struggling_class_idx}")
                    else:
                        # Just use next class in sequence
                        current_idx = np.where(classes == int(struggling_class_idx))[0][0]
                        next_idx = (current_idx + 1) % len(classes)
                        struggling_class_idx = str(classes[next_idx])
                        
                        # Get recall for this class
                        class_mask = (y_val == int(struggling_class_idx))
                        pred_mask = (y_val_pred == int(struggling_class_idx))
                        if np.sum(class_mask) > 0:
                            struggling_recall = np.sum(class_mask & pred_mask) / np.sum(class_mask)
                        else:
                            struggling_recall = 0.0
                        
                        logger.log_event(f"Breaking oscillation by rotating to next class {struggling_class_idx}")
                    
                    # Reset oscillation counter
                    oscillation_count = 0
            else:
                # Reset oscillation counter if we switched classes
                oscillation_count = 0
                previous_struggling_class = struggling_class_idx
            
            # Calculate feature importance for the struggling class
            importance_calculator = FeatureImportanceCalculator()
            class_importances = importance_calculator.calculate_per_class_importance(
                current_model, current_X_train, y_train, selected_feature_names
            )
            
            # Convert numeric class indices to original names for visualisation
            named_importances = {}
            for class_idx, importances in class_importances.items():
                class_name = class_mapping.get(int(class_idx), str(class_idx))
                named_importances[class_name] = importances
            
            # Plot feature importance with original class names
            visualiser.plot_feature_importance(
                named_importances,
                output_path=os.path.join(config.output_dir, f"{model_name.lower()}_feature_importance_iter{iteration}.png")
            )
            
            # Special handling for MONOPOLY cases (class taking over predictions)
            if struggling_recall < -5.0:  # Special code for monopoly
                logger.log_event(f"CRITICAL: Class {struggling_class_name} is monopolising predictions. Applying drastic rebalancing.")
                
                # Skip feature weighting for monopoly cases - just rebalance the weights drastically
                weighted_X_train = current_X_train.copy()
                weighted_X_val = current_X_val.copy()
                
                # Use emergency rebalancing
                optimised_class_weights = class_weight_opt.optimize_weights_balanced(
                    current_class_weights, struggling_class_idx, struggling_recall,
                    0.0  # No base f1 score
                )
                
            # Special handling for overpredicted classes
            elif struggling_recall < 0:
                logger.log_event(f"Overpredicted class: {struggling_class_name}. Applying rebalancing strategy.")
                
                # Calculate feature weights with emphasis on reducing importance of overpredicted features
                class_feature_weights = feature_opt.calculate_feature_weights(
                    current_model, current_X_train, y_train, 
                    selected_feature_names, struggling_class_idx
                )
                
                # Apply feature weights - this will reduce weights for overpredicted class features
                weighted_X_train = feature_opt.apply_feature_weights(
                    current_X_train, class_feature_weights, selected_feature_names
                )
                weighted_X_val = feature_opt.apply_feature_weights(
                    current_X_val, class_feature_weights, selected_feature_names
                )
                
                # Optimise class weights - will reduce weight for overpredicted class
                optimised_class_weights = class_weight_opt.optimize_weights_balanced(
                    current_class_weights, struggling_class_idx, struggling_recall,
                    0.0  # No base f1 score
                )
                
            # Special handling for zero-recall classes    
            elif struggling_recall == 0.0:
                logger.log_event(f"Zero recall detected for class {struggling_class_name}. Applying specialised handling.")
                
                # Apply specialised zero-recall boosting - focus on STABLE features
                weighted_X_train = feature_opt.apply_zero_recall_boosting(
                    current_X_train, y_train, struggling_class_idx, selected_feature_names
                )
                weighted_X_val = feature_opt.apply_zero_recall_boosting(
                    current_X_val, y_val, struggling_class_idx, selected_feature_names
                )
                
                # Create balanced class weights for zero-recall classes
                optimised_class_weights = class_weight_opt.optimize_weights_aggressive(
                    current_class_weights, struggling_class_idx, struggling_recall
                )
                
            else:
                # Normal CAIFO flow for classes with some recall
                logger.log_event(f"Using standard approach for class with recall {struggling_recall:.4f}")
                
                # Calculate optimised feature weights
                class_feature_weights = feature_opt.calculate_feature_weights(
                    current_model, current_X_train, y_train, 
                    selected_feature_names, struggling_class_idx
                )
                
                # Apply feature weights
                weighted_X_train = feature_opt.apply_feature_weights(
                    current_X_train, class_feature_weights, selected_feature_names
                )
                weighted_X_val = feature_opt.apply_feature_weights(
                    current_X_val, class_feature_weights, selected_feature_names
                )
                
                # Use balanced class weight optimisation
                optimised_class_weights = class_weight_opt.optimize_weights_balanced(
                    current_class_weights, struggling_class_idx, struggling_recall,
                    0.0  # No base f1 score
                )
            
            # Apply regularisation to model to prevent overfitting to validation
            updated_model = BaseModel(config)
            
            # Train new model with weighted features and optimised class weights
            updated_model.train(weighted_X_train, y_train, optimised_class_weights)
            updated_model.label_encoder = label_encoder
            updated_model.class_mapping = class_mapping
            
            # Evaluate updated model on validation set
            evaluator = CAIFOEvaluator(config)
            updated_results = evaluator.evaluate_model(
                updated_model, weighted_X_val, y_val, selected_feature_names
            )
            
            # Calculate improvement from previous iteration
            prev_f1 = 0.0 if iteration == 1 else history[-1]['f1_weighted']
            improvement = updated_results['f1_weighted'] - prev_f1
            
            # Update exit criteria
            continue_optimisation = exit_criteria.update(
                iteration, 
                updated_results, 
                struggling_class_idx, 
                updated_results['per_class_metrics']
            )
            
            # Log basic metrics
            progress_bar.set_postfix({
                'f1': f"{updated_results['f1_weighted']:.4f}",
                'recall': f"{updated_results['recall_weighted']:.4f}",
                'imp': f"{improvement:.4f}"
            })
            
            # Only update model and parameters if there was improvement
            if improvement >= 0 or iteration == 1:  # Always accept first iteration
                current_model = updated_model
                current_X_train = weighted_X_train
                current_X_val = weighted_X_val
                current_class_weights = optimised_class_weights
                logger.log_event(f"Model updated - improvement: {improvement:.6f}")
            else:
                logger.log_event(f"No improvement. Keeping previous model.")
            
            # Log iteration metrics
            iteration_metrics = {
                'iteration': iteration,
                'f1_weighted': updated_results['f1_weighted'],
                'recall_weighted': updated_results['recall_weighted'],
                'struggling_class': struggling_class_name,
                'struggling_class_recall': updated_results['per_class_metrics'].get(struggling_class_idx, {}).get('recall', 0),
                'improvement': improvement
            }
            
            logger.log_iteration(iteration, iteration_metrics)
            history.append(iteration_metrics)
            
            # Check if we should stop
            if not continue_optimisation:
                logger.log_event(f"Stopping {model_name} optimisation at iteration {iteration}")
                break
            
        except Exception as e:
            logger.log_error(e, f"Error in {model_name} iteration {iteration}")
            continue
    
    # Make sure we have at least one entry in history to avoid errors
    if not history:
        logger.log_event(f"No successful {model_name} iterations, using placeholder history")
        history = [{
            'iteration': 0,
            'f1_weighted': 0.0,
            'recall_weighted': 0.0,
            'struggling_class': 'unknown',
            'struggling_class_recall': 0.0,
            'improvement': 0.0
        }]
    
    # Save model
    model_path = os.path.join(config.model.save_dir, f"{model_name.lower()}_model.joblib")
    current_model.save(model_path)
    logger.log_event(f"{model_name} model saved to {model_path}")
    
    # Plot optimisation progress
    visualiser.plot_caifo_progress(
        history,
        output_path=os.path.join(config.output_dir, f"{model_name.lower()}_progress.png")
    )
    
    return current_model, history, exit_criteria

def train_caifo_model(config_path=None):
    """
    Train a HAR model with CAIFO optimisation, ensuring proper user-based three-way splitting.
    
    Args:
        config_path: Optional path to configuration file
    
    Returns:
        Dictionary with trained models and evaluation results
    """
    # Load configuration
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        config = Config.from_dict(config_dict)
    else:
        config = Config()
    
    # Initialise logger
    logger = CAIFOLogger(config)
    logger.log_event("Training started")
    
    try:
        # Initialise visualiser
        visualiser = HARVisualizer(config)
        
        # Load data with user-based three-way split
        logger.log_event("Loading data with user-based three-way split")
        data_loader = SensorDataLoader(config.data)
        
        # Ensure user separation is enabled in the config
        if not hasattr(config.data, 'enforce_user_separation'):
            logger.log_event("Adding enforce_user_separation=True to config")
            config.data.enforce_user_separation = True
            
        data_dict = data_loader.prepare_data()
        
        # Verify we have all three splits
        X_train, y_train_orig = data_dict['X_train'], data_dict['y_train']
        X_val, y_val_orig = data_dict['X_val'], data_dict['y_val']
        
        # Check if we have test data
        has_test_data = 'X_test' in data_dict and 'y_test' in data_dict
        if has_test_data:
            X_test, y_test_orig = data_dict['X_test'], data_dict['y_test']
            logger.log_event("Test data available for final evaluation")
        else:
            logger.log_event("WARNING: No test data found. Using validation data for evaluation.")
        
        # Convert all labels to numeric for consistency
        logger.log_event("Converting labels to numeric format")
        if has_test_data:
            (y_train, y_val, y_test), label_encoder, class_mapping = encode_labels(y_train_orig, y_val_orig, y_test_orig)
        else:
            (y_train, y_val), label_encoder, class_mapping = encode_labels(y_train_orig, y_val_orig)
        logger.log_event(f"Class mapping: {class_mapping}")
        
        # Check if we have user IDs
        has_user_ids = 'user_ids_train' in data_dict and data_dict['user_ids_train'] is not None
        if has_user_ids:
            logger.log_event("User IDs available - confirming user separation")
            
            # Check for user overlap between sets
            train_users = set(data_dict['user_ids_train'])
            val_users = set(data_dict['user_ids_val'])
            test_users = set(data_dict['user_ids_test']) if has_test_data else set()
            
            # Verify no overlap
            train_val_overlap = train_users.intersection(val_users)
            train_test_overlap = train_users.intersection(test_users)
            val_test_overlap = val_users.intersection(test_users)
            
            if train_val_overlap:
                logger.log_event(f"WARNING: {len(train_val_overlap)} users appear in both train and validation sets!")
            if train_test_overlap:
                logger.log_event(f"WARNING: {len(train_test_overlap)} users appear in both train and test sets!")
            if val_test_overlap:
                logger.log_event(f"WARNING: {len(val_test_overlap)} users appear in both validation and test sets!")
                
            if not (train_val_overlap or train_test_overlap or val_test_overlap):
                logger.log_event("Confirmed: No user overlap between train, validation, and test sets")
        
        # Visualise class distribution
        visualiser.plot_class_distribution(
            y_train, class_mapping,
            os.path.join(config.output_dir, "class_distribution.png")
        )
        
        # Extract features
        logger.log_event("Extracting features")
        feature_extractor = FeatureExtractor(config.feature)
        X_train_features, feature_names = feature_extractor.extract_features(X_train)
        X_val_features, _ = feature_extractor.extract_features(X_val)
        
        if has_test_data:
            X_test_features, _ = feature_extractor.extract_features(X_test)
            logger.log_event(f"Extracted {len(feature_names)} features for all data splits")
        
        # Feature selection
        logger.log_event("Selecting features")
        X_train_selected, selected_feature_names = feature_extractor.select_features(
            X_train_features, y_train, feature_names
        )
        X_val_selected, _ = feature_extractor.select_features(
            X_val_features, y_val, feature_names
        )
        
        if has_test_data:
            X_test_selected, _ = feature_extractor.select_features(
                X_test_features, y_test, feature_names
            )
            logger.log_event(f"Selected {len(selected_feature_names)} features for all data splits")
            
            # Prepare test data for final evaluation only (not for optimisation)
            test_data = {
                'X_test': X_test_selected,
                'y_test': y_test
            }
            if 'user_ids_test' in data_dict:
                test_data['user_ids_test'] = data_dict['user_ids_test']
        else:
            test_data = None
        
        # Initialise CAIFO components
        logger.log_event("Initialising CAIFO components")
        bayes_opt = BayesianHyperOptimizer(config)
        
        # Train Base Model (without any optimisation)
        logger.log_event("Training base model (no optimisation)")
        base_model = BaseModel(config)
        base_model.train(X_train_selected, y_train)
        base_model.label_encoder = label_encoder
        base_model.class_mapping = class_mapping
        
        # Evaluate base model on validation set
        base_evaluator = CAIFOEvaluator(config)
        base_results = base_evaluator.evaluate_model(
            base_model, X_val_selected, y_val, selected_feature_names
        )
        logger.log_event(f"Base model f1_weighted on validation: {base_results['f1_weighted']:.4f}")
        
        # Save base model
        base_model_path = os.path.join(config.model.save_dir, "base_model.joblib")
        os.makedirs(os.path.dirname(base_model_path), exist_ok=True)
        base_model.save(base_model_path)
        logger.log_event(f"Base model saved to {base_model_path}")
        
        # Step 1: Bayesian optimisation using validation data for evaluation
        logger.log_event("Running Bayesian optimisation (validating on validation set)")
        optimised_params = bayes_opt.optimize(
            X_train_selected, y_train, selected_feature_names, 
            metric='f1_weighted',
            # Pass validation data instead of test data
            test_data={
                'X_test': X_val_selected,
                'y_test': y_val
            },
            user_ids=data_dict.get('user_ids_train')
        )
        
        # Extract optimised parameters
        model_params = optimised_params['model_params']
        initial_class_weights = optimised_params['class_weights']
        
        # Train Bayesian-optimised model
        logger.log_event("Training Bayesian-optimised model")
        bayesian_model = BaseModel(config)
        bayesian_model.train(X_train_selected, y_train, initial_class_weights)
        bayesian_model.label_encoder = label_encoder
        bayesian_model.class_mapping = class_mapping
        
        # Evaluate Bayesian-optimised model on validation set
        bayesian_results = base_evaluator.evaluate_model(
            bayesian_model, X_val_selected, y_val, selected_feature_names
        )
        logger.log_event(f"Bayesian-optimised model f1_weighted on validation: {bayesian_results['f1_weighted']:.4f}")
        
        # Save Bayesian model
        bayesian_model_path = os.path.join(config.model.save_dir, "bayesian_model.joblib")
        bayesian_model.save(bayesian_model_path)
        logger.log_event(f"Bayesian model saved to {bayesian_model_path}")
        
        # Step 2A: Run CAIFO on Base Model (No Bayesian optimisation)
        logger.log_event("\n=== Running CAIFO starting from Base Model ===")
        caifo_base_model, caifo_base_history, caifo_base_exit = run_caifo_optimisation(
            config, 
            X_train_selected, y_train, 
            X_val_selected, y_val, 
            base_model, 
            None,  # No initial class weights
            selected_feature_names, 
            class_mapping,
            label_encoder,
            logger,
            model_name="CAIFO_Base"
        )
        
        # Evaluate CAIFO-Base model
        caifo_base_results = base_evaluator.evaluate_model(
            caifo_base_model, X_val_selected, y_val, selected_feature_names
        )
        logger.log_event(f"CAIFO-Base model f1_weighted on validation: {caifo_base_results['f1_weighted']:.4f}")
        
        # Step 2B: Run CAIFO on Bayesian Model
        logger.log_event("\n=== Running CAIFO starting from Bayesian-Optimised Model ===")
        caifo_bayesian_model, caifo_bayesian_history, caifo_bayesian_exit = run_caifo_optimisation(
            config, 
            X_train_selected, y_train, 
            X_val_selected, y_val, 
            bayesian_model, 
            initial_class_weights,
            selected_feature_names, 
            class_mapping,
            label_encoder,
            logger,
            model_name="CAIFO_Bayesian"
        )
        
        # Evaluate CAIFO-Bayesian model
        caifo_bayesian_results = base_evaluator.evaluate_model(
            caifo_bayesian_model, X_val_selected, y_val, selected_feature_names
        )
        logger.log_event(f"CAIFO-Bayesian model f1_weighted on validation: {caifo_bayesian_results['f1_weighted']:.4f}")

        # ====== VALIDATION COMPARISON REPORT AND VISUALISATIONS ======
        logger.log_event("Generating validation comparison report and visualisations")
        
        # Create evaluation directory
        val_evaluation_dir = os.path.join(config.output_dir, "validation_evaluation")
        os.makedirs(val_evaluation_dir, exist_ok=True)
        
        # Compare all models on validation set
        evaluator = CAIFOEvaluator(config)
        
        # 4-way comparison: Base, Bayesian, CAIFO-Base, CAIFO-Bayesian
        # First compare Base vs Bayesian vs CAIFO-Bayesian
        val_comparison_1 = evaluator.compare_all_stages(
            base_model,            # Original base model
            bayesian_model,        # Model after Bayesian optimisation
            caifo_bayesian_model,  # CAIFO model starting from Bayesian
            X_val_selected, y_val
        )
        
        # Then compare Base vs CAIFO-Base vs CAIFO-Bayesian
        val_comparison_2 = evaluator.compare_all_stages(
            base_model,            # Original base model
            caifo_base_model,      # CAIFO model starting from Base
            caifo_bayesian_model,  # CAIFO model starting from Bayesian
            X_val_selected, y_val
        )
        
        # Generate reports
        evaluator.generate_report(
            val_comparison_1,
            output_dir=os.path.join(val_evaluation_dir, "bayesian_path")
        )
        
        evaluator.generate_report(
            val_comparison_2,
            output_dir=os.path.join(val_evaluation_dir, "base_vs_bayesian_path")
        )

        # Generate confusion matrices for all models
        logger.log_event("Generating confusion matrices for all models")

        # For validation data
        evaluator.plot_confusion_matrix(
            base_results['confusion_matrix'],
            base_results['classes'],
            os.path.join(val_evaluation_dir, "base_confusion_matrix.png")
        )

        evaluator.plot_confusion_matrix(
            bayesian_results['confusion_matrix'],
            bayesian_results['classes'],
            os.path.join(val_evaluation_dir, "bayesian_confusion_matrix.png")
        )

        evaluator.plot_confusion_matrix(
            caifo_base_results['confusion_matrix'],
            caifo_base_results['classes'],
            os.path.join(val_evaluation_dir, "caifo_base_confusion_matrix.png")
        )

        evaluator.plot_confusion_matrix(
            caifo_bayesian_results['confusion_matrix'],
            caifo_bayesian_results['classes'],
            os.path.join(val_evaluation_dir, "caifo_bayesian_confusion_matrix.png")
        )
        
        # Create comparison visualisations
        visualiser.plot_model_comparison(
            val_comparison_1['base_metrics'],
            val_comparison_1['bayesian_metrics'],
            val_comparison_1['caifo_metrics'],
            output_path=os.path.join(val_evaluation_dir, "bayesian_path_comparison.png")
        )
        
        visualiser.plot_model_comparison(
            val_comparison_2['base_metrics'],
            val_comparison_2['bayesian_metrics'],  # This is actually CAIFO-Base now
            val_comparison_2['caifo_metrics'],     # This is CAIFO-Bayesian
            output_path=os.path.join(val_evaluation_dir, "base_vs_bayesian_path_comparison.png")
        )
        
        # Create 4-way comparison chart
        plt.figure(figsize=(12, 8))
        models = ['Base', 'Bayesian', 'CAIFO-Base', 'CAIFO-Bayesian']
        f1_scores = [
            base_results['f1_weighted'],
            bayesian_results['f1_weighted'],
            caifo_base_results['f1_weighted'],
            caifo_bayesian_results['f1_weighted']
        ]
        recall_scores = [
            base_results['recall_weighted'],
            bayesian_results['recall_weighted'],
            caifo_base_results['recall_weighted'],
            caifo_bayesian_results['recall_weighted']
        ]
        
        x = np.arange(len(models))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(14, 8))
        rects1 = ax.bar(x - width/2, f1_scores, width, label='F1 Weighted')
        rects2 = ax.bar(x + width/2, recall_scores, width, label='Recall Weighted')
        
        ax.set_title('Comparison of All Model Variations')
        ax.set_ylabel('Score')
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.legend()
        
        # Add value labels
        def add_labels(rects):
            for rect in rects:
                height = rect.get_height()
                ax.annotate(f'{height:.4f}',
                          xy=(rect.get_x() + rect.get_width() / 2, height),
                          xytext=(0, 3),  # 3 points vertical offset
                          textcoords="offset points",
                          ha='center', va='bottom')
        
        add_labels(rects1)
        add_labels(rects2)
        
        plt.tight_layout()
        plt.savefig(os.path.join(val_evaluation_dir, "four_way_comparison.png"), dpi=300)
        plt.close()
        
        # Print validation performance summary
        logger.log_event("\nValidation Set Performance Summary:")
        logger.log_event(f"Base Model F1: {base_results['f1_weighted']:.4f}")
        logger.log_event(f"Bayesian Model F1: {bayesian_results['f1_weighted']:.4f}")
        logger.log_event(f"CAIFO-Base Model F1: {caifo_base_results['f1_weighted']:.4f}")
        logger.log_event(f"CAIFO-Bayesian Model F1: {caifo_bayesian_results['f1_weighted']:.4f}")
        
        # Final evaluation with test data if available
        if has_test_data:
            logger.log_event("\nPerforming final evaluation on test data")
            
            # Create test evaluation directory
            test_evaluation_dir = os.path.join(config.output_dir, "test_evaluation")
            os.makedirs(test_evaluation_dir, exist_ok=True)
            
            # Compare all models on test data
            test_comparison_1 = evaluator.compare_all_stages(
                base_model,            # Original base model
                bayesian_model,        # Model after Bayesian optimisation
                caifo_bayesian_model,  # CAIFO model starting from Bayesian
                X_test_selected, y_test
            )
            
            # Then compare Base vs CAIFO-Base vs CAIFO-Bayesian
            test_comparison_2 = evaluator.compare_all_stages(
                base_model,            # Original base model
                caifo_base_model,      # CAIFO model starting from Base
                caifo_bayesian_model,  # CAIFO model starting from Bayesian
                X_test_selected, y_test
            )
            
            # Generate reports
            evaluator.generate_report(
                test_comparison_1,
                output_dir=os.path.join(test_evaluation_dir, "bayesian_path")
            )
            
            evaluator.generate_report(
                test_comparison_2,
                output_dir=os.path.join(test_evaluation_dir, "base_vs_bayesian_path")
            )

            # Generate confusion matrices for all models on test data
            logger.log_event("Generating confusion matrices for all models on test data")

            evaluator.plot_confusion_matrix(
                test_comparison_1['base_metrics']['confusion_matrix'],
                test_comparison_1['base_metrics']['classes'],
                os.path.join(test_evaluation_dir, "test_base_confusion_matrix.png")
            )

            evaluator.plot_confusion_matrix(
                test_comparison_1['bayesian_metrics']['confusion_matrix'],
                test_comparison_1['bayesian_metrics']['classes'],
                os.path.join(test_evaluation_dir, "test_bayesian_confusion_matrix.png")
            )

            evaluator.plot_confusion_matrix(
                test_comparison_2['bayesian_metrics']['confusion_matrix'],  # CAIFO-Base
                test_comparison_2['bayesian_metrics']['classes'],
                os.path.join(test_evaluation_dir, "test_caifo_base_confusion_matrix.png")
            )

            evaluator.plot_confusion_matrix(
                test_comparison_1['caifo_metrics']['confusion_matrix'],  # CAIFO-Bayesian
                test_comparison_1['caifo_metrics']['classes'],
                os.path.join(test_evaluation_dir, "test_caifo_bayesian_confusion_matrix.png")
            )
            
            # Create comparison visualisations
            visualiser.plot_model_comparison(
                test_comparison_1['base_metrics'],
                test_comparison_1['bayesian_metrics'],
                test_comparison_1['caifo_metrics'],
                output_path=os.path.join(test_evaluation_dir, "bayesian_path_comparison.png")
            )
            
            visualiser.plot_model_comparison(
                test_comparison_2['base_metrics'],
                test_comparison_2['bayesian_metrics'],  # This is actually CAIFO-Base now
                test_comparison_2['caifo_metrics'],     # This is CAIFO-Bayesian
                output_path=os.path.join(test_evaluation_dir, "base_vs_bayesian_path_comparison.png")
            )
            
            # Create 4-way comparison chart for test data
            test_f1_scores = [
                test_comparison_1['base_metrics']['f1_weighted'],
                test_comparison_1['bayesian_metrics']['f1_weighted'],
                test_comparison_2['bayesian_metrics']['f1_weighted'],  # CAIFO-Base
                test_comparison_1['caifo_metrics']['f1_weighted']      # CAIFO-Bayesian
            ]
            
            test_recall_scores = [
                test_comparison_1['base_metrics']['recall_weighted'],
                test_comparison_1['bayesian_metrics']['recall_weighted'],
                test_comparison_2['bayesian_metrics']['recall_weighted'],  # CAIFO-Base
                test_comparison_1['caifo_metrics']['recall_weighted']      # CAIFO-Bayesian
            ]
            
            fig, ax = plt.subplots(figsize=(14, 8))
            rects1 = ax.bar(x - width/2, test_f1_scores, width, label='F1 Weighted')
            rects2 = ax.bar(x + width/2, test_recall_scores, width, label='Recall Weighted')
            
            ax.set_title('Test Set Performance Comparison')
            ax.set_ylabel('Score')
            ax.set_xticks(x)
            ax.set_xticklabels(models)
            ax.legend()
            
            add_labels(rects1)
            add_labels(rects2)
            
            plt.tight_layout()
            plt.savefig(os.path.join(test_evaluation_dir, "four_way_test_comparison.png"), dpi=300)
            plt.close()
            
            # Print test performance summary
            logger.log_event("\nTest Set Performance Summary:")
            logger.log_event(f"Base Model F1: {test_comparison_1['base_metrics']['f1_weighted']:.4f}")
            logger.log_event(f"Bayesian Model F1: {test_comparison_1['bayesian_metrics']['f1_weighted']:.4f}")
            logger.log_event(f"CAIFO-Base Model F1: {test_comparison_2['bayesian_metrics']['f1_weighted']:.4f}")
            logger.log_event(f"CAIFO-Bayesian Model F1: {test_comparison_1['caifo_metrics']['f1_weighted']:.4f}")
            
            # Create result dictionary
            result = {
                'base_model': base_model,
                'bayesian_model': bayesian_model,
                'caifo_base_model': caifo_base_model,
                'caifo_bayesian_model': caifo_bayesian_model,
                'feature_names': selected_feature_names,
                'label_encoder': label_encoder,
                'class_mapping': class_mapping,
                'validation_evaluation': {
                    'base': base_results,
                    'bayesian': bayesian_results,
                    'caifo_base': caifo_base_results,
                    'caifo_bayesian': caifo_bayesian_results
                },
                'test_evaluation': {
                    'comparison_1': test_comparison_1,
                    'comparison_2': test_comparison_2
                },
                'history': {
                    'caifo_base': caifo_base_history,
                    'caifo_bayesian': caifo_bayesian_history
                }
            }
        else:
            # No test data available, just return validation results
            result = {
                'base_model': base_model,
                'bayesian_model': bayesian_model,
                'caifo_base_model': caifo_base_model,
                'caifo_bayesian_model': caifo_bayesian_model,
                'feature_names': selected_feature_names,
                'label_encoder': label_encoder,
                'class_mapping': class_mapping,
                'validation_evaluation': {
                    'base': base_results,
                    'bayesian': bayesian_results,
                    'caifo_base': caifo_base_results,
                    'caifo_bayesian': caifo_bayesian_results
                },
                'history': {
                    'caifo_base': caifo_base_history,
                    'caifo_bayesian': caifo_bayesian_history
                }
            }

        logger.log_event("Training completed successfully")
        return result
    
    except Exception as e:
        logger.log_error(e, "Error during training")
        raise e
    finally:
        # Save history
        logger.save_history()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train HAR model with CAIFO")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    args = parser.parse_args()
    
    # Train model
    start_time = time.time()
    result = train_caifo_model(args.config)
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")