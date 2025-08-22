"""
Evaluation utilities for CAIFO models.
"""
import numpy as np
import pandas as pd
import os
from typing import Dict, List, Any, Tuple
from sklearn.metrics import (
    f1_score, recall_score, precision_score, confusion_matrix,
    accuracy_score, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

class CAIFOEvaluator:
    """
    Evaluation utilities for CAIFO models.
    """
    
    def __init__(self, config):
        """
        Initialize the evaluator.
        
        Args:
            config: Configuration object
        """
        self.config = config
    
    def evaluate_model(self, model, X: np.ndarray, y: np.ndarray, 
                      feature_names: List[str] = None) -> Dict[str, Any]:
        """
        Comprehensive evaluation of a model.
        
        Args:
            model: Trained model with predict method
            X: Feature matrix
            y: Target labels
            feature_names: Optional list of feature names
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Make predictions
        y_pred = model.predict(X)
        
        # Calculate basic metrics
        accuracy = accuracy_score(y, y_pred)
        classes = np.unique(y)
        
        result = {
            'accuracy': accuracy,
            'f1_weighted': f1_score(y, y_pred, average='weighted', zero_division=0),
            'recall_weighted': recall_score(y, y_pred, average='weighted'),
            'precision_weighted': precision_score(y, y_pred, average='weighted', zero_division=0),
        }
        
        # Per-class metrics
        per_class_recall = recall_score(y, y_pred, average=None, labels=classes)
        per_class_f1 = f1_score(y, y_pred, average=None, labels=classes, zero_division=0)
        per_class_precision = precision_score(y, y_pred, average=None, labels=classes, zero_division=0)
        
        # Create per-class metrics dictionary
        per_class_metrics = {}
        for i, class_name in enumerate(classes):
            per_class_metrics[str(class_name)] = {
                'recall': per_class_recall[i],
                'precision': per_class_precision[i],
                'f1': per_class_f1[i],
                'support': np.sum(y == class_name)
            }
        
        result['per_class_metrics'] = per_class_metrics
        
        # Confusion matrix
        result['confusion_matrix'] = confusion_matrix(y, y_pred, labels=classes)
        result['classes'] = classes
        
        # Classification report (as string)
        result['classification_report'] = classification_report(y, y_pred, labels=classes)
        
        return result
    
    def compare_models(self, base_model, caifo_model, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Compare base model with CAIFO model.
        
        Args:
            base_model: Base model without CAIFO optimization
            caifo_model: Model with CAIFO optimization
            X: Feature matrix
            y: Target labels
            
        Returns:
            Dictionary of comparison metrics
        """
        # Evaluate both models
        base_metrics = self.evaluate_model(base_model, X, y)
        caifo_metrics = self.evaluate_model(caifo_model, X, y)
        
        # Calculate improvements
        improvements = {}
        
        # Overall metrics
        for metric in ['accuracy', 'f1_weighted', 'recall_weighted', 'precision_weighted']:
            improvements[metric] = caifo_metrics[metric] - base_metrics[metric]
        
        # Per-class improvements
        per_class_improvements = {}
        classes = np.unique(y)
        
        for class_name in [str(c) for c in classes]:
            if class_name in base_metrics['per_class_metrics'] and class_name in caifo_metrics['per_class_metrics']:
                base_class = base_metrics['per_class_metrics'][class_name]
                caifo_class = caifo_metrics['per_class_metrics'][class_name]
                
                per_class_improvements[class_name] = {
                    'recall': caifo_class['recall'] - base_class['recall'],
                    'precision': caifo_class['precision'] - base_class['precision'],
                    'f1': caifo_class['f1'] - base_class['f1'],
                    'support': base_class['support']  # Number of samples
                }
        
        # Result dictionary
        result = {
            'base_metrics': base_metrics,
            'caifo_metrics': caifo_metrics,
            'improvements': improvements,
            'per_class_improvements': per_class_improvements
        }
        
        return result
    
    def compare_all_stages(self, base_model, bayesian_model, caifo_model, 
                        X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Compare all three stages of model training: base model, Bayesian optimization, and CAIFO.
        
        Args:
            base_model: Base model without any optimization
            bayesian_model: Model after Bayesian optimization
            caifo_model: Model after CAIFO optimization
            X: Feature matrix
            y: Target labels
            
        Returns:
            Dictionary of comparison metrics
        """
        # Evaluate all models
        base_metrics = self.evaluate_model(base_model, X, y)
        bayesian_metrics = self.evaluate_model(bayesian_model, X, y)
        caifo_metrics = self.evaluate_model(caifo_model, X, y)
        
        # Calculate improvements relative to base model
        bayesian_improvements = {}
        caifo_improvements = {}
        caifo_vs_bayesian = {}
        
        # Overall metrics
        metrics_to_compare = ['accuracy', 'f1_weighted', 'recall_weighted', 'precision_weighted']
        
        for metric in metrics_to_compare:
            bayesian_improvements[metric] = bayesian_metrics[metric] - base_metrics[metric]
            caifo_improvements[metric] = caifo_metrics[metric] - base_metrics[metric]
            caifo_vs_bayesian[metric] = caifo_metrics[metric] - bayesian_metrics[metric]
        
        # Per-class improvements
        bayesian_per_class = {}
        caifo_per_class = {}
        caifo_vs_bayesian_per_class = {}
        
        classes = np.unique(y)
        
        for class_name in [str(c) for c in classes]:
            if (class_name in base_metrics['per_class_metrics'] and 
                class_name in bayesian_metrics['per_class_metrics'] and 
                class_name in caifo_metrics['per_class_metrics']):
                
                base_class = base_metrics['per_class_metrics'][class_name]
                bayesian_class = bayesian_metrics['per_class_metrics'][class_name]
                caifo_class = caifo_metrics['per_class_metrics'][class_name]
                
                bayesian_per_class[class_name] = {
                    'recall': bayesian_class['recall'] - base_class['recall'],
                    'precision': bayesian_class['precision'] - base_class['precision'],
                    'f1': bayesian_class['f1'] - base_class['f1'],
                    'support': base_class['support']
                }
                
                caifo_per_class[class_name] = {
                    'recall': caifo_class['recall'] - base_class['recall'],
                    'precision': caifo_class['precision'] - base_class['precision'],
                    'f1': caifo_class['f1'] - base_class['f1'],
                    'support': base_class['support']
                }
                
                caifo_vs_bayesian_per_class[class_name] = {
                    'recall': caifo_class['recall'] - bayesian_class['recall'],
                    'precision': caifo_class['precision'] - bayesian_class['precision'],
                    'f1': caifo_class['f1'] - bayesian_class['f1'],
                    'support': base_class['support']
                }
        
        # Result dictionary
        result = {
            'base_metrics': base_metrics,
            'bayesian_metrics': bayesian_metrics,
            'caifo_metrics': caifo_metrics,
            'bayesian_improvements': bayesian_improvements,
            'caifo_improvements': caifo_improvements,
            'caifo_vs_bayesian': caifo_vs_bayesian,
            'bayesian_per_class': bayesian_per_class,
            'caifo_per_class': caifo_per_class,
            'caifo_vs_bayesian_per_class': caifo_vs_bayesian_per_class
        }
        
        return result
    
    def generate_report(self, evaluation_results: Dict[str, Any], 
                        output_dir: str = None) -> str:
        """
        Generate evaluation report.
        
        Args:
            evaluation_results: Results from evaluate_model or compare_models
            output_dir: Optional directory to save report and plots
            
        Returns:
            Report as string
        """
        import io
        
        buffer = io.StringIO()
        
        # Header
        buffer.write("# CAIFO Evaluation Report\n\n")
        
        # Overall metrics
        if 'improvements' in evaluation_results:
            # This is a comparison report
            buffer.write("## Model Comparison\n\n")
            
            buffer.write("### Overall Improvements\n\n")
            buffer.write("| Metric | Base Model | CAIFO Model | Improvement |\n")
            buffer.write("|--------|------------|-------------|-------------|\n")
            
            improvements = evaluation_results['improvements']
            base_metrics = evaluation_results['base_metrics']
            caifo_metrics = evaluation_results['caifo_metrics']
            
            for metric in ['accuracy', 'f1_weighted', 'recall_weighted', 'precision_weighted']:
                if metric in improvements:
                    buffer.write(f"| {metric} | {base_metrics[metric]:.4f} | {caifo_metrics[metric]:.4f} | {improvements[metric]:.4f} |\n")
            
            # Per-class improvements
            buffer.write("\n### Per-Class Improvements\n\n")
            buffer.write("| Class | Metric | Base Model | CAIFO Model | Improvement |\n")
            buffer.write("|-------|--------|------------|-------------|-------------|\n")
            
            per_class_improvements = evaluation_results['per_class_improvements']
            
            for class_name, metrics in per_class_improvements.items():
                for metric in ['recall', 'precision', 'f1']:
                    base_value = evaluation_results['base_metrics']['per_class_metrics'][class_name][metric]
                    caifo_value = evaluation_results['caifo_metrics']['per_class_metrics'][class_name][metric]
                    improvement = metrics[metric]
                    
                    buffer.write(f"| {class_name} | {metric} | {base_value:.4f} | {caifo_value:.4f} | {improvement:.4f} |\n")
        
        elif 'caifo_metrics' in evaluation_results:
            # This is a multi-model comparison (base, bayesian, caifo)
            buffer.write("## Multi-Model Comparison\n\n")
            
            buffer.write("### Overall Metrics\n\n")
            buffer.write("| Metric | Base Model | Bayesian Model | CAIFO Model |\n")
            buffer.write("|--------|------------|----------------|-------------|\n")
            
            base_metrics = evaluation_results['base_metrics']
            bayesian_metrics = evaluation_results['bayesian_metrics']
            caifo_metrics = evaluation_results['caifo_metrics']
            
            for metric in ['accuracy', 'f1_weighted', 'recall_weighted', 'precision_weighted']:
                if metric in base_metrics and metric in bayesian_metrics and metric in caifo_metrics:
                    buffer.write(f"| {metric} | {base_metrics[metric]:.4f} | {bayesian_metrics[metric]:.4f} | {caifo_metrics[metric]:.4f} |\n")
            
            # Per-class metrics for CAIFO model
            buffer.write("\n### CAIFO Model Per-Class Metrics\n\n")
            buffer.write("| Class | Recall | Precision | F1 Score | Support |\n")
            buffer.write("|-------|--------|-----------|----------|--------|\n")
            
            if 'per_class_metrics' in caifo_metrics:
                per_class_metrics = caifo_metrics['per_class_metrics']
                
                for class_name, metrics in per_class_metrics.items():
                    buffer.write(f"| {class_name} | {metrics['recall']:.4f} | {metrics['precision']:.4f} | {metrics['f1']:.4f} | {metrics['support']} |\n")
        
        else:
            # This is a single model evaluation
            buffer.write("## Model Evaluation\n\n")
            
            buffer.write("### Overall Metrics\n\n")
            buffer.write("| Metric | Value |\n")
            buffer.write("|--------|-------|\n")
            
            for metric in ['accuracy', 'f1_weighted', 'recall_weighted', 'precision_weighted']:
                if metric in evaluation_results:
                    buffer.write(f"| {metric} | {evaluation_results[metric]:.4f} |\n")
            
            # Per-class metrics
            if 'per_class_metrics' in evaluation_results:
                buffer.write("\n### Per-Class Metrics\n\n")
                buffer.write("| Class | Recall | Precision | F1 Score | Support |\n")
                buffer.write("|-------|--------|-----------|----------|--------|\n")
                
                per_class_metrics = evaluation_results['per_class_metrics']
                
                for class_name, metrics in per_class_metrics.items():
                    buffer.write(f"| {class_name} | {metrics['recall']:.4f} | {metrics['precision']:.4f} | {metrics['f1']:.4f} | {metrics['support']} |\n")
        
        # Classification report
        if 'classification_report' in evaluation_results:
            buffer.write("\n### Classification Report\n\n")
            buffer.write("```\n")
            buffer.write(evaluation_results['classification_report'])
            buffer.write("\n```\n")
        elif 'caifo_metrics' in evaluation_results and 'classification_report' in evaluation_results['caifo_metrics']:
            buffer.write("\n### CAIFO Model Classification Report\n\n")
            buffer.write("```\n")
            buffer.write(evaluation_results['caifo_metrics']['classification_report'])
            buffer.write("\n```\n")
        
        # If output directory is provided, save report
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            
            # Save report
            with open(os.path.join(output_dir, "evaluation_report.md"), 'w') as f:
                f.write(buffer.getvalue())
        
        return buffer.getvalue()
    
    def plot_confusion_matrix(self, cm: np.ndarray, classes: np.ndarray,
                            output_path: str = None) -> plt.Figure:
        """
        Plot confusion matrix.
        
        Args:
            cm: Confusion matrix
            classes: Class labels
            output_path: Optional path to save plot
            
        Returns:
            Matplotlib figure
        """
        plt.figure(figsize=(10, 8))
        
        # Normalize confusion matrix
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_norm = np.nan_to_num(cm_norm)  # Replace NaNs with zeros
        
        # Create heatmap
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', 
                   xticklabels=classes, yticklabels=classes)
        
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title('Normalized Confusion Matrix')
        
        # Save if output path provided
        if output_path:
            plt.savefig(output_path, bbox_inches='tight', dpi=300)
        
        return plt.gcf()