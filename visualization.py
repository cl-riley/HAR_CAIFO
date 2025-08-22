"""
Visualisation utilities for HAR-CAIFO.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple
import os

class HARVisualizer:
    """
    Visualise HAR data and model results.
    """
    
    def __init__(self, config):
        """
        Initialise the visualiser.
        
        Args:
            config: Configuration object
        """
        self.config = config
        
        # Set default style
        sns.set_style("whitegrid")
        
        # Use a modern, readable colour palette
        self.palette = sns.color_palette("viridis", 18)
        
        # Configure matplotlib
        plt.rcParams.update({
            'figure.figsize': (12, 8),
            'axes.labelsize': 12,
            'axes.titlesize': 14,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10
        })
    
    def plot_sensor_data(self, data: pd.DataFrame, window_size: int = None,
                       output_path: str = None) -> plt.Figure:
        """
        Plot sensor data with highlighted window.
        
        Args:
            data: DataFrame containing sensor data
            window_size: Optional window size to highlight
            output_path: Optional path to save plot
            
        Returns:
            Matplotlib figure
        """
        if window_size is None:
            window_size = self.config.data.window_size
        
        plt.figure(figsize=(15, 10))
        
        # Identify sensor columns
        acc_cols = [col for col in data.columns if 'acc' in col.lower()]
        gyro_cols = [col for col in data.columns if 'gyro' in col.lower()]
        
        # Plot accelerometer data
        plt.subplot(2, 1, 1)
        for i, col in enumerate(acc_cols):
            plt.plot(data.index, data[col], label=col, color=self.palette[i % len(self.palette)])
        
        plt.title('Accelerometer Data', fontweight='bold')
        plt.xlabel('Sample')
        plt.ylabel('Acceleration')
        plt.legend()
        
        # Highlight a window if window_size is provided
        if window_size > 0:
            start = len(data) // 3  # Example window starting point
            end = start + window_size
            if end <= len(data):
                plt.axvspan(start, end, alpha=0.2, color='red', label='Window')
        
        # Plot gyroscope data if available
        if gyro_cols:
            plt.subplot(2, 1, 2)
            for i, col in enumerate(gyro_cols):
                plt.plot(data.index, data[col], label=col, color=self.palette[i % len(self.palette)])
            
            plt.title('Gyroscope Data', fontweight='bold')
            plt.xlabel('Sample')
            plt.ylabel('Angular Velocity')
            plt.legend()
            
            # Highlight the same window
            if window_size > 0:
                start = len(data) // 3
                end = start + window_size
                if end <= len(data):
                    plt.axvspan(start, end, alpha=0.2, color='red', label='Window')
        
        plt.tight_layout()
        
        # Save if output path provided
        if output_path:
            plt.savefig(output_path, bbox_inches='tight', dpi=300)
        
        return plt.gcf()
    
    def plot_feature_importance(self, importances: Dict[str, Dict[str, float]],
                              top_n: int = 15, output_path: str = None) -> plt.Figure:
        """
        Plot feature importance for each class.
        
        Args:
            importances: Dictionary of per-class feature importances
            top_n: Number of top features to show
            output_path: Optional path to save plot
            
        Returns:
            Matplotlib figure
        """
        n_classes = len(importances)
        n_cols = min(2, n_classes)
        n_rows = (n_classes + n_cols - 1) // n_cols
        
        plt.figure(figsize=(15, 5 * n_rows))
        
        for i, (class_name, class_importances) in enumerate(importances.items()):
            plt.subplot(n_rows, n_cols, i + 1)
            
            # Get top N features
            top_features = sorted(
                class_importances.items(),
                key=lambda x: x[1],
                reverse=True
            )[:top_n]
            
            # Convert to dataframe for plotting
            feature_names = [item[0] for item in top_features]
            feature_values = [item[1] for item in top_features]
            
            # Create color gradient based on importance
            colors = [self.palette[min(int(val * 9), 8)] for val in feature_values]
            
            # Plot horizontal bar chart
            bars = plt.barh(range(len(feature_names)), feature_values, align='center', color=colors)
            plt.yticks(range(len(feature_names)), feature_names)
            
            plt.title(f'Top Features for Class: {class_name}', fontweight='bold')
            plt.xlabel('Importance')
            plt.ylabel('Feature')
            
            # Add importance values to the end of each bar
            for bar, value in zip(bars, feature_values):
                plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                        f'{value:.3f}', va='center')
            
            # Invert y-axis to have most important at the top
            plt.gca().invert_yaxis()
            
            plt.tight_layout()
        
        # Save if output path provided
        if output_path:
            plt.savefig(output_path, bbox_inches='tight', dpi=300)
        
        return plt.gcf()
    
    def plot_caifo_progress(self, history: List[Dict[str, Any]],
                          output_path: str = None) -> plt.Figure:
        """
        Plot CAIFO optimisation progress.
        
        Args:
            history: List of dictionaries containing metrics at each iteration
            output_path: Optional path to save plot
            
        Returns:
            Matplotlib figure
        """
        plt.figure(figsize=(15, 10))
        
        # Extract metrics
        iterations = [h.get('iteration', i) for i, h in enumerate(history)]
        f1_weighted = [h.get('f1_weighted', 0) for h in history]
        recall_weighted = [h.get('recall_weighted', 0) for h in history]
        
        # Extract struggling class info if available
        struggling_classes = [h.get('struggling_class', None) for h in history]
        
        # Plot overall metrics
        plt.subplot(2, 1, 1)
        plt.plot(iterations, f1_weighted, 'b-o', label='F1 Weighted', linewidth=2)
        plt.plot(iterations, recall_weighted, 'g-^', label='Recall Weighted', linewidth=2)
        
        plt.title('CAIFO Optimisation Progress - Overall Metrics', fontweight='bold')
        plt.xlabel('Iteration')
        plt.ylabel('Score')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Plot struggling classes
        plt.subplot(2, 1, 2)
        
        # Create a DataFrame for better plotting with categorical data
        if struggling_classes and all(sc is not None for sc in struggling_classes):
            unique_classes = sorted(set(struggling_classes))
            class_colors = {cls: self.palette[i % len(self.palette)] for i, cls in enumerate(unique_classes)}
            
            # Create a colorful timeline of which class was addressed in each iteration
            for i, cls in enumerate(struggling_classes):
                plt.axvspan(i-0.4, i+0.4, color=class_colors[cls], alpha=0.3)
            
            # Add text labels for classes
            for i, cls in enumerate(struggling_classes):
                if i % 2 == 0:  # Add labels to every other iteration to avoid crowding
                    plt.text(i, 0.5, cls, ha='center', va='center', fontsize=10)
            
            plt.title('Classes Addressed by CAIFO in Each Iteration', fontweight='bold')
            plt.xlabel('Iteration')
            plt.yticks([])  # Hide y-axis ticks
            
            # Add legend for class colors
            from matplotlib.patches import Patch
            legend_elements = [Patch(facecolor=color, alpha=0.3, label=cls) 
                             for cls, color in class_colors.items()]
            plt.legend(handles=legend_elements, title="Classes", loc='center')
        
        plt.tight_layout()
        
        # Save if output path provided
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        
        return plt.gcf()
    
    def plot_class_distribution(self, y: np.ndarray, class_mapping: Dict = None,
                              output_path: str = None) -> plt.Figure:
        """
        Plot class distribution.
        
        Args:
            y: Array of class labels
            class_mapping: Optional mapping from numeric indices to class names
            output_path: Optional path to save plot
            
        Returns:
            Matplotlib figure
        """
        plt.figure(figsize=(12, 6))
        
        # Count occurrences of each class
        classes, counts = np.unique(y, return_counts=True)
        
        # Convert class indices to names if mapping provided
        if class_mapping:
            class_names = [class_mapping.get(cls, str(cls)) for cls in classes]
        else:
            class_names = [str(cls) for cls in classes]
        
        # Calculate percentages
        percentages = 100 * counts / len(y)
        
        # Create DataFrame for easier plotting
        df = pd.DataFrame({
            'Class': class_names,
            'Count': counts,
            'Percentage': percentages
        })
        
        # Sort by count (descending)
        df = df.sort_values('Count', ascending=False)
        
        # Plot as bar chart
        ax = sns.barplot(x='Class', y='Count', data=df, palette=self.palette)
        
        # Add percentage labels on top of bars
        for i, p in enumerate(ax.patches):
            ax.annotate(
                f'{df.iloc[i]["Percentage"]:.1f}%', 
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='bottom',
                fontweight='bold'
            )
        
        plt.title('Class Distribution', fontweight='bold')
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        # Save if output path provided
        if output_path:
            plt.savefig(output_path, bbox_inches='tight', dpi=300)
        
        return plt.gcf()
    
    def plot_model_comparison(self, base_results: Dict, bayesian_results: Dict,
                            caifo_results: Dict, metric: str = 'f1_weighted',
                            output_path: str = None) -> plt.Figure:
        """
        Plot comparison of models across all three stages.
        
        Args:
            base_results: Results from base model
            bayesian_results: Results from Bayesian-optimised model
            caifo_results: Results from CAIFO model
            metric: Metric to compare ('f1_weighted', 'recall_weighted', etc.)
            output_path: Optional path to save plot
            
        Returns:
            Matplotlib figure
        """
        plt.figure(figsize=(14, 10))
        
        # Extract metrics for overall comparison
        models = ['Base Model', 'Bayesian Optimised', 'CAIFO']
        overall_values = [
            base_results.get(metric, 0),
            bayesian_results.get(metric, 0),
            caifo_results.get(metric, 0)
        ]
        
        # Plot overall comparison
        plt.subplot(2, 1, 1)
        bars = plt.bar(models, overall_values, color=self.palette[:3])
        
        # Add value labels on top of bars
        for bar, value in zip(bars, overall_values):
            plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                   f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
        
        plt.title(f'Model Comparison - Overall {metric.replace("_", " ").title()}', fontweight='bold')
        plt.ylabel(metric.replace('_', ' ').title())
        plt.ylim(top=max(overall_values) * 1.1)  # Add some space for labels
        
        # Extract per-class metrics
        plt.subplot(2, 1, 2)
        
        # Get per-class metrics from each model
        if all(m.get('per_class_metrics') for m in [base_results, bayesian_results, caifo_results]):
            # Get common classes across all models
            base_classes = set(base_results['per_class_metrics'].keys())
            bayesian_classes = set(bayesian_results['per_class_metrics'].keys())
            caifo_classes = set(caifo_results['per_class_metrics'].keys())
            
            common_classes = base_classes.intersection(bayesian_classes, caifo_classes)
            
            # Prepare data for plotting
            class_data = []
            for cls in common_classes:
                base_recall = base_results['per_class_metrics'][cls].get('recall', 0)
                bayesian_recall = bayesian_results['per_class_metrics'][cls].get('recall', 0)
                caifo_recall = caifo_results['per_class_metrics'][cls].get('recall', 0)
                
                class_data.append({
                    'Class': cls,
                    'Base': base_recall,
                    'Bayesian': bayesian_recall,
                    'CAIFO': caifo_recall
                })
            
            # Convert to DataFrame
            df = pd.DataFrame(class_data)
            
            # Calculate class improvement
            df['Improvement'] = df['CAIFO'] - df['Base']
            
            # Sort by improvement (descending)
            df = df.sort_values('Improvement', ascending=False)
            
            # Plot per-class recall comparison
            x = np.arange(len(df))
            width = 0.25
            
            plt.bar(x - width, df['Base'], width, label='Base Model', color=self.palette[0])
            plt.bar(x, df['Bayesian'], width, label='Bayesian Model', color=self.palette[1])
            plt.bar(x + width, df['CAIFO'], width, label='CAIFO Model', color=self.palette[2])
            
            plt.xlabel('Class')
            plt.ylabel('Recall')
            plt.title('Per-Class Recall Comparison', fontweight='bold')
            plt.xticks(x, df['Class'])
            plt.legend()
            
            # Add improvement annotation
            for i, row in df.iterrows():
                plt.annotate(
                    f"{row['Improvement']:+.2f}",
                    xy=(i + width, row['CAIFO'] + 0.01),
                    ha='center',
                    va='bottom',
                    fontsize=8,
                    color='green' if row['Improvement'] > 0 else 'red'
                )
        
        plt.tight_layout()
        
        # Save if output path provided
        if output_path:
            plt.savefig(output_path, bbox_inches='tight', dpi=300)
        
        return plt.gcf()
    
    def plot_per_class_improvement(self, base_results: Dict, caifo_results: Dict, 
                                 class_mapping: Dict = None, metric: str = 'recall',
                                 output_path: str = None) -> plt.Figure:
        """
        Plot per-class improvement between base and CAIFO models.
        
        Args:
            base_results: Results from base model
            caifo_results: Results from CAIFO model
            class_mapping: Optional mapping from class indices to names
            metric: Metric to compare ('recall', 'f1', 'precision')
            output_path: Optional path to save plot
            
        Returns:
            Matplotlib figure
        """
        plt.figure(figsize=(14, 7))
        
        # Get per-class metrics from both models
        base_metrics = base_results.get('per_class_metrics', {})
        caifo_metrics = caifo_results.get('per_class_metrics', {})
        
        # Find common classes
        common_classes = set(base_metrics.keys()).intersection(set(caifo_metrics.keys()))
        
        # Calculate improvement for each class
        improvements = {}
        for cls in common_classes:
            base_value = base_metrics[cls].get(metric, 0)
            caifo_value = caifo_metrics[cls].get(metric, 0)
            improvement = caifo_value - base_value
            
            # Get original class name from mapping if available
            if class_mapping and int(cls) in class_mapping:
                display_name = class_mapping[int(cls)]
            else:
                display_name = cls
                
            improvements[display_name] = improvement
        
        # Sort by improvement (descending)
        sorted_improvements = sorted(improvements.items(), key=lambda x: x[1], reverse=True)
        
        # Extract data for plotting
        classes = [item[0] for item in sorted_improvements]
        values = [item[1] for item in sorted_improvements]
        
        # Plot as bar chart
        bars = plt.bar(classes, values, color=['g' if v >= 0 else 'r' for v in values])
        
        # Add data labels
        for bar in bars:
            height = bar.get_height()
            if height >= 0:
                va = 'bottom'
                y_pos = height + 0.01
            else:
                va = 'top'
                y_pos = height - 0.01
                
            plt.text(
                bar.get_x() + bar.get_width()/2, 
                y_pos,
                f'{height:.3f}',
                ha='center',
                va=va,
                fontweight='bold',
                color='g' if height >= 0 else 'r'
            )
        
        # Add labels and title
        plt.title(f'Per-Class {metric.title()} Improvement (CAIFO vs Base)', fontweight='bold')
        plt.xlabel('Class')
        plt.ylabel(f'{metric.title()} Improvement')
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        plt.grid(axis='y', alpha=0.3)
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        
        # Save if output path provided
        if output_path:
            plt.savefig(output_path, bbox_inches='tight', dpi=300)
            
        return plt.gcf()
    
    def create_evaluation_dashboard(self, base_results: Dict, bayesian_results: Dict, 
                                  caifo_results: Dict, class_mapping: Dict = None,
                                  dataset_name: str = "Dataset", output_dir: str = None) -> None:
        """
        Create a comprehensive evaluation dashboard with multiple visualisations.
        
        Args:
            base_results: Results from base model
            bayesian_results: Results from Bayesian-optimised model
            caifo_results: Results from CAIFO model
            class_mapping: Optional mapping from class indices to names
            dataset_name: Name of the dataset (e.g., "Validation", "Test")
            output_dir: Directory to save visualisations
        """
        if output_dir is None:
            output_dir = os.path.join(self.config.output_dir, f"{dataset_name.lower()}_evaluation")
            
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Overall Model Comparison
        self.plot_model_comparison(
            base_results, bayesian_results, caifo_results,
            output_path=os.path.join(output_dir, "model_comparison.png")
        )
        
        # 2. Per-Class Improvement
        self.plot_per_class_improvement(
            base_results, caifo_results, class_mapping, metric='recall',
            output_path=os.path.join(output_dir, "per_class_recall_improvement.png")
        )
        
        self.plot_per_class_improvement(
            base_results, caifo_results, class_mapping, metric='f1',
            output_path=os.path.join(output_dir, "per_class_f1_improvement.png")
        )
        
        # 3. Create a detailed metrics table as HTML
        metrics_html = self._create_metrics_table_html(
            base_results, bayesian_results, caifo_results, class_mapping
        )
        
        with open(os.path.join(output_dir, "metrics_table.html"), "w") as f:
            f.write(metrics_html)
        
        print(f"Evaluation dashboard for {dataset_name} created in {output_dir}")
    
    def _create_metrics_table_html(self, base_results: Dict, bayesian_results: Dict,
                                caifo_results: Dict, class_mapping: Dict = None) -> str:
        """
        Create an HTML table with detailed metrics for all models.
        
        Args:
            base_results: Results from base model
            bayesian_results: Results from Bayesian-optimised model
            caifo_results: Results from CAIFO model
            class_mapping: Optional mapping from class indices to names
            
        Returns:
            HTML string with metrics table
        """
        # CSS styles for the table
        css = """
        <style>
            table {
                border-collapse: collapse;
                width: 100%;
                margin-bottom: 20px;
                font-family: Arial, sans-serif;
            }
            th, td {
                border: 1px solid #ddd;
                padding: 8px;
                text-align: center;
            }
            th {
                background-color: #4CAF50;
                color: white;
            }
            tr:nth-child(even) {
                background-color: #f2f2f2;
            }
            .improvement-positive {
                color: green;
                font-weight: bold;
            }
            .improvement-negative {
                color: red;
                font-weight: bold;
            }
            .section-header {
                background-color: #333;
                color: white;
                font-weight: bold;
            }
            .header {
                text-align: center;
                font-family: Arial, sans-serif;
                margin-bottom: 20px;
            }
        </style>
        """
        
        # Start of HTML content
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>HAR-CAIFO Evaluation Results</title>
            {css}
        </head>
        <body>
            <div class="header">
                <h1>HAR-CAIFO Evaluation Results</h1>
                <p>Comparison of Base, Bayesian, and CAIFO models</p>
            </div>
        """
        
        # Overall metrics table
        html += """
        <h2>Overall Metrics</h2>
        <table>
            <tr>
                <th>Metric</th>
                <th>Base Model</th>
                <th>Bayesian Model</th>
                <th>CAIFO Model</th>
                <th>Improvement (CAIFO vs Base)</th>
            </tr>
        """
        
        # Add rows for overall metrics
        metrics = ['accuracy', 'f1_weighted', 'recall_weighted', 'precision_weighted']
        for metric in metrics:
            base_value = base_results.get(metric, 0)
            bayesian_value = bayesian_results.get(metric, 0)
            caifo_value = caifo_results.get(metric, 0)
            improvement = caifo_value - base_value
            
            improvement_class = "improvement-positive" if improvement >= 0 else "improvement-negative"
            
            html += f"""
            <tr>
                <td>{metric.replace('_', ' ').title()}</td>
                <td>{base_value:.4f}</td>
                <td>{bayesian_value:.4f}</td>
                <td>{caifo_value:.4f}</td>
                <td class="{improvement_class}">{improvement:+.4f}</td>
            </tr>
            """
        
        html += "</table>"
        
        # Per-class metrics table
        html += """
        <h2>Per-Class Metrics</h2>
        <table>
            <tr>
                <th rowspan="2">Class</th>
                <th colspan="3">Recall</th>
                <th colspan="3">F1 Score</th>
                <th rowspan="2">Support</th>
            </tr>
            <tr>
                <th>Base</th>
                <th>Bayesian</th>
                <th>CAIFO</th>
                <th>Base</th>
                <th>Bayesian</th>
                <th>CAIFO</th>
            </tr>
        """
        
        # Get per-class metrics
        base_class_metrics = base_results.get('per_class_metrics', {})
        bayesian_class_metrics = bayesian_results.get('per_class_metrics', {})
        caifo_class_metrics = caifo_results.get('per_class_metrics', {})
        
        # Find all unique classes
        all_classes = set()
        all_classes.update(base_class_metrics.keys())
        all_classes.update(bayesian_class_metrics.keys())
        all_classes.update(caifo_class_metrics.keys())
        
        # Sort classes numerically if possible
        try:
            sorted_classes = sorted(all_classes, key=lambda x: int(x))
        except:
            sorted_classes = sorted(all_classes)
        
        # Add rows for each class
        for cls in sorted_classes:
            # Get metrics for this class
            base_metrics = base_class_metrics.get(cls, {})
            bayesian_metrics = bayesian_class_metrics.get(cls, {})
            caifo_metrics = caifo_class_metrics.get(cls, {})
            
            # Get support (number of samples)
            support = max(
                base_metrics.get('support', 0),
                bayesian_metrics.get('support', 0),
                caifo_metrics.get('support', 0)
            )
            
            # Get display name from mapping if available
            if class_mapping and int(cls) in class_mapping:
                display_name = f"{cls} ({class_mapping[int(cls)]})"
            else:
                display_name = cls
            
            # Add row for this class
            html += f"""
            <tr>
                <td>{display_name}</td>
                <td>{base_metrics.get('recall', 0):.4f}</td>
                <td>{bayesian_metrics.get('recall', 0):.4f}</td>
                <td>{caifo_metrics.get('recall', 0):.4f}</td>
                <td>{base_metrics.get('f1', 0):.4f}</td>
                <td>{bayesian_metrics.get('f1', 0):.4f}</td>
                <td>{caifo_metrics.get('f1', 0):.4f}</td>
                <td>{support}</td>
            </tr>
            """
        
        html += """
        </table>
        </body>
        </html>
        """
        
        return html