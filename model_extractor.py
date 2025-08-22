#!/usr/bin/env python
"""
Extract parameters from a trained HAR-CAIFO model for C++ implementation.
Optimized for ESP32 with minimal storage requirements.
"""
import joblib
import json
import numpy as np
import os

def extract_compact_model(model, output_dir):
    """Extract a compact representation of the model"""
    # Use only a small subset of trees
    max_trees = 3  # Only use 3 trees for ESP32
    
    trees = []
    n_trees = min(max_trees, len(model.estimators_))
    
    print(f"Extracting {n_trees} trees (compact format)...")
    
    for i in range(n_trees):
        tree = model.estimators_[i]
        tree_ = tree.tree_
        
        nodes = []
        for node_id in range(tree_.node_count):
            if tree_.children_left[node_id] == -1 and tree_.children_right[node_id] == -1:
                # Leaf node - simplified format
                nodes.append({
                    "t": "L",  # Leaf
                    "c": int(np.argmax(tree_.value[node_id]))  # Class prediction
                })
            else:
                # Decision node - simplified format
                nodes.append({
                    "t": "D",  # Decision
                    "f": int(tree_.feature[node_id]),  # Feature index
                    "v": round(float(tree_.threshold[node_id]), 3),  # Threshold with reduced precision
                    "l": int(tree_.children_left[node_id]),  # Left child
                    "r": int(tree_.children_right[node_id])  # Right child
                })
        
        trees.append(nodes)
        
        print(f"Processed tree {i+1}/{n_trees}: {len(nodes)} nodes")
    
    # Create compact model with minimal metadata
    compact_model = {
        "type": "rf",
        "ntrees": n_trees,
        "nclass": int(model.n_classes_),
        "nfeat": int(model.n_features_in_),
        "classes": model.classes_.tolist(),
        "trees": trees
    }
    
    # Save to a single JSON file with minimal whitespace
    with open(os.path.join(output_dir, "model.json"), "w") as f:
        json.dump(compact_model, f, separators=(',', ':'))  # Remove whitespace
    
    # Save model info
    with open(os.path.join(output_dir, "model_info.json"), "w") as f:
        json.dump({
            "model_type": "random_forest",
            "n_trees": n_trees,
            "n_classes": int(model.n_classes_),
            "n_features": int(model.n_features_in_),
            "classes": model.classes_.tolist(),
        }, f, indent=2)

    file_size = os.path.getsize(os.path.join(output_dir, "model.json"))
    print(f"Compact model saved: {file_size} bytes")
    
    return compact_model

def extract_feature_names(model_dict, output_dir):
    """Extract feature names and parameters"""
    if 'label_encoder' in model_dict:
        label_encoder = model_dict['label_encoder']
        classes = label_encoder.classes_.tolist()
        
        # Save class mapping
        class_dict = {
            "classes": classes,
            "n_classes": len(classes)
        }
        
        with open(os.path.join(output_dir, "classes.json"), "w") as f:
            json.dump(class_dict, f, indent=2)
    
    # Check if class_mapping exists
    if 'class_mapping' in model_dict:
        class_mapping = model_dict['class_mapping']
        
        # Save class mapping
        with open(os.path.join(output_dir, "class_mapping.json"), "w") as f:
            json.dump(class_mapping, f, indent=2)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract model parameters for C++ implementation")
    parser.add_argument("--model", type=str, default="models/base_model.joblib", help="Path to the model file")
    parser.add_argument("--output", type=str, default="model_data", help="Output directory for extracted parameters")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    print(f"Loading model from {args.model}...")
    model_dict = joblib.load(args.model)
    
    # Check if it's a dictionary with 'model' key or the model itself
    if isinstance(model_dict, dict) and 'model' in model_dict:
        model = model_dict['model']
        # Extract feature names and class mapping
        extract_feature_names(model_dict, args.output)
    else:
        model = model_dict
    
    # Extract compact model
    compact_model = extract_compact_model(model, args.output)
    
    print(f"Model parameters extracted successfully to {args.output}")