#!/usr/bin/env python3
"""
Example script to evaluate fairness metrics for synthetic datasets.

Usage:
    python evaluate_fairness_example.py \
        --dataset german-credit-data \
        --original-data data/german-credit-data.csv \
        --synthetic-data synthetic_data/gemini_3_flash_preview/20260130_181753.csv \
        --sensitive-attr sex \
        --output-dir fairness_reports
"""

import argparse
import pandas as pd
import numpy as np
import os
from datetime import datetime
from evaluate_fairness import (
    evaluate_fairness_comprehensive,
    print_fairness_report,
    save_fairness_report,
    extract_sensitive_attributes
)
from data_loader import load_german_credit_data_balanced, load_dataset_generic


def load_data_from_csv(csv_path: str, dataset_name: str) -> tuple:
    """
    Load data from CSV file and extract labels.
    
    Args:
        csv_path: Path to CSV file
        dataset_name: Name of dataset
    
    Returns:
        Tuple of (dataframe, labels)
    """
    print(f"Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Normalize column names
    df.columns = [col.lower().strip() for col in df.columns]
    
    # Extract labels
    label_col = None
    for col in ['class', 'class-label', 'class_label', 'credit', 'target', 'label']:
        if col in df.columns:
            label_col = col
            break
    
    if label_col is None:
        raise ValueError(f"Could not find label column in {df.columns.tolist()}")
    
    # Convert labels to binary
    if df[label_col].dtype == 'object' or df[label_col].dtype == 'string':
        labels = (df[label_col].astype(str).str.lower().str.strip() == 'good').astype(int)
    else:
        labels = df[label_col].astype(int).values
    
    # Remove label column from dataframe
    df = df.drop(columns=[label_col])
    
    print(f"  Loaded {len(df)} samples")
    print(f"  Label distribution: {labels.sum()} (1), {len(labels) - labels.sum()} (0)")
    
    return df, labels


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate fairness metrics for synthetic datasets"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["german-credit-data", "adult"],
        help="Dataset name"
    )
    parser.add_argument(
        "--original-data",
        type=str,
        required=True,
        help="Path to original data CSV file"
    )
    parser.add_argument(
        "--synthetic-data",
        type=str,
        required=True,
        help="Path to synthetic data CSV file"
    )
    parser.add_argument(
        "--sensitive-attr",
        type=str,
        default="sex",
        choices=["sex", "age", "race"],
        help="Sensitive attribute to evaluate (default: sex)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="fairness_reports",
        help="Directory to save fairness reports (default: fairness_reports)"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Optional: Path to trained model for predictions"
    )
    parser.add_argument(
        "--use-labels-as-predictions",
        action="store_true",
        help="Use labels as predictions (for evaluation without model)"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*70)
    print("FAIRNESS EVALUATION FOR SYNTHETIC DATA")
    print("="*70)
    print(f"Dataset: {args.dataset}")
    print(f"Sensitive Attribute: {args.sensitive_attr}")
    print(f"Original Data: {args.original_data}")
    print(f"Synthetic Data: {args.synthetic_data}")
    print("="*70)
    
    # Load original data
    print("\n1. Loading Original Data...")
    if args.dataset == "german-credit-data":
        original_sentences, original_labels, original_data, _ = load_german_credit_data_balanced(n_samples=None)
    else:
        original_sentences, original_labels, original_data, _ = load_dataset_generic(args.dataset)
    
    # Load synthetic data
    print("\n2. Loading Synthetic Data...")
    synthetic_data, synthetic_labels = load_data_from_csv(args.synthetic_data, args.dataset)
    
    # Ensure same number of samples (or use all available)
    min_samples = min(len(original_data), len(synthetic_data))
    if len(original_data) != len(synthetic_data):
        print(f"\n⚠ Warning: Different number of samples")
        print(f"  Original: {len(original_data)} samples")
        print(f"  Synthetic: {len(synthetic_data)} samples")
        print(f"  Using first {min_samples} samples from each")
        
        original_data = original_data.iloc[:min_samples].reset_index(drop=True)
        original_labels = original_labels[:min_samples]
        synthetic_data = synthetic_data.iloc[:min_samples].reset_index(drop=True)
        synthetic_labels = synthetic_labels[:min_samples]
    
    # Prepare predictions
    original_predictions = None
    synthetic_predictions = None
    
    if args.use_labels_as_predictions:
        print("\n3. Using labels as predictions...")
        original_predictions = original_labels
        synthetic_predictions = synthetic_labels
    elif args.model_path:
        print("\n3. Loading model and generating predictions...")
        # TODO: Load model and generate predictions
        # For now, use labels as predictions
        print("  ⚠ Model prediction not yet implemented, using labels as predictions")
        original_predictions = original_labels
        synthetic_predictions = synthetic_labels
    else:
        print("\n3. No model provided, using labels as predictions...")
        original_predictions = original_labels
        synthetic_predictions = synthetic_labels
    
    # Prepare features for individual fairness (optional)
    # For now, we'll skip this as it requires feature extraction
    original_features = None
    synthetic_features = None
    
    # Evaluate fairness
    print("\n4. Evaluating Fairness Metrics...")
    results = evaluate_fairness_comprehensive(
        original_data=original_data,
        synthetic_data=synthetic_data,
        original_labels=original_labels,
        synthetic_labels=synthetic_labels,
        original_predictions=original_predictions,
        synthetic_predictions=synthetic_predictions,
        original_features=original_features,
        synthetic_features=synthetic_features,
        dataset_name=args.dataset,
        sensitive_attr_name=args.sensitive_attr
    )
    
    # Print report
    print_fairness_report(results, dataset_name=args.dataset)
    
    # Save report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"fairness_report_{args.dataset}_{args.sensitive_attr}_{timestamp}.json"
    output_path = os.path.join(args.output_dir, output_filename)
    
    save_fairness_report(results, output_path, dataset_name=args.dataset)
    
    print("\n" + "="*70)
    print("EVALUATION COMPLETE")
    print("="*70)
    print(f"Report saved to: {output_path}")


if __name__ == "__main__":
    main()
