"""
Comprehensive Fairness Evaluation for Synthetic Datasets

This module implements various individual fairness and group fairness metrics
to evaluate synthetic datasets against original datasets.

Based on fairness literature and best practices for evaluating synthetic data.

Individual Fairness Metrics:
- Individual Fairness Score (similar individuals get similar predictions)
- Consistency Score
- Individual Discrimination Score

Group Fairness Metrics:
- Demographic Parity (Statistical Parity)
- Equalized Odds
- Equal Opportunity
- Disparate Impact
- Calibration
- Accuracy Parity
- Treatment Equality
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import json
import os
from datetime import datetime


def extract_sensitive_attributes(data: pd.DataFrame, dataset_name: str) -> Dict[str, np.ndarray]:
    """
    Extract sensitive attributes from dataset.
    
    Args:
        data: DataFrame with data
        dataset_name: Name of dataset
    
    Returns:
        Dictionary mapping attribute names to arrays
    """
    sensitive_attrs = {}
    
    if dataset_name == "german-credit-data":
        # Normalize column names
        data.columns = [col.lower().strip() for col in data.columns]
        
        if 'age' in data.columns:
            sensitive_attrs['age'] = data['age'].values
        if 'sex' in data.columns:
            # Normalize sex values
            sex_values = data['sex'].astype(str).str.lower().str.strip()
            sensitive_attrs['sex'] = (sex_values == 'male').astype(int)  # 1 = male, 0 = female
    elif dataset_name == "adult":
        data.columns = [col.lower().strip() for col in data.columns]
        
        if 'sex' in data.columns:
            sex_values = data['sex'].astype(str).str.lower().str.strip()
            sensitive_attrs['sex'] = (sex_values == 'male').astype(int)
        if 'race' in data.columns:
            # Encode race (simplified: 1 = white, 0 = non-white)
            race_values = data['race'].astype(str).str.lower().str.strip()
            sensitive_attrs['race'] = (race_values == 'white').astype(int)
    
    return sensitive_attrs


def demographic_parity(y_pred: np.ndarray, sensitive_attr: np.ndarray, 
                      threshold: float = 0.5) -> Dict[str, float]:
    """
    Calculate Demographic Parity (Statistical Parity).
    
    Demographic Parity: P(Y=1 | A=a) = P(Y=1 | A=b) for all groups a, b
    
    Args:
        y_pred: Predicted labels or probabilities
        sensitive_attr: Binary sensitive attribute (0 or 1)
        threshold: Threshold for binary classification (if y_pred is probability)
    
    Returns:
        Dictionary with metrics
    """
    # Convert to binary if needed
    if y_pred.dtype == float and np.any((y_pred > 1) | (y_pred < 0)):
        y_pred_binary = (y_pred >= threshold).astype(int)
    else:
        y_pred_binary = y_pred.astype(int)
    
    # Calculate positive rate for each group
    group_0_mask = sensitive_attr == 0
    group_1_mask = sensitive_attr == 1
    
    if np.sum(group_0_mask) == 0 or np.sum(group_1_mask) == 0:
        return {
            'demographic_parity': np.nan,
            'positive_rate_group_0': np.nan,
            'positive_rate_group_1': np.nan,
            'parity_difference': np.nan
        }
    
    positive_rate_0 = np.mean(y_pred_binary[group_0_mask])
    positive_rate_1 = np.mean(y_pred_binary[group_1_mask])
    
    # Demographic parity difference
    parity_difference = abs(positive_rate_1 - positive_rate_0)
    
    # Demographic parity ratio (closer to 1 is better)
    if positive_rate_0 > 0:
        parity_ratio = positive_rate_1 / positive_rate_0
    else:
        parity_ratio = np.nan if positive_rate_1 == 0 else np.inf
    
    return {
        'demographic_parity': 1.0 - parity_difference,  # Higher is better (1.0 = perfect parity)
        'positive_rate_group_0': positive_rate_0,
        'positive_rate_group_1': positive_rate_1,
        'parity_difference': parity_difference,
        'parity_ratio': parity_ratio
    }


def equalized_odds(y_true: np.ndarray, y_pred: np.ndarray, sensitive_attr: np.ndarray,
                   threshold: float = 0.5) -> Dict[str, float]:
    """
    Calculate Equalized Odds.
    
    Equalized Odds: P(Y=1 | Y_true=y, A=a) = P(Y=1 | Y_true=y, A=b) for all y, a, b
    
    Args:
        y_true: True labels
        y_pred: Predicted labels or probabilities
        sensitive_attr: Binary sensitive attribute
        threshold: Threshold for binary classification
    
    Returns:
        Dictionary with metrics
    """
    # Convert to binary if needed
    if y_pred.dtype == float and np.any((y_pred > 1) | (y_pred < 0)):
        y_pred_binary = (y_pred >= threshold).astype(int)
    else:
        y_pred_binary = y_pred.astype(int)
    
    y_true = y_true.astype(int)
    
    group_0_mask = sensitive_attr == 0
    group_1_mask = sensitive_attr == 1
    
    if np.sum(group_0_mask) == 0 or np.sum(group_1_mask) == 0:
        return {
            'equalized_odds': np.nan,
            'tpr_group_0': np.nan,
            'tpr_group_1': np.nan,
            'fpr_group_0': np.nan,
            'fpr_group_1': np.nan,
            'tpr_difference': np.nan,
            'fpr_difference': np.nan
        }
    
    # Calculate TPR and FPR for each group
    def calculate_tpr_fpr(y_true_group, y_pred_group):
        tn, fp, fn, tp = confusion_matrix(y_true_group, y_pred_group, labels=[0, 1]).ravel()
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        return tpr, fpr
    
    tpr_0, fpr_0 = calculate_tpr_fpr(y_true[group_0_mask], y_pred_binary[group_0_mask])
    tpr_1, fpr_1 = calculate_tpr_fpr(y_true[group_1_mask], y_pred_binary[group_1_mask])
    
    tpr_difference = abs(tpr_1 - tpr_0)
    fpr_difference = abs(fpr_1 - fpr_0)
    
    # Equalized odds score (higher is better, 1.0 = perfect)
    equalized_odds_score = 1.0 - max(tpr_difference, fpr_difference)
    
    return {
        'equalized_odds': equalized_odds_score,
        'tpr_group_0': tpr_0,
        'tpr_group_1': tpr_1,
        'fpr_group_0': fpr_0,
        'fpr_group_1': fpr_1,
        'tpr_difference': tpr_difference,
        'fpr_difference': fpr_difference
    }


def equal_opportunity(y_true: np.ndarray, y_pred: np.ndarray, sensitive_attr: np.ndarray,
                     threshold: float = 0.5) -> Dict[str, float]:
    """
    Calculate Equal Opportunity.
    
    Equal Opportunity: P(Y=1 | Y_true=1, A=a) = P(Y=1 | Y_true=1, A=b)
    (Same TPR for all groups)
    
    Args:
        y_true: True labels
        y_pred: Predicted labels or probabilities
        sensitive_attr: Binary sensitive attribute
        threshold: Threshold for binary classification
    
    Returns:
        Dictionary with metrics
    """
    # Convert to binary if needed
    if y_pred.dtype == float and np.any((y_pred > 1) | (y_pred < 0)):
        y_pred_binary = (y_pred >= threshold).astype(int)
    else:
        y_pred_binary = y_pred.astype(int)
    
    y_true = y_true.astype(int)
    
    group_0_mask = sensitive_attr == 0
    group_1_mask = sensitive_attr == 1
    
    if np.sum(group_0_mask) == 0 or np.sum(group_1_mask) == 0:
        return {
            'equal_opportunity': np.nan,
            'tpr_group_0': np.nan,
            'tpr_group_1': np.nan,
            'tpr_difference': np.nan
        }
    
    # Calculate TPR for each group
    def calculate_tpr(y_true_group, y_pred_group):
        tn, fp, fn, tp = confusion_matrix(y_true_group, y_pred_group, labels=[0, 1]).ravel()
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        return tpr
    
    tpr_0 = calculate_tpr(y_true[group_0_mask], y_pred_binary[group_0_mask])
    tpr_1 = calculate_tpr(y_true[group_1_mask], y_pred_binary[group_1_mask])
    
    tpr_difference = abs(tpr_1 - tpr_0)
    
    # Equal opportunity score (higher is better, 1.0 = perfect)
    equal_opportunity_score = 1.0 - tpr_difference
    
    return {
        'equal_opportunity': equal_opportunity_score,
        'tpr_group_0': tpr_0,
        'tpr_group_1': tpr_1,
        'tpr_difference': tpr_difference
    }


def disparate_impact(y_pred: np.ndarray, sensitive_attr: np.ndarray,
                    threshold: float = 0.5, threshold_value: float = 0.8) -> Dict[str, float]:
    """
    Calculate Disparate Impact.
    
    Disparate Impact: min(P(Y=1|A=0), P(Y=1|A=1)) / max(P(Y=1|A=0), P(Y=1|A=1))
    Rule of thumb: ratio should be >= 0.8 (80% rule)
    
    Args:
        y_pred: Predicted labels or probabilities
        sensitive_attr: Binary sensitive attribute
        threshold: Threshold for binary classification
        threshold_value: Minimum acceptable ratio (default 0.8)
    
    Returns:
        Dictionary with metrics
    """
    # Convert to binary if needed
    if y_pred.dtype == float and np.any((y_pred > 1) | (y_pred < 0)):
        y_pred_binary = (y_pred >= threshold).astype(int)
    else:
        y_pred_binary = y_pred.astype(int)
    
    group_0_mask = sensitive_attr == 0
    group_1_mask = sensitive_attr == 1
    
    if np.sum(group_0_mask) == 0 or np.sum(group_1_mask) == 0:
        return {
            'disparate_impact': np.nan,
            'disparate_impact_ratio': np.nan,
            'meets_threshold': False
        }
    
    positive_rate_0 = np.mean(y_pred_binary[group_0_mask])
    positive_rate_1 = np.mean(y_pred_binary[group_1_mask])
    
    # Disparate impact ratio
    if positive_rate_0 == 0 and positive_rate_1 == 0:
        ratio = 1.0
    elif positive_rate_0 == 0 or positive_rate_1 == 0:
        ratio = 0.0
    else:
        ratio = min(positive_rate_0, positive_rate_1) / max(positive_rate_0, positive_rate_1)
    
    meets_threshold = ratio >= threshold_value
    
    return {
        'disparate_impact': ratio,
        'disparate_impact_ratio': ratio,
        'positive_rate_group_0': positive_rate_0,
        'positive_rate_group_1': positive_rate_1,
        'meets_threshold': meets_threshold,
        'threshold_value': threshold_value
    }


def calibration(y_true: np.ndarray, y_pred_proba: np.ndarray, sensitive_attr: np.ndarray,
                n_bins: int = 10) -> Dict[str, float]:
    """
    Calculate Calibration (Calibration by Group).
    
    Calibration: P(Y_true=1 | Y_pred=p, A=a) = p for all p, a
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        sensitive_attr: Binary sensitive attribute
        n_bins: Number of bins for calibration
    
    Returns:
        Dictionary with metrics
    """
    y_true = y_true.astype(int)
    
    group_0_mask = sensitive_attr == 0
    group_1_mask = sensitive_attr == 1
    
    def calculate_calibration_error(y_true_group, y_pred_group, n_bins):
        if len(y_true_group) == 0:
            return np.nan, np.nan
        
        # Bin predictions
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0.0  # Expected Calibration Error
        mce = 0.0  # Maximum Calibration Error
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_pred_group > bin_lower) & (y_pred_group <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = y_true_group[in_bin].mean()
                avg_confidence_in_bin = y_pred_group[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
                mce = max(mce, np.abs(avg_confidence_in_bin - accuracy_in_bin))
        
        return ece, mce
    
    ece_0, mce_0 = calculate_calibration_error(
        y_true[group_0_mask], y_pred_proba[group_0_mask], n_bins
    )
    ece_1, mce_1 = calculate_calibration_error(
        y_true[group_1_mask], y_pred_proba[group_1_mask], n_bins
    )
    
    # Calibration difference between groups
    if not (np.isnan(ece_0) or np.isnan(ece_1)):
        calibration_difference = abs(ece_1 - ece_0)
    else:
        calibration_difference = np.nan
    
    return {
        'calibration_ece_group_0': ece_0,
        'calibration_ece_group_1': ece_1,
        'calibration_mce_group_0': mce_0,
        'calibration_mce_group_1': mce_1,
        'calibration_difference': calibration_difference
    }


def accuracy_parity(y_true: np.ndarray, y_pred: np.ndarray, sensitive_attr: np.ndarray,
                   threshold: float = 0.5) -> Dict[str, float]:
    """
    Calculate Accuracy Parity.
    
    Accuracy Parity: P(Y_pred=Y_true | A=a) = P(Y_pred=Y_true | A=b)
    
    Args:
        y_true: True labels
        y_pred: Predicted labels or probabilities
        sensitive_attr: Binary sensitive attribute
        threshold: Threshold for binary classification
    
    Returns:
        Dictionary with metrics
    """
    # Convert to binary if needed
    if y_pred.dtype == float and np.any((y_pred > 1) | (y_pred < 0)):
        y_pred_binary = (y_pred >= threshold).astype(int)
    else:
        y_pred_binary = y_pred.astype(int)
    
    y_true = y_true.astype(int)
    
    group_0_mask = sensitive_attr == 0
    group_1_mask = sensitive_attr == 1
    
    if np.sum(group_0_mask) == 0 or np.sum(group_1_mask) == 0:
        return {
            'accuracy_parity': np.nan,
            'accuracy_group_0': np.nan,
            'accuracy_group_1': np.nan,
            'accuracy_difference': np.nan
        }
    
    accuracy_0 = np.mean(y_pred_binary[group_0_mask] == y_true[group_0_mask])
    accuracy_1 = np.mean(y_pred_binary[group_1_mask] == y_true[group_1_mask])
    
    accuracy_difference = abs(accuracy_1 - accuracy_0)
    
    # Accuracy parity score (higher is better, 1.0 = perfect)
    accuracy_parity_score = 1.0 - accuracy_difference
    
    return {
        'accuracy_parity': accuracy_parity_score,
        'accuracy_group_0': accuracy_0,
        'accuracy_group_1': accuracy_1,
        'accuracy_difference': accuracy_difference
    }


def individual_fairness_score(features: np.ndarray, y_pred: np.ndarray,
                              k: int = 5, metric: str = 'cosine') -> Dict[str, float]:
    """
    Calculate Individual Fairness Score.
    
    Individual Fairness: Similar individuals should receive similar predictions.
    
    Args:
        features: Feature matrix [n_samples, n_features]
        y_pred: Predicted labels or probabilities
        k: Number of nearest neighbors
        metric: Distance metric ('cosine', 'euclidean', etc.)
    
    Returns:
        Dictionary with metrics
    """
    if len(features) < k + 1:
        return {
            'individual_fairness': np.nan,
            'consistency_score': np.nan,
            'mean_prediction_variance': np.nan
        }
    
    # Find k nearest neighbors for each sample
    nbrs = NearestNeighbors(n_neighbors=k+1, metric=metric).fit(features)
    distances, indices = nbrs.kneighbors(features)
    
    # Exclude self (first neighbor is always self)
    neighbor_indices = indices[:, 1:]  # [n_samples, k]
    neighbor_distances = distances[:, 1:]
    
    # Calculate prediction variance for each sample's neighbors
    prediction_variances = []
    consistency_scores = []
    
    for i in range(len(features)):
        neighbor_preds = y_pred[neighbor_indices[i]]
        prediction_variance = np.var(neighbor_preds)
        prediction_variances.append(prediction_variance)
        
        # Consistency: how similar are predictions for similar individuals
        # Lower variance = higher consistency
        if prediction_variance > 0:
            consistency = 1.0 / (1.0 + prediction_variance)
        else:
            consistency = 1.0
        consistency_scores.append(consistency)
    
    mean_prediction_variance = np.mean(prediction_variances)
    mean_consistency = np.mean(consistency_scores)
    
    # Individual fairness score (higher is better, 1.0 = perfect)
    individual_fairness_score = mean_consistency
    
    return {
        'individual_fairness': individual_fairness_score,
        'consistency_score': mean_consistency,
        'mean_prediction_variance': mean_prediction_variance
    }


def treatment_equality(y_true: np.ndarray, y_pred: np.ndarray, sensitive_attr: np.ndarray,
                      threshold: float = 0.5) -> Dict[str, float]:
    """
    Calculate Treatment Equality.
    
    Treatment Equality: FNR / FPR should be equal across groups
    
    Args:
        y_true: True labels
        y_pred: Predicted labels or probabilities
        sensitive_attr: Binary sensitive attribute
        threshold: Threshold for binary classification
    
    Returns:
        Dictionary with metrics
    """
    # Convert to binary if needed
    if y_pred.dtype == float and np.any((y_pred > 1) | (y_pred < 0)):
        y_pred_binary = (y_pred >= threshold).astype(int)
    else:
        y_pred_binary = y_pred.astype(int)
    
    y_true = y_true.astype(int)
    
    group_0_mask = sensitive_attr == 0
    group_1_mask = sensitive_attr == 1
    
    if np.sum(group_0_mask) == 0 or np.sum(group_1_mask) == 0:
        return {
            'treatment_equality': np.nan,
            'fnr_group_0': np.nan,
            'fnr_group_1': np.nan,
            'fpr_group_0': np.nan,
            'fpr_group_1': np.nan,
            'fnr_fpr_ratio_group_0': np.nan,
            'fnr_fpr_ratio_group_1': np.nan
        }
    
    def calculate_fnr_fpr(y_true_group, y_pred_group):
        tn, fp, fn, tp = confusion_matrix(y_true_group, y_pred_group, labels=[0, 1]).ravel()
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        return fnr, fpr
    
    fnr_0, fpr_0 = calculate_fnr_fpr(y_true[group_0_mask], y_pred_binary[group_0_mask])
    fnr_1, fpr_1 = calculate_fnr_fpr(y_true[group_1_mask], y_pred_binary[group_1_mask])
    
    # Calculate FNR/FPR ratio for each group
    fnr_fpr_ratio_0 = fnr_0 / fpr_0 if fpr_0 > 0 else np.inf
    fnr_fpr_ratio_1 = fnr_1 / fpr_1 if fpr_1 > 0 else np.inf
    
    # Treatment equality difference
    if not (np.isinf(fnr_fpr_ratio_0) or np.isinf(fnr_fpr_ratio_1)):
        treatment_difference = abs(fnr_fpr_ratio_1 - fnr_fpr_ratio_0)
    else:
        treatment_difference = np.nan
    
    # Treatment equality score (higher is better, 1.0 = perfect)
    if not np.isnan(treatment_difference):
        treatment_equality_score = 1.0 / (1.0 + treatment_difference)
    else:
        treatment_equality_score = np.nan
    
    return {
        'treatment_equality': treatment_equality_score,
        'fnr_group_0': fnr_0,
        'fnr_group_1': fnr_1,
        'fpr_group_0': fpr_0,
        'fpr_group_1': fpr_1,
        'fnr_fpr_ratio_group_0': fnr_fpr_ratio_0,
        'fnr_fpr_ratio_group_1': fnr_fpr_ratio_1,
        'treatment_difference': treatment_difference
    }


def evaluate_fairness_comprehensive(
    original_data: pd.DataFrame,
    synthetic_data: pd.DataFrame,
    original_labels: np.ndarray,
    synthetic_labels: np.ndarray,
    original_predictions: Optional[np.ndarray] = None,
    synthetic_predictions: Optional[np.ndarray] = None,
    original_features: Optional[np.ndarray] = None,
    synthetic_features: Optional[np.ndarray] = None,
    dataset_name: str = "german-credit-data",
    sensitive_attr_name: str = "sex"
) -> Dict[str, Dict]:
    """
    Comprehensive fairness evaluation comparing original and synthetic datasets.
    
    Args:
        original_data: Original dataset DataFrame
        synthetic_data: Synthetic dataset DataFrame
        original_labels: True labels for original data
        synthetic_labels: True labels for synthetic data
        original_predictions: Model predictions for original data (optional)
        synthetic_predictions: Model predictions for synthetic data (optional)
        original_features: Feature vectors for original data (for individual fairness)
        synthetic_features: Feature vectors for synthetic data (for individual fairness)
        dataset_name: Name of dataset
        sensitive_attr_name: Name of sensitive attribute to evaluate
    
    Returns:
        Dictionary with all fairness metrics
    """
    results = {
        'original': {},
        'synthetic': {},
        'comparison': {}
    }
    
    # Extract sensitive attributes
    original_sensitive = extract_sensitive_attributes(original_data, dataset_name)
    synthetic_sensitive = extract_sensitive_attributes(synthetic_data, dataset_name)
    
    if sensitive_attr_name not in original_sensitive or sensitive_attr_name not in synthetic_sensitive:
        print(f"Warning: Sensitive attribute '{sensitive_attr_name}' not found in data")
        return results
    
    original_attr = original_sensitive[sensitive_attr_name]
    synthetic_attr = synthetic_sensitive[sensitive_attr_name]
    
    # Use labels as predictions if predictions not provided
    if original_predictions is None:
        original_predictions = original_labels
    if synthetic_predictions is None:
        synthetic_predictions = synthetic_labels
    
    # Group Fairness Metrics
    print("\n" + "="*70)
    print("GROUP FAIRNESS METRICS")
    print("="*70)
    
    # Original data metrics
    print("\nOriginal Data:")
    print("  Calculating demographic parity...")
    results['original']['demographic_parity'] = demographic_parity(
        original_predictions, original_attr
    )
    print("  Calculating equalized odds...")
    results['original']['equalized_odds'] = equalized_odds(
        original_labels, original_predictions, original_attr
    )
    print("  Calculating equal opportunity...")
    results['original']['equal_opportunity'] = equal_opportunity(
        original_labels, original_predictions, original_attr
    )
    print("  Calculating disparate impact...")
    results['original']['disparate_impact'] = disparate_impact(
        original_predictions, original_attr
    )
    print("  Calculating accuracy parity...")
    results['original']['accuracy_parity'] = accuracy_parity(
        original_labels, original_predictions, original_attr
    )
    print("  Calculating treatment equality...")
    results['original']['treatment_equality'] = treatment_equality(
        original_labels, original_predictions, original_attr
    )
    
    # Synthetic data metrics
    print("\nSynthetic Data:")
    print("  Calculating demographic parity...")
    results['synthetic']['demographic_parity'] = demographic_parity(
        synthetic_predictions, synthetic_attr
    )
    print("  Calculating equalized odds...")
    results['synthetic']['equalized_odds'] = equalized_odds(
        synthetic_labels, synthetic_predictions, synthetic_attr
    )
    print("  Calculating equal opportunity...")
    results['synthetic']['equal_opportunity'] = equal_opportunity(
        synthetic_labels, synthetic_predictions, synthetic_attr
    )
    print("  Calculating disparate impact...")
    results['synthetic']['disparate_impact'] = disparate_impact(
        synthetic_predictions, synthetic_attr
    )
    print("  Calculating accuracy parity...")
    results['synthetic']['accuracy_parity'] = accuracy_parity(
        synthetic_labels, synthetic_predictions, synthetic_attr
    )
    print("  Calculating treatment equality...")
    results['synthetic']['treatment_equality'] = treatment_equality(
        synthetic_labels, synthetic_predictions, synthetic_attr
    )
    
    # Calibration (requires probability predictions)
    if original_predictions is not None and synthetic_predictions is not None:
        if original_predictions.dtype == float and np.all((original_predictions >= 0) & (original_predictions <= 1)):
            results['original']['calibration'] = calibration(
                original_labels, original_predictions, original_attr
            )
        if synthetic_predictions.dtype == float and np.all((synthetic_predictions >= 0) & (synthetic_predictions <= 1)):
            results['synthetic']['calibration'] = calibration(
                synthetic_labels, synthetic_predictions, synthetic_attr
            )
    
    # Individual Fairness Metrics (if features provided)
    if original_features is not None and synthetic_features is not None:
        print("\n" + "="*70)
        print("INDIVIDUAL FAIRNESS METRICS")
        print("="*70)
        
        print("\nOriginal Data:")
        print("  Calculating individual fairness...")
        results['original']['individual_fairness'] = individual_fairness_score(
            original_features, original_predictions
        )
        
        print("\nSynthetic Data:")
        print("  Calculating individual fairness...")
        results['synthetic']['individual_fairness'] = individual_fairness_score(
            synthetic_features, synthetic_predictions
        )
    
    # Comparison metrics
    print("\n" + "="*70)
    print("COMPARISON METRICS")
    print("="*70)
    
    def compare_metrics(metric_name, original_metric, synthetic_metric):
        """Compare a metric between original and synthetic"""
        if isinstance(original_metric, dict) and isinstance(synthetic_metric, dict):
            comparison = {}
            for key in original_metric.keys():
                if key in synthetic_metric:
                    orig_val = original_metric[key]
                    synth_val = synthetic_metric[key]
                    if not (np.isnan(orig_val) or np.isnan(synth_val)):
                        comparison[key] = {
                            'original': orig_val,
                            'synthetic': synth_val,
                            'difference': abs(synth_val - orig_val),
                            'relative_change': (synth_val - orig_val) / orig_val if orig_val != 0 else np.nan
                        }
            return comparison
        return None
    
    for metric_name in ['demographic_parity', 'equalized_odds', 'equal_opportunity', 
                        'disparate_impact', 'accuracy_parity', 'treatment_equality']:
        if metric_name in results['original'] and metric_name in results['synthetic']:
            comparison = compare_metrics(
                metric_name,
                results['original'][metric_name],
                results['synthetic'][metric_name]
            )
            if comparison:
                results['comparison'][metric_name] = comparison
    
    return results


def print_fairness_report(results: Dict[str, Dict], dataset_name: str = "german-credit-data"):
    """
    Print a comprehensive fairness report.
    
    Args:
        results: Results dictionary from evaluate_fairness_comprehensive
        dataset_name: Name of dataset
    """
    print("\n" + "="*70)
    print(f"FAIRNESS EVALUATION REPORT - {dataset_name.upper()}")
    print("="*70)
    
    # Group Fairness Metrics
    print("\n" + "-"*70)
    print("GROUP FAIRNESS METRICS")
    print("-"*70)
    
    metrics_to_print = [
        ('demographic_parity', 'Demographic Parity'),
        ('equalized_odds', 'Equalized Odds'),
        ('equal_opportunity', 'Equal Opportunity'),
        ('disparate_impact', 'Disparate Impact'),
        ('accuracy_parity', 'Accuracy Parity'),
        ('treatment_equality', 'Treatment Equality')
    ]
    
    for metric_key, metric_name in metrics_to_print:
        print(f"\n{metric_name}:")
        if metric_key in results['original']:
            orig = results['original'][metric_key]
            synth = results['synthetic'][metric_key]
            
            # Print main score if available
            if 'demographic_parity' in orig:
                print(f"  Original:   {orig['demographic_parity']:.4f}")
                print(f"  Synthetic:  {synth['demographic_parity']:.4f}")
            elif 'equalized_odds' in orig:
                print(f"  Original:   {orig['equalized_odds']:.4f}")
                print(f"  Synthetic:  {synth['equalized_odds']:.4f}")
            elif 'equal_opportunity' in orig:
                print(f"  Original:   {orig['equal_opportunity']:.4f}")
                print(f"  Synthetic:  {synth['equal_opportunity']:.4f}")
            elif 'disparate_impact' in orig:
                print(f"  Original:   {orig['disparate_impact']:.4f} (meets threshold: {orig.get('meets_threshold', False)})")
                print(f"  Synthetic:  {synth['disparate_impact']:.4f} (meets threshold: {synth.get('meets_threshold', False)})")
            elif 'accuracy_parity' in orig:
                print(f"  Original:   {orig['accuracy_parity']:.4f}")
                print(f"  Synthetic:  {synth['accuracy_parity']:.4f}")
            elif 'treatment_equality' in orig:
                print(f"  Original:   {orig['treatment_equality']:.4f}")
                print(f"  Synthetic:  {synth['treatment_equality']:.4f}")
            
            # Print comparison if available
            if metric_key in results['comparison']:
                comp = results['comparison'][metric_key]
                if 'difference' in comp:
                    print(f"  Difference: {comp['difference']:.4f}")
    
    # Individual Fairness Metrics
    if 'individual_fairness' in results['original']:
        print("\n" + "-"*70)
        print("INDIVIDUAL FAIRNESS METRICS")
        print("-"*70)
        
        orig_if = results['original']['individual_fairness']
        synth_if = results['synthetic']['individual_fairness']
        
        print(f"\nIndividual Fairness Score:")
        print(f"  Original:   {orig_if['individual_fairness']:.4f}")
        print(f"  Synthetic:  {synth_if['individual_fairness']:.4f}")
        
        print(f"\nConsistency Score:")
        print(f"  Original:   {orig_if['consistency_score']:.4f}")
        print(f"  Synthetic:  {synth_if['consistency_score']:.4f}")
    
    print("\n" + "="*70)


def save_fairness_report(results: Dict[str, Dict], output_path: str, dataset_name: str = "german-credit-data"):
    """
    Save fairness report to JSON file.
    
    Args:
        results: Results dictionary
        output_path: Path to save JSON file
        dataset_name: Name of dataset
    """
    # Convert numpy types to native Python types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj) if isinstance(obj, np.floating) else int(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        elif isinstance(obj, bool):
            return obj
        elif pd.isna(obj):
            return None
        return obj
    
    serializable_results = convert_to_serializable(results)
    
    report = {
        'dataset_name': dataset_name,
        'timestamp': datetime.now().isoformat(),
        'results': serializable_results
    }
    
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n✓ Fairness report saved to: {output_path}")


if __name__ == "__main__":
    # Example usage
    print("Fairness Evaluation Module")
    print("Import this module and use evaluate_fairness_comprehensive() function")
    print("to evaluate fairness metrics for synthetic datasets.")
