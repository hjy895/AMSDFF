"""
Evaluation metrics for Enhanced AMSDFF
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report, roc_auc_score
)
from typing import Dict, List, Optional, Union


def calculate_metrics(
    y_true: Union[List, np.ndarray],
    y_pred: Union[List, np.ndarray],
    average: str = 'weighted',
    include_regression_metrics: bool = True
) -> Dict[str, float]:
    """
    Calculate comprehensive evaluation metrics
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        average: Averaging method for multi-class metrics
        include_regression_metrics: Whether to include regression-style metrics
        
    Returns:
        Dictionary of metrics
    """
    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Classification metrics
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred, average=average, zero_division=0),
        'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
        'recall': recall_score(y_true, y_pred, average=average, zero_division=0)
    }
    
    # Per-class metrics
    if len(np.unique(y_true)) > 2:
        metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['f1_micro'] = f1_score(y_true, y_pred, average='micro', zero_division=0)
    
    # Regression-style metrics (for compatibility with paper format)
    if include_regression_metrics:
        metrics['MAE'] = np.mean(np.abs(y_pred - y_true))
        metrics['RMSE'] = np.sqrt(np.mean((y_pred - y_true) ** 2))
        metrics['MAPE'] = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    
    return metrics


def calculate_confusion_matrix(
    y_true: Union[List, np.ndarray],
    y_pred: Union[List, np.ndarray],
    labels: Optional[List] = None
) -> np.ndarray:
    """
    Calculate confusion matrix
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: List of label names
        
    Returns:
        Confusion matrix
    """
    return confusion_matrix(y_true, y_pred, labels=labels)


def calculate_classification_report(
    y_true: Union[List, np.ndarray],
    y_pred: Union[List, np.ndarray],
    labels: Optional[List] = None,
    target_names: Optional[List[str]] = None
) -> str:
    """
    Generate classification report
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: List of label indices
        target_names: List of label names
        
    Returns:
        Classification report string
    """
    return classification_report(
        y_true, y_pred, 
        labels=labels, 
        target_names=target_names,
        zero_division=0
    )


def calculate_roc_auc(
    y_true: Union[List, np.ndarray],
    y_proba: np.ndarray,
    multi_class: str = 'ovr'
) -> float:
    """
    Calculate ROC AUC score
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        multi_class: Multi-class strategy ('ovr' or 'ovo')
        
    Returns:
        ROC AUC score
    """
    try:
        return roc_auc_score(y_true, y_proba, multi_class=multi_class)
    except ValueError:
        # Handle cases where ROC AUC cannot be calculated
        return 0.0


def calculate_performance_improvements(
    baseline_metrics: Dict[str, float],
    enhanced_metrics: Dict[str, float]
) -> Dict[str, float]:
    """
    Calculate performance improvements over baseline
    
    Args:
        baseline_metrics: Baseline model metrics
        enhanced_metrics: Enhanced model metrics
        
    Returns:
        Dictionary of percentage improvements
    """
    improvements = {}
    
    for metric in baseline_metrics:
        if metric in enhanced_metrics:
            baseline_val = baseline_metrics[metric]
            enhanced_val = enhanced_metrics[metric]
            
            if baseline_val > 0:
                # For error metrics (MAE, RMSE, MAPE), improvement is reduction
                if metric in ['MAE', 'RMSE', 'MAPE']:
                    improvement = (baseline_val - enhanced_val) / baseline_val * 100
                else:
                    # For accuracy metrics, improvement is increase
                    improvement = (enhanced_val - baseline_val) / baseline_val * 100
                
                improvements[f'{metric}_improvement'] = improvement
    
    return improvements


def calculate_statistical_significance(
    predictions_1: np.ndarray,
    predictions_2: np.ndarray,
    y_true: np.ndarray,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95
) -> Dict[str, float]:
    """
    Calculate statistical significance using bootstrap
    
    Args:
        predictions_1: Predictions from model 1
        predictions_2: Predictions from model 2
        y_true: True labels
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level for intervals
        
    Returns:
        Dictionary with p-value and confidence intervals
    """
    n_samples = len(y_true)
    
    # Calculate observed difference
    acc_1 = accuracy_score(y_true, predictions_1)
    acc_2 = accuracy_score(y_true, predictions_2)
    observed_diff = acc_2 - acc_1
    
    # Bootstrap
    differences = []
    for _ in range(n_bootstrap):
        # Sample with replacement
        indices = np.random.choice(n_samples, n_samples, replace=True)
        
        # Calculate accuracies for bootstrap sample
        boot_acc_1 = accuracy_score(y_true[indices], predictions_1[indices])
        boot_acc_2 = accuracy_score(y_true[indices], predictions_2[indices])
        
        differences.append(boot_acc_2 - boot_acc_1)
    
    differences = np.array(differences)
    
    # Calculate p-value
    p_value = np.mean(differences <= 0) if observed_diff > 0 else np.mean(differences >= 0)
    
    # Calculate confidence intervals
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    ci_lower = np.percentile(differences, lower_percentile)
    ci_upper = np.percentile(differences, upper_percentile)
    
    return {
        'observed_difference': observed_diff,
        'p_value': p_value,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'significant': p_value < alpha
    }