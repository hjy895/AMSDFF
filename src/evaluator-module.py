"""
Model Evaluator for Enhanced AMSDFF
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple

from .metrics import (
    calculate_metrics,
    calculate_confusion_matrix,
    calculate_classification_report,
    calculate_roc_auc
)


class ModelEvaluator:
    """
    Comprehensive model evaluator
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: Optional[str] = None
    ):
        self.model = model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
    
    def evaluate(
        self,
        dataloader: DataLoader,
        return_predictions: bool = False,
        return_probabilities: bool = False
    ) -> Dict:
        """
        Evaluate model on a dataset
        
        Args:
            dataloader: DataLoader for evaluation
            return_predictions: Whether to return predictions
            return_probabilities: Whether to return probabilities
            
        Returns:
            Dictionary of evaluation metrics
        """
        self.model.eval()
        
        all_predictions = []
        all_labels = []
        all_probabilities = []
        total_loss = 0.0
        
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                logits = self.model(input_ids, attention_mask)
                
                # Calculate loss
                loss = criterion(logits, labels)
                total_loss += loss.item()
                
                # Get predictions
                predictions = logits.argmax(dim=-1)
                probabilities = torch.softmax(logits, dim=-1)
                
                # Store results
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # Convert to numpy arrays
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        all_probabilities = np.array(all_probabilities)
        
        # Calculate metrics
        metrics = calculate_metrics(all_labels, all_predictions)
        metrics['loss'] = total_loss / len(dataloader)
        
        # Add confusion matrix
        metrics['confusion_matrix'] = calculate_confusion_matrix(
            all_labels, all_predictions
        )
        
        # Add ROC AUC if binary or probability scores available
        if len(np.unique(all_labels)) == 2:
            metrics['roc_auc'] = calculate_roc_auc(
                all_labels, all_probabilities[:, 1]
            )
        
        # Return additional data if requested
        if return_predictions:
            metrics['predictions'] = all_predictions
        
        if return_probabilities:
            metrics['probabilities'] = all_probabilities
        
        return metrics
    
    def evaluate_with_analysis(
        self,
        dataloader: DataLoader,
        class_names: Optional[List[str]] = None
    ) -> Dict:
        """
        Evaluate with detailed analysis
        
        Args:
            dataloader: DataLoader for evaluation
            class_names: Optional list of class names
            
        Returns:
            Dictionary with evaluation results and analysis
        """
        # Get basic evaluation metrics
        results = self.evaluate(
            dataloader,
            return_predictions=True,
            return_probabilities=True
        )
        
        # Add classification report
        results['classification_report'] = calculate_classification_report(
            results['labels'],
            results['predictions'],
            target_names=class_names
        )
        
        # Add per-class accuracy
        confusion_matrix = results['confusion_matrix']
        per_class_accuracy = np.diag(confusion_matrix) / confusion_matrix.sum(axis=1)
        results['per_class_accuracy'] = per_class_accuracy
        
        # Add prediction confidence statistics
        probabilities = results['probabilities']
        max_probs = np.max(probabilities, axis=1)
        
        results['confidence_stats'] = {
            'mean': np.mean(max_probs),
            'std': np.std(max_probs),
            'min': np.min(max_probs),
            'max': np.max(max_probs),
            'median': np.median(max_probs)
        }
        
        # Identify misclassified samples
        misclassified_indices = np.where(
            results['predictions'] != results['labels']
        )[0]
        
        results['misclassified'] = {
            'count': len(misclassified_indices),
            'percentage': len(misclassified_indices) / len(results['labels']) * 100,
            'indices': misclassified_indices
        }
        
        return results
    
    def cross_validate(
        self,
        dataset,
        n_folds: int = 5,
        stratified: bool = True
    ) -> Dict:
        """
        Perform cross-validation
        
        Args:
            dataset: Dataset to evaluate
            n_folds: Number of folds
            stratified: Whether to use stratified splits
            
        Returns:
            Cross-validation results
        """
        from sklearn.model_selection import KFold, StratifiedKFold
        
        # Prepare data
        all_indices = list(range(len(dataset)))
        all_labels = [dataset[i]['labels'].item() for i in all_indices]
        
        # Create folds
        if stratified:
            kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
            splits = kfold.split(all_indices, all_labels)
        else:
            kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
            splits = kfold.split(all_indices)
        
        # Store results for each fold
        fold_results = []
        
        for fold_idx, (train_indices, val_indices) in enumerate(splits):
            print(f"\nFold {fold_idx + 1}/{n_folds}")
            
            # Create subset datasets
            train_subset = torch.utils.data.Subset(dataset, train_indices)
            val_subset = torch.utils.data.Subset(dataset, val_indices)
            
            # Create dataloaders
            train_loader = DataLoader(train_subset, batch_size=16, shuffle=True)
            val_loader = DataLoader(val_subset, batch_size=16, shuffle=False)
            
            # Train model (simplified - you might want to use full trainer)
            # This is a placeholder - implement actual training
            
            # Evaluate
            fold_metrics = self.evaluate(val_loader)
            fold_results.append(fold_metrics)
        
        # Aggregate results
        aggregated_results = {}
        metric_names = fold_results[0].keys()
        
        for metric in metric_names:
            if metric not in ['confusion_matrix', 'predictions', 'probabilities']:
                values = [fold[metric] for fold in fold_results]
                aggregated_results[f'{metric}_mean'] = np.mean(values)
                aggregated_results[f'{metric}_std'] = np.std(values)
        
        aggregated_results['fold_results'] = fold_results
        
        return aggregated_results