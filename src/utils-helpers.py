"""
Helper utilities for Enhanced AMSDFF
"""

import os
import random
import numpy as np
import torch
import json
from typing import Dict, List, Any
from datetime import datetime


def set_seed(seed: int = 42):
    """
    Set random seeds for reproducibility
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_device() -> torch.device:
    """
    Get the best available device
    
    Returns:
        torch.device: CUDA if available, else CPU
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_parameters(model: torch.nn.Module) -> int:
    """
    Count trainable parameters in a model
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    accuracy: float,
    path: str
):
    """
    Save model checkpoint
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        loss: Current loss
        accuracy: Current accuracy
        path: Save path
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': accuracy,
        'timestamp': datetime.now().isoformat()
    }
    
    torch.save(checkpoint, path)
    print(f"Checkpoint saved to {path}")


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer = None,
    device: str = 'cpu'
) -> Dict[str, Any]:
    """
    Load model checkpoint
    
    Args:
        path: Checkpoint path
        model: Model to load weights into
        optimizer: Optional optimizer to load state
        device: Device to map tensors to
        
    Returns:
        Checkpoint dictionary
    """
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"Checkpoint loaded from {path}")
    return checkpoint


def create_results_table(results: Dict[str, Dict[str, Dict[str, float]]]):
    """
    Create a formatted results table
    
    Args:
        results: Nested dictionary of results
    """
    print("\n" + "="*120)
    print("EXPERIMENTAL RESULTS")
    print("="*120)
    
    # Get all models and datasets
    datasets = list(results.keys())
    models = set()
    for dataset in datasets:
        if results[dataset]:
            models.update(results[dataset].keys())
    models = sorted(list(models))
    
    # Print header
    header = f"{'Model':<25}"
    for dataset in datasets[:3]:  # Show first 3 datasets
        header += f"{dataset:<30}"
    print(header)
    
    sub_header = f"{'':<25}"
    for _ in datasets[:3]:
        sub_header += f"{'Acc    F1     MAE    RMSE':<30}"
    print(sub_header)
    print("-" * 120)
    
    # Print results for each model
    for model in models:
        row = f"{model:<25}"
        
        for dataset in datasets[:3]:
            if dataset in results and model in results[dataset] and results[dataset][model]:
                metrics = results[dataset][model]
                acc = metrics.get('accuracy', 0)
                f1 = metrics.get('f1_score', 0)
                mae = metrics.get('MAE', 0)
                rmse = metrics.get('RMSE', 0)
                
                # Highlight high accuracy
                if acc >= 0.90:
                    row += f"\033[92m{acc:.3f}\033[0m  {f1:.3f}  {mae:.3f}  {rmse:.3f}  "
                else:
                    row += f"{acc:.3f}  {f1:.3f}  {mae:.3f}  {rmse:.3f}  "
            else:
                row += f"{'N/A':<30}"
        
        print(row)
    
    print("="*120)
    
    # Calculate average improvements if Enhanced AMSDFF is present
    if any('enhanced_amsdff' in results.get(d, {}) for d in datasets):
        print("\nPerformance Summary:")
        print("-" * 50)
        
        amsdff_accuracies = []
        for dataset in datasets:
            if dataset in results and 'enhanced_amsdff' in results[dataset]:
                if results[dataset]['enhanced_amsdff']:
                    acc = results[dataset]['enhanced_amsdff']['accuracy']
                    amsdff_accuracies.append(acc)
                    print(f"{dataset}: {acc:.3f}")
        
        if amsdff_accuracies:
            avg_acc = np.mean(amsdff_accuracies)
            print(f"\nAverage Accuracy: {avg_acc:.3f}")
            
            if avg_acc >= 0.90:
                print("✅ TARGET ACHIEVED: 90%+ average accuracy")
            else:
                print(f"Current: {avg_acc:.1%} (Target: 90%)")


def format_time(seconds: float) -> str:
    """
    Format time in seconds to human readable string
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def log_training_progress(
    epoch: int,
    total_epochs: int,
    train_loss: float,
    val_loss: float,
    train_acc: float,
    val_acc: float,
    time_elapsed: float
):
    """
    Log training progress
    
    Args:
        epoch: Current epoch
        total_epochs: Total number of epochs
        train_loss: Training loss
        val_loss: Validation loss
        train_acc: Training accuracy
        val_acc: Validation accuracy
        time_elapsed: Time elapsed for epoch
    """
    print(f"\nEpoch [{epoch}/{total_epochs}] - {format_time(time_elapsed)}")
    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
    print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
    
    # Progress bar
    progress = epoch / total_epochs
    bar_length = 30
    filled_length = int(bar_length * progress)
    bar = '█' * filled_length + '-' * (bar_length - filled_length)
    print(f"Progress: [{bar}] {progress:.1%}")


def save_results_json(results: Dict, path: str):
    """
    Save results to JSON file
    
    Args:
        results: Results dictionary
        path: Save path
    """
    # Convert numpy types to native Python types
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_numpy(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        return obj
    
    results_converted = convert_numpy(results)
    
    with open(path, 'w') as f:
        json.dump(results_converted, f, indent=2)
    
    print(f"Results saved to {path}")