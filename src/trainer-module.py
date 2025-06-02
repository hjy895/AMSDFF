"""
Training Module for Enhanced AMSDFF
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm
import numpy as np
from typing import Dict, Tuple, Optional

from .loss_functions import FocalLoss, LabelSmoothingLoss
from ..evaluation.metrics import calculate_metrics


class AMSDFFTrainer:
    """
    Trainer class for Enhanced AMSDFF model
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = None,
        config: dict = None
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config or self._get_default_config()
        
        # Move model to device
        self.model.to(self.device)
        
        # Initialize optimizer and scheduler
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # Initialize loss functions
        self.criterion = self._create_loss_function()
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_accuracy': [],
            'val_accuracy': []
        }
        
        # Best model tracking
        self.best_val_accuracy = 0.0
        self.best_model_state = None
    
    def _get_default_config(self) -> dict:
        """Get default training configuration"""
        return {
            'learning_rate': 5e-6,
            'weight_decay': 0.01,
            'warmup_ratio': 0.1,
            'epochs': 5,
            'gradient_clip': 1.0,
            'loss_weights': {
                'ce': 0.4,
                'focal': 0.4,
                'label_smoothing': 0.2
            },
            'focal_alpha': 1.0,
            'focal_gamma': 2.0,
            'label_smoothing': 0.1
        }
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer"""
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay'],
            betas=(0.9, 0.999),
            eps=1e-8
        )
    
    def _create_scheduler(self) -> torch.optim.lr_scheduler._LRScheduler:
        """Create learning rate scheduler"""
        total_steps = len(self.train_loader) * self.config['epochs']
        warmup_steps = int(total_steps * self.config['warmup_ratio'])
        
        return get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
    
    def _create_loss_function(self):
        """Create combined loss function"""
        return CombinedLoss(
            focal_alpha=self.config['focal_alpha'],
            focal_gamma=self.config['focal_gamma'],
            label_smoothing=self.config['label_smoothing'],
            weights=self.config['loss_weights']
        )
    
    def train(self, epochs: int = None) -> Dict[str, list]:
        """
        Train the model
        
        Args:
            epochs: Number of epochs to train (overrides config)
            
        Returns:
            Training history dictionary
        """
        epochs = epochs or self.config['epochs']
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            print("-" * 50)
            
            # Training phase
            train_loss, train_acc = self._train_epoch()
            
            # Validation phase
            val_loss, val_acc, val_metrics = self._validate()
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_accuracy'].append(train_acc)
            self.history['val_accuracy'].append(val_acc)
            
            # Print epoch results
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
            print(f"Val F1: {val_metrics['f1_score']:.4f}")
            
            # Save best model
            if val_acc > self.best_val_accuracy:
                self.best_val_accuracy = val_acc
                self.best_model_state = self.model.state_dict().copy()
                print(f"New best model! Accuracy: {val_acc:.4f}")
        
        # Load best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            
        return self.history
    
    def _train_epoch(self) -> Tuple[float, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(self.train_loader, desc="Training")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            logits = self.model(input_ids, attention_mask)
            
            # Calculate loss
            loss = self.criterion(logits, labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.config['gradient_clip']
            )
            
            # Update weights
            self.optimizer.step()
            self.scheduler.step()
            
            # Update metrics
            total_loss += loss.item()
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{correct/total:.4f}"
            })
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def _validate(self) -> Tuple[float, float, dict]:
        """Validate the model"""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                logits = self.model(input_ids, attention_mask)
                
                # Calculate loss
                loss = self.criterion(logits, labels)
                total_loss += loss.item()
                
                # Get predictions
                preds = logits.argmax(dim=-1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        avg_loss = total_loss / len(self.val_loader)
        metrics = calculate_metrics(all_labels, all_preds)
        
        return avg_loss, metrics['accuracy'], metrics
    
    def evaluate(self, test_loader: DataLoader) -> dict:
        """
        Evaluate model on test set
        
        Args:
            test_loader: Test data loader
            
        Returns:
            Dictionary of evaluation metrics
        """
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluation"):
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                logits = self.model(input_ids, attention_mask)
                
                # Get predictions
                preds = logits.argmax(dim=-1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate comprehensive metrics
        metrics = calculate_metrics(all_labels, all_preds)
        
        return metrics
    
    def save_model(self, path: str):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
            'history': self.history,
            'best_val_accuracy': self.best_val_accuracy
        }
        torch.save(checkpoint, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.config = checkpoint['config']
        self.history = checkpoint['history']
        self.best_val_accuracy = checkpoint['best_val_accuracy']
        print(f"Model loaded from {path}")


class CombinedLoss(nn.Module):
    """Combined loss function for Enhanced AMSDFF"""
    
    def __init__(
        self,
        focal_alpha: float = 1.0,
        focal_gamma: float = 2.0,
        label_smoothing: float = 0.1,
        weights: dict = None
    ):
        super().__init__()
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.label_smoothing_loss = LabelSmoothingLoss(smoothing=label_smoothing)
        self.weights = weights or {'ce': 0.4, 'focal': 0.4, 'label_smoothing': 0.2}
    
    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Calculate combined loss"""
        ce_loss = F.cross_entropy(logits, labels)
        focal_loss = self.focal_loss(logits, labels)
        smooth_loss = self.label_smoothing_loss(logits, labels)
        
        total_loss = (
            self.weights['ce'] * ce_loss +
            self.weights['focal'] * focal_loss +
            self.weights['label_smoothing'] * smooth_loss
        )
        
        return total_loss