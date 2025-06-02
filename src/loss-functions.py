"""
Custom loss functions for Enhanced AMSDFF
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance
    
    Reference:
    Lin et al. "Focal Loss for Dense Object Detection"
    """
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate focal loss
        
        Args:
            inputs: Predictions [batch_size, num_classes]
            targets: Ground truth labels [batch_size]
            
        Returns:
            Focal loss value
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        p_t = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class LabelSmoothingLoss(nn.Module):
    """
    Label smoothing loss for better generalization
    """
    
    def __init__(self, smoothing: float = 0.1, reduction: str = 'mean'):
        super().__init__()
        self.smoothing = smoothing
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate label smoothing loss
        
        Args:
            inputs: Predictions [batch_size, num_classes]
            targets: Ground truth labels [batch_size]
            
        Returns:
            Label smoothing loss value
        """
        n_classes = inputs.size(-1)
        
        # Create smoothed labels
        one_hot = torch.zeros_like(inputs).scatter(1, targets.unsqueeze(1), 1)
        smoothed_labels = one_hot * (1 - self.smoothing) + self.smoothing / n_classes
        
        # Calculate loss
        log_probs = F.log_softmax(inputs, dim=-1)
        loss = -(smoothed_labels * log_probs).sum(dim=-1)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for learning better representations
    """
    
    def __init__(self, temperature: float = 0.07, reduction: str = 'mean'):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
    
    def forward(
        self, 
        features: torch.Tensor, 
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate contrastive loss
        
        Args:
            features: Feature representations [batch_size, feature_dim]
            labels: Ground truth labels [batch_size]
            
        Returns:
            Contrastive loss value
        """
        # Normalize features
        features = F.normalize(features, p=2, dim=1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        
        # Create mask for positive pairs
        labels = labels.unsqueeze(0)
        mask = torch.eq(labels, labels.T).float()
        
        # Compute log probabilities
        exp_sim = torch.exp(similarity_matrix)
        log_prob = similarity_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True))
        
        # Compute mean of log-likelihood for positive pairs
        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / mask.sum(dim=1)
        
        # Loss is negative log-likelihood
        loss = -mean_log_prob_pos
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class MultiTaskLoss(nn.Module):
    """
    Multi-task loss for auxiliary objectives
    """
    
    def __init__(self,