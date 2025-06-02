"""
Enhanced AMSDFF Model Implementation
Advanced Multi-Scale Dynamic Feature Fusion for Text Classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from .components.attention import EnhancedMultiScaleAttention
from .components.fusion import SimplifiedDynamicFusion
from .components.pooling import AdvancedPoolingStrategy


class EnhancedAMSDFF(nn.Module):
    """
    Enhanced AMSDFF: Advanced Multi-Scale Dynamic Feature Fusion Model
    
    This model combines state-of-the-art techniques for text classification:
    - Multi-scale attention mechanisms
    - Dynamic feature fusion
    - Advanced pooling strategies
    - Regularized classification head
    
    Args:
        num_classes (int): Number of output classes
        hidden_size (int): Hidden dimension size (default: 768)
        pretrained_model (str): Name of pretrained transformer model
        dropout_rate (float): Dropout rate for regularization
    """
    
    def __init__(
        self, 
        num_classes: int,
        hidden_size: int = 768,
        pretrained_model: str = 'distilbert-base-uncased',
        dropout_rate: float = 0.1
    ):
        super().__init__()
        
        # Load pretrained transformer
        self.transformer = AutoModel.from_pretrained(pretrained_model)
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        
        # Core components
        self.multi_scale_attention = EnhancedMultiScaleAttention(
            hidden_size=hidden_size,
            scales=[1, 3, 5]
        )
        
        # Feature enhancement layer
        self.feature_enhancer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )
        
        # Dynamic fusion module
        self.dynamic_fusion = SimplifiedDynamicFusion(
            hidden_size=hidden_size,
            num_experts=3
        )
        
        # Advanced pooling strategy
        self.pooling_strategy = AdvancedPoolingStrategy(hidden_size)
        
        # Regularized classification head
        self.classifier = self._build_classifier(hidden_size, num_classes)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _build_classifier(self, hidden_size: int, num_classes: int) -> nn.Module:
        """Build regularized classification head"""
        return nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU(),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.LayerNorm(hidden_size // 4),
            nn.GELU(),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_size // 4, num_classes)
        )
    
    def _init_weights(self, module):
        """Initialize model weights"""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.constant_(module.bias, 0)
            torch.nn.init.constant_(module.weight, 1.0)
    
    def forward(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor,
        return_features: bool = False
    ) -> torch.Tensor:
        """
        Forward pass of Enhanced AMSDFF
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            return_features: Whether to return intermediate features
            
        Returns:
            logits: Classification logits [batch_size, num_classes]
            features (optional): Intermediate features if return_features=True
        """
        
        # Get transformer outputs
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        sequence_output = outputs.last_hidden_state
        
        # Feature enhancement
        enhanced_features = self.feature_enhancer(sequence_output)
        
        # Multi-scale attention
        ms_features = self.multi_scale_attention(enhanced_features, attention_mask)
        
        # Dynamic fusion
        fused_features = self.dynamic_fusion(ms_features)
        
        # Advanced pooling
        pooled_features = self.pooling_strategy(fused_features, attention_mask)
        
        # Classification
        logits = self.classifier(pooled_features)
        
        if return_features:
            return {
                'logits': logits,
                'pooled_features': pooled_features,
                'sequence_features': fused_features
            }
        
        return logits
    
    def get_attention_weights(self):
        """Get attention weights for visualization"""
        return self.pooling_strategy.get_attention_weights()