"""
Multi-Scale Attention Module for Enhanced AMSDFF
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EnhancedMultiScaleAttention(nn.Module):
    """
    Enhanced multi-scale attention mechanism that captures features
    at different granularities using depthwise separable convolutions
    and attention-based fusion.
    """
    
    def __init__(self, hidden_size: int, scales: list = [1, 3, 5]):
        super().__init__()
        self.scales = scales
        self.num_scales = len(scales)
        
        # Depthwise separable convolutions for each scale
        self.scale_convs = nn.ModuleList([
            self._build_scale_conv(hidden_size, scale) 
            for scale in scales
        ])
        
        # Scale fusion attention
        self.scale_fusion = nn.MultiheadAttention(
            hidden_size, 
            num_heads=8, 
            batch_first=True, 
            dropout=0.1
        )
        
        # Output projection
        self.output_projection = nn.Linear(hidden_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        
    def _build_scale_conv(self, hidden_size: int, scale: int) -> nn.Module:
        """Build depthwise separable convolution for a specific scale"""
        return nn.Sequential(
            # Depthwise convolution
            nn.Conv1d(
                hidden_size, 
                hidden_size, 
                kernel_size=scale,
                padding=scale//2, 
                groups=hidden_size
            ),
            # Pointwise convolution
            nn.Conv1d(hidden_size, hidden_size // self.num_scales, 1),
            nn.BatchNorm1d(hidden_size // self.num_scales),
            nn.GELU()
        )
    
    def forward(
        self, 
        x: torch.Tensor, 
        attention_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Apply multi-scale attention
        
        Args:
            x: Input features [batch_size, seq_len, hidden_size]
            attention_mask: Optional attention mask [batch_size, seq_len]
            
        Returns:
            Enhanced features [batch_size, seq_len, hidden_size]
        """
        batch_size, seq_len, hidden_size = x.shape
        
        # Apply multi-scale convolutions
        x_conv = x.transpose(1, 2)  # [B, H, S]
        scale_outputs = []
        
        for conv in self.scale_convs:
            scale_out = conv(x_conv)  # [B, H//num_scales, S]
            scale_out = scale_out.transpose(1, 2)  # [B, S, H//num_scales]
            scale_outputs.append(scale_out)
        
        # Concatenate scale outputs
        multi_scale_features = torch.cat(scale_outputs, dim=-1)  # [B, S, H]
        
        # Apply attention fusion
        if attention_mask is not None:
            attended_features, _ = self.scale_fusion(
                multi_scale_features, 
                multi_scale_features, 
                multi_scale_features,
                key_padding_mask=~attention_mask.bool()
            )
        else:
            attended_features, _ = self.scale_fusion(
                multi_scale_features, 
                multi_scale_features, 
                multi_scale_features
            )
        
        # Output projection and residual connection
        output = self.output_projection(attended_features)
        output = self.layer_norm(output + x)  # Residual connection
        
        return output