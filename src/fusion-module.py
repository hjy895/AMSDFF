"""
Dynamic Feature Fusion Module for Enhanced AMSDFF
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimplifiedDynamicFusion(nn.Module):
    """
    Simplified dynamic fusion module that adaptively combines
    features using a mixture of experts approach.
    """
    
    def __init__(self, hidden_size: int, num_experts: int = 3):
        super().__init__()
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        
        # Expert networks
        self.experts = nn.ModuleList([
            self._build_expert(hidden_size) 
            for _ in range(num_experts)
        ])
        
        # Gating network
        self.gate = nn.Sequential(
            nn.Linear(hidden_size, num_experts),
            nn.Softmax(dim=-1)
        )
        
    def _build_expert(self, hidden_size: int) -> nn.Module:
        """Build an expert network"""
        return nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply dynamic fusion
        
        Args:
            x: Input features [batch_size, seq_len, hidden_size]
            
        Returns:
            Fused features [batch_size, seq_len, hidden_size]
        """
        # Compute expert outputs
        expert_outputs = torch.stack([
            expert(x) for expert in self.experts
        ], dim=-1)  # [B, S, H, num_experts]
        
        # Compute gates using pooled features
        pooled_x = x.mean(dim=1)  # [B, H]
        gates = self.gate(pooled_x)  # [B, num_experts]
        
        # Reshape gates for broadcasting
        gates = gates.unsqueeze(1).unsqueeze(-1)  # [B, 1, 1, num_experts]
        
        # Weighted combination of expert outputs
        output = (expert_outputs * gates).sum(dim=-1)  # [B, S, H]
        
        return output


class AdaptiveFusion(nn.Module):
    """
    Alternative adaptive fusion module with cross-attention mechanism
    """
    
    def __init__(self, hidden_size: int, num_heads: int = 8):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Cross-attention for adaptive fusion
        self.cross_attention = nn.MultiheadAttention(
            hidden_size,
            num_heads=num_heads,
            batch_first=True,
            dropout=0.1
        )
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size * 4, hidden_size)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        
    def forward(
        self, 
        x: torch.Tensor,
        context: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Apply adaptive fusion with optional context
        
        Args:
            x: Input features [batch_size, seq_len, hidden_size]
            context: Optional context features
            
        Returns:
            Fused features [batch_size, seq_len, hidden_size]
        """
        # Use self-attention if no context provided
        if context is None:
            context = x
            
        # Cross-attention
        attended, _ = self.cross_attention(x, context, context)
        x = self.norm1(x + attended)
        
        # Feed-forward
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        return x