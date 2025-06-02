"""
Advanced Pooling Strategies for Enhanced AMSDFF
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdvancedPoolingStrategy(nn.Module):
    """
    Advanced pooling strategy that combines multiple pooling methods:
    - Attention-based pooling
    - CLS token pooling
    - Max pooling
    """
    
    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Attention pooling
        self.attention_pooling = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1)
        )
        
        # Learnable CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_size))
        
        # Multi-head attention for CLS
        self.cls_attention = nn.MultiheadAttention(
            hidden_size,
            num_heads=8,
            batch_first=True,
            dropout=0.1
        )
        
        # Pooling fusion layer
        self.pooling_fusion = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),  # 3 pooling methods
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size)
        )
        
        # Store attention weights for visualization
        self._attention_weights = None
        
    def forward(
        self, 
        x: torch.Tensor, 
        attention_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Apply advanced pooling strategy
        
        Args:
            x: Input features [batch_size, seq_len, hidden_size]
            attention_mask: Optional attention mask [batch_size, seq_len]
            
        Returns:
            Pooled features [batch_size, hidden_size]
        """
        batch_size = x.size(0)
        
        # 1. Attention-based pooling
        attention_weights = self.attention_pooling(x)  # [B, S, 1]
        
        if attention_mask is not None:
            attention_weights = attention_weights.masked_fill(
                ~attention_mask.unsqueeze(-1).bool(),
                float('-inf')
            )
        
        attention_weights = F.softmax(attention_weights, dim=1)
        self._attention_weights = attention_weights.detach()  # Store for visualization
        
        attended_pooled = (x * attention_weights).sum(dim=1)  # [B, H]
        
        # 2. CLS token pooling
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # [B, 1, H]
        
        if attention_mask is not None:
            cls_output, _ = self.cls_attention(
                cls_tokens, x, x,
                key_padding_mask=~attention_mask.bool()
            )
        else:
            cls_output, _ = self.cls_attention(cls_tokens, x, x)
        
        cls_pooled = cls_output.squeeze(1)  # [B, H]
        
        # 3. Max pooling
        if attention_mask is not None:
            masked_x = x.masked_fill(
                ~attention_mask.unsqueeze(-1).bool(),
                float('-inf')
            )
        else:
            masked_x = x
            
        max_pooled = masked_x.max(dim=1)[0]  # [B, H]
        
        # Combine all pooling methods
        combined = torch.cat([attended_pooled, cls_pooled, max_pooled], dim=-1)
        output = self.pooling_fusion(combined)
        
        return output
    
    def get_attention_weights(self):
        """Get stored attention weights for visualization"""
        return self._attention_weights


class HierarchicalPooling(nn.Module):
    """
    Alternative hierarchical pooling strategy
    """
    
    def __init__(self, hidden_size: int, num_levels: int = 3):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_levels = num_levels
        
        # Hierarchical pooling layers
        self.pooling_layers = nn.ModuleList([
            nn.Conv1d(hidden_size, hidden_size, kernel_size=2, stride=2)
            for _ in range(num_levels)
        ])
        
        # Fusion layer
        self.fusion = nn.Linear(hidden_size * num_levels, hidden_size)
        
    def forward(
        self, 
        x: torch.Tensor,
        attention_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Apply hierarchical pooling
        
        Args:
            x: Input features [batch_size, seq_len, hidden_size]
            attention_mask: Optional attention mask
            
        Returns:
            Pooled features [batch_size, hidden_size]
        """
        # Transpose for conv1d
        x = x.transpose(1, 2)  # [B, H, S]
        
        pooled_features = []
        current_features = x
        
        for pool_layer in self.pooling_layers:
            current_features = pool_layer(current_features)
            # Global average pooling at each level
            pooled = current_features.mean(dim=-1)  # [B, H]
            pooled_features.append(pooled)
        
        # Combine hierarchical features
        combined = torch.cat(pooled_features, dim=-1)  # [B, H * num_levels]
        output = self.fusion(combined)  # [B, H]
        
        return output