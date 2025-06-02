"""
Baseline models for comparison with Enhanced AMSDFF
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseModel(nn.Module):
    """Base class for all models"""
    
    def __init__(self, pretrained_model, num_classes: int, hidden_size: int = 768):
        super().__init__()
        self.transformer = pretrained_model
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(0.1)


class LSTMModel(BaseModel):
    """LSTM-based text classification model"""
    
    def __init__(self, pretrained_model, num_classes: int):
        super().__init__(pretrained_model, num_classes)
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            self.hidden_size,
            self.hidden_size // 2,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=0.1
        )
        
        # Classification head
        self.classifier = nn.Linear(self.hidden_size, num_classes)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # Get transformer outputs
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        
        # Apply LSTM
        lstm_out, (hidden, cell) = self.lstm(sequence_output)
        
        # Concatenate final hidden states from both directions
        hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        
        # Classification
        return self.classifier(self.dropout(hidden))


class GRUModel(BaseModel):
    """GRU-based text classification model"""
    
    def __init__(self, pretrained_model, num_classes: int):
        super().__init__(pretrained_model, num_classes)
        
        # Bidirectional GRU
        self.gru = nn.GRU(
            self.hidden_size,
            self.hidden_size // 2,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=0.1
        )
        
        # Classification head
        self.classifier = nn.Linear(self.hidden_size, num_classes)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # Get transformer outputs
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        
        # Apply GRU
        gru_out, hidden = self.gru(sequence_output)
        
        # Concatenate final hidden states from both directions
        hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        
        # Classification
        return self.classifier(self.dropout(hidden))


class TransformerModel(BaseModel):
    """Pure transformer-based text classification model"""
    
    def __init__(self, pretrained_model, num_classes: int):
        super().__init__(pretrained_model, num_classes)
        
        # Additional transformer layer
        self.attention = nn.MultiheadAttention(
            self.hidden_size,
            num_heads=8,
            batch_first=True,
            dropout=0.1
        )
        
        # Pooling layer
        self.pooler = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Tanh()
        )
        
        # Classification head
        self.classifier = nn.Linear(self.hidden_size, num_classes)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # Get transformer outputs
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        
        # Apply additional attention
        if attention_mask is not None:
            attn_output, _ = self.attention(
                sequence_output,
                sequence_output,
                sequence_output,
                key_padding_mask=~attention_mask.bool()
            )
        else:
            attn_output, _ = self.attention(
                sequence_output,
                sequence_output,
                sequence_output
            )
        
        # Mean pooling
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).float()
            sum_embeddings = torch.sum(attn_output * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            pooled_output = sum_embeddings / sum_mask
        else:
            pooled_output = attn_output.mean(dim=1)
        
        # Apply pooler
        pooled_output = self.pooler(pooled_output)
        
        # Classification
        return self.classifier(self.dropout(pooled_output))


class CNNModel(BaseModel):
    """CNN-based text classification model"""
    
    def __init__(self, pretrained_model, num_classes: int):
        super().__init__(pretrained_model, num_classes)
        
        # Multiple kernel sizes for different n-grams
        self.kernel_sizes = [3, 4, 5]
        self.num_filters = 100
        
        # Convolutional layers
        self.convs = nn.ModuleList([
            nn.Conv1d(self.hidden_size, self.num_filters, kernel_size=k)
            for k in self.kernel_sizes
        ])
        
        # Classification head
        self.classifier = nn.Linear(
            len(self.kernel_sizes) * self.num_filters,
            num_classes
        )
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # Get transformer outputs
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        
        # Transpose for conv1d
        sequence_output = sequence_output.transpose(1, 2)
        
        # Apply convolutions
        conv_outputs = []
        for conv in self.convs:
            conv_out = F.relu(conv(sequence_output))
            pooled = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)
            conv_outputs.append(pooled)
        
        # Concatenate all conv outputs
        concatenated = torch.cat(conv_outputs, dim=1)
        
        # Classification
        return self.classifier(self.dropout(concatenated))


class AttentionModel(BaseModel):
    """Attention-based text classification model"""
    
    def __init__(self, pretrained_model, num_classes: int):
        super().__init__(pretrained_model, num_classes)
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.Tanh(),
            nn.Linear(self.hidden_size // 2, 1)
        )
        
        # Classification head
        self.classifier = nn.Linear(self.hidden_size, num_classes)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # Get transformer outputs
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        
        # Calculate attention weights
        attention_weights = self.attention(sequence_output)
        
        if attention_mask is not None:
            attention_weights = attention_weights.masked_fill(
                ~attention_mask.unsqueeze(-1).bool(),
                float('-inf')
            )
        
        attention_weights = F.softmax(attention_weights, dim=1)
        
        # Apply attention
        attended_output = (sequence_output * attention_weights).sum(dim=1)
        
        # Classification
        return self.classifier(self.dropout(attended_output))