"""
Hypergraph Neural Network Architecture
Deep HGNN with residual connections and attention
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import MessagePassing, global_mean_pool, global_max_pool


class HypergraphConv(MessagePassing):
    """
    Hypergraph Convolution Layer with message passing.
    
    Implements the operation:
        H^(l+1) = σ(D_v^(-1/2) H W D_e^(-1) H^T D_v^(-1/2) H^(l) Θ)
    
    Args:
        in_channels: Input feature dimensionality
        out_channels: Output feature dimensionality
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.3
    ) -> None:
        super().__init__(aggr='add', flow='source_to_target')
        self.lin1 = nn.Linear(in_channels, out_channels)
        self.lin2 = nn.Linear(out_channels, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.bn = nn.BatchNorm1d(out_channels)
        
    def forward(
        self,
        x: Tensor,
        hyperedge_index: Tensor,
        hyperedge_weight: Optional[Tensor] = None
    ) -> Tensor:
        """
        Forward pass through hypergraph convolution.
        
        Args:
            x: Node features [num_nodes, in_channels]
            hyperedge_index: Edge connectivity [2, num_edges]
            hyperedge_weight: Optional edge weights
            
        Returns:
            Transformed node features [num_nodes, out_channels]
        """
        x = self.lin1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.propagate(hyperedge_index, x=x, edge_weight=hyperedge_weight)
        
        x = self.lin2(x)
        x = self.bn(x)
        x = F.relu(x)
        
        return x
    
    def message(self, x_j: Tensor, edge_weight: Optional[Tensor]) -> Tensor:
        """Define message computation."""
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j


class ResidualHypergraphBlock(nn.Module):
    """
    Residual block for deep hypergraph networks.
    
    Args:
        channels: Feature dimensionality
        dropout: Dropout probability
    """
    
    def __init__(self, channels: int, dropout: float = 0.3) -> None:
        super().__init__()
        self.conv1 = HypergraphConv(channels, channels, dropout)
        self.conv2 = HypergraphConv(channels, channels, dropout)
        self.bn1 = nn.BatchNorm1d(channels)
        self.bn2 = nn.BatchNorm1d(channels)
        
    def forward(
        self,
        x: Tensor,
        hyperedge_index: Tensor,
        hyperedge_weight: Optional[Tensor] = None
    ) -> Tensor:
        """Forward with residual connection."""
        residual = x
        
        x = self.conv1(x, hyperedge_index, hyperedge_weight)
        x = self.bn1(x)
        x = F.relu(x)
        
        x = self.conv2(x, hyperedge_index, hyperedge_weight)
        x = self.bn2(x)
        
        x = x + residual  # Residual connection
        x = F.relu(x)
        
        return x


class HypergraphNeuralNetwork(nn.Module):
    """
    Advanced Hypergraph Neural Network for lung nodule classification.
    
    Architecture:
        - Input projection layer
        - Multiple residual HGNN blocks
        - Global pooling (mean + max)
        - Multi-task heads (classification + regression)
    
    Args:
        in_channels: Input feature dimension
        hidden_channels: Hidden layer dimension
        num_classes: Number of output classes
        num_layers: Number of HGNN layers
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 256,
        num_classes: int = 2,
        num_layers: int = 4,
        dropout: float = 0.3
    ) -> None:
        super().__init__()
        
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_classes = num_classes
        self.num_layers = num_layers
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Residual HGNN blocks
        self.conv_layers = nn.ModuleList([
            ResidualHypergraphBlock(hidden_channels, dropout)
            for _ in range(num_layers)
        ])
        
        # Global attention pooling
        self.global_attention = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.Tanh(),
            nn.Linear(hidden_channels // 2, 1)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.BatchNorm1d(hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, num_classes)
        )
        
        # Regression head
        self.regressor = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1),
            nn.Sigmoid()
        )
        
    def forward(
        self,
        x: Tensor,
        hyperedge_index: Tensor,
        batch: Tensor,
        hyperedge_weight: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward pass.
        
        Args:
            x: Node features [num_nodes, in_channels]
            hyperedge_index: Edge connectivity [2, num_edges]
            batch: Batch assignment [num_nodes]
            hyperedge_weight: Optional edge weights
            
        Returns:
            Tuple of (classification_logits, malignancy_scores)
        """
        # Initial projection
        x = self.input_proj(x)
        
        # Deep HGNN layers
        for conv_layer in self.conv_layers:
            x = conv_layer(x, hyperedge_index, hyperedge_weight)
        
        # Global pooling with attention
        attention_weights = torch.softmax(self.global_attention(x), dim=0)
        x_attended = global_mean_pool(x * attention_weights, batch)
        x_max = global_max_pool(x, batch)
        
        # Concatenate pooling strategies
        x_global = torch.cat([x_attended, x_max], dim=1)
        
        # Task heads
        logits = self.classifier(x_global)
        malignancy_score = self.regressor(x_global)
        
        return logits, malignancy_score