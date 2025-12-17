"""
Advanced Loss Functions for Medical Imaging
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    Reference: Lin et al., "Focal Loss for Dense Object Detection" (2017)
    """
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        """
        Args:
            inputs: Logits [N, C]
            targets: Class labels [N]
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


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Label Smoothing Cross Entropy for regularization.
    """
    def __init__(self, epsilon: float = 0.1, reduction: str = 'mean'):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction
    
    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        """
        Args:
            inputs: Logits [N, C]
            targets: Class labels [N]
        """
        num_classes = inputs.size(-1)
        log_probs = F.log_softmax(inputs, dim=-1)
        
        # One-hot with smoothing
        targets_one_hot = torch.zeros_like(inputs)
        targets_one_hot.scatter_(1, targets.unsqueeze(1), 1)
        targets_smooth = (1 - self.epsilon) * targets_one_hot + \
                        self.epsilon / num_classes
        
        loss = -(targets_smooth * log_probs).sum(dim=-1)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


def compute_class_weights(labels: Tensor, num_classes: int) -> Tensor:
    """
    Compute inverse frequency class weights.
    
    Args:
        labels: Class labels [N]
        num_classes: Total number of classes
    
    Returns:
        Class weights [num_classes]
    """
    # Count samples per class
    counts = torch.bincount(labels, minlength=num_classes).float()
    
    # Avoid division by zero
    counts = torch.clamp(counts, min=1.0)
    
    # Inverse frequency
    weights = 1.0 / counts
    
    # Normalize
    weights = weights / weights.sum() * num_classes
    
    return weights