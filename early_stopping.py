"""
Early Stopping Implementation
Prevents overfitting with best model restoration
"""

from typing import Optional, Dict, Any
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer


class EarlyStopping:
    """
    Early stopping to prevent overfitting.
    
    Monitors validation loss and stops training when no improvement
    is observed for a specified number of epochs.
    
    Args:
        patience: Number of epochs to wait for improvement
        verbose: Print messages
        delta: Minimum change to qualify as improvement
        path: Checkpoint save path
    """
    
    def __init__(
        self,
        patience: int = 15,
        verbose: bool = True,
        delta: float = 0.0001,
        path: str = 'checkpoint.pt'
    ) -> None:
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score: Optional[float] = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path
        self.best_model_state: Optional[Dict[str, Any]] = None
        
    def __call__(
        self,
        val_loss: float,
        model: nn.Module,
        optimizer: Optimizer,
        epoch: int,
        metrics: Dict[str, float]
    ) -> None:
        """
        Check if early stopping criteria met.
        
        Args:
            val_loss: Current validation loss
            model: Model to save
            optimizer: Optimizer to save
            epoch: Current epoch
            metrics: Dictionary of metrics
        """
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, optimizer, epoch, metrics)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, optimizer, epoch, metrics)
            self.counter = 0
            
    def save_checkpoint(
        self,
        val_loss: float,
        model: nn.Module,
        optimizer: Optimizer,
        epoch: int,
        metrics: Dict[str, float]
    ) -> None:
        """Save model checkpoint."""
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} → {val_loss:.6f}). Saving model...')
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'metrics': metrics,
            'best_score': self.best_score
        }
        torch.save(checkpoint, self.path)
        self.val_loss_min = val_loss
        self.best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    
    def load_best_model(self, model: nn.Module) -> nn.Module:
        """Load best model state."""
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)
            if self.verbose:
                print(f'Loaded best model with validation loss: {self.val_loss_min:.6f}')
        return model