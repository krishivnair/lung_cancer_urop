"""
Advanced Training Framework - Research Grade
Fixed validation logic with correct prediction threshold
"""

from typing import Dict, List, Tuple, Any
from pathlib import Path
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm.auto import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report

from .early_stopping import EarlyStopping
from .config import ExperimentConfig
from .losses import FocalLoss, LabelSmoothingCrossEntropy, compute_class_weights


class HGNNTrainer:
    """Research-grade training framework with advanced techniques."""
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        output_dir: Path,
        patience: int = 25,
        config: ExperimentConfig = None
    ) -> None:
        self.model = model.to(device)
        self.device = device
        self.output_dir = output_dir
        self.patience = patience
        self.config = config or ExperimentConfig()
        
        self.train_history: Dict[str, List[float]] = defaultdict(list)
        self.val_history: Dict[str, List[float]] = defaultdict(list)
        
        self.early_stopping = EarlyStopping(
            patience=patience,
            verbose=True,
            delta=0.0001,
            path=str(output_dir / 'best_model.pth')
        )
        
        self.alpha_cls = self.config.alpha_classification
        self.alpha_reg = self.config.alpha_regression
        
        if self.config.use_focal_loss:
            self.cls_criterion = FocalLoss(
                alpha=self.config.focal_loss_alpha,
                gamma=self.config.focal_loss_gamma
            )
        elif self.config.use_label_smoothing:
            self.cls_criterion = LabelSmoothingCrossEntropy(
                epsilon=self.config.label_smoothing_eps
            )
        else:
            self.cls_criterion = None
        
        self.class_weights = None
        self.current_epoch = 0
        
    def setup_training(
        self,
        lr: float = 0.001,
        weight_decay: float = 1e-4
    ) -> None:
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        self.scheduler_cosine = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=self.config.cosine_t0,
            T_mult=self.config.cosine_t_mult,
            eta_min=self.config.cosine_eta_min
        )
        
        self.scheduler_plateau = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=self.config.lr_factor,
            patience=self.config.lr_patience
        )
        
        if self.config.use_mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
    
    def _warmup_lr(self, epoch: int) -> None:
        if epoch < self.config.warmup_epochs:
            lr_scale = (epoch + 1) / self.config.warmup_epochs
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.config.learning_rate * lr_scale
    
    def compute_loss(
        self,
        logits: Tensor,
        malignancy_pred: Tensor,
        labels: Tensor,
        malignancy_true: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        if self.cls_criterion is not None:
            cls_loss = self.cls_criterion(logits, labels)
        elif self.config.use_class_weights and self.class_weights is not None:
            cls_loss = F.cross_entropy(logits, labels, weight=self.class_weights)
        else:
            cls_loss = F.cross_entropy(logits, labels)
        
        reg_loss = F.mse_loss(malignancy_pred.squeeze(), malignancy_true.float())
        total_loss = self.alpha_cls * cls_loss + self.alpha_reg * reg_loss
        
        return total_loss, cls_loss, reg_loss
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train model for one epoch with advanced techniques."""
        self.model.train()
        total_loss = 0.0
        total_cls_loss = 0.0
        total_reg_loss = 0.0
        all_preds: List[int] = []
        all_labels: List[int] = []
        
        if self.config.use_class_weights and self.class_weights is None:
            all_train_labels = []
            for batch in train_loader:
                if batch is None:
                    continue
                
                if batch.y.dim() == 0:
                    all_train_labels.append(batch.y.item())
                elif batch.y.size(0) == batch.x.size(0):
                    unique_batch_ids = torch.unique(batch.batch, sorted=True)
                    for graph_id in unique_batch_ids:
                        graph_mask = (batch.batch == graph_id)
                        all_train_labels.append(batch.y[graph_mask][0].item())
                else:
                    all_train_labels.extend(batch.y.cpu().numpy().tolist())
            
            # ======================================================================
            # CHANGE: AGGRESSIVE CLASS WEIGHTING
            # ======================================================================
            all_train_labels_tensor = torch.tensor(all_train_labels, device=self.device)
            
            # Count samples per class
            class_counts = torch.bincount(all_train_labels_tensor, minlength=self.config.num_classes)
            total_samples = class_counts.sum().float()
            
            # Inverse frequency with POWER SCALING (more aggressive)
            self.class_weights = (total_samples / (class_counts.float() + 1.0)) ** 2.0  # ← SQUARED
            
            # Normalize
            self.class_weights = self.class_weights / self.class_weights.sum() * self.config.num_classes
            self.class_weights = self.class_weights.to(self.device)
            
            print(f"  Class counts: {class_counts.cpu().numpy()}")
            print(f"  Class weights (AGGRESSIVE): {self.class_weights.cpu().numpy()}")
            # ======================================================================
        
        self.optimizer.zero_grad()
        pbar = tqdm(train_loader, desc="Training", leave=False)
        
        for batch_idx, batch in enumerate(pbar):
            if batch is None:
                continue
            
            batch = batch.to(self.device)
            
            if batch.y.dim() == 0:
                labels_for_loss = batch.y.unsqueeze(0)
            elif batch.y.size(0) == batch.x.size(0):
                unique_batch_ids = torch.unique(batch.batch, sorted=True)
                labels_for_loss = torch.stack([
                    batch.y[batch.batch == graph_id][0]
                    for graph_id in unique_batch_ids
                ])
            else:
                labels_for_loss = batch.y
            
            malignancy_true = labels_for_loss.float() / (self.config.num_classes - 1)
            
            if self.config.use_mixed_precision:
                with torch.cuda.amp.autocast():
                    logits, malignancy_pred = self.model(
                        batch.x, batch.edge_index, batch.batch
                    )
                    loss, cls_loss, reg_loss = self.compute_loss(
                        logits, malignancy_pred, labels_for_loss, malignancy_true
                    )
                    loss = loss / self.config.gradient_accumulation_steps
                
                self.scaler.scale(loss).backward()
                
                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        max_norm=self.config.gradient_clip_norm
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                logits, malignancy_pred = self.model(
                    batch.x, batch.edge_index, batch.batch
                )
                loss, cls_loss, reg_loss = self.compute_loss(
                    logits, malignancy_pred, labels_for_loss, malignancy_true
                )
                loss = loss / self.config.gradient_accumulation_steps
                loss.backward()
                
                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        max_norm=self.config.gradient_clip_norm
                    )
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            
            total_loss += loss.item() * self.config.gradient_accumulation_steps
            total_cls_loss += cls_loss.item()
            total_reg_loss += reg_loss.item()
            
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            labels = labels_for_loss.cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'cls': f'{cls_loss.item():.4f}'
            })
        
        avg_loss = total_loss / len(train_loader)
        avg_cls_loss = total_cls_loss / len(train_loader)
        avg_reg_loss = total_reg_loss / len(train_loader)
        accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
        
        return {
            'loss': avg_loss,
            'cls_loss': avg_cls_loss,
            'reg_loss': avg_reg_loss,
            'accuracy': accuracy
        }
    
    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Dict[str, Any]:
        """Validate model with binary classification metrics."""
        self.model.eval()
        total_loss = 0.0
        all_preds: List[int] = []
        all_probs: List[np.ndarray] = []
        all_labels: List[int] = []
        
        for batch in tqdm(val_loader, desc="Validating", leave=False):
            if batch is None:
                continue
            
            batch = batch.to(self.device)
            
            if batch.y.dim() == 0:
                labels_for_loss = batch.y.unsqueeze(0)
            elif batch.y.size(0) == batch.x.size(0):
                unique_batch_ids = torch.unique(batch.batch, sorted=True)
                labels_for_loss = torch.stack([
                    batch.y[batch.batch == graph_id][0]
                    for graph_id in unique_batch_ids
                ])
            else:
                labels_for_loss = batch.y
            
            malignancy_true = labels_for_loss.float() / (self.config.num_classes - 1)
            
            logits, malignancy_pred = self.model(
                batch.x, batch.edge_index, batch.batch
            )
            
            loss, _, _ = self.compute_loss(
                logits, malignancy_pred, labels_for_loss, malignancy_true
            )
            
            total_loss += loss.item()
            
            probs = F.softmax(logits, dim=1).cpu().numpy()
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            labels = labels_for_loss.cpu().numpy()
            
            all_preds.extend(preds.tolist())
            all_probs.extend(probs.tolist())
            all_labels.extend(labels.tolist())
        
        avg_loss = total_loss / len(val_loader)
        all_preds_arr = np.array(all_preds)
        all_probs_arr = np.array(all_probs)
        all_labels_arr = np.array(all_labels)
        
        accuracy = (all_preds_arr == all_labels_arr).mean()
        
        # Multi-class metrics (FIXED FOR BINARY)
        try:
            if self.config.num_classes == 2:
                unique_labels = np.unique(all_labels_arr)
                if len(unique_labels) > 1:
                    auc = roc_auc_score(all_labels_arr, all_probs_arr[:, 1])
                else:
                    auc = 0.0
            else:
                from sklearn.preprocessing import label_binarize
                y_bin = label_binarize(all_labels_arr, classes=range(self.config.num_classes))
                auc = roc_auc_score(y_bin, all_probs_arr, average='weighted', multi_class='ovr')
        except Exception as e:
            print(f"⚠️ AUC calculation failed: {str(e)}")
            auc = 0.0
        
        f1 = f1_score(all_labels_arr, all_preds_arr, average='weighted', zero_division=0)
        precision = precision_score(all_labels_arr, all_preds_arr, average='weighted', zero_division=0)
        recall = recall_score(all_labels_arr, all_preds_arr, average='weighted', zero_division=0)
        
        cm = confusion_matrix(all_labels_arr, all_preds_arr)
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'auc': auc,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'confusion_matrix': cm,
            'all_probs': all_probs_arr,
            'all_labels': all_labels_arr
        }
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
        tensorboard_writer: SummaryWriter = None
    ) -> Tuple[Dict[str, List[float]], Dict[str, List[float]]]:
        """Complete training loop with warmup."""
        print("\n" + "=" * 80)
        print("STARTING RESEARCH-GRADE TRAINING")
        print("=" * 80)
        print(f"Multi-class: {self.config.num_classes} classes")
        print(f"Focal Loss: {self.config.use_focal_loss}")
        print(f"Label Smoothing: {self.config.use_label_smoothing}")
        print(f"Class Weights: {self.config.use_class_weights}")
        print("=" * 80)
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 40)
            
            self._warmup_lr(epoch)
            
            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.validate(val_loader)
            
            if epoch >= self.config.warmup_epochs:
                self.scheduler_cosine.step()
                self.scheduler_plateau.step(val_metrics['loss'])
            
            for key, value in train_metrics.items():
                self.train_history[key].append(value)
            
            for key, value in val_metrics.items():
                if key not in ['confusion_matrix', 'all_probs', 'all_labels']:
                    self.val_history[key].append(value)
            
            if tensorboard_writer:
                tensorboard_writer.add_scalar('Loss/train', train_metrics['loss'], epoch)
                tensorboard_writer.add_scalar('Loss/val', val_metrics['loss'], epoch)
                tensorboard_writer.add_scalar('Accuracy/train', train_metrics['accuracy'], epoch)
                tensorboard_writer.add_scalar('Accuracy/val', val_metrics['accuracy'], epoch)
                tensorboard_writer.add_scalar('F1/val', val_metrics['f1'], epoch)
            
            print(f"\nTrain Loss: {train_metrics['loss']:.4f} | Train Acc: {train_metrics['accuracy']:.4f}")
            print(f"Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['accuracy']:.4f}")
            print(f"Val AUC: {val_metrics['auc']:.4f} | Val F1: {val_metrics['f1']:.4f}")
            print(f"Val Precision: {val_metrics['precision']:.4f} | Val Recall: {val_metrics['recall']:.4f}")
            
            self.early_stopping(
                val_metrics['loss'],
                self.model,
                self.optimizer,
                epoch,
                val_metrics
            )
            
            if self.early_stopping.early_stop:
                print(f"\n🛑 Early stopping triggered after {epoch + 1} epochs")
                break
            
            if (epoch + 1) % self.config.checkpoint_interval == 0:
                checkpoint_path = self.output_dir / f'checkpoint_epoch_{epoch + 1}.pth'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                }, checkpoint_path)
                print(f"✓ Checkpoint saved: {checkpoint_path.name}")
        
        print("\n" + "=" * 80)
        print("LOADING BEST MODEL")
        print("=" * 80)
        self.model = self.early_stopping.load_best_model(self.model)
        
        return self.train_history, self.val_history