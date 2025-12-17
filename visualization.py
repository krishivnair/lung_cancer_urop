"""
Visualization & Results Analysis
Publication-ready figures and tables
"""

from typing import Dict, List, Any
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve


class ResultsVisualizer:
    """
    Comprehensive visualization suite for research paper figures.
    
    Args:
        output_dir: Directory to save figures
    """
    
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True, parents=True)
    
    def plot_training_curves(
        self,
        train_history: Dict[str, List[float]],
        val_history: Dict[str, List[float]],
        save_name: str = 'training_curves.png'
    ) -> None:
        """Plot training and validation curves."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Loss curves
        axes[0, 0].plot(train_history['loss'], label='Train', linewidth=2, color='steelblue')
        axes[0, 0].plot(val_history['loss'], label='Validation', linewidth=2, color='coral')
        axes[0, 0].set_title('Loss Curves', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy curves
        axes[0, 1].plot(train_history['accuracy'], label='Train', linewidth=2, color='steelblue')
        axes[0, 1].plot(val_history['accuracy'], label='Validation', linewidth=2, color='coral')
        axes[0, 1].set_title('Accuracy Curves', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # AUC curve
        axes[1, 0].plot(val_history['auc'], label='Validation AUC', 
                       color='green', linewidth=2)
        axes[1, 0].set_title('ROC-AUC Score', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('AUC')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # F1 Score
        axes[1, 1].plot(val_history['f1'], label='Validation F1', 
                       color='purple', linewidth=2)
        axes[1, 1].set_title('F1 Score', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('F1')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / save_name, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_confusion_matrix(
        self,
        cm: np.ndarray,
        class_names: List[str] = ['Benign', 'Malignant'],
        save_name: str = 'confusion_matrix.png'
    ) -> None:
        """Plot confusion matrix."""
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names,
                   cbar_kws={'label': 'Count'})
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        plt.savefig(self.output_dir / save_name, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_roc_curve(
        self,
        labels: np.ndarray,
        probs: np.ndarray,
        save_name: str = 'roc_curve.png'
    ) -> None:
        """Plot ROC curve."""
        fpr, tpr, _ = roc_curve(labels, probs)
        from sklearn.metrics import auc
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('Receiver Operating Characteristic (ROC) Curve', 
                 fontsize=14, fontweight='bold')
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / save_name, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_precision_recall_curve(
        self,
        labels: np.ndarray,
        probs: np.ndarray,
        save_name: str = 'pr_curve.png'
    ) -> None:
        """Plot Precision-Recall curve."""
        precision, recall, _ = precision_recall_curve(labels, probs)
        from sklearn.metrics import average_precision_score
        ap = average_precision_score(labels, probs)
        
        plt.figure(figsize=(10, 8))
        plt.plot(recall, precision, color='purple', lw=2,
                label=f'PR curve (AP = {ap:.3f})')
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curve', fontsize=14, fontweight='bold')
        plt.legend(loc="lower left", fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / save_name, dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_results_table(
        self,
        metrics_dict: Dict[str, Any],
        save_name: str = 'results_table.csv'
    ) -> None:
        """Generate LaTeX-ready results table."""
        df = pd.DataFrame([metrics_dict])
        
        print("\n" + "=" * 80)
        print("RESULTS TABLE (Copy for LaTeX)")
        print("=" * 80)
        print(df.to_latex(index=False, float_format="%.4f"))
        
        df.to_csv(self.output_dir / save_name, index=False)
        print(f"\n✓ Results table saved to {self.output_dir / save_name}")