"""
Configuration Management - RESEARCH GRADE
"""

from dataclasses import dataclass, asdict
from typing import Tuple
import yaml
from pathlib import Path


@dataclass
class ExperimentConfig:
    """Research-grade configuration with advanced features."""
    
    # ==================== Experiment Metadata ====================
    experiment_name: str = "HGNN_LungCancer_MultiClass_v1.0"
    random_seed: int = 999
    project_name: str = "Duke_Lung_HGNN"
    
    # ==================== Data Configuration ====================
    base_path: str = "/content/drive/MyDrive/duke_lung_data"
    subset_name: str = "subset01"
    annotations_file: str = "DLCSD24_Annotations.csv"
    metadata_file: str = "DLCSD24_metadata_v1.1.csv"
    num_patients: int = 500  # INCREASED: Use more data
    train_split: float = 0.8
    
    # ==================== Model Architecture (KEEP COMPLEX) ====================
    hidden_channels: int = 256  # Keep high capacity
    num_classes: int = 2  # CHANGED
    num_layers: int = 4  # Keep deep
    num_heads: int = 4
    dropout: float = 0.4  # Slightly increased
    
    # ==================== Hypergraph Construction ====================
    k_neighbors: int = 8
    spatial_threshold: float = 50.0
    feature_similarity_threshold: float = 0.7
    
    # ==================== Training Configuration ====================
    batch_size: int = 8  # INCREASED: Better gradient estimates
    num_epochs: int = 150  # INCREASED: More training
    learning_rate: float = 0.0005  # REDUCED: More stable
    weight_decay: float = 5e-4  # INCREASED: More regularization
    patience: int = 25  # INCREASED: Give more time
    gradient_clip_norm: float = 1.0
    warmup_epochs: int = 5  # NEW: Learning rate warmup
    
    # ==================== Multi-Task Loss (REBALANCED) ====================
    alpha_classification: float = 1.0
    alpha_regression: float = 0.05  # REDUCED: Less emphasis on regression
    
    # ==================== Advanced Training Features ====================
    use_mixed_precision: bool = True
    use_class_weights: bool = True  # NEW: Handle imbalance
    use_focal_loss: bool = True  # NEW: Better for imbalanced data
    focal_loss_alpha: float = 0.10
    focal_loss_gamma: float = 3.0
    use_label_smoothing: bool = True  # NEW: Regularization
    label_smoothing_eps: float = 0.1
    gradient_accumulation_steps: int = 2  # NEW: Larger effective batch
    
    # ==================== Optimization ====================
    num_workers: int = 2
    pin_memory: bool = True
    
    # Learning rate scheduling
    cosine_t0: int = 20
    cosine_t_mult: int = 2
    cosine_eta_min: float = 1e-7
    lr_factor: float = 0.5
    lr_patience: int = 8
    plateau_factor: float = 0.5
    plateau_patience: int = 8
    
    # ==================== Preprocessing ====================
    target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    patch_size: Tuple[int, int, int] = (64, 64, 32)
    hu_window_center: int = -600
    hu_window_width: int = 1500
    
    # ==================== Augmentation ====================
    use_augmentation: bool = True
    augmentation_prob: float = 0.5
    
    # ==================== Device Configuration ====================
    device: str = "cuda"
    use_multi_gpu: bool = False
    
    # ==================== Logging & Checkpointing ====================
    log_interval: int = 10
    checkpoint_interval: int = 10
    save_best_only: bool = True
    
    # ==================== Version Control ====================
    code_version: str = "2.0.0"
    
    def save_config(self, path: str) -> None:
        """Save configuration to YAML file."""
        with open(path, 'w') as f:
            yaml.dump(asdict(self), f, default_flow_style=False)
    
    @classmethod
    def load_config(cls, path: str) -> 'ExperimentConfig':
        """Load configuration from YAML file."""
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)