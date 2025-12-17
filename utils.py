"""
Utility Functions
Helper functions for setup and reproducibility
"""

import random
import os
import numpy as np
import torch
from pathlib import Path
from typing import Tuple


def set_global_seed(seed: int = 42) -> None:
    """
    Set all random seeds for full reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set environment variables for additional determinism
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    
    # PyTorch 1.8+ deterministic algorithms
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except:
        pass  # Older PyTorch versions


def setup_directories(base_path: Path, experiment_name: str) -> Tuple[Path, Path, Path, Path, Path]:
    """
    Create output directory structure.
    
    Args:
        base_path: Base data directory
        experiment_name: Name of experiment
        
    Returns:
        Tuple of (output_path, models_path, results_path, logs_path, config_path)
    """
    output_path = base_path / "outputs" / experiment_name
    models_path = output_path / "models"
    results_path = output_path / "results"
    logs_path = output_path / "logs"
    config_path = output_path / "config"
    checkpoints_path = output_path / "checkpoints"
    
    for path in [output_path, models_path, results_path, logs_path, config_path, checkpoints_path]:
        path.mkdir(exist_ok=True, parents=True)
    
    return output_path, models_path, results_path, logs_path, config_path


def worker_init_fn(worker_id: int, seed: int = 42) -> None:
    """
    Initialize worker for DataLoader with unique seed.
    
    Args:
        worker_id: Worker process ID
        seed: Base random seed
    """
    worker_seed = seed + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)