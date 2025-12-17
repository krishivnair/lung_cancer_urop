# Hypergraph Neural Networks for Lung Cancer Detection

Research-grade implementation for Scopus-indexed publication.

## Project Structure
```
lung_cancer_hgnn/
├── src/                    # Source code modules
│   ├── config.py          # Configuration management
│   ├── preprocessing.py   # Medical image preprocessing
│   ├── hypergraph.py      # Hypergraph construction
│   ├── models.py          # HGNN architecture
│   ├── dataset.py         # Dataset loader
│   ├── early_stopping.py  # Early stopping
│   ├── trainer.py         # Training framework
│   ├── visualization.py   # Results visualization
│   └── utils.py           # Utility functions
├── notebooks/             # Jupyter notebooks
├── configs/               # Configuration files
├── requirements.txt       # Dependencies
└── README.md             # This file
```

## Installation
```bash
pip install -r requirements.txt
```

## Usage in Google Colab
```python
# Mount drive and add to path
from google.colab import drive
drive.mount('/content/drive')

import sys
sys.path.append('/content/drive/MyDrive/lung_cancer_hgnn')

# Import modules
from src import *

# Initialize
config = ExperimentConfig()
preprocessor = AdvancedPreprocessor()
model = HypergraphNeuralNetwork(in_channels=122)

# Train
trainer = HGNNTrainer(model, device, output_dir)
trainer.setup_training()
train_history, val_history = trainer.train(train_loader, val_loader, num_epochs=100)
```

## Features

- ✅ Research-grade preprocessing
- ✅ Multi-type hypergraph construction
- ✅ Deep residual HGNN architecture
- ✅ Mixed precision training
- ✅ Early stopping with best model restoration
- ✅ Comprehensive evaluation metrics
- ✅ Publication-ready visualizations
- ✅ Full reproducibility (seeds, configs)

## Citation
```bibtex
@article{your_paper_2025,
  title={Hypergraph Neural Networks for Lung Nodule Classification},
  author={Your Name},
  journal={Journal Name},
  year={2025}
}
```