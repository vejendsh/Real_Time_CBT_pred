# CoreTempAI Project

A physics-based machine learning framework for predicting core body temperature using advanced neural architectures and computational fluid dynamics (CFD) data.

## Overview

CoreTempAI combines state-of-the-art machine learning techniques with physics-based simulations (CFD) to provide accurate predictions of core body temperature. The project leverages both traditional neural networks and advanced transformer-based architectures (neural operators) to process various input parameters including both scalar values and temporal profiles.

## Project Structure

```
CoreTempAI/
├── cases/              # CFD simulation case files
├── data/               # Training and validation datasets
├── CoreTempAI/         # Main package directory
│   ├── data_aggregation/    # Data collection and aggregation utilities
│   ├── data_generation/     # CFD simulation data generation scripts
│   ├── data_preprocessing/  # Data preprocessing and transformation
│   ├── input_parameters/    # Input parameter definitions and validation
│   ├── model/              # Machine learning model implementations
│   ├── training/           # Training scripts and utilities
│   └── utils/             # Helper functions and utilities
├── setup.py           # Package installation configuration
└── requirements.txt   # Project dependencies
```

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended for training)
- Ansys Fluent (for data generation)

### Installation Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/vejendsh/CoreTempAI.git
   cd CoreTempAI
   ```

2. Install the package and dependencies:
   ```bash
   pip install -e .
   ```

## Key Features

- **Multiple Model Architectures**:
  - Standard feedforward neural networks for scalar inputs
  - Neural Operator for processing temporal profiles
  - Physics-informed neural networks for enhanced prediction accuracy

- **Comprehensive Data Pipeline**:
  - Automated CFD simulation data generation
  - Robust data preprocessing and augmentation
  - Efficient data aggregation and management

- **Advanced Training Capabilities**:
  - Multi-GPU training support
  - Wandb integration for experiment tracking
  - Customizable training configurations

## Usage

For detailed usage instructions and examples, please refer to the documentation in the `CoreTempAI` directory.


## Development

The project is actively maintained and welcomes contributions. For development setup and guidelines, please refer to the documentation in the `CoreTempAI` directory.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Authors

- Sai Yeshwanth Vejendla (vejendsh@mail.uc.edu)

## Citation

If you use this project in your research, please cite:

```bibtex
@software{CoreTempAI2024,
  author = {Vejendla, Sai Yeshwanth},
  title = {CoreTempAI: Physics-based Machine Learning for Core Body Temperature Prediction},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/vejendsh/CoreTempAI.git}
}
``` 