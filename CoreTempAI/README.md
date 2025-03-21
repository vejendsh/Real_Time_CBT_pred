# CoreTempAI Package Documentation

This package provides the core functionality for temperature prediction using physics-based neural network and operator architectures.

## Project Structure

- `config.py`: Configuration parameters for models and simulations
- `example_usage.py`: Example script demonstrating the full data pipeline
- `model/`: Neural network model implementations
  - `neural_network_model.py`: Standard neural network model
  - `neural_operator_model.py`: Neural Operator model implementation
- `training/`: Training utilities
  - `neural_network_training.py`: Neural network training utilities
  - `neural_operator_training.py`: Neural operator training utilities
- `data_preprocessing/`: Data preprocessing utilities
  - `data_preprocessor.py`: Unified preprocessing for all model types
  - `model_processor.py`: Specialized processors for different models
- `data_aggregation/`: Data collection and organization utilities
  - `data_aggregator.py`: Main aggregator interface
  - `aggregator_base.py`: Base class for all aggregators
  - `default_aggregator.py`: Standard data aggregation
  - `custom_aggregator.py`: Custom data format handling
- `data_generation/`: Utilities for generating training data using Ansys Fluent
  - `data_generator.py`: Script for generating simulation data
  - `set_params.py`: Parameter configuration utilities
  - `run_case.py`: Simulation execution utilities
- `input_parameters/`: Simulation parameter definitions
  - `input_parameters.py`: Defines scalar and profile parameters with their ranges
- `utils/`: Utility functions
  - `utils.py`: General utility functions

## Package Components

### Input Parameters (`input_parameters/`)

The `input_parameters` module defines the simulation parameters used throughout the package:

#### Scalar Parameters
- Constant values used throughout the simulation (e.g., metabolic rates, ambient temperature)
- Each parameter has a defined range: `[min_value, max_value, "units"]`
- Example: `"T_amb": [300, 310, "K"]` (ambient temperature)

#### Profile Parameters
- Time-varying parameters that change during simulation (e.g., heart rate)
- Similar format to scalar parameters: `[min_value, max_value, "units"]`
- Example: `"HR": [3600, 7200, "s^-1"]` (heart rate profile)

Parameters can be customized by modifying values within their defined ranges, which allows researchers to explore different physiological conditions without modifying the core code. The system automatically samples from these ranges to generate diverse simulation scenarios.

### Data Processing

#### Data Generation (`data_generation/`)
- CFD simulation automation using Ansys Fluent
- Parameter space sampling
- Simulation monitoring and validation
- Implementation available in `data_generator.py`, `set_params.py`, and `run_case.py`

The training data is generated using Ansys Fluent, a computational fluid dynamics (CFD) solver. The `data_generator.py` script automates the process of running multiple simulations with different input parameters to generate a dataset for training the models. This enables rapid creation of synthetic training data without the need for expensive experimental setups.

#### Data Aggregation (`data_aggregation/`)
- Unified system for collecting and organizing data from multiple simulation runs
- Supports various simulation formats through custom and default aggregators
- Cache mechanism for faster data access
- Utilities for selecting and filtering runs

#### Data Preprocessing (`data_preprocessing/`)
- Converts aggregated data into training-ready formats
- Supports both neural network and neural operator models
- Features for normalization, FFT transformation, and data splitting
- Specialized processors for different model architectures

### Models

#### Neural Network (`model/neural_network_model.py`)
- Standard feedforward neural network for temperature prediction
- Flexible architecture that adapts to input parameters
- Support for custom hidden layer configurations

#### Neural Operator (`model/neural_operator_model.py`)
- Advanced operator-based architecture for spatiotemporal predictions
- Transformer-based implementation with multi-head attention
- Supports processing of both scalar parameters and temporal profiles

The neural operator model extends beyond traditional neural networks by incorporating attention mechanisms that can process both scalar parameters and temporal profiles. This architecture is particularly effective for capturing complex spatiotemporal patterns in the data, leading to more accurate predictions of temperature distributions over time.

### Training (`training/`)

#### Neural Network Training (`neural_network_training.py`)
- Comprehensive training pipeline for neural network models
- Support for early stopping, learning rate scheduling, and model checkpointing
- Evaluation and visualization utilities for model assessment

#### Neural Operator Training (`neural_operator_training.py`)
- Specialized training utilities for operator-based models
- Advanced optimization strategies with gradient accumulation
- Customizable training configurations via NeuralOperatorConfig 


## Workflow

The complete workflow consists of five main steps:

1. **Input Parameter Definition**: Configure simulation parameters with their allowed ranges
2. **Data Generation**: Generate simulation data using CFD tools and defined parameters
3. **Data Aggregation**: Collect and organize raw data from multiple simulations
4. **Data Preprocessing**: Transform aggregated data into model-ready formats
5. **Model Training and Evaluation**: Train models on preprocessed data and evaluate performance

## Usage Example

The `example_usage.py` and 'test_data_pipeline.py' file demonstrates how to use the CoreTempAI data pipeline:

```python
from CoreTempAI.data_aggregation import DataAggregator
from CoreTempAI.data_preprocessing import DataPreprocessor

# Define directories
raw_data_dir = "data/raw"
cache_dir = "data/cache"
processed_dir = "data/processed"

# Initialize the data aggregator
aggregator = DataAggregator(raw_data_dir, cache_dir)

# Discover available run directories
run_dirs = aggregator.discover_run_directories()

# Select runs to process
selected_runs = aggregator.select_runs_by_range(start=1, end=3)

# Aggregate data from selected runs
aggregated_data = aggregator.aggregate_all_data(
    use_cache=True,
    run_selection=selected_runs
)

# Get the path to the aggregated data file
aggregated_data_file = aggregated_data.get('data_file')

# Initialize the data preprocessor
preprocessor = DataPreprocessor(processed_dir)

# Prepare data for neural network model
nn_result = preprocessor.prepare_data_from_file(
    aggregated_data_file=aggregated_data_file,
    split_ratio=0.8,
    random_seed=42,
    apply_fft=True,
    normalize=True,
    for_neural_operator=False
)

# Prepare data for neural operator model
no_result = preprocessor.prepare_data_from_file(
    aggregated_data_file=aggregated_data_file,
    split_ratio=0.8,
    random_seed=42,
    normalize=True,
    for_neural_operator=True
)
```

### Model Training and Evaluation

The following example demonstrates how to train and evaluate both model types:

```python
# Import training modules
from CoreTempAI.training.neural_network_training import (
    load_data_from_preprocessor as nn_load_data,
    train_model as nn_train_model,
    evaluate_model as nn_evaluate_model
)
from CoreTempAI.training.neural_operator_training import (
    load_data_from_preprocessor as no_load_data,
    train_model as no_train_model,
    evaluate_model as no_evaluate_model
)

# Set paths
PROCESSED_DATA_DIR = "data/processed"
NN_MODEL_DIR = "data/models/neural_network"
NO_MODEL_DIR = "data/models/neural_operator"

# Set parameters
APPLY_FFT = True
RANDOM_SEED = 42
NN_EPOCHS = 500
NO_EPOCHS = 500
USE_WANDB = False  # Set to True to use Weights & Biases for tracking
PROJECT_NAME = "CoreTempAI"

# Neural Network Training
print("Loading Neural Network data...")
nn_train_dataloader, nn_test_dataloader, nn_metadata, nn_train_dataset, nn_test_dataset = nn_load_data(
    processed_data_dir=PROCESSED_DATA_DIR,
    apply_fft=APPLY_FFT
)

# Train Neural Network model
nn_model = nn_train_model(
    train_dataloader=nn_train_dataloader,
    test_dataloader=nn_test_dataloader,
    metadata=nn_metadata,
    epochs=NN_EPOCHS,
    save_dir=NN_MODEL_DIR,
    use_wandb=USE_WANDB,
    project_name=PROJECT_NAME
)

# Evaluate Neural Network model
nn_metrics = nn_evaluate_model(
    model=nn_model,
    test_dataset=nn_test_dataset,
    save_dir=NN_MODEL_DIR,
    use_wandb=USE_WANDB
)

print(f"Neural Network metrics: {nn_metrics}")

# Neural Operator Training
no_train_dataset, no_test_dataset, no_metadata = no_load_data(
    processed_data_dir=PROCESSED_DATA_DIR,
    apply_fft=APPLY_FFT
)

# Train Neural Operator model
no_model = no_train_model(
    train_dataset=no_train_dataset,
    test_dataset=no_test_dataset,
    metadata=no_metadata,
    epochs=NO_EPOCHS,
    batch_size=32,
    save_dir=NO_MODEL_DIR,
    use_wandb=USE_WANDB,
    project_name=PROJECT_NAME
)

# Evaluate Neural Operator model
no_metrics = no_evaluate_model(
    model=no_model,
    test_dataset=no_test_dataset,
    save_dir=NO_MODEL_DIR,
    use_wandb=USE_WANDB
)

print(f"Neural Operator metrics: {no_metrics}")
```

## Configuration

The project uses a centralized configuration system. You can customize parameters by modifying the config objects:

```python
from CoreTempAI.config import NeuralOperatorConfig

# Create and customize config
config = NeuralOperatorConfig()
config.learning_rate = 0.0005
config.batch_size = 64
```

## License

This project is licensed under the MIT License - see the LICENSE file for details. 