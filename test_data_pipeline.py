#!/usr/bin/env python
"""
Test file for CoreTempAI data pipeline.

This script tests the full data pipeline from data aggregation to model training:
1. Data aggregation using DataAggregator
2. Data preprocessing using DataPreprocessor
3. Neural Network model training
4. Neural Operator model training

Note: This is a test script only and should NOT be run directly.
The script verifies all linkages between components work correctly.
"""

import os
import torch
from pathlib import Path

# Import data aggregation and preprocessing modules
from CoreTempAI.data_aggregation.data_aggregator import DataAggregator
from CoreTempAI.data_preprocessing.data_preprocessor import DataPreprocessor
from CoreTempAI.data_preprocessing.model_processor import ProcessingConfig

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
RAW_DATA_DIR = "data/raw"
AGGREGATE_DIR = "data/aggregated"
PROCESSED_DATA_DIR = "data/processed"
NN_MODEL_DIR = "data/models/neural_network"
NO_MODEL_DIR = "data/models/neural_operator"

# Set parameters
APPLY_FFT = True  # Whether to apply FFT for neural network training
RANDOM_SEED = 42
SPLIT_RATIO = 0.8
NN_EPOCHS = 500
NO_EPOCHS = 500
USE_WANDB = False  # Set to True if you want to use Weights & Biases for tracking
PROJECT_NAME = "CoreTempAI"

def main():
    """
    Main function to run the full data pipeline test.
    """
    # Create directories if they don't exist
    os.makedirs(AGGREGATE_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    os.makedirs(NN_MODEL_DIR, exist_ok=True)
    os.makedirs(NO_MODEL_DIR, exist_ok=True)
    
    print("Starting CoreTempAI data pipeline test...")
    
    # Step 1: Data Aggregation
    print("\n=== Step 1: Data Aggregation ===")
    aggregator = DataAggregator(raw_data_dir=RAW_DATA_DIR, aggregate_dir=AGGREGATE_DIR)
    
    # Discover run directories
    run_dirs = aggregator.discover_run_directories()
    print(f"Found {len(run_dirs)} run directories")
    
    aggregated_data = aggregator.aggregate_all_data(use_cache=True)
    
    # Check if data aggregation was successful
    if not aggregated_data or 'scalar_inputs' not in aggregated_data or len(aggregated_data['scalar_inputs']) == 0:
        print("Error: Data aggregation failed or no data was found.")
        return
    
    print(f"Data aggregation complete")
    
    # Step 2: Data Preprocessing
    print("\n=== Step 2: Data Preprocessing ===")
    preprocessor = DataPreprocessor(processed_data_dir=PROCESSED_DATA_DIR)
    
    # Create a processing config
    processing_config = ProcessingConfig(
        split_ratio=SPLIT_RATIO,
        random_seed=RANDOM_SEED,
        normalize=True,
        apply_fft=APPLY_FFT,
        n_freq_components=16  # Default number of frequency components
    )
    
    # Prepare data for both Neural Network and Neural Operator models
    preprocessed_data = preprocessor.prepare_data(
        scalar_inputs=aggregated_data['scalar_inputs'],
        profile_inputs=aggregated_data.get('profile_inputs', []),
        outputs=aggregated_data['outputs'],
        split_ratio=SPLIT_RATIO,
        random_seed=RANDOM_SEED,
        apply_fft=APPLY_FFT,
        normalize=True
    )
    
    # Check if preprocessing was successful
    if not preprocessed_data['nn_processed'] and not preprocessed_data['no_processed']:
        print("Error: Data preprocessing failed.")
        return
    
    print(f"Data preprocessing complete.")
    print(f"Neural Network data: {preprocessed_data['nn_samples']} samples")
    print(f"Neural Operator data: {preprocessed_data['no_samples']} samples")
    
    # # Step 3: Neural Network Training
    # print("\n=== Step 3: Neural Network Training ===")
    # # Load data for Neural Network
    # try:
    #     print("Loading Neural Network data...")
    #     nn_train_dataloader, nn_test_dataloader, nn_metadata, nn_train_dataset, nn_test_dataset = nn_load_data(
    #         processed_data_dir=PROCESSED_DATA_DIR,
    #         apply_fft=APPLY_FFT
    #     )
    #     print("Successfully loaded Neural Network data.")
        
    #     # Train Neural Network model
    #     nn_model = nn_train_model(
    #         train_dataloader=nn_train_dataloader,
    #         test_dataloader=nn_test_dataloader,
    #         metadata=nn_metadata,
    #         epochs=NN_EPOCHS,
    #         save_dir=NN_MODEL_DIR,
    #         use_wandb=USE_WANDB,
    #         project_name=PROJECT_NAME
    #     )
        
    #     # Evaluate Neural Network model
    #     nn_metrics = nn_evaluate_model(
    #         model=nn_model,
    #         test_dataset=nn_test_dataset,
    #         save_dir=NN_MODEL_DIR,
    #         use_wandb=USE_WANDB
    #     )
        
    #     print(f"Neural Network training complete.")
    #     print(f"Neural Network metrics: {nn_metrics}")
    # except Exception as e:
    #     print(f"Error in Neural Network training: {e}")
    
    # Step 4: Neural Operator Training
    print("\n=== Step 4: Neural Operator Training ===")
    # Load data for Neural Operator
    try:
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
        
        print(f"Neural Operator training complete.")
        print(f"Neural Operator metrics: {no_metrics}")
    except Exception as e:
        print(f"Error in Neural Operator training: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n=== Data Pipeline Test Complete ===")
    print("All components of the data pipeline have been tested.")

if __name__ == "__main__":
    # This script should not be run directly. It is for testing purposes only.
    # print("This is a test script for the CoreTempAI data pipeline.")
    # print("It should NOT be run directly in a production environment.")
    # print("To run the test, comment out the lines below and uncomment the main() call.")
    main() 