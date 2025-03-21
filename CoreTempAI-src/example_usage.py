"""
Example usage of the CoreTempAI data aggregation and preprocessing system.

This script demonstrates how to use the new data aggregation and preprocessing system
to prepare data for machine learning models.
"""

import os
from pathlib import Path

# Import the necessary classes
from CoreTempAI.data_aggregation import DataAggregator
from CoreTempAI.data_preprocessing import DataPreprocessor

def main():
    """
    Example of using the data aggregation and preprocessing system.
    """
    # Define directories
    raw_data_dir = "data/raw"
    cache_dir = "data/cache"
    processed_dir = "data/processed"
    
    # Create directories if they don't exist
    os.makedirs(raw_data_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)
    
    # Step 1: Initialize the data aggregator
    print("\n=== Step 1: Initialize Data Aggregator ===")
    aggregator = DataAggregator(raw_data_dir, cache_dir)
    
    # Step 2: Discover available run directories
    print("\n=== Step 2: Discover Run Directories ===")
    run_dirs = aggregator.discover_run_directories()
    print(f"Found {len(run_dirs)} run directories")
    
    # Step 3: Select runs to process (e.g., using numerical range)
    print("\n=== Step 3: Select Runs to Process ===")
    selected_runs = aggregator.select_runs_by_range(start=1, end=3)
    print(f"Selected runs: {selected_runs}")
    
    # Step 4: Aggregate data from selected runs
    print("\n=== Step 4: Aggregate Data ===")
    aggregated_data = aggregator.aggregate_all_data(
        use_cache=True,
        run_selection=selected_runs
    )
    
    # Get the path to the aggregated data file
    aggregated_data_file = aggregated_data.get('data_file')
    print(f"Aggregated data saved to: {aggregated_data_file}")
    
    # Step 5: Initialize the data preprocessor
    print("\n=== Step 5: Initialize Data Preprocessor ===")
    preprocessor = DataPreprocessor(processed_dir)
    
    # Step 6: Prepare data for neural network model
    print("\n=== Step 6: Prepare Data for Neural Network ===")
    nn_result = preprocessor.prepare_data_from_file(
        aggregated_data_file=aggregated_data_file,
        split_ratio=0.8,
        random_seed=42,
        apply_fft=True,
        normalize=True,
        for_neural_operator=False
    )
    
    # Step 7: Prepare data for neural operator model (optional)
    print("\n=== Step 7: Prepare Data for Neural Operator (Optional) ===")
    no_result = preprocessor.prepare_data_from_file(
        aggregated_data_file=aggregated_data_file,
        split_ratio=0.8,
        random_seed=42,
        normalize=True,
        for_neural_operator=True
    )
    
    # Print results
    print("\n=== Results ===")
    print(f"Neural Network processed: {nn_result['nn_processed']}")
    print(f"Neural Operator processed: {no_result['no_processed']}")
    print(f"Number of NN samples: {nn_result['nn_samples']}")
    print(f"Number of NO samples: {no_result['no_samples']}")
    print(f"Output directories:")
    for k, v in nn_result['output_dirs'].items():
        print(f"  {k}: {v}")

if __name__ == "__main__":
    main() 