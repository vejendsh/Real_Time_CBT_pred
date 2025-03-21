"""
Unified data preprocessor for CoreTempAI.

This module provides functionality to preprocess aggregated data for both
Neural Network and Neural Operator model architectures.
"""

import os
import time
from pathlib import Path
import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union, Optional, Any
from tqdm import tqdm
import torch.serialization

# Add safe globals for numpy arrays
torch.serialization.add_safe_globals(['numpy.core.multiarray._reconstruct', 'numpy.core.multiarray.scalar'])

from CoreTempAI.data_preprocessing.model_processor import NeuralNetworkProcessor, NeuralOperatorProcessor

class DataPreprocessor:
    """
    Unified data preprocessor for CoreTempAI models.
    
    This class is responsible for preparing data for both Neural Network and 
    Neural Operator model architectures based on previously aggregated data.
    """
    
    def __init__(self, processed_data_dir: str):
        """
        Initialize the data preprocessor.
        
        Args:
            processed_data_dir (str): Path to store processed data
        """
        self.processed_data_dir = Path(processed_data_dir)
            
        # Create processed directory if it doesn't exist
        os.makedirs(self.processed_data_dir, exist_ok=True)
        
        # Feature metadata
        self.feature_info = {}
        
    def prepare_data_from_file(self, aggregated_data_file: str, split_ratio: float = 0.8, 
                              random_seed: int = 42, apply_fft: bool = False, 
                              normalize: bool = True) -> Dict[str, Any]:
        """
        Prepare data for models using aggregated data file.
        
        Args:
            aggregated_data_file (str): Path to the aggregated data file
            split_ratio (float): Train/test split ratio
            random_seed (int): Random seed for reproducibility
            apply_fft (bool): Whether to apply FFT to neural network output data
            normalize (bool): Whether to normalize the data
            
        Returns:
            Dict[str, Any]: Dictionary with information about the processed data
        """
        # Load aggregated data
        print(f"Loading aggregated data from {aggregated_data_file}...")
        aggregated_data = torch.load(aggregated_data_file)
        
        # Extract data components
        scalar_inputs = aggregated_data.get('scalar_inputs', [])
        profile_inputs = aggregated_data.get('profile_inputs', [])
        outputs = aggregated_data.get('outputs', [])
        self.feature_info = aggregated_data.get('feature_info', {})
        
        # Check if we have data
        if not scalar_inputs or not outputs:
            raise ValueError("No input or output data found in the aggregated data file")
        
        print(f"Loaded {len(scalar_inputs)} samples from aggregated data file")
        
        # Prepare data for the selected model type
        return self.prepare_data(
            scalar_inputs=scalar_inputs,
            profile_inputs=profile_inputs,
            outputs=outputs,
            split_ratio=split_ratio,
            random_seed=random_seed,
            apply_fft=apply_fft,
            normalize=normalize
        )
    
    def prepare_data(self, scalar_inputs: List[Dict[str, float]], profile_inputs: List[Any], 
                    outputs: List[np.ndarray], split_ratio: float = 0.8, random_seed: int = 42, 
                    apply_fft: bool = False, normalize: bool = True) -> Dict[str, Any]:
        """
        Prepare data for both Neural Network and Neural Operator models.
        
        Args:
            scalar_inputs (List[Dict[str, float]]): List of scalar input dictionaries
            profile_inputs (List[Any]): List of profile inputs (DataFrames)
            outputs (List[np.ndarray]): List of output arrays
            split_ratio (float): Train/test split ratio
            random_seed (int): Random seed for reproducibility
            apply_fft (bool): Whether to apply FFT to neural network output data
            normalize (bool): Whether to normalize the data
            
        Returns:
            Dict[str, Any]: Dictionary with information about the processed data for both model types
        """
        result = {
            'nn_processed': False,
            'no_processed': False,
            'nn_samples': 0,
            'no_samples': 0,
            'output_dirs': {
                'base': str(self.processed_data_dir)
            }
        }
        
        try:
            # Process data for both model types
            
            # 1. Process data for Neural Network
            print("Processing data for Neural Network...")
            nn_processor = NeuralNetworkProcessor(self.processed_data_dir)
            
            # Prepare data for Neural Network
            try:
                # Prepare data
                from CoreTempAI.data_preprocessing.model_processor import ProcessingConfig
                nn_config = ProcessingConfig(
                    split_ratio=split_ratio,
                    random_seed=random_seed,
                    normalize=normalize,
                    apply_fft=apply_fft
                )
                
                # Prepare data
                nn_inputs, nn_outputs = nn_processor.prepare_data(
                    scalar_inputs=scalar_inputs,
                    profile_inputs=profile_inputs,
                    outputs=outputs,
                    config=nn_config
                )
                
                # Split data
                nn_data = nn_processor.split_data(
                    inputs=nn_inputs,
                    outputs=nn_outputs,
                    config=nn_config
                )
                
                # Create metadata
                nn_metadata = {
                    'feature_info': self.feature_info,
                    'apply_fft': apply_fft,
                    'normalize': normalize,
                    'split_ratio': split_ratio,
                    'random_seed': random_seed,
                    'total_samples': len(nn_inputs),
                    'input_dim': nn_inputs.shape[1] if hasattr(nn_inputs, 'shape') else len(nn_inputs[0]) if isinstance(nn_inputs, list) else None,
                    'output_dim': nn_outputs.shape[1] if hasattr(nn_outputs, 'shape') else len(nn_outputs[0]) if isinstance(nn_outputs, list) else None,
                }
                
                # Add normalization parameters for FFT
                norm_params = getattr(nn_processor, 'normalization_params', {})
                if apply_fft and norm_params:
                    # Make sure keys match what neural_network_training.py expects
                    if 'y_min' in norm_params and 'y_max' in norm_params:
                        nn_metadata['normalization_params'] = {
                            'Y_fft_min': norm_params['y_min'],
                            'Y_fft_max': norm_params['y_max'],
                            **norm_params  # Keep original keys too
                        }
                    else:
                        nn_metadata['normalization_params'] = norm_params
                
                # Save data
                nn_result = nn_processor.save_data(nn_data, nn_metadata)
                
                result['nn_processed'] = True
                result['nn_samples'] = nn_result['num_samples']['train'] + nn_result['num_samples']['test']
                result['output_dirs']['nn'] = nn_result['output_dir']
                
                print(f"Neural Network data processed and saved:")
                print(f"  Train samples: {nn_result['num_samples']['train']}")
                print(f"  Test samples: {nn_result['num_samples']['test']}")
                print(f"  Input dimension: {nn_result['input_dim']}")
                print(f"  Output dimension: {nn_result['output_dim']}")
                print(f"  Saved to: {nn_result['output_dir']}")
            except Exception as e:
                print(f"Error processing Neural Network data: {e}")
                import traceback
                traceback.print_exc()
            
            # 2. Process data for Neural Operator
            print("\nProcessing data for Neural Operator...")
            no_processor = NeuralOperatorProcessor(self.processed_data_dir)
            
            try:
                # Create processing config for neural operator
                from CoreTempAI.data_preprocessing.model_processor import ProcessingConfig
                no_config = ProcessingConfig(
                    split_ratio=split_ratio,
                    random_seed=random_seed,
                    normalize=normalize,
                    apply_fft=True,  # Always apply FFT for neural operator
                    n_freq_components=16  # Use a reasonable default
                )
                
                # Prepare data
                no_inputs, no_outputs = no_processor.prepare_data(
                    scalar_inputs=scalar_inputs,
                    profile_inputs=profile_inputs,
                    outputs=outputs,
                    config=no_config
                )
                
                # Split data
                no_data = no_processor.split_data(
                    inputs=no_inputs,
                    outputs=no_outputs,
                    config=no_config
                )
                
                # Create metadata
                no_metadata = {
                    'feature_info': no_processor.feature_info,
                    'normalize': normalize,
                    'split_ratio': split_ratio,
                    'random_seed': random_seed,
                    'apply_fft': True,
                    'n_freq_components': getattr(no_processor.feature_info, 'profile_freq_components', no_config.n_freq_components),
                    'profile_freq_components': getattr(no_processor.feature_info, 'profile_freq_components', no_config.n_freq_components),
                    'profile_input_length': getattr(no_processor.feature_info, 'profile_input_length', None),
                    'output_profile_freq_components': getattr(no_processor.feature_info, 'output_profile_freq_components', None),
                    'output_profile_length': getattr(no_processor.feature_info, 'output_profile_length', None),
                    'total_samples': len(no_inputs[0]) if isinstance(no_inputs, list) else len(no_inputs),
                }
                
                # Ensure feature_info has the fields neural_operator_training.py expects
                no_metadata['feature_info'] = {
                    'defined_scalar_params': getattr(no_processor.feature_info, 'defined_scalar_params', []),
                    'defined_profile_params': getattr(no_processor.feature_info, 'defined_profile_params', []),
                    # Keep original fields too
                    **{k: v for k, v in no_processor.feature_info.__dict__.items() 
                       if k not in ['defined_scalar_params', 'defined_profile_params']}
                }
                
                # Add output_norm_params structure expected by neural_operator_training.py
                norm_params = getattr(no_processor, 'normalization_params', {})
                no_metadata['output_norm_params'] = {
                    'fft': True,  # Always True for Neural Operator
                    # Include min/max if available
                    'min': norm_params.get('y_min', None),
                    'max': norm_params.get('y_max', None),
                    # Include the full normalization params too
                    **norm_params
                }
                
                # Save data
                no_result = no_processor.save_data(no_data, no_metadata)
                
                result['no_processed'] = True
                result['no_samples'] = no_result['num_samples']['train'] + no_result['num_samples']['test']
                result['output_dirs']['no'] = no_result['output_dir']
                
                print(f"Neural Operator data processed and saved:")
                print(f"  Train samples: {no_result['num_samples']['train']}")
                print(f"  Test samples: {no_result['num_samples']['test']}")
                if 'input_dims' in no_result:
                    print(f"  Input dimensions: {no_result['input_dims']}")
                else:
                    print(f"  Input dimension: {no_result.get('input_dim', 'unknown')}")
                print(f"  Output dimension: {no_result['output_dim']}")
                print(f"  Saved to: {no_result['output_dir']}")
            except Exception as e:
                print(f"Error processing Neural Operator data: {e}")
                import traceback
                traceback.print_exc()
            
            return result
                
        except Exception as e:
            print(f"Error in prepare_data: {e}")
            import traceback
            traceback.print_exc()
            return result
            
    def convert_to_tensors(self, scalar_inputs: List[Dict[str, float]], outputs: List[np.ndarray]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert input dictionaries and output arrays to tensors.
        
        Args:
            scalar_inputs (List[Dict[str, float]]): List of scalar input dictionaries
            outputs (List[np.ndarray]): List of output arrays
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Input and output tensors
        """
        # Get all unique feature names
        all_features = set()
        for d in scalar_inputs:
            all_features.update(d.keys())
        all_features = sorted(list(all_features))
        
        # Store feature names
        self.feature_info['scalar_features'] = all_features
        
        # Create empty input tensor
        X = torch.zeros((len(scalar_inputs), len(all_features)))
        
        # Fill input tensor with scalar values
        print(f"Converting {len(scalar_inputs)} scalar inputs to tensor...")
        for i, d in enumerate(tqdm(scalar_inputs, desc="Processing scalar inputs")):
            for j, feature in enumerate(all_features):
                X[i, j] = d.get(feature, 0.0)  # Default to 0.0 if feature not present
        
        # Process outputs
        print(f"Converting {len(outputs)} outputs to tensor...")
        filtered_outputs = [out for out in outputs if isinstance(out, np.ndarray) and len(out) > 0]
        Y = torch.stack([torch.tensor(out, dtype=torch.float32) for out in filtered_outputs])
        
        print(f"Created tensors: X shape={X.shape}, Y shape={Y.shape}")
        return X, Y