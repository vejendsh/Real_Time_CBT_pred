"""
Model data processors for CoreTempAI.

This module provides functionality to process data for different model architectures.
"""

from abc import ABC, abstractmethod
from pathlib import Path
import os
import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union, Optional, Any, NamedTuple, TypeVar, Generic
from tqdm import tqdm
import torch.serialization
import logging
from contextlib import contextmanager
from functools import wraps
from dataclasses import dataclass, field
from datetime import datetime
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add safe globals for numpy arrays
torch.serialization.add_safe_globals(['numpy.core.multiarray._reconstruct', 'numpy.core.multiarray.scalar'])

# Type variables for generic types
T = TypeVar('T')
InputT = TypeVar('InputT')
OutputT = TypeVar('OutputT')

@dataclass
class ProcessingConfig:
    """Configuration for data processing."""
    split_ratio: float = 0.8
    random_seed: int = 42
    normalize: bool = True
    apply_fft: bool = True
    n_freq_components: int = 16
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class FeatureInfo:
    """Information about features in the dataset."""
    scalar_features: List[str] = field(default_factory=list)
    profile_features: List[str] = field(default_factory=list)
    input_dim: Optional[int] = None
    output_dim: Optional[int] = None
    feature_ranges: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    # Add aliases for compatibility with both training modules
    defined_scalar_params: List[str] = field(default_factory=list)
    defined_profile_params: List[str] = field(default_factory=list)
    profile_freq_components: Optional[int] = None
    output_profile_freq_components: Optional[int] = None
    profile_input_length: Optional[int] = None
    output_profile_length: Optional[int] = None

@dataclass
class ProcessingMetadata:
    """Metadata about the processing operation."""
    feature_info: FeatureInfo
    config: ProcessingConfig
    total_samples: int
    train_samples: int
    test_samples: int
    processing_time: float
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class ProcessingResult(Generic[InputT, OutputT]):
    """Result of a data processing operation."""
    train_inputs: InputT
    train_outputs: OutputT
    test_inputs: InputT
    test_outputs: OutputT
    train_indices: torch.Tensor
    test_indices: torch.Tensor
    metadata: ProcessingMetadata
    output_dir: Path

class DataProcessingError(Exception):
    """Base exception for data processing errors."""
    pass

class ValidationError(DataProcessingError):
    """Exception raised when data validation fails."""
    pass

class NormalizationError(DataProcessingError):
    """Exception raised when data normalization fails."""
    pass

class SaveError(DataProcessingError):
    """Exception raised when saving data fails."""
    pass

def log_operation(operation_name: str):
    """
    Decorator to log operation execution and handle errors.
    
    Args:
        operation_name (str): Name of the operation being logged
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger.info(f"Starting {operation_name}...")
            try:
                result = func(*args, **kwargs)
                logger.info(f"Completed {operation_name} successfully")
                return result
            except Exception as e:
                logger.error(f"Error during {operation_name}: {str(e)}")
                raise
        return wrapper
    return decorator

@contextmanager
def error_context(operation_name: str):
    """
    Context manager for handling errors during operations.
    
    Args:
        operation_name (str): Name of the operation being performed
    """
    try:
        logger.info(f"Starting {operation_name}...")
        yield
        logger.info(f"Completed {operation_name} successfully")
    except Exception as e:
        logger.error(f"Error during {operation_name}: {str(e)}")
        raise

class BaseDataProcessor(ABC, Generic[InputT, OutputT]):
    """
    Base class for all data processors with common functionality.
    """
    
    def __init__(self, processed_data_dir: Path):
        """
        Initialize the base processor.
        
        Args:
            processed_data_dir (Path): Path to the processed data directory
        """
        self.processed_data_dir = processed_data_dir
        self.output_dir = processed_data_dir / self.__class__.__name__.lower().replace("processor", "")
        os.makedirs(self.output_dir, exist_ok=True)
        self.normalization_params: Dict[str, torch.Tensor] = {}
        self.feature_info = FeatureInfo()
        logger.info(f"Initialized {self.__class__.__name__} with output directory: {self.output_dir}")
    
    def validate_inputs(self, scalar_inputs: List[Dict[str, float]], profile_inputs: List[Any], outputs: List[np.ndarray]) -> None:
        """
        Validate input data.
        
        Args:
            scalar_inputs (List[Dict[str, float]]): List of scalar inputs
            profile_inputs (List[Any]): List of profile inputs
            outputs (List[np.ndarray]): List of outputs
            
        Raises:
            ValidationError: If inputs are invalid
        """
        with error_context("input validation"):
            if not scalar_inputs or not outputs:
                raise ValidationError("No input or output data provided")
            
            # Validate outputs
            valid_outputs = [out for out in outputs if isinstance(out, np.ndarray) and len(out) > 0]
            if not valid_outputs:
                raise ValidationError("No valid output arrays found")
                
            # Check if all arrays have the same length
            output_lengths = [len(out) for out in valid_outputs]
            if len(set(output_lengths)) > 1:
                raise ValidationError(f"Output arrays have different lengths: {set(output_lengths)}. All outputs must have the same length.")
    
    @log_operation("tensor normalization")
    def normalize_tensor(self, tensor: torch.Tensor, name: str = "") -> torch.Tensor:
        """
        Normalize a tensor using min-max normalization.
        
        Args:
            tensor (torch.Tensor): Tensor to normalize
            name (str): Name of the tensor for storing normalization params
            
        Returns:
            torch.Tensor: Normalized tensor
            
        Raises:
            NormalizationError: If normalization fails
        """
        try:
            min_vals, _ = torch.min(tensor, dim=0)
            max_vals, _ = torch.max(tensor, dim=0)
            
            # Handle case where min_vals equals max_vals (avoid division by zero)
            eps = 1e-8
            diff = max_vals - min_vals
            diff[diff == 0] = eps
            
            normalized = (tensor - min_vals) / diff
            
            # Store normalization parameters
            if name:
                self.normalization_params[f'{name}_min'] = min_vals
                self.normalization_params[f'{name}_max'] = max_vals
                
                # Store a single range value for metadata if possible, otherwise skip
                if min_vals.numel() == 1 and max_vals.numel() == 1:
                    self.feature_info.feature_ranges[name] = (min_vals.item(), max_vals.item())
                else:
                    # Just store the first element's range as representative sample
                    min_sample = min_vals.flatten()[0].item() if min_vals.numel() > 0 else 0.0
                    max_sample = max_vals.flatten()[0].item() if max_vals.numel() > 0 else 1.0
                    self.feature_info.feature_ranges[name] = (min_sample, max_sample)
            
            return normalized
        except Exception as e:
            raise NormalizationError(f"Failed to normalize tensor {name}: {str(e)}")
    
    @log_operation("scalar tensor creation")
    def create_scalar_tensor(self, scalar_inputs: List[Dict[str, float]]) -> Tuple[torch.Tensor, List[str]]:
        """
        Create tensor from scalar inputs.
        
        Args:
            scalar_inputs (List[Dict[str, float]]): List of scalar input dictionaries
            
        Returns:
            Tuple[torch.Tensor, List[str]]: Input tensor and feature names
            
        Raises:
            ValidationError: If input data is invalid
        """
        try:
            # Get all unique feature names
            all_features = set()
            for d in scalar_inputs:
                all_features.update(d.keys())
            all_features = sorted(list(all_features))
            self.feature_info.scalar_features = all_features
            # Set the alias for compatibility
            self.feature_info.defined_scalar_params = all_features
            
            # Create empty input tensor
            X = torch.zeros((len(scalar_inputs), len(all_features)))
            
            # Fill input tensor with scalar values
            for i, d in enumerate(scalar_inputs):
                for j, feature in enumerate(all_features):
                    X[i, j] = d.get(feature, 0.0)  # Default to 0.0 if feature not present
            
            return X, all_features
        except Exception as e:
            raise ValidationError(f"Failed to create scalar tensor: {str(e)}")
    
    @log_operation("output tensor creation")
    def create_output_tensor(self, outputs: List[np.ndarray]) -> torch.Tensor:
        """
        Create tensor from output arrays.
        
        Args:
            outputs (List[np.ndarray]): List of output arrays
            
        Returns:
            torch.Tensor: Output tensor
            
        Raises:
            ValidationError: If output data is invalid
        """
        try:
            valid_outputs = [out for out in outputs if isinstance(out, np.ndarray) and len(out) > 0]
            return torch.stack([torch.tensor(out, dtype=torch.float32) for out in valid_outputs])
        except Exception as e:
            raise ValidationError(f"Failed to create output tensor: {str(e)}")
    
    @abstractmethod
    def prepare_data(self, scalar_inputs: List[Dict[str, float]], profile_inputs: List[Any], outputs: List[np.ndarray], 
                     config: ProcessingConfig) -> Tuple[InputT, OutputT]:
        """
        Prepare data for a specific model type.
        
        Args:
            scalar_inputs (List[Dict[str, float]]): List of scalar inputs
            profile_inputs (List[Any]): List of profile inputs
            outputs (List[np.ndarray]): List of outputs
            config (ProcessingConfig): Processing configuration
            
        Returns:
            Tuple[InputT, OutputT]: Prepared input and output data
        """
        pass
    
    @abstractmethod
    def split_data(self, inputs: InputT, outputs: OutputT, 
                   config: ProcessingConfig) -> Dict[str, Union[InputT, OutputT, torch.Tensor]]:
        """
        Split data into training and testing sets.
        
        Args:
            inputs (InputT): Input data
            outputs (OutputT): Output data
            config (ProcessingConfig): Processing configuration
            
        Returns:
            Dict[str, Union[InputT, OutputT, torch.Tensor]]: Dictionary with train/test data
        """
        pass
    
    @abstractmethod
    def save_data(self, data: Dict[str, Union[InputT, OutputT, torch.Tensor]], 
                  metadata: ProcessingMetadata) -> ProcessingResult[InputT, OutputT]:
        """
        Save prepared data to disk.
        
        Args:
            data (Dict[str, Union[InputT, OutputT, torch.Tensor]]): Dictionary with prepared data
            metadata (ProcessingMetadata): Processing metadata
            
        Returns:
            ProcessingResult[InputT, OutputT]: Result of the processing operation
            
        Raises:
            SaveError: If saving data fails
        """
        pass
    
    def get_normalization_params(self) -> Dict[str, torch.Tensor]:
        """Get normalization parameters."""
        return self.normalization_params

class NeuralNetworkProcessor(BaseDataProcessor[torch.Tensor, torch.Tensor]):
    """
    Data processor for Neural Network models.
    
    This class prepares data for traditional Neural Network architectures.
    """
    
    @log_operation("neural network data preparation")
    def prepare_data(self, scalar_inputs: List[Dict[str, float]], profile_inputs: List[Any], 
                     outputs: List[np.ndarray], config: ProcessingConfig) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare data for the neural network model.
        
        Args:
            scalar_inputs (List[Dict[str, float]] or torch.Tensor): List of scalar input dictionaries or tensor
            profile_inputs (List[Any]): Not used for neural network, included for interface compatibility
            outputs (List[np.ndarray] or torch.Tensor): List of output arrays or tensor
            config (ProcessingConfig): Processing configuration
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Prepared input and output tensors
            
        Raises:
            ValidationError: If input data is invalid
            NormalizationError: If normalization fails
        """
        self.validate_inputs(scalar_inputs, profile_inputs, outputs)
        
        # Check if scalar_inputs and outputs are already tensors
        if isinstance(scalar_inputs, torch.Tensor) and isinstance(outputs, torch.Tensor):
            logger.info(f"Inputs and outputs are already tensors: X shape={scalar_inputs.shape}, Y shape={outputs.shape}")
            X = scalar_inputs
            Y = outputs
        else:
            # Create tensors from inputs
            X, self._feature_names = self.create_scalar_tensor(scalar_inputs)
            Y = self.create_output_tensor(outputs)
            
            # If applying FFT, transform the tensor using rfft
            if config.apply_fft:
                logger.info("Applying rfft to output tensor...")
                fft_result = torch.fft.rfft(Y, dim=1)
                Y_real = fft_result.real
                Y_imag = fft_result.imag
                Y = torch.cat([Y_real, Y_imag], dim=1)
                logger.info(f"FFT applied: Y shape={Y.shape}")
        
        # Normalize if requested
        if config.normalize:
            logger.info("Normalizing data...")
            X = self.normalize_tensor(X, 'x')
            Y = self.normalize_tensor(Y, 'y')
        
        # Set additional properties
        self._input_dim = X.shape[1]
        self._output_dim = Y.shape[1]
        self._apply_fft = config.apply_fft
        
        logger.info(f"Prepared data: X shape={X.shape}, Y shape={Y.shape}")
        return X, Y
    
    @log_operation("neural network data splitting")
    def split_data(self, inputs: torch.Tensor, outputs: torch.Tensor, 
                   config: ProcessingConfig) -> Dict[str, Union[torch.Tensor, torch.Tensor]]:
        """
        Split data into training and testing sets.
        
        Args:
            inputs (torch.Tensor): Input tensor
            outputs (torch.Tensor): Output tensor
            config (ProcessingConfig): Processing configuration
            
        Returns:
            Dict[str, Union[torch.Tensor, torch.Tensor]]: Dictionary with train/test data
        """
        # Set random seed for reproducibility
        torch.manual_seed(config.random_seed)
        np.random.seed(config.random_seed)
        
        # Get total number of samples
        num_samples = inputs.shape[0]
        num_train = int(num_samples * config.split_ratio)
        
        # Create random permutation of indices
        indices = torch.randperm(num_samples)
        
        # Split data
        train_indices = indices[:num_train]
        test_indices = indices[num_train:]
        
        train_inputs = inputs[train_indices]
        train_outputs = outputs[train_indices]
        
        test_inputs = inputs[test_indices]
        test_outputs = outputs[test_indices]
        
        logger.info(f"Split completed: {len(train_indices)} training samples, {len(test_indices)} testing samples")
        
        return {
            'train_inputs': train_inputs,
            'train_outputs': train_outputs,
            'test_inputs': test_inputs,
            'test_outputs': test_outputs,
            'train_indices': train_indices,
            'test_indices': test_indices,
            'num_samples': {
                'train': len(train_indices),
                'test': len(test_indices)
            },
            'input_dim': inputs.shape[1],
            'output_dim': outputs.shape[1]
        }
    
    @log_operation("neural network data saving")
    def save_data(self, data: Dict[str, Union[torch.Tensor, torch.Tensor]], 
                  metadata: ProcessingMetadata) -> Dict[str, Any]:
        """
        Save prepared data to disk.
        
        Args:
            data (Dict[str, Union[torch.Tensor, torch.Tensor]]): Dictionary with prepared data
            metadata (ProcessingMetadata): Processing metadata
            
        Returns:
            Dict[str, Any]: Result dictionary with saved data information
            
        Raises:
            SaveError: If saving data fails
        """
        try:
            # Create output directory
            output_dir = self.processed_data_dir / 'neuralnetwork'
            os.makedirs(output_dir, exist_ok=True)
            
            # Save data
            torch.save(data['train_inputs'], output_dir / 'train_inputs.pt')
            torch.save(data['train_outputs'], output_dir / 'train_outputs.pt')
            torch.save(data['test_inputs'], output_dir / 'test_inputs.pt')
            torch.save(data['test_outputs'], output_dir / 'test_outputs.pt')
            
            # Save metadata as-is
            torch.save(metadata, output_dir / 'metadata.pt')
            
            # Create result dictionary
            result = {
                'train_inputs': data['train_inputs'],
                'train_outputs': data['train_outputs'],
                'test_inputs': data['test_inputs'],
                'test_outputs': data['test_outputs'],
                'train_indices': data.get('train_indices'),
                'test_indices': data.get('test_indices'),
                'num_samples': data.get('num_samples', {
                    'train': len(data['train_indices']),
                    'test': len(data['test_indices'])
                }),
                'input_dim': data.get('input_dim', data['train_inputs'].shape[1]),
                'output_dim': data.get('output_dim', data['train_outputs'].shape[1]),
                'metadata': metadata,
                'output_dir': str(output_dir)
            }
            
            logger.info(f"Saved data to {output_dir}")
            return result
            
        except Exception as e:
            raise SaveError(f"Failed to save neural network data: {str(e)}")


class NeuralOperatorProcessor(BaseDataProcessor[List[torch.Tensor], torch.Tensor]):
    """
    Data processor for Neural Operator models.
    
    This class prepares data for Neural Operator architectures.
    """
    
    @log_operation("neural operator data preparation")
    def prepare_data(self, scalar_inputs: List[Dict[str, float]], profile_inputs: List[Any], 
                     outputs: List[np.ndarray], config: ProcessingConfig) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Prepare data for the neural operator model.
        
        Args:
            scalar_inputs (List[Dict[str, float]] or List[torch.Tensor]): List of scalar input dictionaries or tensors
            profile_inputs (List[Any]): List of profile inputs (pandas DataFrames)
            outputs (List[np.ndarray] or torch.Tensor): List of output arrays or tensor
            config (ProcessingConfig): Processing configuration
            
        Returns:
            Tuple[List[torch.Tensor], torch.Tensor]: Prepared input and output tensors
            
        Raises:
            ValidationError: If input data is invalid
            NormalizationError: If normalization fails
        """
        self.validate_inputs(scalar_inputs, profile_inputs, outputs)
        
        # Check if inputs are already in the expected format
        if (isinstance(scalar_inputs, list) and len(scalar_inputs) >= 2 and 
            isinstance(scalar_inputs[0], torch.Tensor) and isinstance(outputs, torch.Tensor)):
            logger.info("Inputs and outputs are already tensors")
            return scalar_inputs, outputs
        
        # Step 1: Extract profile features if available
        profile_features = set()
        for profile in profile_inputs:
            if isinstance(profile, pd.DataFrame) and not profile.empty:
                profile_features.update([col for col in profile.columns if col != 'time'])
        
        profile_features = sorted(list(profile_features))
        self.feature_info.profile_features = profile_features
        # Set the alias for compatibility
        self.feature_info.defined_profile_params = profile_features
        
        # Step 2: Extract scalar features
        scalar_features = set()
        for d in scalar_inputs:
            scalar_features.update(d.keys())
        scalar_features = sorted(list(scalar_features))
        self.feature_info.scalar_features = scalar_features
        
        # Determine number of samples to process (minimum of available inputs and outputs)
        n_samples = len(scalar_inputs)
        
        # Check if we have any samples to process
        if n_samples == 0:
            raise ValidationError("No valid samples found. Make sure scalar_inputs and outputs contain data.")
        
        # Step 3: Verify output length consistency and extract profile input length
        # Find first valid output to determine length
        output_length = None
        for out in outputs:
            if isinstance(out, np.ndarray) and len(out) > 0:
                output_length = len(out)
                break
                
        if output_length is None:
            raise ValidationError("No valid output arrays found")
        
        # Check all outputs have the same length
        for i in range(n_samples):
            if i < len(outputs) and isinstance(outputs[i], np.ndarray) and len(outputs[i]) > 0:
                if len(outputs[i]) != output_length:
                    raise ValidationError(f"Output arrays have different lengths. Expected {output_length}, found {len(outputs[i])} at index {i}.")
        
        # Store the original output length for metadata
        self.feature_info.output_profile_length = output_length
        
        # Calculate the number of frequency components for outputs (rfft output size)
        n_rfft = output_length // 2 + 1
        # Store this for the model
        self.feature_info.output_profile_freq_components = n_rfft
        
        # Find profile input length from first valid profile
        profile_input_length = None
        for profile in profile_inputs:
            if isinstance(profile, pd.DataFrame) and not profile.empty and profile_features:
                if profile_features[0] in profile.columns:
                    profile_input_length = len(profile[profile_features[0]])
                    break
        
        # Use profile length to calculate n_freq_components or fallback to config
        if profile_input_length:
            n_freq_components = profile_input_length // 2 + 1
            self.feature_info.profile_input_length = profile_input_length
            logger.info(f"Using profile input length {profile_input_length} to calculate n_freq_components={n_freq_components}")
        else:
            n_freq_components = config.n_freq_components
            logger.info(f"No valid profile input found, using config n_freq_components={n_freq_components}")
        
        # Store the number of frequency components for future reference
        self.feature_info.profile_freq_components = n_freq_components
        
        # Step 4: Create tensors for processing
        n_scalar_features = len(scalar_features)
        
        # Create scalar input tensor [batch_size, n_features, 1]
        X_scalar = torch.zeros((n_samples, n_scalar_features, 1))
        logger.info(f"Processing {n_samples} scalar inputs...")
        for i in tqdm(range(n_samples), desc="Processing scalar inputs"):
            for j, feature in enumerate(scalar_features):
                if feature in scalar_inputs[i]:
                    X_scalar[i, j, 0] = scalar_inputs[i].get(feature, 0.0)
        
        # Create profile input tensor with FFT [batch_size, n_freq_components, 2]
        X_profile = torch.zeros((n_samples, n_freq_components, 2))
        logger.info(f"Processing {n_samples} profile inputs with FFT...")
        for i in tqdm(range(n_samples), desc="Processing profile inputs with FFT"):
            if i < len(profile_inputs) and isinstance(profile_inputs[i], pd.DataFrame) and not profile_inputs[i].empty:
                df = profile_inputs[i]
                if profile_features and profile_features[0] in df.columns:
                    profile_vals = df[profile_features[0]].values
                    fft = np.fft.rfft(profile_vals)  # Use rfft (real FFT) for real inputs
                    for j in range(min(n_freq_components, len(fft))):
                        X_profile[i, j, 0] = np.real(fft[j])
                        X_profile[i, j, 1] = np.imag(fft[j])
        
        # Create output tensor with FFT [batch_size, n_rfft, 2]
        Y = torch.zeros((n_samples, n_rfft, 2))
        logger.info(f"Processing {n_samples} outputs with FFT...")
        for i in tqdm(range(n_samples), desc="Processing outputs with FFT"):
            if i < len(outputs) and isinstance(outputs[i], np.ndarray) and len(outputs[i]) > 0:
                output_vals = outputs[i]
                fft = np.fft.rfft(output_vals)  # Use rfft for real inputs
                for j in range(len(fft)):
                    Y[i, j, 0] = np.real(fft[j])
                    Y[i, j, 1] = np.imag(fft[j])
        
        # Step 5: Normalize if requested
        if config.normalize:
            logger.info("Normalizing data...")
            X_scalar = self.normalize_tensor(X_scalar, 'x_scalar')
            X_profile = self.normalize_tensor(X_profile, 'x_profile')
            Y = self.normalize_tensor(Y, 'y')
            
            # Store normalization parameters in a format expected by training modules
            if 'y_min' in self.normalization_params and 'y_max' in self.normalization_params:
                # Create output_norm_params structure that neural_operator_training.py expects
                self.normalization_params['output_norm_params'] = {
                    'fft': True,
                    'min': self.normalization_params['y_min'],
                    'max': self.normalization_params['y_max']
                }
        
        # Store metadata
        self.feature_info.input_dim = {
            'scalar': X_scalar.shape,
            'profile': X_profile.shape
        }
        self.feature_info.output_dim = Y.shape
        
        logger.info(f"Prepared data: X_scalar shape={X_scalar.shape}, X_profile shape={X_profile.shape}, Y shape={Y.shape}")
        return [X_scalar, X_profile], Y
    
    @log_operation("neural operator data splitting")
    def split_data(self, inputs: List[torch.Tensor], outputs: torch.Tensor, 
                   config: ProcessingConfig) -> Dict[str, Union[List[torch.Tensor], torch.Tensor]]:
        """
        Split data into training and testing sets.
        
        Args:
            inputs (List[torch.Tensor]): List of input tensors [scalar_input, profile_input]
            outputs (torch.Tensor): Output tensor with shape [batch_size, n_rfft, 2]
            config (ProcessingConfig): Processing configuration
            
        Returns:
            Dict[str, Union[List[torch.Tensor], torch.Tensor]]: Dictionary with train/test data
        """
        # Set random seed for reproducibility
        torch.manual_seed(config.random_seed)
        np.random.seed(config.random_seed)
        
        # Get total number of samples
        num_samples = outputs.shape[0]
        num_train = int(num_samples * config.split_ratio)
        
        # Create random permutation of indices
        indices = torch.randperm(num_samples)
        
        # Split data
        train_indices = indices[:num_train]
        test_indices = indices[num_train:]
        
        # Unpack inputs
        scalar_input, profile_input = inputs
        
        # Split inputs and outputs
        train_scalar_input = scalar_input[train_indices]
        train_profile_input = profile_input[train_indices]
        train_outputs = outputs[train_indices]
        
        test_scalar_input = scalar_input[test_indices]
        test_profile_input = profile_input[test_indices]
        test_outputs = outputs[test_indices]
        
        logger.info(f"Split completed: {len(train_indices)} training samples, {len(test_indices)} testing samples")
        logger.info(f"Train outputs shape: {train_outputs.shape}, Test outputs shape: {test_outputs.shape}")
        
        return {
            'train_inputs': [train_scalar_input, train_profile_input],
            'train_outputs': train_outputs,
            'test_inputs': [test_scalar_input, test_profile_input],
            'test_outputs': test_outputs,
            'train_indices': train_indices,
            'test_indices': test_indices,
            'num_samples': {
                'train': len(train_indices),
                'test': len(test_indices)
            },
            'input_dims': {
                'scalar': scalar_input.shape,
                'profile': profile_input.shape
            },
            'output_dim': outputs.shape
        }
    
    @log_operation("neural operator data saving")
    def save_data(self, data: Dict[str, Union[List[torch.Tensor], torch.Tensor]], 
                  metadata: ProcessingMetadata) -> ProcessingResult[List[torch.Tensor], torch.Tensor]:
        """
        Save prepared data to disk.
        
        Args:
            data (Dict[str, Union[List[torch.Tensor], torch.Tensor]]): Dictionary with prepared data
            metadata (ProcessingMetadata): Processing metadata
            
        Returns:
            ProcessingResult[List[torch.Tensor], torch.Tensor]: Result of the processing operation
            
        Raises:
            SaveError: If saving data fails
        """
        try:
            # Create output directory
            output_dir = self.processed_data_dir / 'neuraloperator'
            os.makedirs(output_dir, exist_ok=True)
            
            # Save data
            torch.save(data['train_inputs'], output_dir / 'train_inputs.pt')
            torch.save(data['train_outputs'], output_dir / 'train_outputs.pt')
            torch.save(data['test_inputs'], output_dir / 'test_inputs.pt')
            torch.save(data['test_outputs'], output_dir / 'test_outputs.pt')
            
            # Save metadata as-is
            torch.save(metadata, output_dir / 'metadata.pt')
            
            # Create a comprehensive result
            result = {
                'train_inputs': data['train_inputs'],
                'train_outputs': data['train_outputs'],
                'test_inputs': data['test_inputs'],
                'test_outputs': data['test_outputs'],
                'train_indices': data.get('train_indices'),
                'test_indices': data.get('test_indices'),
                'num_samples': data.get('num_samples', {
                    'train': len(data['train_indices']),
                    'test': len(data['test_indices'])
                }),
                'input_dims': data.get('input_dims', {
                    'scalar': data['train_inputs'][0].shape,
                    'profile': data['train_inputs'][1].shape
                }),
                'output_dim': data.get('output_dim', data['train_outputs'].shape),
                'metadata': metadata,
                'output_dir': str(output_dir)
            }
            
            logger.info(f"Saved data to {output_dir}")
            return result
            
        except Exception as e:
            raise SaveError(f"Failed to save neural operator data: {str(e)}") 