"""
Training utilities for Neural Operator models in CoreTempAI.

This module provides training functionality for Neural Operator models,
compatible with the data preprocessor.
"""

import sys
import os
from pathlib import Path

# Add the project root directory to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Import standard libraries
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
import numpy as np
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
import wandb  # For experiment tracking

# Import CoreTempAI modules
from CoreTempAI.model.neural_operator_model import NeuralOperator
from CoreTempAI.data_preprocessing.data_preprocessor import DataPreprocessor
from CoreTempAI.config import NeuralOperatorConfig
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TrainState:
    """Track number of steps, examples, and tokens processed"""

    def __init__(self):
        self.step = 0  # Steps in the current epoch
        self.accum_step = 0  # Number of gradient accumulation steps
        self.samples = 0  # total # of examples used
        self.tokens = 0  # total # of tokens processed


class NODataset(Dataset):
    """
    Dataset class for Neural Operator model.
    """
    
    def __init__(self, input_data=None, output_data=None, data_dir=None, subset='train', 
                 apply_fft=False, device=device):
        """
        Initialize the dataset.
        
        Args:
            input_data (List[torch.Tensor], optional): List of input tensors 
            output_data (torch.Tensor, optional): Target values
            data_dir (str, optional): Path to directory containing processed data
            subset (str, optional): Which subset to load ('train', 'test', 'val')
            apply_fft (bool): Whether FFT has already been applied to the output data
            device (torch.device, optional): Device to store tensors on
        """
        # If data_dir is provided, load data from files
        if data_dir is not None:
            data_dir = Path(data_dir) if not isinstance(data_dir, Path) else data_dir
            
            input_file = data_dir / f'{subset}_inputs.pt'
            output_file = data_dir / f'{subset}_outputs.pt'
            metadata_file = data_dir / 'metadata.pt'
            
            if not input_file.exists() or not output_file.exists():
                raise FileNotFoundError(f"Data files not found in {data_dir}")
            
            input_data = torch.load(input_file)
            output_data = torch.load(output_file)
            
            # Check if metadata exists and load FFT info
            if metadata_file.exists():
                metadata = torch.load(metadata_file)
                if 'output_norm_params' in metadata and 'fft' in metadata['output_norm_params']:
                    apply_fft = metadata['output_norm_params']['fft']
                    print(f"Setting apply_fft={apply_fft} based on metadata.")
        
        # Validate that we have data
        if input_data is None or output_data is None:
            raise ValueError("Either provide input_data and output_data, or a valid data_dir.")
        
        # Store inputs as a list of tensors, each moved to the device
        if isinstance(input_data, list):
            self.input_data = [tensor.to(device) for tensor in input_data]
        else:
            # Handle case where input_data might be a single tensor
            self.input_data = [input_data.to(device)]
        
        # Verify tensor shapes and make adjustments if needed
        if len(self.input_data) >= 1:
            # First tensor should be scalar parameters with shape [batch_size, num_scalars, scalar_dim]
            scalar_tensor = self.input_data[0]
            # If shape is [batch_size, scalar_dim], add an extra dimension
            if len(scalar_tensor.shape) == 2:
                batch_size, scalar_dim = scalar_tensor.shape
                self.input_data[0] = scalar_tensor.unsqueeze(-1)
                print(f"Reshaped scalar tensor from {scalar_tensor.shape} to {self.input_data[0].shape}")
        
        # Store output data
        self.output_data = output_data.to(device)
        
        # Store this flag to know if the output is in frequency domain
        self.fft_applied = apply_fft
    
    def __len__(self):
        """
        Get the number of samples in the dataset.
        
        Returns:
            int: Number of samples
        """
        # Use the first tensor to determine the batch size
        return self.input_data[0].shape[0]
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Args:
            idx (int): Index of the sample
            
        Returns:
            tuple: (inputs, output) pair where inputs is a list of tensors
        """
        # Extract the inputs for the given index
        inputs = [tensor[idx] for tensor in self.input_data]
        
        # Return the inputs and output
        return inputs, self.output_data[idx]


class Batch:
    """Object for holding a batch of data with mask during training."""

    def __init__(self, inputs, outputs):
        """
        Initialize a batch.
        
        Args:
            inputs (List[torch.Tensor]): List of input tensors for this batch
            outputs (torch.Tensor): Output tensor with shape (batch_size, output_dim)
        """
        # Store the inputs
        self.inputs = inputs
        self.outputs = outputs
        
        # Set ntokens for loss normalization
        # Use the product of the dimensions of the output tensor
        if isinstance(outputs, torch.Tensor):
            self.ntokens = outputs.numel()  # Number of elements in the tensor
            print(f"ntokens: {self.ntokens}")
        else:
            # Fallback to avoid potential errors
            self.ntokens = 1
        
    
    def to(self, device):
        """
        Move batch to device.
        
        Args:
            device: Device to move batch to
        """
        if isinstance(self.inputs, list):
            # Move each tensor in the list to the device
            self.inputs = [tensor.to(device) for tensor in self.inputs]
        else:
            # Move single tensor to device
            self.inputs = self.inputs.to(device)
            
        self.outputs = self.outputs.to(device)
        return self


def load_data_from_preprocessor(processed_data_dir=None, apply_fft=True):
    """
    Load data prepared by the DataPreprocessor.
    
    Args:
        processed_data_dir (str, optional): Path to processed data directory
        apply_fft (bool): Whether FFT has been applied to the data
        
    Returns:
        tuple: (train_dataset, test_dataset, metadata)
    """
    # Default processed data directory
    if processed_data_dir is None:
        processed_data_dir = Path("data/processed")
    else:
        processed_data_dir = Path(processed_data_dir)
    
    # Check if processed data exists
    if not processed_data_dir.exists():
        raise FileNotFoundError(f"Processed data directory not found: {processed_data_dir}")
    
    # Try different possible directories for neural operator data
    possible_dirs = [
        processed_data_dir / 'neuraloperator',
        processed_data_dir / 'fluent_data' / 'neuraloperator',
    ]
    
    no_dir = None
    for directory in possible_dirs:
        if directory.exists() and (directory / 'train_inputs.pt').exists():
            no_dir = directory
            print(f"Found neural operator data in {no_dir}")
            break
    
    if no_dir is None:
        raise FileNotFoundError(f"Could not find neural operator data in {processed_data_dir}")
    
    # Load neural operator data
    train_inputs = torch.load(no_dir / 'train_inputs.pt')
    train_outputs = torch.load(no_dir / 'train_outputs.pt')
    test_inputs = torch.load(no_dir / 'test_inputs.pt')
    test_outputs = torch.load(no_dir / 'test_outputs.pt')
    
    # Load metadata and check if FFT was applied
    metadata_path = no_dir / 'metadata.pt'
    if metadata_path.exists():
        metadata = torch.load(metadata_path)
        
        # Check if FFT info is in metadata
        if 'output_norm_params' in metadata and 'fft' in metadata['output_norm_params']:
            apply_fft = metadata['output_norm_params']['fft']
            print(f"Setting apply_fft={apply_fft} based on metadata.")
    else:
        metadata = {}
        print("Warning: No metadata file found. Using default settings.")
    
    # Create datasets
    train_dataset = NODataset(train_inputs, train_outputs, apply_fft=apply_fft)
    test_dataset = NODataset(test_inputs, test_outputs, apply_fft=apply_fft)
    
    return train_dataset, test_dataset, metadata


def data_gen(batch_size, dataset):
    """
    Generate batches from dataset.
    
    Args:
        batch_size (int): Batch size
        dataset (NODataset): Dataset to generate batches from
        
    Yields:
        Batch: Batch object containing inputs and outputs
    """
    # Validate dataset and batch size
    dataset_size = len(dataset)
    if dataset_size == 0:
        print("Warning: Dataset is empty! No batches will be generated.")
        return
    
    if batch_size > dataset_size:
        print(f"Warning: Batch size ({batch_size}) is larger than dataset size ({dataset_size}).")
        batch_size = max(1, dataset_size // 2)  # Ensure at least one item per batch
        print(f"Adjusting batch size to {batch_size}")
    # Create dataloader
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )
    
    # Check if dataloader will yield any batches
    num_batches = len(dataloader)
    print(f"DataLoader created with {num_batches} batches (dataset size: {dataset_size}, batch size: {batch_size})")
    
    if num_batches == 0:
        print("Warning: DataLoader will not yield any batches! Check batch size and drop_last settings.")
        return
    
    # Yield batches
    for inputs, outputs in dataloader:
        yield Batch(inputs, outputs)



class SimpleLossCompute:
    """A simple loss compute and train function."""

    def __init__(self, generator, criterion):
        """
        Initialize loss compute.
        
        Args:
            generator: Generator model (usually the output layer of the model)
            criterion: Loss criterion
        """
        self.generator = generator
        self.criterion = criterion

    def __call__(self, x, y, norm):
        """
        Compute loss.
        
        Args:
            x: Model output
            y: Target output
            norm: Normalization factor
            
        Returns:
            tuple: (loss, loss_node)
        """
        # Compute loss
        loss = self.criterion(x, y)
        
        # Return both the loss value and the loss node for backward
        return loss, loss


def run_epoch(
    data_iter,
    epoch,
    model,
    loss_compute,
    optimizer,
    scheduler,
    mode="train",
    accum_iter=1,
    train_state=None,
    use_wandb=False
):
    """
    Run one epoch of training or evaluation.
    
    Args:
        data_iter: Data iterator
        epoch: Current epoch number
        model: Model to train/evaluate
        loss_compute: Loss computation function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        mode: "train" or "eval"
        accum_iter: Number of gradient accumulation steps
        train_state: TrainState object
        use_wandb: Whether to log metrics to wandb
        
    Returns:
        tuple: (average_loss, train_state)
    """
    if train_state is None:
        train_state = TrainState()
        
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    n_accum = 0
    i = 0

    try:
        with tqdm(data_iter, dynamic_ncols=True) as tqdmDataLoader:
            for batch in tqdmDataLoader:
                batch.to(device)
                
                # Validate input shapes
                if isinstance(batch.inputs, list) and len(batch.inputs) > 0:
                    scalar_inputs = batch.inputs[0]  # First tensor is scalar inputs
                    if len(scalar_inputs.shape) == 2:
                        # Add an extra dimension for the vector dimension if needed
                        # Shape should be [batch_size, num_vectors, vector_dim]
                        scalar_inputs = scalar_inputs.unsqueeze(1)
                        batch.inputs[0] = scalar_inputs
                        print(f"Reshaped scalar inputs to {scalar_inputs.shape}")
                    
                    # Check if we have function inputs
                    if len(batch.inputs) > 1:
                        function_inputs = batch.inputs[1]  # Second tensor is function inputs
                        print(f"Function inputs shape: {function_inputs.shape}")
                
                # Process the batch
                print(f"Input shapes before model: {[inp.shape for inp in batch.inputs] if isinstance(batch.inputs, list) else batch.inputs.shape}")
                out = model(batch.inputs)
                print(f"Output shape from model: {out.shape}")
                
                loss, loss_node = loss_compute(out, batch.outputs, batch.ntokens)
                
                if mode == "train":
                    loss_node.backward()
                    train_state.step += 1
                    
                    # Update samples count
                    if isinstance(batch.inputs, list) and len(batch.inputs) > 0:
                        train_state.samples += batch.inputs[0].shape[0]
                    elif hasattr(batch.inputs, 'shape') and len(batch.inputs.shape) > 0:
                        train_state.samples += batch.inputs.shape[0]
                    else:
                        # Fallback to a default value
                        train_state.samples += 1
                        
                    train_state.tokens += batch.ntokens
                    
                    if i % accum_iter == 0:
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                        n_accum += 1
                        train_state.accum_step += 1
                    
                    # Update scheduler
                    if isinstance(scheduler, ReduceLROnPlateau):
                        scheduler.step(loss_node)
                    else:
                        scheduler.step()

                    # Log metrics to wandb if enabled
                    if use_wandb:
                        wandb.log({
                            "epoch": epoch+1, 
                            "loss": loss.item(), 
                            "loss_node": loss_node.item(), 
                            "LR": optimizer.param_groups[0]["lr"]
                        })

                    tqdmDataLoader.set_postfix(
                        ordered_dict={
                            "epoch": epoch + 1,
                            "loss": loss.item(),
                            "loss_node": loss_node.item(),
                            "LR": optimizer.param_groups[0]["lr"]
                        }
                    )
                else:
                    # Log metrics to wandb if enabled
                    if use_wandb:
                        wandb.log({
                            "eval_epoch": epoch+1, 
                            "eval_loss": loss.item(), 
                            "eval_loss_node": loss_node.item()
                        })
                        
                    tqdmDataLoader.set_postfix(
                        ordered_dict={
                            "eval_epoch": epoch + 1,
                            "eval_loss": loss.item(),
                            "eval_loss_node": loss_node.item()
                        }
                    )

                total_loss += loss.item()
                total_tokens += batch.ntokens
                tokens += batch.ntokens
                i += 1
    except Exception as e:
        import traceback
        print(f"Error during run_epoch: {e}")
        traceback.print_exc()
        # If an exception occurs during the first batch, we might still have total_tokens = 0
        if total_tokens == 0:
            print("No batches were processed. Setting total_tokens to 1 to avoid division by zero.")
            total_tokens = 1

    # Avoid division by zero if no batches were processed
    if total_tokens == 0:
        print("Warning: No tokens processed in this epoch. Setting total_tokens to 1 to avoid division by zero.")
        total_tokens = 1
        
    avg_loss = total_loss / total_tokens
    return avg_loss, train_state

def make_model(config=None, metadata=None, vectors=None, num_functions=None):
    """
    Create a Neural Operator model.
    
    Args:
        config (NeuralOperatorConfig, optional): Configuration object
        metadata (dict, optional): Metadata from preprocessor
        vectors (list, optional): List of tuples (num_vectors, vector_dim) to manually specify vector dimensions
        num_functions (int, optional): Number of functions to manually specify
        
    Returns:
        NeuralOperator: Neural Operator model
    """

    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # If config is provided, use it directly
    if config is not None:
        # If config is a dictionary, convert it to NeuralOperatorConfig object
        if isinstance(config, dict):
            config = NeuralOperatorConfig.from_dict(config)
        # If config is not a NeuralOperatorConfig object, warn the user
        elif not isinstance(config, NeuralOperatorConfig):
            print(f"Warning: config is of type {type(config)}, expected NeuralOperatorConfig. Creating default config.")
            config = NeuralOperatorConfig()
    if config is None:
        config = NeuralOperatorConfig()

    
    # If vectors and num_functions are directly provided, use them
    if vectors is not None and num_functions is not None:
        print(f"Using provided vectors={vectors} and num_functions={num_functions}")
    # Otherwise, try to determine vectors and num_functions from metadata if available
    elif metadata is not None:
        # Scalar dimension - usually 1 for each scalar parameter
        scalar_dim = 1
        
        # Get the number of scalar parameters
        defined_scalar_params = []
        if 'feature_info' in metadata and 'defined_scalar_params' in metadata['feature_info']:
            defined_scalar_params = metadata['feature_info']['defined_scalar_params']
        
        num_scalars = len(defined_scalar_params)
        print(f"Found {num_scalars} scalar parameters: {defined_scalar_params}")
        
        if num_scalars > 0:
            # Define vectors as a list of tuples (num_vectors, vector_dim)
            # For scalar parameters, we have num_scalars vectors, each with dimension scalar_dim
            vectors = [(num_scalars, scalar_dim)]
            print(f"Setting vectors to {vectors}")
        
        # Get the number of function/profile features
        num_profile_features = 0
        if 'feature_info' in metadata and 'defined_profile_params' in metadata['feature_info']:
            defined_profile_params = metadata['feature_info']['defined_profile_params']
            num_profile_features = len(defined_profile_params)
            print(f"Found {num_profile_features} profile features: {defined_profile_params}")
        
        if num_profile_features > 0:
            # Set num_functions to the number of profile features
            num_functions = num_profile_features
            print(f"Setting num_functions to {num_functions}")
    else:
        # If neither metadata nor direct parameters are provided, use defaults
        print("No metadata or direct parameters provided. Using default model configuration.")
        vectors = [(1, 1)]  # Default: 1 vector of dimension 1
        num_functions = 1   # Default: 1 function
        print(f"Using default vectors={vectors} and num_functions={num_functions}")
    
    # Create model
    print(f"Creating Neural Operator with vectors={vectors}, num_functions={num_functions}")
    model = NeuralOperator(vectors=vectors, num_functions=num_functions, config=config)
    
    return model.to(device)


def train_model(
    train_dataset, 
    test_dataset, 
    metadata, 
    config=None, 
    epochs=1000, 
    batch_size=32, 
    lr=1e-3, 
    save_dir=None, 
    use_wandb=False, 
    project_name="CoreTempAI"
):
    """
    Train the Neural Operator model.
    
    Args:
        train_dataset: Training dataset
        test_dataset: Test dataset
        metadata: Metadata from preprocessor
        config: Model configuration
        epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
        save_dir: Directory to save model
        use_wandb: Whether to use wandb for logging
        project_name: Name of wandb project
        
    Returns:
        NeuralOperator: Trained model
    """
    # Ensure batch size is appropriate for dataset size
    train_size = len(train_dataset)
    test_size = len(test_dataset)
    
    if batch_size >= train_size:
        adjusted_batch_size = max(1, train_size // 2)  # Ensure at least 2 batches
        print(f"Warning: Batch size ({batch_size}) is larger than or equal to training dataset size ({train_size}).")
        print(f"Adjusting batch size to {adjusted_batch_size}")
        batch_size = adjusted_batch_size
    
    # Initialize wandb if enabled
    if use_wandb:
        wandb.init(project=project_name, config={
            "model_type": "NeuralOperator",
            "learning_rate": lr,
            "epochs": epochs,
            "batch_size": batch_size,
            "dataset_train_size": train_size,
            "dataset_test_size": test_size,
            "device": str(device),
            "run_selection": metadata.get('run_selection')
        })
    
    # Create model
    model = make_model(config, metadata)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Define loss function and optimizer
    criterion = nn.MSELoss(reduction="mean")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Define scheduler
    if config is not None and hasattr(config, 'scheduler_type'):
        if config.scheduler_type == 'exponential':
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.99999, last_epoch=-1)
        elif config.scheduler_type == 'constant':
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0)
        elif config.scheduler_type == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 20000)
        else:  # 'reduce_plateau'
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 'min', 
                factor=getattr(config, 'scheduler_factor', 0.5), 
                patience=getattr(config, 'scheduler_patience', 50), 
                cooldown=getattr(config, 'scheduler_cooldown', 50)
            )
    else:
        # Default scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', factor=0.5, patience=50, cooldown=50
        )
    
    # Watch model in wandb if enabled
    if use_wandb:
        wandb.watch(model, log="all")
    
    # Training loop
    start_time = time.time()
    train_losses = []
    test_losses = []
    train_state = TrainState()
    best_loss = float('inf')
    
    # Create save directory if it doesn't exist
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss, train_state = run_epoch(
            data_gen(batch_size, train_dataset),
            epoch,
            model,
            SimpleLossCompute(model.generator, criterion),
            optimizer,
            scheduler,
            mode="train",
            accum_iter=getattr(config, 'accum_iter', 1) if config else 1,
            train_state=train_state,
            use_wandb=use_wandb
        )
        train_losses.append(train_loss)
        
        # Evaluation
        model.eval()
        with torch.no_grad():
            test_loss, _ = run_epoch(
                data_gen(batch_size, test_dataset),
                epoch,
                model,
                SimpleLossCompute(model.generator, criterion),
                optimizer,
                scheduler,
                mode="eval",
                train_state=train_state,
                use_wandb=use_wandb
            )
            test_losses.append(test_loss)
        
        # Print progress
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}")
        
        # Save best model
        if test_loss < best_loss and save_dir is not None:
            best_loss = test_loss
            model_path = os.path.join(save_dir, 'best_neural_operator_model.pt')
            model.save(model_path)
            print(f"Saved best model with loss {test_loss:.6f} to {model_path}")
        
        # Save checkpoint periodically
        if (epoch + 1) % 10 == 0 and save_dir is not None:
            checkpoint_path = os.path.join(save_dir, f'neural_operator_model_epoch_{epoch+1}.pt')
            model.save(checkpoint_path)
            print(f"Saved checkpoint at epoch {epoch+1} to {checkpoint_path}")
    
    end_time = time.time()
    total_time = end_time - start_time

    print("Training completed!")
    print(f"Total training time: {total_time:.2f} seconds")
    
    # Save final model
    if save_dir is not None:
        final_model_path = os.path.join(save_dir, 'neural_operator_model_final.pt')
        model.save(final_model_path)
        print(f"Saved final model to {final_model_path}")
        
        # Save training history
        history = {
            'train_losses': train_losses,
            'test_losses': test_losses,
            'metadata': metadata,
            'training_time': total_time,
            'epochs': epochs,
            'learning_rate': lr
        }
        history_path = os.path.join(save_dir, 'training_history.pt')
        torch.save(history, history_path)
        print(f"Training history saved to {history_path}")
        
        # Plot training curves
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(test_losses, label='Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Test Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, 'loss_curves.png'), dpi=300)
        plt.close()
        
        # Log final model to wandb if enabled
        if use_wandb:
            wandb.save(final_model_path)
            wandb.save(history_path)
            wandb.save(os.path.join(save_dir, 'loss_curves.png'))
            
            # Log summary metrics
            wandb.run.summary["final_train_loss"] = train_losses[-1]
            wandb.run.summary["final_test_loss"] = test_losses[-1]
            wandb.run.summary["best_test_loss"] = best_loss
            wandb.run.summary["training_time"] = total_time
    
    # Finish wandb run if enabled
    if use_wandb:
        wandb.finish()

    return model


def evaluate_model(model, test_dataset, num_samples=5, save_dir=None, use_wandb=False):
    """
    Evaluate the model and visualize predictions.
    
    Args:
        model: Trained model
        test_dataset: Test dataset
        num_samples: Number of samples to visualize
        save_dir: Directory to save visualizations
        use_wandb: Whether to log results to wandb
        
    Returns:
        dict: Evaluation metrics
    """
    model.eval()
    
    # Limit the number of samples to visualize
    num_samples = min(num_samples, len(test_dataset))
    
    # Create directory for visualizations if provided
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
    
    # Compute predictions for all test samples
    with torch.no_grad():
        # Create a batch with all test data
        print(f"Computing predictions for all {len(test_dataset)} test samples...")
        all_inputs = test_dataset.input_data
        all_outputs = test_dataset.output_data
        
        # Ensure inputs are properly formatted for the model
        if isinstance(all_inputs, list):
            print(f"Input shapes: {[inp.shape for inp in all_inputs]}")
        else:
            print(f"Input shape: {all_inputs.shape}")
        
        # Get model predictions
        try:
            all_preds = model(all_inputs)
            print(f"Predictions shape: {all_preds.shape}")
        except Exception as e:
            print(f"Error during prediction: {e}")
            print("Evaluation failed. Returning empty metrics.")
            return {'mse': 0, 'mae': 0, 'r2': 0, 'pearson_r': 0}
    
    # If FFT was applied, convert back to time domain if needed
    if hasattr(test_dataset, 'fft_applied') and test_dataset.fft_applied:
        # Denormalize predictions and outputs if needed
        if hasattr(test_dataset, 'output_min') and hasattr(test_dataset, 'output_max'):
            pred_unnorm = test_dataset.output_min + ((test_dataset.output_max - test_dataset.output_min) * all_preds)
            output_unnorm = test_dataset.output_min + ((test_dataset.output_max - test_dataset.output_min) * all_outputs)
        else:
            pred_unnorm = all_preds
            output_unnorm = all_outputs
        
        # If we need to convert back to time domain for visualization
        # Reshape to separate real and imaginary parts
        batch_size = pred_unnorm.shape[0]
        num_freq = pred_unnorm.shape[1] // 2
        
        # Extract real and imaginary parts
        pred_real_part = pred_unnorm[:, :, ]  # Every even index
        pred_imag_part = pred_unnorm[:, 1::2]  # Every odd index
        pred_complex = torch.complex(pred_real_part, pred_imag_part)
        
        output_real_part = output_unnorm[:, ::2]
        output_imag_part = output_unnorm[:, 1::2]
        output_complex = torch.complex(output_real_part, output_imag_part)
        
        # Convert back to time domain using inverse FFT
        # Assuming output length of 100 points, adjust as needed
        output_length = 100
        pred_final = torch.fft.irfft(pred_complex, output_length, dim=1)
        output_final = torch.fft.irfft(output_complex, output_length, dim=1)
    else:
        # Use the predictions and outputs directly if no FFT was applied
        pred_final = all_preds
        output_final = all_outputs
    
    # Compute metrics
    mse = torch.mean((output_final - pred_final) ** 2).item()
    mae = torch.mean(torch.abs(output_final - pred_final)).item()
    
    # Compute R-squared
    output_flat = output_final.view(-1)
    pred_flat = pred_final.view(-1)
    y_mean = torch.mean(output_flat)
    ss_res = torch.sum((output_flat - pred_flat) ** 2)
    ss_tot = torch.sum((output_flat - y_mean) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    # Compute correlation coefficient
    corr = 0
    for i in range(pred_final.shape[0]):
        corr += torch.corrcoef(torch.stack((pred_final[i], output_final[i])))[0, 1]
    corr_mean = corr/pred_final.shape[0]
    
    print(f"Model Evaluation Metrics:")
    print(f"  MSE: {mse:.6f}")
    print(f"  MAE: {mae:.6f}")
    print(f"  R-squared: {r2.item():.6f}")
    print(f"  Pearson r coefficient: {corr_mean.item():.6f}")
    
    # Log metrics to wandb if enabled
    if use_wandb:
        wandb.log({
            "test_mse": mse,
            "test_mae": mae,
            "test_r2": r2.item(),
            "test_pearson_r": corr_mean.item()
        })
    
    # Visualize predictions for selected samples
    for i in range(num_samples):
        plt.figure(figsize=(10, 6))
        
        # Get sample data
        output = output_final[i].cpu().numpy()
        pred = pred_final[i].cpu().numpy()
        
        # Create x-axis (time or position values)
        x = np.linspace(0, output.shape[0]-1, output.shape[0])
        
        # Plot
        plt.plot(x, output, label='True', linewidth=2)
        plt.plot(x, pred, label='Predicted', linestyle='--', linewidth=2)
        
        # Labels and title
        if hasattr(test_dataset, 'fft_applied') and test_dataset.fft_applied:
            plt.xlabel('Time')
            plt.ylabel('Temperature')
            plt.title(f'Sample {i+1} - Temperature Profile')
        else:
            plt.xlabel('Position')
            plt.ylabel('Value')
            plt.title(f'Sample {i+1}')
            
        plt.legend()
        plt.grid(True)
        
        # Save figure
        if save_dir is not None:
            fig_path = os.path.join(save_dir, f'prediction_sample_{i+1}.png')
            plt.savefig(fig_path, dpi=300)
            
            # Log figure to wandb if enabled
            if use_wandb:
                wandb.log({f"prediction_sample_{i+1}": wandb.Image(fig_path)})
                
            plt.close()
        else:
            plt.show()
    
    return {
        'mse': mse,
        'mae': mae,
        'r2': r2.item(),
        'pearson_r': corr_mean.item()
    }


def main(processed_data_dir=None, config=None, epochs=1000, batch_size=32, lr=1e-3, save_dir=None, use_wandb=False, project_name="CoreTempAI", apply_fft=True):
    """
    Main function to train and evaluate the Neural Operator model.
    
    Args:
        processed_data_dir: Path to processed data directory
        config: Model configuration
        epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
        save_dir: Directory to save model and results
        use_wandb: Whether to use wandb for logging
        project_name: Name of wandb project
        apply_fft: Whether the data has been preprocessed with FFT
        
    Returns:
        tuple: (model, metrics)
    """
    # Load data
    train_dataset, test_dataset, metadata = load_data_from_preprocessor(
        processed_data_dir=processed_data_dir,
        apply_fft=apply_fft
    )
    print(f"metadata: {metadata}")
    
    # Adjust batch size if necessary to prevent empty DataLoaders
    train_size = len(train_dataset)
    test_size = len(test_dataset)
    if batch_size >= train_size:
        adjusted_batch_size = max(1, train_size // 2)  # Ensure at least 2 batches
        print(f"Warning: Batch size ({batch_size}) is larger than or equal to training dataset size ({train_size}).")
        print(f"Adjusting batch size to {adjusted_batch_size}")
        batch_size = adjusted_batch_size
    
    print(f"Using batch size: {batch_size} for datasets - Train: {train_size}, Test: {test_size}")
    
    # Train model
    model = train_model(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        metadata=metadata,
        config=config,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        save_dir=save_dir,
        use_wandb=use_wandb,
        project_name=project_name
    )
    
    # Evaluate model
    metrics = evaluate_model(
        model=model,
        test_dataset=test_dataset,
        num_samples=5,
        save_dir=save_dir,
        use_wandb=use_wandb
    )
    
    return model, metrics


if __name__ == "__main__":
    # Example usage
    main(
        processed_data_dir="data/processed",
        epochs=1000,
        batch_size=32,
        lr=1e-3,
        save_dir="data/models/neural_operator",
        use_wandb=True,
        project_name="CoreTempAI",
        apply_fft=True  # Set to True if the data has been preprocessed with FFT
    ) 