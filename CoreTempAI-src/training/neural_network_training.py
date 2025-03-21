"""
Training utilities for neural networks in CoreTempAI.

This module provides training functionality for neural network models,
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
from CoreTempAI.model.neural_network_model import NeuralNetwork
from CoreTempAI.data_preprocessing.data_preprocessor import DataPreprocessor

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CBTDataset(Dataset):
    """
    Dataset class for CBT temperature prediction.
    """
    
    def __init__(self, input_data, output_data, apply_fft=False):
        """
        Initialize the dataset.
        
        Args:
            input_data (torch.Tensor): Input features
            output_data (torch.Tensor): Target values
            apply_fft (bool): Flag indicating whether the output data has already had FFT applied
        """
        self.input_data = input_data.to(device)
        self.output_data = output_data.to(device)
        self.fft_applied = apply_fft
        
        # If FFT was applied, we should have normalization metadata
        if apply_fft:
            # Store min and max for potential denormalization during evaluation
            # These will be set by the dataloader when loading from metadata
            self.output_min = None
            self.output_max = None
    
    def __len__(self):
        """
        Get the number of samples in the dataset.
        
        Returns:
            int: Number of samples
        """
        return self.input_data.shape[0]
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Args:
            idx (int): Index of the sample
            
        Returns:
            tuple: (X, y) pair
        """
        return self.input_data[idx], self.output_data[idx]


def load_data_from_preprocessor(processed_data_dir=None, apply_fft=False):
    """
    Load data prepared by the DataPreprocessor.
    
    Args:
        processed_data_dir (str, optional): Path to processed data directory
        apply_fft (bool): Flag indicating whether to use FFT-processed data (if available)
        
    Returns:
        tuple: (train_dataloader, test_dataloader, metadata, train_dataset, test_dataset)
    """

    # Default processed data directory
    if processed_data_dir is None:
        processed_data_dir = Path("data/processed")
    else:
        processed_data_dir = Path(processed_data_dir)
    
    # Check if processed data exists
    if not processed_data_dir.exists():
        raise FileNotFoundError(f"Processed data directory not found: {processed_data_dir}")
    
    # Load metadata first to check if FFT was applied during preprocessing
    metadata = torch.load(processed_data_dir / 'neuralnetwork' / 'metadata.pt')
    # Check if FFT was applied during preprocessing
    data_has_fft = metadata.get('apply_fft', False)
    # Log warning if there's a mismatch between request and data
    if apply_fft != data_has_fft:
        print(f"Warning: Requested apply_fft={apply_fft} but data has apply_fft={data_has_fft}")
        print(f"Using data as is (apply_fft={data_has_fft})")
        apply_fft = data_has_fft
    
    # Load neural network data
    nn_dir = processed_data_dir / 'neuralnetwork'
    train_inputs = torch.load(nn_dir / 'train_inputs.pt')
    train_outputs = torch.load(nn_dir / 'train_outputs.pt')
    test_inputs = torch.load(nn_dir / 'test_inputs.pt')
    test_outputs = torch.load(nn_dir / 'test_outputs.pt')
    
    # Create datasets with the apply_fft flag from the metadata
    train_dataset = CBTDataset(train_inputs, train_outputs, apply_fft=apply_fft)
    test_dataset = CBTDataset(test_inputs, test_outputs, apply_fft=apply_fft)
    
    # Set normalization parameters if FFT was applied
    if apply_fft and 'normalization_params' in metadata:
        norm_params = metadata['normalization_params']
        if 'Y_fft_min' in norm_params and 'Y_fft_max' in norm_params:
            train_dataset.output_min = norm_params['Y_fft_min']
            train_dataset.output_max = norm_params['Y_fft_max']
            test_dataset.output_min = norm_params['Y_fft_min']
            test_dataset.output_max = norm_params['Y_fft_max']
    
    # Create dataloaders
    batch_size = min(128, len(train_dataset))
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
    
    print(f"Data loaded successfully:")
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Test samples: {len(test_dataset)}")
    print(f"  Input dimension: {train_inputs.shape[1]}")
    print(f"  Output dimension: {train_outputs.shape[1]}")
    print(f"  FFT applied: {apply_fft}")
    
    return train_dataloader, test_dataloader, metadata, train_dataset, test_dataset


def run_epoch(dataloader, epoch, model, loss_fn, optimizer, scheduler, mode="train", use_wandb=False):
    """
    Run one epoch of training or evaluation.
    
    Args:
        dataloader: DataLoader for the dataset
        epoch: Current epoch number
        model: Model to train/evaluate
        loss_fn: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        mode: "train" or "test"
        use_wandb: Whether to log metrics to wandb
        
    Returns:
        float: Average loss for the epoch
    """
    if mode == "train":
        model.train()
    else:
        model.eval()
        
    loss_accum = 0
    step = 0

    with tqdm(dataloader, dynamic_ncols=True) as tqdmDataLoader:
        for batch, (X, y) in enumerate(tqdmDataLoader):
            # Compute prediction and loss
            if mode == "train":
                optimizer.zero_grad()
                
            pred = model(X)
            loss_node = loss_fn(pred, y)
            loss_accum += loss_node.item()
            step += 1
            loss = loss_accum/step

            if mode == "train":
                # Backpropagation
                loss_node.backward()
                optimizer.step()
                if isinstance(scheduler, ReduceLROnPlateau):
                    scheduler.step(loss_node)
                else:
                    scheduler.step()

                # Log metrics to wandb if enabled
                if use_wandb:
                    wandb.log({
                        "train_epoch": epoch + 1, 
                        "train_loss": loss, 
                        "train_loss_node": loss_node.item(), 
                        "LR": optimizer.param_groups[0]["lr"]
                    })

                tqdmDataLoader.set_postfix(
                    ordered_dict={
                        "train_epoch": epoch + 1,
                        "train_loss": loss,
                        "train_loss_node": loss_node.item(),
                        "LR": optimizer.param_groups[0]["lr"]
                    }
                )
            else:
                # Log metrics to wandb if enabled
                if use_wandb:
                    wandb.log({
                        "test_epoch": epoch + 1, 
                        "test_loss": loss, 
                        "test_loss_node": loss_node.item()
                    })
                    
                tqdmDataLoader.set_postfix(
                    ordered_dict={
                        "test_epoch": epoch + 1,
                        "test_loss": loss,
                        "test_loss_node": loss_node.item()
                    }
                )

    return loss


def train_model(train_dataloader, test_dataloader, metadata, epochs=1000, lr=1e-3, save_dir=None, use_wandb=False, project_name="CoreTempAI"):
    """
    Train the neural network model.
    
    Args:
        train_dataloader: DataLoader for training data
        test_dataloader: DataLoader for test data
        metadata: Metadata from the preprocessor
        epochs: Number of training epochs
        lr: Initial learning rate
        save_dir: Directory to save the model
        use_wandb: Whether to use Weights & Biases for logging
        project_name: Name of the wandb project
        
    Returns:
        NeuralNetwork: Trained model
    """
    # Initialize wandb if enabled
    if use_wandb:
        wandb.init(project=project_name, config={
            "learning_rate": lr,
            "epochs": epochs,
            "input_dim": metadata.get('input_dim'),
            "output_dim": metadata.get('output_dim'),
            "batch_size": train_dataloader.batch_size,
            "optimizer": "Adam",
            "scheduler": "ReduceLROnPlateau",
            "device": str(device),
            "run_selection": metadata.get('run_selection')
        })
    
    # Get input and output dimensions from metadata
    input_dim = metadata.get('input_dim')
    output_dim = metadata.get('output_dim')
    
    # Initialize model
    print(f"Initializing model with input_dim={input_dim}, output_dim={output_dim}")
    model = NeuralNetwork(input_dim=input_dim, output_dim=output_dim).to(device)
    
    # Define loss function and optimizer
    loss_fn = nn.MSELoss(reduction="sum")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Define schedulers
    reduce_lr_scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=50, cooldown=50)
    lambda_scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0)
    
    # Watch model in wandb if enabled
    if use_wandb:
        wandb.watch(model, log="all")
    
    # Training loop
    start_time = time.time()
    train_losses = []
    test_losses = []
    
    for epoch in range(epochs):
        # Training
        train_loss = run_epoch(train_dataloader, epoch, model, loss_fn, optimizer, reduce_lr_scheduler, mode="train", use_wandb=use_wandb)
        train_losses.append(train_loss)
        
        # Evaluation
        with torch.no_grad():
            test_loss = run_epoch(test_dataloader, epoch, model, loss_fn, optimizer, lambda_scheduler, mode="test", use_wandb=use_wandb)
            test_losses.append(test_loss)
        
        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
    
    end_time = time.time()
    total_time = end_time - start_time

    print("Training completed!")
    print(f"Total training time: {total_time:.2f} seconds")
    
    # Save the model if a save directory is provided
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        model_path = os.path.join(save_dir, 'neural_network_model.pt')
        model.save(model_path)
        print(f"Model saved to {model_path}")
        
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
            wandb.save(model_path)
            wandb.save(history_path)
            wandb.save(os.path.join(save_dir, 'loss_curves.png'))
            
            # Log summary metrics
            wandb.run.summary["final_train_loss"] = train_losses[-1]
            wandb.run.summary["final_test_loss"] = test_losses[-1]
            wandb.run.summary["best_test_loss"] = min(test_losses)
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
        save_dir: Directory to save the visualizations
        use_wandb: Whether to log results to wandb
    """
    model.eval()
    
    # Limit the number of samples to visualize
    num_samples = min(num_samples, len(test_dataset))
    
    # Create directory for visualizations if provided
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
    
    # Compute predictions for all test samples
    with torch.no_grad():
        all_inputs = test_dataset.input_data
        all_outputs = test_dataset.output_data
        all_preds = model(all_inputs)
    
    # If FFT was applied, convert back to time domain
    if hasattr(test_dataset, 'fft_applied') and test_dataset.fft_applied:
        # Check if we have normalization parameters
        if not hasattr(test_dataset, 'output_min') or test_dataset.output_min is None:
            print("Warning: FFT was applied but normalization parameters are missing.")
            # Use the data as is
            pred_final = all_preds
            output_final = all_outputs
        else:
            # Denormalize predictions and outputs
            pred_unnorm = test_dataset.output_min + ((test_dataset.output_max - test_dataset.output_min) * all_preds)
            output_unnorm = test_dataset.output_min + ((test_dataset.output_max - test_dataset.output_min) * all_outputs)
            
            # Reshape to separate real and imaginary parts
            batch_size = pred_unnorm.shape[0]
            pred_real_part = pred_unnorm[:, ::2]
            pred_imag_part = pred_unnorm[:, 1::2]
            pred_fft = torch.complex(pred_real_part, pred_imag_part)
            
            output_real_part = output_unnorm[:, ::2]
            output_imag_part = output_unnorm[:, 1::2]
            output_fft = torch.complex(output_real_part, output_imag_part)
            
            # Convert back to time domain
            pred_final = torch.fft.irfft(pred_fft, 100, dim=1)
            output_final = torch.fft.irfft(output_fft, 100, dim=1)
    else:
        # Use outputs and predictions directly
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
    x = np.linspace(0, 30, 100)  # Assuming 30 minutes and 100 points
    
    for i in range(num_samples):
        plt.figure(figsize=(10, 6))
        
        # Convert to numpy and Celsius if needed
        output_temp = output_final[i].cpu().numpy()
        pred_temp = pred_final[i].cpu().numpy()
        
        # Convert from Kelvin to Celsius if values are above 100
        if np.mean(output_temp) > 100:
            output_temp -= 273.15
            pred_temp -= 273.15
        
        plt.plot(x, output_temp, "-", lw=1.5, label="True Core Body Temperature")
        plt.plot(x, pred_temp, "--", lw=2, label="Predicted Core Body Temperature")
        plt.xlabel("Time (min)")
        plt.ylabel("Core Body Temperature (°C)")
        plt.title(f"Sample {i+1}")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
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


def main(processed_data_dir=None, apply_fft=True, epochs=1000, lr=1e-3, save_dir=None, use_wandb=False, project_name="CoreTempAI"):
    """
    Main function to train and evaluate the neural network model.
    
    Args:
        processed_data_dir: Path to processed data directory
        apply_fft: Whether to use FFT-processed data
        epochs: Number of training epochs
        lr: Initial learning rate
        save_dir: Directory to save the model and results
        use_wandb: Whether to use Weights & Biases for logging
        project_name: Name of the wandb project
    """
    # Load data
    train_dataloader, test_dataloader, metadata, train_dataset, test_dataset = load_data_from_preprocessor(
        processed_data_dir=processed_data_dir,
        apply_fft=apply_fft
    )
    
    # Train model
    model = train_model(
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        metadata=metadata,
        epochs=epochs,
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
        apply_fft=True,     # Use the FFT-processed data (should match what was used in DataPreprocessor.prepare_data)
        epochs=1000,
        lr=1e-3,
        save_dir="data/models/neural_network",
        use_wandb=True,     # Enable wandb logging
        project_name="CoreTempAI"
    )