"""
Neural Network model implementation for CoreTempAI.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from pathlib import Path

class NeuralNetwork(nn.Module):
    """
    Neural Network model for temperature prediction.
    
    This is a flexible feedforward neural network that automatically adapts
    to the number of input parameters defined in input_parameters.py.
    """
    
    def __init__(self, input_dim=None, output_dim=None, hidden_dims=None):
        """
        Initialize the neural network with flexible architecture.
        
        Args:
            input_dim (int, optional): Dimension of input features.
                If None, will be determined from metadata.
            output_dim (int, optional): Dimension of output.
                If None, will be determined from metadata.
            hidden_dims (list, optional): List of hidden layer dimensions.
                If None, will use a default architecture scaled to input size.
        """
        super().__init__()
        
        # Try to load metadata if dimensions not provided
        if input_dim is None or output_dim is None:
            metadata = self._load_metadata()
            input_dim = input_dim or metadata.get('input_dim', 5)
            output_dim = output_dim or metadata.get('output_dim', 102)
        
        # Default hidden dimensions if not provided
        if hidden_dims is None:
            # Scale architecture based on input dimension
            scale_factor = max(1, input_dim // 5)  # Scale based on input size
            hidden_dims = [
                32 * scale_factor,
                64 * scale_factor,
                128 * scale_factor,
                256 * scale_factor,
                256 * scale_factor,
                256 * scale_factor,
                128 * scale_factor,
                64 * scale_factor,
                32 * scale_factor
            ]
        
        # Store dimensions for reference
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        
        # Build the network architecture
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(nn.Tanh())
        
        # Hidden layers
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            layers.append(nn.Tanh())
        
        # Output layer
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        layers.append(nn.Tanh())
        
        # Create sequential model
        self.linear_relu_stack = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier initialization."""
        for layer in self.linear_relu_stack:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain('tanh'))
                nn.init.zeros_(layer.bias)
    
    def _load_metadata(self):
        """
        Load metadata from processed data directory.
        
        Returns:
            dict: Metadata dictionary or empty dict if not found
        """
        try:
            # Try to find metadata in standard locations
            possible_paths = [
                Path("data/processed/metadata.pt"),
                Path("../data/processed/metadata.pt"),
                Path("../../data/processed/metadata.pt")
            ]
            
            for path in possible_paths:
                if path.exists():
                    return torch.load(path)
            
            print("Warning: Could not find metadata file. Using default dimensions.")
            return {}
        except Exception as e:
            print(f"Error loading metadata: {e}")
            return {}

    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim)
        """
        # Validate input dimension
        if x.shape[1] != self.input_dim:
            raise ValueError(f"Expected input dimension {self.input_dim}, got {x.shape[1]}")
        
        pred = self.linear_relu_stack(x)
        return pred
    
    def save(self, filepath):
        """
        Save the model to a file.
        
        Args:
            filepath (str): Path to save the model
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save model with architecture information
        torch.save({
            'state_dict': self.state_dict(),
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'hidden_dims': self.hidden_dims
        }, filepath)
        print(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath, device=None):
        """
        Load a model from a file.
        
        Args:
            filepath (str): Path to the saved model
            device (torch.device, optional): Device to load the model to
            
        Returns:
            NeuralNetwork: Loaded model
        """
        # Load checkpoint
        checkpoint = torch.load(filepath, map_location=device)
        
        # Create model with saved architecture
        model = cls(
            input_dim=checkpoint.get('input_dim'),
            output_dim=checkpoint.get('output_dim'),
            hidden_dims=checkpoint.get('hidden_dims')
        )
        
        # Load state dict
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        
        return model 
