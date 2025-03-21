"""
Neural Operator model implementation for CoreTempAI.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
from pathlib import Path
from CoreTempAI.config import NeuralOperatorConfig

config = NeuralOperatorConfig()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class NeuralOperator(nn.Module):
    """
    Neural Operator model for temperature prediction.
    
    This is a flexible implementation of a neural operator that automatically adapts
    to the number of input parameters defined in input_parameters.py.
    """
    
    def __init__(self, vectors=None, num_functions=None, config=None):
        """
        Initialize the neural operator.
        
        Args:
            vectors (list, optional): List of tuples (num_vectors, vector_dim).
                If None, will be determined from metadata.
            num_functions (int, optional): Number of functions.
                If None, will be determined from metadata.
            config (NeuralOperatorConfig, optional): Configuration object with model parameters.
                If None, will use default configuration.
        """
        super(NeuralOperator, self).__init__()
        
        # Try to load metadata if parameters not provided
        metadata = None
        if vectors is None or num_functions is None:
            metadata = self._load_metadata()
            
            if vectors is None:
                input_dim = metadata.get('input_dim', 5)
                vectors = [(input_dim, 1)]
            
            if num_functions is None:
                num_functions = metadata.get('num_profile_features', 1)
        
        # Use provided config or create default
        if config is None:
            try:
                from ..config import NeuralOperatorConfig
                config = NeuralOperatorConfig()
            except ImportError:
                # Create a simple default config if import fails
                config = type('NeuralOperatorConfig', (), {
                    'num_freqs': 51,
                    'mlp_hidden_dim': 128,
                    'd_model': 256,
                    'mlp_neg_slope': 0.2,
                    'n_heads': 4,
                    'd_ff': 512,
                    'dropout': 0.1,
                    'n_layers': 4,
                    'generator_mode': 'slice',
                    'slice_range': [0, -1]
                })
        
        # Store configuration
        self.config = config
        self.vectors = vectors
        self.num_functions = num_functions
            
        # Initialize embedders for both vectors and functions
        embedders = []
        self.num_freqs = getattr(config, 'num_freqs', 51)
        
        # Vector embedders - one per vector
        for num_vectors, vector_dim in vectors:
            embedders.extend([MLP([vector_dim, config.mlp_hidden_dim, config.d_model], 
                                 neg_slope=config.mlp_neg_slope) for _ in range(num_vectors)])
        
        # Function embedders - one per function
        embedders.extend([MLP([2, config.mlp_hidden_dim, config.d_model], 
                             neg_slope=config.mlp_neg_slope) for _ in range(num_functions)])
        
        self.num_vector_embedders = sum(num_vectors for num_vectors, _ in vectors)
        self.embedders = nn.ModuleList(embedders)
        self.decoder = Decoder(config.n_heads, config.d_model, config.d_ff, config.dropout, config.n_layers)
        
        # Get output_profile_freq_components from metadata if available, otherwise use default
        num_tokens = None
        if metadata is not None and 'output_profile_freq_components' in metadata:
            num_tokens = metadata['output_profile_freq_components']
            print(f"Using output_profile_freq_components={num_tokens} from metadata for num_tokens")
        
        # Initialize generator with num_tokens if available
        self.generator = Generator(config.d_model, 
                                  mode=config.generator_mode, 
                                  slice_range=getattr(config, 'slice_range', [0, -1]),
                                  num_tokens=num_tokens or getattr(config, 'output_profile_freq_components', 13))
    
    def _load_metadata(self):
        """
        Load metadata from processed data directory.
        
        Returns:
            dict: Metadata dictionary or empty dict if not found
        """
        try:
            # Try to find neural operator specific metadata first
            possible_no_paths = [
                Path("data/processed/neural_operator/metadata.pt"),
                Path("../data/processed/neural_operator/metadata.pt"),
                Path("../../data/processed/neural_operator/metadata.pt"),
                Path("data/processed/fluent_data/neural_operator/metadata.pt"),  # Add Fluent data paths
                Path("../data/processed/fluent_data/neural_operator/metadata.pt"),
                Path("../../data/processed/fluent_data/neural_operator/metadata.pt"),
                Path("data/processed/metadata.pt"),  # Fallback to legacy metadata
                Path("../data/processed/metadata.pt"),
                Path("../../data/processed/metadata.pt")
            ]
            
            for path in possible_no_paths:
                if path.exists():
                    metadata = torch.load(path)
                    print(f"Loaded metadata from {path}")
                    return metadata
            
            print("Warning: Could not find metadata file. Using default dimensions.")
            return {}
        except Exception as e:
            print(f"Error loading metadata: {e}")
            return {}
    
    def forward(self, input):
        "Take in and process masked src and target sequences."
        return self.generator(self.decode(input))

    def decode(self, input):
        """
        Decode input tensors into embeddings.
        
        Args:
            input: List of input tensors [scalar_tensor, function_tensor]
        
        Returns:
            torch.Tensor: Decoded embeddings
        """
        embeddings = []
        
        # Verify input format
        if not isinstance(input, list):
            raise ValueError(f"Expected input to be a list of tensors, got {type(input)}")
        
        if len(input) == 0:
            raise ValueError("Input list is empty")
        
        # Process vectors from first tensor (scalar inputs)
        # Reshape if needed: input[0] should be [batch_size, num_scalars]
        scalar_input = input[0]
        
        # Handle different input tensor shapes
        if len(scalar_input.shape) == 2:
            # Shape is [batch_size, num_scalars]
            # Reshape to [batch_size, num_scalars, 1] for vector embedders
            scalar_input = scalar_input.unsqueeze(-1)
        elif len(scalar_input.shape) == 3 and scalar_input.shape[2] != 1:
            # If the last dimension isn't 1, reshape appropriately
            print(f"Warning: Unusual scalar input shape {scalar_input.shape}, reshaping")
        
        # Process each scalar input with its embedder
        for i in range(min(self.num_vector_embedders, scalar_input.shape[1])):
            vector_embedding = self.embedders[i](scalar_input[:, i:i+1, :])
            embeddings.append(vector_embedding)
        
        # Process function inputs if available
        if len(input) > 1 and input[1] is not None:
            function_input = input[1]
            
            # Check function input shape
            if len(function_input.shape) == 3:  # Expected shape: [batch_size, seq_len, 1 or 2]
                # If the third dimension is 1, we need to expand it to 2 for embedders
                if function_input.shape[2] == 1:
                    # Add a zero column to match expected input
                    # Print warning about unusual function input shape
                    print(f"Warning: Unusual function input shape {function_input.shape}, expanding dimension")
                    zeros = torch.zeros_like(function_input)
                    function_input = torch.cat([function_input, zeros], dim=2)
                
                # Process the function with embedders
                for i in range(self.num_vector_embedders, len(self.embedders)):
                    func_idx = i - self.num_vector_embedders
                    
                    # For Fluent data, we typically just have one function
                    if func_idx == 0:
                        func_embedding = self.embedders[i](function_input)
                        embeddings.append(func_embedding)
                    else:
                        # Skip additional function embedders if we don't have data for them
                        print(f"Skipping function embedder {i} as we only have one function input")
            else:
                print(f"Warning: Unexpected function input shape {function_input.shape}")
        
        # Ensure we have at least one embedding
        if not embeddings:
            raise ValueError("No embeddings were generated from the input data")
        
        # Ensure all embeddings have 3 dimensions [batch_size, seq_len, embedding_dim]
        for i in range(len(embeddings)):
            # If shape is [batch_size, embedding_dim], add a sequence dimension
            if len(embeddings[i].shape) == 2:
                print(f"Warning: Embedding {i} has shape {embeddings[i].shape}, adding sequence dimension")
                embeddings[i] = embeddings[i].unsqueeze(1)
        
        # Concatenate embeddings along sequence dimension (dim=1)
        try:
            # Check dimensions for compatibility
            shapes = [emb.shape for emb in embeddings]
            embed_dims = [shape[2] for shape in shapes]
            
            # Verify all embedding dimensions match
            if not all(dim == embed_dims[0] for dim in embed_dims):
                print(f"Warning: Embedding dimensions don't match: {shapes}")
                # Try to reshape to match dimensions
                for i in range(len(embeddings)):
                    if embeddings[i].shape[2] != embed_dims[0]:
                        print(f"Reshaping embedding {i} from {embeddings[i].shape}")
                        # Use a projection layer to match dimensions
                        proj = nn.Linear(embeddings[i].shape[2], embed_dims[0]).to(embeddings[i].device)
                        embeddings[i] = proj(embeddings[i])
            
            stacked = torch.cat(embeddings, dim=1)
            return self.decoder(stacked)
        except RuntimeError as e:
            print(f"Error stacking embeddings: {e}")
            print(f"Embedding shapes: {[emb.shape for emb in embeddings]}")
            # Fallback: use just the first embedding if we can't stack
            return self.decoder(embeddings[0])
            
        
    
    def save(self, filepath):
        """
        Save the model to a file.
        
        Args:
            filepath (str): Path to save the model
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save model with architecture information
        config_dict = {k: getattr(self.config, k) for k in dir(self.config) 
                      if not k.startswith('_') and not callable(getattr(self.config, k))}
        
        torch.save({
            'state_dict': self.state_dict(),
            'vectors': self.vectors,
            'num_functions': self.num_functions,
            'config': config_dict
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
            NeuralOperator: Loaded model
        """
        # Load checkpoint
        checkpoint = torch.load(filepath, map_location=device)
        
        # Create config object
        config_dict = checkpoint.get('config', {})
        config = type('NeuralOperatorConfig', (), config_dict)
        
        # Create model with saved architecture
        model = cls(
            vectors=checkpoint.get('vectors'),
            num_functions=checkpoint.get('num_functions'),
            config=config
        )
        
        # Load state dict
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        
        return model


class Generator(nn.Module):

    def __init__(self, d_model, mode='slice', slice_range=None, num_tokens=None):
        super(Generator, self).__init__()
        self.d_model = d_model
        self.mode = mode
        
        # Set default values if not provided
        if slice_range is None:
            slice_range = [0, -1]  # Default to use the whole sequence
            
        if mode == 'project' and num_tokens is None:
            # Use a default value if no explicit num_tokens is provided
            num_tokens = 13  # Default value
            print(f"Warning: Using default num_tokens={num_tokens} for Generator in project mode")
            
        self.slice_range = slice_range
        self.num_tokens = num_tokens
        self.unembedder = MLP([d_model, 32, 2])
        
        if mode == 'project':
            self.projection = nn.Linear(d_model, d_model)
            self.pool = nn.Parameter(torch.randn(num_tokens, d_model))

    def forward(self, x):
        if self.mode == 'slice':
            # Handle negative indices and ensure end index doesn't exceed sequence length
            start_idx = self.slice_range[0] if self.slice_range[0] >= 0 else x.size(1) + self.slice_range[0]
            end_idx = self.slice_range[1] if self.slice_range[1] >= 0 else x.size(1) + self.slice_range[1]
            end_idx = min(end_idx + 1, x.size(1))  # Add 1 for inclusive end, but don't exceed sequence length
            x = x[:, start_idx:end_idx, :]
        else:  # project mode
            projected = self.projection(x)
            attention_weights = torch.matmul(projected, self.pool.T)
            attention_weights = attention_weights.softmax(dim=1)
            x = torch.matmul(attention_weights.transpose(1, 2), x)
            
        return self.unembedder(x)


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class Decoder(nn.Module):
    "Generic N layer decoder with masking."

    def __init__(self,  h, d_model, d_ff, dropout, N):
        super(Decoder, self).__init__()
        self.layer_blocks = []
        for _ in range(N):
            self.layer_blocks.append(DecoderLayer(h, d_model, d_ff, dropout))
        self.layers = nn.ModuleList(self.layer_blocks)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"
    
    def __init__(self, h, d_model, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadedAttention(h, d_model)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm = LayerNorm(d_model)

    def forward(self, x):
        x_n = self.norm(x)
        h = x + self.self_attn(x_n, x_n, x_n)

        x = h + self.feed_forward(self.norm(h))
        return x


def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = scores.softmax(dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        self.d_model = d_model
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = [MLP([d_model, d_model]) for i in range(4)]

        self.linears = nn.ModuleList(self.linears)

        for i in self.linears:
            i.layers[0].bias = None

        self.attn = None
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(p=self.dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [
            lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
        ]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(
            query, key, value, mask=mask, dropout=self.dropout_layer
        )

        # 3) "Concat" using a view and apply a final linear.
        x = (
            x.transpose(1, 2)
            .contiguous()
            .view(nbatches, -1, self.h * self.d_k)
        )
        del query
        del key
        del value
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        return self.w_2(self.dropout(torch.relu(self.w_1(x))))


class MLP(nn.Module):
    def __init__(self, layer_sizes, neg_slope=2):
        super(MLP, self).__init__()
        self.layer_sizes = layer_sizes
        self.neg_slope = neg_slope
        self.layers = []

        for i in range(len(layer_sizes) - 2):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            self.layers.append(nn.LeakyReLU(negative_slope=neg_slope))
        self.layers.append(nn.Linear(layer_sizes[-2], layer_sizes[-1]))

        self.layers = nn.ModuleList(self.layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def save(self, filepath):
        """
        Save the model to a file.
        
        Args:
            filepath (str): Path to save the model
        """
        torch.save({
            'state_dict': self.state_dict(),
            'config': self.config.to_dict()
        }, filepath)
    
    @classmethod
    def load(cls, filepath):
        """
        Load a model from a file.
        
        Args:
            filepath (str): Path to the saved model
            
        Returns:
            NeuralOperator: Loaded model
        """
        from CoreTempAI.config import NeuralOperatorConfig
        
        checkpoint = torch.load(filepath)
        config = NeuralOperatorConfig.from_dict(checkpoint['config'])
        
        model = cls(config.vectors, config.num_functions, config)
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        
        return model 