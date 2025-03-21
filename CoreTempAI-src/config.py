"""
Configuration settings for the project.

This module contains all configuration settings organized into dictionaries
for better structure and easier access.
"""

import os
import torch
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Directory structure configuration
DIRECTORIES = {
    "raw": os.path.join(PROJECT_ROOT, "data", "raw"),
    "root": PROJECT_ROOT,
    "case": os.path.join(PROJECT_ROOT, "cases", "exercise", "with_sweating", "case_1"),
}

# File paths configuration
FILES = {
    "case_file": os.path.join(DIRECTORIES["case"], "Case1.cas.h5"),
    # Uncomment to enable journal files
    # "params_journal": os.path.join(DIRECTORIES["case"], "journals", "params.log"),
    # "run_journal": os.path.join(DIRECTORIES["case"], "journals", "run.log"),
}

# Profile generation settings
PROFILE_CONFIG = {
    "duration": 1800,  # Duration in seconds (1 hour)
    "step_size": 18,   # Step size in seconds (1 minute)
}

# Fourier sampler settings
FOURIER_CONFIG = {
    "max_freq": 10,       # Maximum frequency for Fourier sampler
    "nu_range": [0, 1],   # Default range for normalized values
}

# Simulation parameters
SIMULATION_CONFIG = {
    "processor_count": 4,
    "precision": "DOUBLE",
    "dimension": "THREE",
}

DATA_GENERATOR_CONFIG = {
    "num_data": 10,  # Number of data points to be generated
}

# Neural Operator Configuration
class NeuralOperatorConfig:
    """
    Configuration class for Neural Operator model and related components.
    This centralizes all parameters used across the codebase.
    """
    
    def __init__(self):
        # ===== DOMAIN PARAMETERS =====
        # Spatial domain
        self.x_min = 0
        self.x_max = 1
        
        # Temporal domain
        self.t_min = 0
        self.t_max = 0.01
        self.dt = 0.00001
        self.dx = 0.01
        
        # ===== SIMULATION PARAMETERS =====
        # Sample size
        self.n = 1000  # Number of samples
        
        # Frequency parameters
        self.u_max_freq = 30  # Maximum frequency for initial condition
        self.nu_max_freq = 5  # Maximum frequency for viscosity
        self.u_min = -1 # Minimum value for initial condition


        # ===== MODEL ARCHITECTURE =====
        # Transformer parameters
        self.d_model = 64      # Dimension of model
        self.d_ff = 128        # Dimension of feed-forward network
        self.n_heads = 8       # Number of attention heads
        self.n_layers = 10     # Number of decoder layers
        self.dropout = 0.1     # Dropout rate
        
        # MLP parameters
        self.mlp_hidden_dim = 32  # Hidden dimension for MLPs
        self.mlp_neg_slope = 2    # Negative slope for LeakyReLU
        
        # Generator parameters
        self.generator_mode = 'project'  # 'slice' or 'project'
        self.slice_range = [0, -1]  # Default slice range
        
        # Frequency parameters
        self.num_freqs = 51     # Number of frequencies for function embedders
        
        # ===== TRAINING PARAMETERS =====
        # Optimization
        self.batch_size = 1000
        self.learning_rate = 0.001
        self.epochs = 80
        self.accum_iter = 1    # Gradient accumulation steps
        
        # Scheduler
        self.scheduler_type = 'reduce_plateau'  # 'exponential', 'constant', 'cosine', 'reduce_plateau'
        self.scheduler_factor = 0.5             # Factor by which to reduce learning rate (for ReduceLROnPlateau)
        self.scheduler_patience = 1000          # Number of epochs with no improvement after which LR will be reduced
        self.scheduler_cooldown = 1000          # Number of epochs to wait before resuming normal operation
        
        # Loss function
        self.loss_type = 'mse'  # 'mse', 'l1', etc.
        
        # ===== SYSTEM PARAMETERS =====
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Paths
        self.model_save_path = os.path.join("models", "neural_operator")
        self.results_path = os.path.join("results", "neural_operator")
        
        # Logging
        self.wandb_project = "cbt-pred-neural-operator"
        self.log_interval = 1
        self.use_wandb = False  # Whether to use Weights & Biases for logging
    
    @classmethod
    def from_dict(cls, param_dict):
        """Create config from dictionary of parameters"""
        config = cls()
        for key, value in param_dict.items():
            if hasattr(config, key):
                setattr(config, key, value)
            else:
                print(f"Warning: Parameter '{key}' not found in config")
        return config
    
    def to_dict(self):
        """Convert config to dictionary"""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
    
    def save(self, filepath, overwrite_protection=True):
        """
        Save config to file with optional overwrite protection.
        
        Args:
            filepath (str): Path where the config should be saved
            overwrite_protection (bool): Whether to use overwrite protection
            
        Returns:
            str: The actual filepath used for saving
        """
        from CoreTempAI.utils.utils import safe_save
        if overwrite_protection:
            return safe_save(filepath, torch.save, self.to_dict())
        else:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            torch.save(self.to_dict(), filepath)
            return filepath
    
    @classmethod
    def load(cls, filepath):
        """Load config from file"""
        config_dict = torch.load(filepath)
        return cls.from_dict(config_dict)




# # Journal file paths
# PARAMS_JOURNAL = CASE_DIR / r"journals" / r"params.log"
# RUN_JOURNAL = CASE_DIR / r"journals" / r"run.log"


