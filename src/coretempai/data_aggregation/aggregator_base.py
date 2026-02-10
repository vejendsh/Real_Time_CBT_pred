"""
Abstract base aggregator for CoreTempAI.

This module defines the abstract base class for data aggregators.
"""

from abc import ABC, abstractmethod
from pathlib import Path
import re
from typing import Dict, List, Tuple, Union, Optional, Any
import numpy as np
import pandas as pd

class DataAggregatorBase(ABC):
    """
    Abstract base class for data aggregators.
    
    This class defines the interface that all data aggregators must implement.
    Concrete implementations handle specific data formats like CSV files,
    Fluent case files with .out files, etc.
    """
    
    def __init__(self, run_dir: Path):
        """
        Initialize the data aggregator.
        
        Args:
            run_dir (Path): Path to the run directory
        """
        self.run_dir = run_dir
        self.run_name = run_dir.name
        self.available_scalar_features = []
        self.available_profile_features = []
        
    @abstractmethod
    def aggregate_scalar_inputs(self, parameter_names: Optional[List[str]] = None) -> Dict[int, Any]:
        """
        Aggregate scalar input parameters from a run directory.
        
        Args:
            parameter_names (List[str], optional): Names of parameters to extract.
                                                If None, all parameters will be extracted.
            
        Returns:
            Dict[int, Any]: Dictionary mapping iteration to scalar input data
        """
        pass
    
    @abstractmethod
    def aggregate_profile_inputs(self) -> Dict[int, pd.DataFrame]:
        """
        Aggregate profile input parameters from a run directory.
        
        Returns:
            Dict[int, pd.DataFrame]: Dictionary mapping iteration to profile DataFrame
        """
        pass
    
    @abstractmethod
    def aggregate_outputs(self) -> Dict[int, np.ndarray]:
        """
        Aggregate output data from a run directory.
        
        Returns:
            Dict[int, np.ndarray]: Dictionary mapping iteration to output data
        """
        pass
    
    @staticmethod
    def extract_iteration(filename: str) -> int:
        """
        Extract iteration number from a filename.
        
        Args:
            filename (str): Filename to extract iteration from
            
        Returns:
            int: Iteration number
        """
        # Traditional format (e.g., '_1234' or '-1234')
        match = re.search(r'[-_](\d+)', filename)
        if match:
            return int(match.group(1))
        
        if ".out" in filename:
            # Report-def format (e.g., 'report-def-0-rfile-5-00000.out')
            match = re.search(r'_(\d+)_', filename)
            if match:
                return int(match.group(1))
            else:
                return 0
        
        # Handle the case for the first iteration (no number in filename)
        return 0
    
    def check_directory_exists(self, directory: Path, name: str = "Directory") -> None:
        """
        Check if a directory exists and raise a FileNotFoundError if it doesn't.
        
        Args:
            directory (Path): Directory to check
            name (str): Name of the directory (for error message)
            
        Raises:
            FileNotFoundError: If the directory doesn't exist
        """
        if not directory.exists():
            raise FileNotFoundError(f"{name} not found: {directory}")

    def get_available_features(self) -> Dict[str, List[str]]:
        """
        Get information about available features.
        
        Returns:
            Dict[str, List[str]]: Dictionary containing feature information
        """
        return {
            "scalar_features": self.available_scalar_features,
            "profile_features": self.available_profile_features
        } 