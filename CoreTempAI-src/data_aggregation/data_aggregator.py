"""
Data aggregator for CoreTempAI.

This module provides functionality to aggregate data from simulation run directories
and create a unified cache for further processing.
"""

import os
import re
import time
import glob
from pathlib import Path
import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union, Optional, Any
from tqdm import tqdm
import importlib
import torch.serialization

# Add safe globals for numpy arrays
torch.serialization.add_safe_globals(['numpy.core.multiarray._reconstruct', 'numpy.core.multiarray.scalar'])


class DataAggregator:
    """
    Unified data aggregator for CoreTempAI.
    
    This class is responsible for analyzing the run directories, aggregating data,
    and creating a cache for faster subsequent access.
    """
    
    def __init__(self, raw_data_dir: str, aggregate_dir: str = None):
        """
        Initialize the data aggregator.
        
        Args:
            raw_data_dir (str): Path to the raw data directory
            cache_dir (str, optional): Path to store cache data
        """
        self.raw_data_dir = Path(raw_data_dir)
        
        if aggregate_dir is None:
            # Default to a 'cache' directory at the same level as raw
            self.aggregate_dir = self.raw_data_dir.parent / 'aggregated'
        else:
            self.aggregate_dir = Path(aggregate_dir)
            
        # Create cache directory if it doesn't exist
        os.makedirs(self.aggregate_dir, exist_ok=True)
        
        # Discover run directories
        self.run_dirs = []
        
        # Load parameter definitions from input_parameters.py
        self.load_parameter_definitions()
        
    def load_parameter_definitions(self):
        """
        Load parameter definitions from input_parameters.py.
        
        This allows the aggregator to automatically adapt to any changes
        in the parameter definitions.
        """
        try:
            # Dynamically import the input_parameters module
            input_params = importlib.import_module('CoreTempAI.input_parameters.input_parameters')
            
            # Get scalar and profile parameters
            self.scalar_params = getattr(input_params, 'scalar_params', {})
            self.profile_params = getattr(input_params, 'profile_params', {})
            
            # Extract parameter names
            self.scalar_param_names = list(self.scalar_params.keys())
            self.profile_param_names = list(self.profile_params.keys())
            
            print(f"Loaded parameter definitions:")
            print(f"  Scalar parameters: {self.scalar_param_names}")
            print(f"  Profile parameters: {self.profile_param_names}")
            
        except (ImportError, AttributeError) as e:
            print(f"Warning: Could not load parameter definitions from input_parameters.py: {e}")
            print("Using empty parameter definitions.")
            self.scalar_params = {}
            self.profile_params = {}
            self.scalar_param_names = []
            self.profile_param_names = []
            
    def discover_run_directories(self) -> List[Path]:
        """
        Discover all run directories in the raw data directory.
        
        Returns:
            List[Path]: List of paths to run directories
        """
        run_pattern = re.compile(r"Run_(\d+)")
        run_dirs = [d for d in self.raw_data_dir.glob("Run_*") 
                   if d.is_dir() and run_pattern.match(d.name)]
        
        # Sort by run number
        run_dirs.sort(key=lambda x: int(re.search(r"Run_(\d+)", x.name).group(1)))
        
        self.run_dirs = run_dirs
        return run_dirs
        
    def has_out_files(self, run_dir: Path) -> bool:
        """
        Check if the run directory contains .out files in the output_profile directory.
        
        Args:
            run_dir (Path): Path to the run directory
            
        Returns:
            bool: True if .out files are found, False otherwise
        """
        output_dir = run_dir / "output_profile"
        if output_dir.exists():
            out_files = list(output_dir.glob("*.out"))
            return len(out_files) > 0
        return False
        
    def create_appropriate_aggregator(self, run_dir: Path) -> 'DataAggregatorBase':
        """
        Create the appropriate aggregator based on the run directory content.
        
        Args:
            run_dir (Path): Path to the run directory
            
        Returns:
            DataAggregatorBase: Appropriate aggregator instance
        """
        from CoreTempAI.data_aggregation.default_aggregator import DefaultAggregator
        from CoreTempAI.data_aggregation.custom_aggregator import CustomAggregator
        
        # Simply check for .out files
        if self.has_out_files(run_dir):
            print(f"Using CustomAggregator for {run_dir.name} (found .out files)")
            return CustomAggregator(run_dir)
        else:
            print(f"Using DefaultAggregator for {run_dir.name}")
            return DefaultAggregator(run_dir)
            
    def aggregate_run_data(self, run_dir: Path, parameter_names: Optional[List[str]] = None,
                         use_cache: bool = True) -> Tuple[Dict[int, Any], Dict[int, Any], Dict[int, np.ndarray]]:
        """
        Aggregate data from a single run directory with automatic type detection.
        
        Args:
            run_dir (Path): Path to the run directory
            parameter_names (List[str], optional): Names of parameters to extract.
                                                If None, all parameters will be extracted.
            use_cache (bool): Whether to use cached data if available
            
        Returns:
            Tuple containing:
            - Dictionary of scalar inputs by iteration
            - Dictionary of profile inputs by iteration
            - Dictionary of core temperature outputs by iteration
        """
        run_name = run_dir.name
        aggregate_file = self.aggregate_dir / f"{run_name}_aggregate.pt"
        
        # Check if cache exists and is valid
        if use_cache and aggregate_file.exists():
            try:
                # Get modification time of run directory and cache file
                run_mtime = max(f.stat().st_mtime for f in run_dir.glob('**/*') if f.is_file())
                aggregate_mtime = aggregate_file.stat().st_mtime
                
                # Cache is valid if it's newer than the most recently modified file in run directory
                if aggregate_mtime > run_mtime:
                    print(f"Loading data for {run_name} from cache")
                    aggregate_data = torch.load(aggregate_file, weights_only=False)
                    
                    # Get the cached data
                    scalar_inputs = aggregate_data.get('scalar_inputs', {})
                    profile_inputs = aggregate_data.get('profile_inputs', {})
                    outputs = aggregate_data.get('outputs', {})
                    
                    # # Filter scalar inputs if parameter_names is provided
                    # if parameter_names and isinstance(next(iter(scalar_inputs.values()), {}), dict):
                    #     filtered_scalar_inputs = {}
                    #     for iter_num, iter_data in scalar_inputs.items():
                    #         filtered_scalar_inputs[iter_num] = {
                    #             k: v for k, v in iter_data.items() if k in parameter_names
                    #         }
                    #     scalar_inputs = filtered_scalar_inputs
                    
                    return scalar_inputs, profile_inputs, outputs
            except Exception as e:
                print(f"Error checking cache for {run_name}: {e}")
        
        print(f"Aggregating data for {run_name} from source files")
        
        # Create appropriate aggregator
        aggregator = self.create_appropriate_aggregator(run_dir)
        
        # Aggregate data
        try:
            scalar_inputs = aggregator.aggregate_scalar_inputs(parameter_names)
            profile_inputs = aggregator.aggregate_profile_inputs()
            outputs = aggregator.aggregate_outputs()

                        
            # Add size information for profile inputs and outputs if available
            if profile_inputs:
                profile_size = profile_inputs[0].shape
            else:
                profile_size = None
            if outputs:
                output_size = outputs[0].shape
            else:
                output_size = None
            
            # Cache the data
            aggregate_data = {
                'scalar_inputs': scalar_inputs,
                'profile_inputs': profile_inputs,
                'profile_size': profile_size,
                'output_size': output_size,
                'outputs': outputs,
                'features': aggregator.get_available_features(),
                'timestamp': time.time()
            }


            torch.save(aggregate_data, aggregate_file, pickle_protocol=4)
            print(f"Aggregate data for {run_name}: {aggregate_data}")
            
            return scalar_inputs, profile_inputs, outputs
            
        except Exception as e:
            print(f"Error aggregating data from {run_name}: {e}")
            import traceback
            traceback.print_exc()
            return {}, {}, {}
    
    def aggregate_all_data(self, use_cache: bool = True, run_selection: List[str] = None,
                         parameter_names: List[str] = None) -> Dict[str, Any]:
        """
        Aggregate data from all selected run directories and save it to disk.
        
        Args:
            use_cache (bool): Whether to use cached data if available
            run_selection (List[str], optional): List of run names to include
            parameter_names (List[str], optional): Names of parameters to extract.
                                                If None, all parameters will be extracted.
            
        Returns:
            Dict[str, Any]: Dictionary containing loaded data and metadata
        """
        # If parameter_names is not provided, use names from scalar_params
        if parameter_names is None:
            parameter_names = self.scalar_param_names
            print(f"Using parameter names from scalar_params: {parameter_names}")
            
        # Discover run directories if not already done
        if not self.run_dirs:
            self.discover_run_directories()
        
        # Filter run directories based on selection
        selected_runs = self.run_dirs
        if run_selection is not None:
            selected_runs = [run_dir for run_dir in self.run_dirs if run_dir.name in run_selection]
            print(f"Selected {len(selected_runs)} out of {len(self.run_dirs)} run directories")
        
        # Check if we have any runs to process
        if not selected_runs:
            print("No run directories found or selected")
            return {}
            
        # Create a unique name for the aggregate cache based on run selection
        run_names = "_".join([run_dir.name for run_dir in selected_runs[:3]])
        if len(selected_runs) > 3:
            run_names += f"_and_{len(selected_runs)-3}_more"
        
        # Define aggregate cache file path
        aggregate_file = self.aggregate_dir / f"aggregate_{run_names}.pt"
        
        # Check if cache exists and should be used
        if use_cache and aggregate_file.exists():
            try:
                # Check if any run directory is newer than the cache file
                newest_run = max(
                    [max(f.stat().st_mtime for f in run_dir.glob('**/*') if f.is_file()) 
                     for run_dir in selected_runs],
                    default=0
                )
                aggregate_mtime = aggregate_file.stat().st_mtime
                
                # Cache is valid if it's newer than the most recently modified file
                if aggregate_mtime > newest_run:
                    print(f"Loading combined aggregate data from: {aggregate_file}")
                    aggregated_data = torch.load(aggregate_file, weights_only=False)
                    if 'scalar_inputs' in aggregated_data and 'outputs' in aggregated_data:
                        print(f"Loaded {len(aggregated_data['scalar_inputs'])} samples from aggregate cache")
                        return aggregated_data
                    else:
                        print("Aggregate file does not contain required data. Re-aggregating...")
            except Exception as e:
                print(f"Error loading from aggregate cache: {e}")
                
        # Aggregate data from all selected run directories
        all_scalar_inputs = []
        all_profile_inputs = []
        all_outputs = []
        all_features = set()
        
        print(f"Aggregating data from {len(selected_runs)} run directories")
        for run_dir in tqdm(selected_runs, desc="Processing runs"):
            # Aggregate data for this run
            scalar_inputs, profile_inputs, outputs = self.aggregate_run_data(
                run_dir, parameter_names, use_cache
            )
            
            # Skip empty results
            if not scalar_inputs:
                print(f"Skipping {run_dir.name} (no data)")
                continue
                
            # Handle different return formats from different aggregators
            if isinstance(scalar_inputs, dict):
                # Convert dict of iterations to list of scalar inputs
                for iter_num, data in scalar_inputs.items():
                    if data:  # Skip empty or None data
                        all_scalar_inputs.append(data)
                        # Keep track of features for normalization
                        if isinstance(data, dict):
                            all_features.update(data.keys())
                        
                # Also convert profile inputs
                if profile_inputs:
                    for iter_num, data in profile_inputs.items():
                        if data is not None:
                            all_profile_inputs.append(data)
                
                # And outputs
                if outputs:
                    for iter_num, data in outputs.items():
                        if data is not None and len(data) > 0:
                            all_outputs.append(data)
                    
        # Construct the aggregated data dictionary
        aggregated_data = {
            'scalar_inputs': all_scalar_inputs,
            'profile_inputs': all_profile_inputs,
            'outputs': all_outputs,
            'feature_info': {
                'scalar_features': sorted(list(all_features))
            },
            'timestamp': time.time()
        }
        
        # Save the aggregated data to cache
        print(f"Saving combined aggregate data ({len(all_scalar_inputs)} samples) to: {aggregate_file}")
        torch.save(aggregated_data, aggregate_file, pickle_protocol=4)
        
        return aggregated_data
    
    def get_parameter_names(self) -> List[str]:
        """
        Get parameter names from scalar_params.
        
        Returns:
            List[str]: List of parameter names from scalar_params
        """
        if not self.scalar_param_names:
            self.load_parameter_definitions()
        
        return self.scalar_param_names
    
    def get_aggregate_info(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about available aggregate files.
        
        Returns:
            Dict[str, Dict[str, Any]]: Dictionary with cache information
        """
        if not self.run_dirs:
            self.discover_run_directories()
        
        aggregate_info = {}
        for run_dir in self.run_dirs:
            run_name = run_dir.name
            aggregate_file = self.aggregate_dir / f"{run_name}_aggregate.pt"
            
            # Check if cache exists and is valid
            if aggregate_file.exists():
                try:
                    # Get modification time of run directory and cache file
                    run_mtime = max(f.stat().st_mtime for f in run_dir.glob('**/*') if f.is_file())
                    aggregate_mtime = aggregate_file.stat().st_mtime
                    
                    # Cache is valid if it's newer than the most recently modified file in run directory
                    is_valid = aggregate_mtime > run_mtime
                    
                    # Load cache data to get timestamp (with weights_only=False)
                    aggregate_data = torch.load(aggregate_file, weights_only=False)
                    
                    aggregate_info[run_name] = {
                        'exists': True,
                        'is_valid': is_valid,
                        'timestamp': aggregate_data.get('timestamp', 0),
                        'size_bytes': os.path.getsize(aggregate_file)
                    }
                except Exception as e:
                    aggregate_info[run_name] = {
                        'exists': True,
                        'is_valid': False,
                        'error': str(e)
                    }
            else:
                aggregate_info[run_name] = {
                    'exists': False
                }
        
        return aggregate_info
    
    def clear_aggregate(self, run_names: Optional[List[str]] = None) -> List[str]:
        """
        Clear aggregate files for specified runs or all runs.
        
        Args:
            run_names (List[str], optional): List of run names to clear aggregate for.
                                       If None, all aggregates will be cleared.
        
        Returns:
            List[str]: List of aggregate files that were cleared
        """
        if not self.aggregate_dir.exists():
            return []
        
        cleared_files = []
        
        if run_names is None:
            # Clear all aggregate files
            for aggregate_file in self.aggregate_dir.glob('*_aggregate.pt'):
                aggregate_file.unlink()
                cleared_files.append(str(aggregate_file))
            print(f"Cleared all aggregate files ({len(cleared_files)} files)")
        else:
            # Clear specific aggregate files
            for run_name in run_names:
                aggregate_file = self.aggregate_dir / f"{run_name}_aggregate.pt"
                if aggregate_file.exists():
                    aggregate_file.unlink()
                    cleared_files.append(str(aggregate_file))
            print(f"Cleared aggregate for: {', '.join(run_names)}")
        
        return cleared_files

    def select_runs_by_range(self, start: Optional[int] = None, end: Optional[int] = None) -> List[str]:
        """
        Select runs by numerical range.
        
        Args:
            start (int, optional): Starting run number (inclusive)
            end (int, optional): Ending run number (inclusive)
        
        Returns:
            List[str]: List of selected run names
        """
        if not self.run_dirs:
            self.discover_run_directories()
        
        selected_runs = []
        for run_dir in self.run_dirs:
            # Extract run number from name (e.g., "Run_5" -> 5)
            match = re.search(r'Run_(\d+)', run_dir.name)
            if match:
                run_number = int(match.group(1))
                
                # Check if run number is in range
                if (start is None or run_number >= start) and (end is None or run_number <= end):
                    selected_runs.append(run_dir.name)
        
        print(f"Selected runs by range: {selected_runs}")
        return selected_runs 