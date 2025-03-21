"""
Default data aggregator for CoreTempAI.

This module provides functionality to aggregate data from CSV files.
"""

import re
from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
from tqdm import tqdm

from CoreTempAI.data_aggregation.aggregator_base import DataAggregatorBase

class DefaultAggregator(DataAggregatorBase):
    """
    Default data aggregator for CSV file format.
    
    This class aggregates scalar inputs, profile inputs, and outputs from CSV files.
    """
    
    def __init__(self, run_dir: Path):
        """
        Initialize the default data aggregator.
        
        Args:
            run_dir (Path): Path to the run directory
        """
        super().__init__(run_dir)
        
    def aggregate_scalar_inputs(self, parameter_names: Optional[List[str]] = None) -> Dict[int, Dict[str, float]]:
        """
        Aggregate scalar input parameters from CSV files.
        
        Args:
            parameter_names (List[str], optional): Names of parameters to extract.
                                                If None, all parameters will be extracted.
            
        Returns:
            Dict[int, Dict[str, float]]: Dictionary mapping iteration to scalar parameters
        """
        scalar_file = self.run_dir / "input_scalars" / "input_scalars.csv"
        
        if not scalar_file.exists():
            print(f"Warning: Scalar input file not found: {scalar_file}")
            return {}
        
        # Read the CSV file
        df = pd.read_csv(scalar_file)
        
        # Extract numeric values from strings like "712.887 [W/m^3]"
        for col in df.columns:
            if col != "Iteration":
                try:
                    df[col] = df[col].str.extract(r'(\d+\.\d+)').astype(float)
                except (AttributeError, ValueError):
                    # If extraction fails, keep the column as is (might already be numeric)
                    pass
        
        # Filter columns if parameter_names is provided
        if parameter_names is not None:
            # Check which requested parameters are available
            available_columns = [col for col in parameter_names if col in df.columns]
            missing_columns = [col for col in parameter_names if col not in df.columns]
            
            if missing_columns:
                print(f"Warning: The following parameters were not found in the CSV file: {missing_columns}")
            
            # Update available features with requested parameters that are available
            self.available_scalar_features = available_columns
        else:
            # Use all columns except Iteration if no specific parameters are requested
            self.available_scalar_features = [col for col in df.columns if col != "Iteration"]
        
        # Convert to dictionary by iteration
        scalar_inputs = {}
        for _, row in df.iterrows():
            iteration = int(row['Iteration'])
            scalar_inputs[iteration] = {col: row[col] for col in self.available_scalar_features}
        
        # Sort the inputs by iteration number
        scalar_inputs = dict(sorted(scalar_inputs.items(), key=lambda item: item[0]))
        
        return scalar_inputs
    
    def aggregate_profile_inputs(self) -> Dict[int, pd.DataFrame]:
        """
        Aggregate profile input parameters from CSV files.
        
        Returns:
            Dict[int, pd.DataFrame]: Dictionary mapping iteration to profile DataFrame
        """
        profile_dir = self.run_dir / "input_profile"
        
        if not profile_dir.exists():
            print(f"Warning: Profile directory not found: {profile_dir}")
            return {}
        
        # Find all profile files
        profile_files = list(profile_dir.glob("*.csv"))
        
        if not profile_files:
            print(f"Warning: No profile files found in {profile_dir}")
            return {}
        
        # Group by iteration
        profiles_by_iteration = {}
        
        for file in tqdm(profile_files, desc="Loading profile inputs"):
            # Extract iteration number from filename
            iteration = self.extract_iteration(file.name)
                
            try:
                # Read the profile data
                with open(file, 'r') as f:
                    lines = f.readlines()
                
                # Find the start of data section
                data_start = 0
                for i, line in enumerate(lines):
                    if line.strip() == "[Data]":
                        data_start = i + 1
                        break
                
                # Parse the data
                if data_start > 0:
                    # Get header
                    header = lines[data_start].strip().split(',')
                    
                    # Parse data rows
                    data = []
                    for line in lines[data_start+1:]:
                        if line.strip():
                            values = line.strip().split(',')
                            data.append([float(v) for v in values])
                    
                    # Create DataFrame
                    df = pd.DataFrame(data, columns=header)
                    profiles_by_iteration[iteration] = df
                    
                    # Update available profile features
                    if not self.available_profile_features and len(header) > 1:
                        self.available_profile_features = [col for col in header if col != "time"]
                else:
                    print(f"Warning: No profile data found in {file.name}")
            except Exception as e:
                print(f"Error reading profile file {file}: {e}")
        
        # Sort the inputs by iteration number
        profiles_by_iteration = dict(sorted(profiles_by_iteration.items(), key=lambda item: item[0]))
        
        return profiles_by_iteration
    
    def aggregate_outputs(self) -> Dict[int, np.ndarray]:
        """
        Aggregate core temperature output profiles from CSV files.
        
        Returns:
            Dict[int, np.ndarray]: Dictionary mapping iteration to core temp profile
        """
        output_dir = self.run_dir / "output_profile"
        
        if not output_dir.exists():
            print(f"Warning: Output directory not found: {output_dir}")
            return {}
        
        # Find all output files
        output_files = list(output_dir.glob("*"))
        
        if not output_files:
            print(f"Warning: No output files found in {output_dir}")
            return {}
        
        # Group by iteration
        outputs_by_iteration = {}
        
        for file in tqdm(output_files, desc="Loading outputs"):
            # Extract iteration number from filename
            iteration = self.extract_iteration(file.name)
            
            try:
                # Read the temperature data
                with open(file, 'r') as f:
                    lines = f.readlines()
                
                # Parse data - skip header line
                temp_data = []
                for line in lines[3:]:
                    if line.strip():
                        values = line.strip().split()
                        if len(values) >= 2:  # At least position and temperature
                            temp_data.append(float(values[1]))  # Temperature is the second column
                
                # Convert to numpy array
                temp_array = np.array(temp_data)
                
                if len(temp_array) > 0:
                    outputs_by_iteration[iteration] = temp_array

                else:
                    print(f"Warning: No temperature data found in .txt file {file}")

            except Exception as e:
                print(f"Error processing .txt file {file}: {e}")
        
        
        # Sort the outputs by iteration number
        outputs_by_iteration = dict(sorted(outputs_by_iteration.items(), key=lambda item: item[0]))
        
        return outputs_by_iteration 