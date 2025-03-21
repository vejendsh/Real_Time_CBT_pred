"""
Custom data aggregator for CoreTempAI.

This module provides functionality to aggregate data from Ansys Fluent case files 
and .out files for core temperature profiles.
"""

import os
import re
import glob
from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
from tqdm import tqdm

from CoreTempAI.data_aggregation.aggregator_base import DataAggregatorBase

# Import for Ansys Fluent case file reading
try:
    import ansys.fluent.core as pyfluent
    from ansys.fluent.core.filereader.case_file import CaseFile
    ANSYS_AVAILABLE = True
except ImportError:
    ANSYS_AVAILABLE = False
    print("Warning: ansys.fluent.core module not available. Fluent case file loading will be disabled.")

class CustomAggregator(DataAggregatorBase):
    """
    Custom data aggregator for Ansys Fluent case files and .out files.
    
    This class aggregates scalar inputs from Fluent case files and
    core temperature outputs from .out files.
    """
    
    def __init__(self, run_dir: Path):
        """
        Initialize the custom data aggregator.
        
        Args:
            run_dir (Path): Path to the run directory
        """
        super().__init__(run_dir)
        
        # Find all case files in the run directory
        self.case_files = self._find_case_files()
        
        # Find all .out files in the output_profile directory
        self.out_files = self._find_out_files()
        
    def _find_case_files(self) -> List[str]:
        """
        Find all Fluent case files in the run directory.
        
        Returns:
            List[str]: List of case file paths
        """
        # Find all case files in the run directory (recursively)
        case_files = []
        for file_path in glob.glob(os.path.join(self.run_dir, "**", "*.cas.h5"), recursive=True):
            case_files.append(file_path)
        
        if not case_files:
            # Try the non-h5 format as fallback
            for file_path in glob.glob(os.path.join(self.run_dir, "**", "*.cas"), recursive=True):
                if not file_path.endswith(".cas.h5"):  # Avoid double-counting .cas.h5 files
                    case_files.append(file_path)
        
        # Sort case files by iteration number
        case_files.sort(key=self._get_iteration_number)
        
        return case_files
    
    def _find_out_files(self) -> List[str]:
        """
        Find all .out files in the output_profile directory.
        
        Returns:
            List[str]: List of .out file paths
        """
        output_dir = self.run_dir / "output_profile"
        
        if not output_dir.exists():
            return []
        
        # Find all .out files
        out_files = list(output_dir.glob("*.out"))
        
        # Sort by iteration number
        out_files.sort(key=lambda f: self.extract_iteration(f.name))
        
        return out_files
        
    def _get_iteration_number(self, filename: str) -> int:
        """
        Extract iteration number from a case filename.
        
        Args:
            filename (str): Case filename
            
        Returns:
            int: Iteration number
        """
        return self.extract_iteration(os.path.basename(filename))
    
    def aggregate_scalar_inputs(self, parameter_names: Optional[List[str]] = None) -> Dict[int, Any]:
        """
        Aggregate scalar input parameters from Fluent case files.
        
        Args:
            parameter_names (List[str], optional): Names of parameters to extract.
                                                If None, all parameters will be extracted.
            
        Returns:
            Dict[int, Any]: Dictionary mapping iteration to scalar input data
        """
        if not ANSYS_AVAILABLE:
            print("Warning: Ansys Fluent core module not available. Cannot load Fluent data.")
            return {}
        
        if not self.case_files:
            print(f"Warning: No Fluent case files found in: {self.run_dir}")
            return {}
        
        # Process each case file
        scalar_inputs = {}
        
        print(f"Loading scalar inputs from {len(self.case_files)} Fluent case files...")
        for case_file in tqdm(self.case_files, desc="Processing case files"):
            try:
                # Extract iteration number from filename
                iteration = self._get_iteration_number(case_file)
                
                # Read the case file
                reader = CaseFile(case_file_name=case_file)
                
                # Get input parameters
                input_params = reader.input_parameters()
                
                # Create a name-to-parameter mapping
                param_map = {p.name: p for p in input_params}
                
                # Extract parameter values
                if parameter_names is not None:
                    # Extract only the specified parameters by name
                    param_dict = {}
                    for name in parameter_names:
                        if name in param_map:
                            # Extract numeric value from the parameter string (e.g., "712.887 [W/m^3]")
                            param_value = param_map[name].value
                            match = re.search(r'(\d+\.?\d*)', str(param_value))
                            if match:
                                param_dict[name] = float(match.group(1))
                            else:
                                print(f"Warning: Could not extract numeric value from '{param_value}' for parameter '{name}'")
                                param_dict[name] = 0.0
                        else:
                            print(f"Warning: Parameter '{name}' not found in Fluent case file {case_file}")
                            param_dict[name] = 0.0  # Default value
                    
                    # Update available features the first time
                    if not self.available_scalar_features:
                        self.available_scalar_features = list(param_dict.keys())
                else:
                    # Extract all parameters
                    param_dict = {}
                    for p in input_params:
                        # Extract numeric value from the parameter string
                        match = re.search(r'(\d+\.?\d*)', str(p.value))
                        if match:
                            param_dict[p.name] = float(match.group(1))
                        else:
                            print(f"Warning: Could not extract numeric value from '{p.value}' for parameter '{p.name}'")
                            param_dict[p.name] = 0.0
                    
                    # Update available features the first time
                    if not self.available_scalar_features:
                        self.available_scalar_features = list(param_dict.keys())
                
                scalar_inputs[iteration] = param_dict
                
            except Exception as e:
                print(f"Error processing case file {case_file}: {e}")
                continue
        
        # Sort the inputs by iteration number
        scalar_inputs = dict(sorted(scalar_inputs.items(), key=lambda item: item[0]))
        
        return scalar_inputs
    
    def aggregate_profile_inputs(self) -> Dict[int, pd.DataFrame]:
        """
        Aggregate profile input parameters.
        
        Note: Profile inputs are not directly extracted for custom aggregator.
        
        Returns:
            Dict[int, pd.DataFrame]: Empty dictionary
        """
        # Profile inputs are not directly extracted in the custom aggregator
        return {}
    
    def aggregate_outputs(self) -> Dict[int, np.ndarray]:
        """
        Aggregate core temperature output profiles from .out files.
        
        Returns:
            Dict[int, np.ndarray]: Dictionary mapping iteration to core temp profile
        """
        if not self.out_files:
            print(f"Warning: No .out files found in output_profile directory")
            return {}
        
        outputs_by_iteration = {}
        valid_files = []
        
        print(f"Processing all {len(self.out_files)} .out files for flow-time data...")
        
        for out_file in tqdm(self.out_files, desc="Checking .out files"):
            try:
                # Check if file contains flow-time in third line
                with open(out_file, 'r') as f:
                    lines = f.readlines()
                    if len(lines) > 2 and 'flow-time' in lines[2].lower():
                        # Add to valid files if it contains flow-time in the third line
                        valid_files.append(out_file)
            except Exception as e:
                print(f"Error checking .out file {out_file}: {e}")
        
        print(f"Found {len(valid_files)} valid flow-time .out files")
        
        # Sort valid files by iteration number to ensure consistent step size
        valid_files.sort(key=lambda f: self.extract_iteration(f.name))
        
        # Step size for iteration numbering (to handle gaps)
        step_size = 1
        
        # Process each valid .out file
        for i, out_file in enumerate(tqdm(valid_files, desc="Processing flow-time files")):
            try:
                # Extract original iteration number from filename
                orig_iteration = self.extract_iteration(out_file.name)
                
                # Set new iteration number with consistent step size
                iteration = i * step_size
                
                # Read the temperature data
                with open(out_file, 'r') as f:
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
                    print(f"Warning: No temperature data found in .out file {out_file}")
                    
                    # Optional: Map the original iteration to the new one if needed
                    if orig_iteration != iteration:
                        print(f"Mapped original iteration {orig_iteration} to {iteration}")
            except Exception as e:
                print(f"Error processing .out file {out_file}: {e}")
        
        return outputs_by_iteration 