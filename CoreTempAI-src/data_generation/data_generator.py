"""
This script uses Ansys Fluent solver to solve a given case for different values of case parameters.
The case parameters are defined in the utils.parameters file.
"""

import sys
import os
from pathlib import Path

# Add the project root directory to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Importing necessary libraries
import ansys.fluent.core as pyfluent  # Python API for Ansys Fluent
import time  # For timing operations
import CoreTempAI.input_parameters.input_parameters as input_parameters  # Custom module for parameter handling
from CoreTempAI.config import PROJECT_ROOT, FILES, SIMULATION_CONFIG, DATA_GENERATOR_CONFIG
from CoreTempAI.data_generation.run_case import run_case
from CoreTempAI.data_generation.set_params import set_input_parameters
from CoreTempAI.utils.utils import exit_solver, create_run_directory


def generate_data(num_data=None, case_file_path=None):
    """
    Generates simulation data by running Ansys Fluent for a specified number of cases.
    
    Args:
        num_data (int, optional): Number of data points to generate. 
                                 If None, uses the value from DATA_GENERATOR_CONFIG.
    """
    if num_data is None:
        num_data = DATA_GENERATOR_CONFIG["num_data"]

    if case_file_path is None:
        current_case_file_path = FILES["case_file"]

    # Create a new run directory with subdirectories
    run_dir, n = create_run_directory()
        
    for i in range(num_data): 
        cwd_path = os.path.join(run_dir, "output_profile")
        solver_session = pyfluent.launch_fluent(precision=pyfluent.Precision.DOUBLE, dimension=pyfluent.Dimension.THREE,
                                    processor_count=SIMULATION_CONFIG["processor_count"],
                                    cwd=rf"{cwd_path}",
                                    ui_mode="gui",
                                    py=True)

        # Read the initial case file        
        solver_session.file.read_case(file_name=rf"{current_case_file_path}")

        # Set parameters and run the initial case
        set_input_parameters(solver_session, input_parameters.scalar_params, input_parameters.profile_params, run_dir, i)
        output_file_path = os.path.join(run_dir, "case_and_data", f"iter{i}.cas.h5")
        run_case(solver_session, i, current_case_file_path, output_file_path)

        # Exit Fluent
        exit_solver(solver_session, cleanup_dir=cwd_path)

        # Handle output files
        
        # Find all .out files
        for file in os.listdir(cwd_path):
            file_path = os.path.join(cwd_path, file)
            if file.endswith(".out"):
                if "core_temp" in file:
                    lines = open(file_path, 'r').readlines()
                    if len(lines) > 2 and 'flow-time' in lines[2].lower():
                        new_name = f"core_temp_profile_{i}.txt"
                        os.rename(file_path, os.path.join(cwd_path, new_name))
                    else:
                        print(f"Deleting {file_path}")
                        time.sleep(10)
                        os.remove(file_path)
                else:
                    # Delete blood_temp and wavg files
                    print(f"Deleting {file_path}")
                    time.sleep(10)
                    os.remove(file_path)




if __name__ == "__main__":
    # Start timing the execution
    start_time = time.time()

    generate_data()

    # Calculate and print the total execution time
    end_time = time.time()

    time_taken = end_time - start_time
    print(f"Total time taken: {time_taken:.2f} seconds")


