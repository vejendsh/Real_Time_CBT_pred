"""
This code scripts Ansys Fluent solver to simulate CT for HR and ST profiles to generate training data for Machine Learning models.
The case parameters are defined in the utils.parameters file.
"""

import os
import shutil

# Importing necessary libraries
import time  # For timing operations
import coretempai.input_parameters.input_parameters_st_indep as input_parameters  # Custom module for parameter handling
from coretempai.config import FILES, SIMULATION_CONFIG
from coretempai.data_generation.st_indep.run_case import run_case
from coretempai.data_generation.st_indep.set_params_transient import set_transient_params
from coretempai.utils.utils import exit_solver, create_run_directory
 

def generate_data(num_sims=SIMULATION_CONFIG["num_simulations"], case_file_path=FILES["case_file_st_indep"]):
    """
    Main function for scripting Ansys Fluent solver to simulate CT for different HR and ST profiles.
    
    Args:
        num_sims (int, optional): Number of simulations to run. 
                                 If None, uses the value from SIMULATION_CONFIG.
    """

    # Create a new run directory with subdirectories
    run_dir, n = create_run_directory()
    working_directory_path = os.path.join(run_dir, "output_params")
    udf_path = SIMULATION_CONFIG["udf_file_general_st_indep"]

    # Run simulations
    for i in range(num_sims): 

        # Generate new transient input parameters
        transient_params = set_transient_params(udf_path, input_parameters.profile_params, input_parameters.initial_profile_params)

        # Save the transient input parameters as txt files in input_params directory
        hr_profile = transient_params.get("hr_profile")
        st_profile = transient_params.get("st_profile")
        input_time = transient_params.get("input_time")
        hr_file_path = os.path.join(run_dir, "input_params", "HR", f"iter{i}.txt")
        st_file_path = os.path.join(run_dir, "input_params", "ST", f"iter{i}.txt")

        with open(hr_file_path, "w") as f:
            f.write("time,HR\n")
            for idx in range(len(input_time)):
                f.write(f"{input_time[idx]},{hr_profile[idx]}\n")
                
        with open(st_file_path, "w") as f:
            f.write("time,ST\n")
            for idx in range(len(input_time)):
                f.write(f"{input_time[idx]},{st_profile[idx]}\n")

        # Create a new case file for each simulation
        output_file_path = os.path.join(run_dir, "case_and_data", f"iter{i}.cas.h5")

        # Run case
        run_case(working_directory_path, output_file_path, case_file_path)

        # Give the solver time to release .out file handles (avoids WinError 32 on Windows)
        time.sleep(2)

        # Move each .out file into its respective subdirectory in output_params directory
        for file in os.listdir(working_directory_path):
            if not file.endswith(".out") or "-rfile" not in file:
                continue
            src_path = os.path.join(working_directory_path, file)
            subdir = file.split("-rfile")[0]
            dest_dir = os.path.join(run_dir, "output_params", subdir)
            name, ext = os.path.splitext(file)
            dest_path = os.path.join(dest_dir, f"iter{i}{ext}")
            # os.rename(os.path.join(working_directory_path, file), dest_path)
            os.makedirs(dest_dir, exist_ok=True)
            # Copy then remove avoids "file in use" on Windows if rename fails
            for attempt in range(5):
                try:
                    shutil.copy2(src_path, dest_path)
                    os.remove(src_path)
                    break
                except PermissionError:
                    if attempt < 4:
                        time.sleep(1)
                    else:
                        raise


        steady_state_CT = 37.2969477485438
        # Add steady state CT as first entry in CT/iter{i}.txt file
        with open(os.path.join(run_dir, "output_params", "core_temp", f"iter{i}.out"), "r") as f:
            lines = f.readlines()
        lines.insert(3, f"0 {steady_state_CT} 0\n") 
        with open(os.path.join(run_dir, "output_params", "core_temp", f"iter{i}.out"), "w") as f:
            f.writelines(lines)

    # Rename core_temp dir as CT
    os.rename(os.path.join(run_dir, "output_params", "core_temp"), os.path.join(run_dir, "output_params", "CT"))
    

if __name__ == "__main__":
    # Start timing the execution
    start_time = time.time()

    generate_data()

    # Calculate and print the total execution time
    end_time = time.time()

    time_taken = end_time - start_time
    print(f"Total time taken: {time_taken:.2f} seconds")


