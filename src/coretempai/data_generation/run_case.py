"""
This script uses PyFluent API to run Ansys Fluent simulations.
It replaces the functionality of the run.log journal file.
"""


import ansys.fluent.core as pyfluent
from coretempai.config import DIRECTORIES, FILES, PROJECT_ROOT, SIMULATION_CONFIG
from coretempai.utils.utils import exit_solver

def run_case(working_directory_path, case_file_path=FILES["case_file"], output_file_path=DIRECTORIES["raw"]):
    """
    Run the Fluent case.

    Args:
        working_directory_path (str): Path to the working directory.
        case_file_path (str): Path to the case file.
        output_file_path (str): Path to the output file.    
    """

    solver_session = pyfluent.launch_fluent(precision=pyfluent.Precision.DOUBLE, 
                                           dimension=pyfluent.Dimension.THREE,
                                           processor_count=SIMULATION_CONFIG["processor_count"],
                                           ui_mode=pyfluent.UIMode.GUI,cwd=rf"{working_directory_path}",
                                           py=True)
    
    # Read the case file
    solver_session.file.read_case(file_name=rf"{case_file_path}")

    # Set the solver to steady
    solver_session.setup.general.solver.time.set_state("steady")
    
    # Initialize the solution
    solver_session.solution.initialization.initialize()
    
    # Run calculation
    solver_session.solution.run_calculation.calculate()
    
    # Switch to transient simulation
    print("Switching to transient simulation...")
    solver_session.setup.general.solver.time.set_state("unsteady-2nd-order")
    
    # Run transient calculation
    print("Running transient calculation...")
    solver_session.solution.run_calculation.calculate()
    
    # Save the case and data
    print(f"Saving results to {output_file_path}...")
    current_output_file_path = output_file_path
    solver_session.file.write(file_type="data", file_name=rf"{current_output_file_path}")    

    # Exit solver and cleanup
    exit_solver(solver_session, cleanup=False, cleanup_dir=working_directory_path)  


def main():
    """
    Example usage of the run_simulation function.
    """
    # Run the case
    run_case()
    


if __name__ == "__main__":
    main() 