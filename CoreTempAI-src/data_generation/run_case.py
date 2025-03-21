"""
This script uses PyFluent API to run Ansys Fluent simulations.
It replaces the functionality of the run.log journal file.
"""

import sys
import os
from pathlib import Path

# Add the project root directory to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

import ansys.fluent.core as pyfluent
from CoreTempAI.config import DIRECTORIES, FILES, PROJECT_ROOT
from CoreTempAI.utils.utils import exit_solver

def run_case(solver_session, case_number, case_file_path, output_file_path):
    """
    Run the Fluent case with the current parameters.
    """
    # # Read the case file
    # solver_session.file.read_case(file_name=case_file_path)

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


def main():
    """
    Example usage of the run_simulation function.
    """
    # Launch Fluent
    solver_session = pyfluent.launch_fluent(precision=pyfluent.Precision.DOUBLE, 
                                           dimension=pyfluent.Dimension.THREE,
                                           processor_count=4,
                                           ui_mode=pyfluent.UIMode.GUI,
                                           py=True)
    
    # Read the case file
    case_file_path = FILES["case_file"]
    solver_session.file.read_case(file_name=rf"{case_file_path}")
    
    # Run the simulation
    run_case(solver_session, case_number=0, case_file_path=case_file_path)
    
    # Exit Fluent when done
    exit_solver(solver_session, DIRECTORIES["case"])


if __name__ == "__main__":
    main() 