# =============================================================================
# QUICK GUIDE TO MODIFY PARAMETERS:
# 1. To modify existing parameter: Change values within [min, max] range only
#    Example: "T_amb": [300, 310, "K"] -> "T_amb": [305, 315, "K"]
#
# 2. To add new parameter: Follow the format below
#    For constant values: Add to scalar_params
#    For time-varying values: Add to profile_params
#    Format: "parameter_name": [min_value, max_value, "units"]
# =============================================================================

# Define scalar parameters for the simulation
# These are constant values used throughout the simulation

 # Do not forget comma when adding a new parameter 

initial_profile_params = {
    "HR": [60, 100, "bpm"],
}

# Define profile parameters for the simulation
# These parameters can vary over time during the simulation
profile_params = {
    "HR": [60, 180, "bpm"],             # Heart rate profile
}

scalar_params = {
    "h": [1, 50, "W/m^2/K"],
    "T_amb": [20, 50, "C"],
}



