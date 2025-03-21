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

# =============================================================================
# SIMULATION PARAMETERS
# =============================================================================
# This file contains all parameters that will be used in the simulation.
# To customize the simulation, modify the values within the min/max ranges provided.
#
# PARAMETER FORMAT:
# Each parameter is defined as: "parameter_name": [min_value, max_value, "units"]
# You can set any value between min_value and max_value (inclusive).
#
# IMPORTANT: Do not change parameter names or units, only modify the values.
# =============================================================================

# Define scalar parameters for the simulation
# These are constant values used throughout the simulation

 # Do not forget comma when adding a new parameter 

scalar_params = {
"metab_muscle": [500, 1000, "W/m^3"],  
"metab_organ": [100, 200, "W/m^3"],   
"metab_head": [200, 300, "W/m^3"],     
"T_amb": [300, 310, "K"],              
"h": [10, 20, "W/m^2/K"],  
}

# Define profile parameters for the simulation
# These parameters can vary over time during the simulation
profile_params = {
"HR": [3600, 7200, "s^-1"]             # Heart rate profile
}

