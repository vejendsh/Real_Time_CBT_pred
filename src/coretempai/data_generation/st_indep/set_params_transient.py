"""
This script generates transient simulation parameters and modifies UDF.c file.
It generates random HR profiles using Fourier sampler and random scalar parameters,
then injects them into the UDF.c file.
"""

import os
import re
import numpy as np
from numpy.random import noncentral_f
from coretempai.config import SIMULATION_CONFIG, PROFILE_CONFIG, DIRECTORIES
from coretempai.utils.utils import Fourier
import coretempai.input_parameters.input_parameters_st_indep as input_parameters


def generate_profile(max_freq, duration, step_size, range, initial_range=None, initial_value=None):
    """
    Generates a random HR profile using Fourier sampler.
    
    Args:
        max_freq (int): Maximum frequency of profiles
        duration (float): Duration of the profile in seconds
        step_size (float): Time step size in seconds
        range (list): [min, max] range for HR values in min^-1
    Returns:
        numpy.ndarray: Array of HR values with length = (duration / step_size) + 1
    """
    # Generate time points
    time_points = np.arange(0, duration + step_size, step_size)
    print(f"Time points: {time_points}")
    n_points = len(time_points)
    
    # Normalize time points to [0, 1] for the Fourier sampler
    normalized_time = time_points / duration
    
    # Create a Fourier sampler
    sampler = Fourier(normalized_time)
    
    # Generate a random profile within the specified range
    # n=1 means generate 1 profile, max_freq=max_freq, range=hr_range

    profile_values = sampler.sample(1, max_freq=max_freq, range=range, initial_range=initial_range, initial_value=initial_value)[0]


    return profile_values


def format_hr_array_c(hr_array):
    """
    Formats HR array as a C array string.
    
    Args:
        hr_array (numpy.ndarray): Array of HR values
        
    Returns:
        str: C array string like "real heartrate[3601]={60.5,61.2,...};"
    """
    # Format each value with appropriate precision
    hr_values_str = ",".join([f"{val:.4f}" for val in hr_array])
    array_size = len(hr_array)
    return f"real heartrate[{array_size}]={{{hr_values_str}}};"


def format_st_array_c(st_array):
    """
    Formats ST array as a C array string.
    
    Args:
        st_array (numpy.ndarray): Array of ST values
        
    Returns:
        str: C array string like "real skintemp[3601]={60.5,61.2,...};"
    """
    # Format each value with appropriate precision
    st_values_str = ",".join([f"{val:.4f}" for val in st_array])
    array_size = len(st_array)
    return f"real skintemp[{array_size}]={{{st_values_str}}};"


def format_time_array_c(time_array):
    """
    Formats time array as a C array string.
    
    Args:
        time_array (numpy.ndarray): Array of time values in seconds
        
    Returns:
        str: C array string like "real inputtime[3601]={0,1,2,...};"
    """
    # Format each value as integer (time points are typically integers)
    # Use .0f to ensure no decimal point for whole numbers
    time_values_str = ",".join([f"{int(val)}" if val == int(val) else f"{val:.1f}" for val in time_array])
    array_size = len(time_array)
    return f"real inputtime[{array_size}]={{{time_values_str}}};"


def read_skin_temperature_rfile(filepath):
    """
    Read skin temperature and flow-time from a Fluent skin_temperature-rfile.
    
    File format (first 3 lines are header):
        "skin_temperature-rfile"
        "Time Step" "skin_temperature etc.."
        ("Time Step" "skin_temperature" "flow-time")
        <time_step> <skin_temperature_K> <flow_time>
        ...
    
    Args:
        filepath (str): Path to the rfile (e.g. cases/general/skin_temperature-rfile_6_1.out).
        
    Returns:
        tuple: (skin_temps_degC, flow_times) as 1D numpy arrays.
               skin_temps_degC is converted from Kelvin to Celsius.
    """
    with open(filepath, "r") as f:
        lines = f.readlines()
    
    # Skip 3 header lines
    data_lines = [ln.strip() for ln in lines[3:] if ln.strip()]
    skin_temps_K = []
    flow_times = []
    for ln in data_lines:
        parts = ln.split()
        if len(parts) >= 3:
            # time_step, skin_temperature (K), flow_time
            skin_temps_K.append(float(parts[1]))
            flow_times.append(float(parts[2]))
    
    skin_temps_K = np.array(skin_temps_K)
    flow_times = np.array(flow_times)
    # Convert Kelvin to Celsius
    skin_temps_degC = skin_temps_K - 273.15
    return skin_temps_degC, flow_times


def read_core_temp_rfile(filepath):
    """
    Read core temperature and flow-time from a Fluent core_temp-rfile.

    File format (first 3 lines are header):
        "core_temp-rfile"
        "Time Step" "core_temp etc.."
        ("Time Step" "core_temp" "flow-time")
        <time_step> <core_temp> <flow_time>
        ...

    Args:
        filepath (str): Path to the rfile (e.g. cases/general/core_temp-rfile_6_1.out).

    Returns:
        tuple: (core_temps, flow_times) as 1D numpy arrays.
    """
    with open(filepath, "r") as f:
        lines = f.readlines()

    data_lines = [ln.strip() for ln in lines[3:] if ln.strip()]
    core_temps = []
    flow_times = []
    for ln in data_lines:
        parts = ln.split()
        if len(parts) >= 3:
            core_temps.append(float(parts[1]))
            flow_times.append(float(parts[2]))

    return np.array(core_temps), np.array(flow_times)


def apply_skintemp_from_rfile(rfile_path, udf_path, initial_skintemp=33.5):
    """
    Read skin temperature from a Fluent rfile and update the skintemp (and inputtime)
    arrays in the UDF file. Index 0 of skintemp is set to initial_skintemp (default 33.5);
    data from the file fills indices 1, 2, ... Array sizes are adjusted to match the file.
    
    Args:
        rfile_path (str): Path to the skin_temperature-rfile (e.g. .../skin_temperature-rfile_6_1.out).
        udf_path (str): Path to UDF_st_indep.c to modify.
        initial_skintemp (float): Value for skintemp[0] (Celsius). Default 33.5.
        
    Returns:
        dict: With keys "skintemp", "inputtime", "n_points" (number of data rows from file).
    """
    skin_temps_degC, flow_times = read_skin_temperature_rfile(rfile_path)
    n = len(skin_temps_degC)
    if n != len(flow_times):
        raise ValueError("Length mismatch between skin_temperature and flow_time in rfile.")
    
    # skintemp[0] = initial_skintemp, skintemp[1:] = data from file
    skintemp = np.concatenate([[initial_skintemp], skin_temps_degC])
    # inputtime[0] = 0, inputtime[1:] = flow_time from file
    inputtime = np.concatenate([[0.0], flow_times])
    
    with open(udf_path, "r") as f:
        content = f.read()
    
    st_array_str = format_st_array_c(skintemp)
    time_array_str = format_time_array_c(inputtime)
    
    pattern_st = r"real\s+skintemp\[\d+\]\s*=\s*\{.*?\};"
    pattern_inputtime = r"real\s+inputtime\[\d+\]\s*=\s*\{.*?\};"
    
    if not re.search(pattern_st, content, flags=re.DOTALL):
        raise ValueError(f"Pattern 'real skintemp[...]' not found in {udf_path}")
    if not re.search(pattern_inputtime, content, flags=re.DOTALL):
        raise ValueError(f"Pattern 'real inputtime[...]' not found in {udf_path}")
    
    content = re.sub(pattern_st, st_array_str, content, flags=re.DOTALL)
    content = re.sub(pattern_inputtime, time_array_str, content, flags=re.DOTALL)
    
    with open(udf_path, "w") as f:
        f.write(content)
    
    print(f"Updated UDF: {udf_path}")
    print(f"  - skintemp size: {len(skintemp)} (index 0 = {initial_skintemp}, indices 1..{n} from rfile)")
    print(f"  - inputtime size: {len(inputtime)}")
    
    return {"skintemp": skintemp, "inputtime": inputtime, "n_points": n}


def modify_udf_file(udf_path, hr_array, st_array, time_points):
    """
    Modifies UDF.c file to inject HR profile, ST array, and time points,
    Uses pattern matching instead of line numbers for robustness.
    
    Args:
        udf_path (str): Path to UDF.c file
        hr_array (numpy.ndarray): HR profile array
        st_array (numpy.ndarray): ST profile array
        time_points (numpy.ndarray): Time points array corresponding to HR profile

    Returns:
        None
    """
    import re
    
    # Read the UDF file
    with open(udf_path, 'r') as f:
        content = f.read()
    
    # Format HR array, ST array and time array as C code
    hr_array_str = format_hr_array_c(hr_array)
    st_array_str = format_st_array_c(st_array)
    time_array_str = format_time_array_c(time_points)
    
    
    # Pattern 1: Replace HR array
    # Matches: real heartrate[<any_size>]={<any_values>};
    # This pattern handles both single-line and multi-line arrays
    # Uses non-greedy matching to stop at the first closing brace-semicolon
    pattern_hr = r'real\s+heartrate\[\d+\]\s*=\s*\{.*?\};'
    replacement_hr = hr_array_str
    if not re.search(pattern_hr, content, flags=re.DOTALL):
        raise ValueError(f"Pattern 'real heartrate[...]' not found in {udf_path}")
    # Use DOTALL flag to allow . to match newlines across multiple lines
    content = re.sub(pattern_hr, replacement_hr, content, flags=re.DOTALL)
    

    # Pattern 2: Replace inputtime array
    # Matches: real inputtime[<any_size>]={<any_values>};
    # This pattern handles both single-line and multi-line arrays
    pattern_inputtime = r'real\s+inputtime\[\d+\]\s*=\s*\{.*?\};'
    replacement_inputtime = time_array_str
    if not re.search(pattern_inputtime, content, flags=re.DOTALL):
        raise ValueError(f"Pattern 'real inputtime[...]' not found in {udf_path}")
    # Use DOTALL flag to allow . to match newlines across multiple lines
    content = re.sub(pattern_inputtime, replacement_inputtime, content, flags=re.DOTALL)

    # Pattern 3: Replace ST array
    pattern_st = r'real\s+skintemp\[\d+\]\s*=\s*\{.*?\};'
    replacement_st = st_array_str
    if not re.search(pattern_st, content, flags=re.DOTALL):
        raise ValueError(f"Pattern 'real skintemp[...]' not found in {udf_path}")
    # Use DOTALL flag to allow . to match newlines across multiple lines
    content = re.sub(pattern_st, replacement_st, content, flags=re.DOTALL)


    
    # Write modified file back
    with open(udf_path, 'w') as f:
        f.write(content)
    
    print(f"Modified UDF file: {udf_path}")
    print(f"  - HR profile array size: {len(hr_array)}")
    print(f"  - ST profile array size: {len(st_array)}")
    print(f"  - Time points array size: {len(time_points)}")


def set_transient_params(udf_path, profile_params, initial_profile_params, max_freq=PROFILE_CONFIG["max_freq"], 
                         duration=PROFILE_CONFIG["profile_duration"], step_size=PROFILE_CONFIG["step_size"]):
    """
    Main function to generate and set transient parameters in UDF.c.
    
    Args:
        udf_path (str): Path to UDF.c file
        profile_params (dict): Dictionary of profile parameter ranges
        initial_profile_params (dict): Dictionary of initial profile parameter ranges
        max_freq (int): Maximum frequency of profiles
        duration (float): Profile duration in seconds. Defaults to PROFILE_CONFIG.
        step_size (float): Step size in seconds. Defaults to PROFILE_CONFIG.
        
    Returns:
        dict: Dictionary containing generated parameter values
    """
    
    # Generate time points (must match HR profile generation)
    time_points = np.arange(0, duration + step_size, step_size)
    
    # Generate HR profile
    hr_range = profile_params.get("HR", [60, 120])  # Default range if not specified
    initial_hr_range = initial_profile_params.get("HR", [60, 100])  # Default range if not specified
    hr_array = generate_profile(max_freq, duration, step_size, hr_range[:2], initial_hr_range[:2])  # Get [min, max]

    # Generate ST profile
    st_range = profile_params.get("ST", [30, 40])  # Default range if not specified
    # initial_st_range = initial_profile_params.get("ST", [30, 40])
    st_array = generate_profile(max_freq, duration, step_size, st_range[:2])  # Get [min, max]
    st_array = np.insert(st_array, 0, 33.500) # Initial skin temperature is always 33.5 C
    
    # Ensure time_points and hr_array have the same length
    if len(time_points) != len(hr_array):
        raise ValueError(f"Time points array length ({len(time_points)}) does not match HR array length ({len(hr_array)})")

    # Modify UDF file
    modify_udf_file(udf_path, hr_array, st_array, time_points)
    
    # Return generated values for logging/saving
    return {
        "hr_profile": hr_array,
        "st_profile": st_array,
        "time_points": time_points,
        "hr_range": hr_range[:2],
        "max_freq": max_freq,
        "duration": duration,
        "step_size": step_size
    }



def main():
    """
    Example usage of the set_transient_params function.
    """
    udf_path = SIMULATION_CONFIG["udf_file_general_st_indep"]

    profile_params = input_parameters.profile_params
    initial_profile_params = input_parameters.initial_profile_params
    result = set_transient_params(
        udf_path=udf_path,
        profile_params=profile_params,
        initial_profile_params=initial_profile_params,
        max_freq=PROFILE_CONFIG["max_freq"]
    )

    print("\nGenerated parameters:")
    print(f"  HR profile length: {len(result['hr_profile'])}")
    print(f"  HR range: {result['hr_range']}")

    from matplotlib import pyplot as plt
    fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    axs[0].plot(np.arange(0, len(result['hr_profile'])) * PROFILE_CONFIG["step_size"], result['hr_profile'], label='HR profile')
    axs[0].set_ylabel('HR (bpm)')
    axs[0].set_ylim((profile_params.get("HR")[0], profile_params.get("HR")[1]))
    axs[0].set_title('HR Profile')
    axs[0].legend()
    axs[0].grid(True)

    axs[1].plot(np.arange(0, len(result['st_profile'])) * PROFILE_CONFIG["step_size"], result['st_profile'], label='ST profile')
    axs[1].set_ylabel('ST (K)')
    axs[1].set_xlabel('Time (s)')
    axs[1].set_ylim((profile_params.get("ST")[0], profile_params.get("ST")[1]))
    axs[1].set_title('ST Profile')
    axs[1].grid(True)
    axs[1].legend()
    plt.tight_layout()
    plt.show()


# Default skin_temperature rfile name in cases/general (change if needed)
DEFAULT_SKIN_TEMP_RFILE = "skin_temperature-rfile_3_1.out"


# def main():
#     """
#     Run the full routine: read skin temperature from the Fluent rfile in cases/general,
#     then update skintemp and inputtime arrays in UDF_st_indep.c (index 0 = 33.5, rest from file).
#     """
#     rfile_path = os.path.join(DIRECTORIES["case"], DEFAULT_SKIN_TEMP_RFILE)
#     udf_path = SIMULATION_CONFIG["udf_file_general_st_indep"]

#     if not os.path.isfile(rfile_path):
#         raise FileNotFoundError(
#             f"Skin temperature rfile not found: {rfile_path}\n"
#             f"Set DEFAULT_SKIN_TEMP_RFILE or ensure the file exists in cases/general."
#         )

#     result = apply_skintemp_from_rfile(rfile_path, udf_path, initial_skintemp=33.5)
#     print(f"\nDone. Loaded {result['n_points']} points from rfile.")


# def main():
#     """
#     Load core_temp-rfile_6_1.out from cases/general and core_temp-rfile_1_1.out from
#     cases/general_st_indep, truncate the longer series to the shorter length, and plot
#     core temperature vs flow time with legend and labeled axes.
#     """
#     from matplotlib import pyplot as plt

#     case_general = DIRECTORIES["case"]
#     case_st_indep = DIRECTORIES["case_st_indep"]

#     file_general = os.path.join(case_general, "core_temp-rfile_3_1.out")
#     file_st_indep = os.path.join(case_st_indep, "core_temp-rfile_14_1.out")

#     for path in (file_general, file_st_indep):
#         if not os.path.isfile(path):
#             raise FileNotFoundError(f"Core temp rfile not found: {path}")

#     core_temp_general, flow_time_general = read_core_temp_rfile(file_general)
#     core_temp_st_indep, flow_time_st_indep = read_core_temp_rfile(file_st_indep)

#     n = min(len(core_temp_general), len(core_temp_st_indep))
#     core_temp_general = core_temp_general[:n]
#     flow_time_general = flow_time_general[:n]
#     core_temp_st_indep = core_temp_st_indep[:n]
#     flow_time_st_indep = flow_time_st_indep[:n]

#     plt.figure(figsize=(8, 5))
#     plt.plot(flow_time_general, core_temp_general, label="Heat flux BC")
#     plt.plot(flow_time_st_indep, core_temp_st_indep, label="Temperature BC")
#     plt.xlabel("Time (s)")
#     plt.ylabel("Core temperature (C)")
#     # plt.ylim(36, 38)
#     plt.legend()
#     plt.title("Core temperature comparison")
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()


if __name__ == "__main__":
        main()

