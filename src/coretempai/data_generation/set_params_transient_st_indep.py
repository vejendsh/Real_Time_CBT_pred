"""
This script generates transient simulation parameters and modifies UDF.c file.
It generates random HR profiles using Fourier sampler and random scalar parameters,
then injects them into the UDF.c file.
"""


import numpy as np
from numpy.random import noncentral_f
from coretempai.config import SIMULATION_CONFIG, PROFILE_CONFIG
from coretempai.utils.utils import Fourier
import coretempai.input_parameters.input_parameters_st_indep as input_parameters


def generate_profile(max_waves, duration, step_size, range, initial_range=None, initial_value=None):
    """
    Generates a random HR profile using Fourier sampler.
    
    Args:
        max_waves (int): Maximum number of Fourier waves (max frequency)
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
    # n=1 means generate 1 profile, max_freq=max_waves, range=hr_range

    profile_values = sampler.sample(1, max_freq=max_waves, range=range, initial_range=initial_range, initial_value=initial_value)[0]


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

    pattern_st = r'real\s+skintemp\[\d+\]\s*=\s*\{.*?\};'
    replacement_st = st_array_str
    if not re.search(pattern_st, content, flags=re.DOTALL):
        raise ValueError(f"Pattern 'real heartrate[...]' not found in {udf_path}")
    # Use DOTALL flag to allow . to match newlines across multiple lines
    content = re.sub(pattern_st, replacement_st, content, flags=re.DOTALL)


    
    # Write modified file back
    with open(udf_path, 'w') as f:
        f.write(content)
    
    print(f"Modified UDF file: {udf_path}")
    print(f"  - HR profile array size: {len(hr_array)}")
    print(f"  - ST profile array size: {len(st_array)}")
    print(f"  - Time points array size: {len(time_points)}")


def set_transient_params(udf_path, profile_params, initial_profile_params, max_waves=5, 
                         duration=None, step_size=None):
    """
    Main function to generate and set transient parameters in UDF.c.
    
    Args:
        udf_path (str): Path to UDF.c file
        profile_params (dict): Dictionary of profile parameter ranges
        initial_profile_params (dict): Dictionary of initial profile parameter ranges
        max_waves (int): Maximum number of Fourier waves for profiles
        duration (float, optional): Profile duration in seconds. Defaults to PROFILE_CONFIG.
        step_size (float, optional): Step size in seconds. Defaults to PROFILE_CONFIG.
        
    Returns:
        dict: Dictionary containing generated parameter values
    """
    # Use default values from config if not provided
    if duration is None:
        duration = PROFILE_CONFIG["profile_duration"]
    if step_size is None:
        step_size = PROFILE_CONFIG["step_size"]
    
    
    # Generate time points (must match HR profile generation)
    time_points = np.arange(0, duration + step_size, step_size)
    
    # Generate HR profile
    hr_range = profile_params.get("HR", [60, 120])  # Default range if not specified
    initial_hr_range = initial_profile_params.get("HR", [60, 100])  # Default range if not specified
    hr_array = generate_profile(max_waves, duration, step_size, hr_range[:2], initial_hr_range[:2])  # Get [min, max]

    # Generate ST profile
    st_range = profile_params.get("ST", [30, 40])  # Default range if not specified
    # initial_st_range = initial_profile_params.get("ST", [30, 40])
    st_array = generate_profile(max_waves, duration, step_size, st_range[:2], initial_value=33.5)  # Get [min, max]
    
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
        "max_waves": max_waves,
        "duration": duration,
        "step_size": step_size
    }


def main():
    """
    Example usage of the set_transient_params function.
    """
    udf_path = SIMULATION_CONFIG["udf_file_general_st_indep"]
    
    # Get parameters from input_parameters module
    profile_params = input_parameters.profile_params
    initial_profile_params = input_parameters.initial_profile_params
    # Set transient parameters
    result = set_transient_params(
        udf_path=udf_path,
        profile_params=profile_params,
        initial_profile_params=initial_profile_params,
        max_waves=PROFILE_CONFIG["max_freq"]
    )
    
    print("\nGenerated parameters:")
    print(f"  HR profile length: {len(result['hr_profile'])}")
    print(f"  HR range: {result['hr_range']}")

    from matplotlib import pyplot as plt
    fig, axs = plt.subplots(2, 1, figsize=(8,6), sharex=True)
    axs[0].plot(np.arange(0, len(result['hr_profile'])) * PROFILE_CONFIG["step_size"], result['hr_profile'], label='HR profile')
    axs[0].set_ylabel('HR (bpm)')
    axs[0].set_ylim((profile_params.get("HR")[0], profile_params.get("HR")[1]))
    # axs[0].set_ylim(-200, 200)
    axs[0].set_title('HR Profile')
    axs[0].legend() 
    axs[0].grid(True)

    axs[1].plot(np.arange(0, len(result['st_profile'])) * PROFILE_CONFIG["step_size"], result['st_profile'], label='ST profile')
    axs[1].set_ylabel('ST (K)')
    axs[1].set_xlabel('Time (s)')
    axs[1].set_ylim((profile_params.get("ST")[0], profile_params.get("ST")[1]))
    # axs[1].set_ylim(-200, 200)
    axs[1].set_title('ST Profile')
    axs[1].grid(True)
    axs[1].legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

