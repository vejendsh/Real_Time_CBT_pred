"""
This script generates transient simulation parameters and modifies UDF.c file.
It generates random HR profiles using Fourier sampler and random scalar parameters,
then injects them into the UDF.c file.
"""


import numpy as np
from coretempai.config import SIMULATION_CONFIG, PROFILE_CONFIG
from coretempai.utils.utils import Fourier
import coretempai.input_parameters.input_parameters as input_parameters


def generate_hr_profile(max_waves, duration, step_size, hr_range, initial_hr_range):
    """
    Generates a random HR profile using Fourier sampler.
    
    Args:
        max_waves (int): Maximum number of Fourier waves (max frequency)
        duration (float): Duration of the profile in seconds
        step_size (float): Time step size in seconds
        hr_range (list): [min, max] range for HR values in min^-1
        initial_hr_range (list): [min, max] range for initial HR values in min^-1
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

    profile_values = np.zeros(shape=(1,1))
    i=0

    while not (initial_hr_range[0] <= profile_values[0] <= initial_hr_range[1]):
        profile_values = sampler.sample(1, max_freq=max_waves, range=hr_range, initial_range=initial_hr_range)[0]
        i=i+1
    print(f"Took {i} attempt(s) to generate the required HR profile")


    return profile_values


def generate_scalar_params(scalar_params, initial_scalar_params):
    """
    Randomly samples scalar parameters from their specified ranges.
    
    Args:
        scalar_params (dict): Dictionary of parameter names and their ranges
                             Example: {"T_amb": [20, 50, "C"], "h": [1, 10, "W/m^2/K"]}
        
    Returns:
        dict: Dictionary of parameter names and their randomly sampled values
              Example: {"T_amb": 35.5, "h": 6.2}
    """
    scalar_values = {}
    for param_name, param_range in scalar_params.items():
        min_val = param_range[0]
        max_val = param_range[1]
        random_value = round(np.random.uniform(min_val, max_val), 3)
        scalar_values[param_name] = random_value

    initial_scalar_values = {}
    for param_name, param_range in initial_scalar_params.items():
        min_val = param_range[0]
        max_val = param_range[1]
        random_value = round(np.random.uniform(min_val, max_val), 3)
        initial_scalar_values[param_name] = random_value
    
    return scalar_values, initial_scalar_values


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


def modify_udf_file(udf_path, hr_array, time_points, t_amb, h_coefficient, t_amb_initial, h_coefficient_initial):
    """
    Modifies UDF.c file to inject HR profile, time points, T_amb, and heat transfer coefficient.
    Uses pattern matching instead of line numbers for robustness.
    
    Args:
        udf_path (str): Path to UDF.c file
        hr_array (numpy.ndarray): HR profile array
        time_points (numpy.ndarray): Time points array corresponding to HR profile
        t_amb (float): Ambient temperature in Celsius
        h_coefficient (float): Heat transfer coefficient in W/m^2/K
        t_amb_initial (float): Initial ambient temperature in Celsius
        h_coefficient_initial (float): Initial heat transfer coefficient in W/m^2/K
    Returns:
        None
    """
    import re
    
    # Read the UDF file
    with open(udf_path, 'r') as f:
        content = f.read()
    
    # Format HR array and time array as C code
    hr_array_str = format_hr_array_c(hr_array)
    time_array_str = format_time_array_c(time_points)
    
    # Pattern 1: Replace heat transfer coefficient
    # Matches: #define heattransfercoefficient <any_value>
    pattern_htc = r'real\s+heattransfercoefficient\s*=\s*[\d.]+;'
    replacement_htc = f'real heattransfercoefficient = {h_coefficient};'
    if not re.search(pattern_htc, content):
        raise ValueError(f"Pattern 'real heattransfercoefficient' not found in {udf_path}")
    content = re.sub(pattern_htc, replacement_htc, content)
    
    # Pattern 2: Replace ambient temperature
    # Matches: #define ambienttemperature <any_value>
    pattern_tamb = r'real\s+ambienttemperature\s*=\s*[\d.]+;'
    replacement_tamb = f'real ambienttemperature = {t_amb};'
    if not re.search(pattern_tamb, content):
        raise ValueError(f"Pattern 'real ambienttemperature' not found in {udf_path}")
    content = re.sub(pattern_tamb, replacement_tamb, content)
    
    # Pattern 3: Replace HR array
    # Matches: real heartrate[<any_size>]={<any_values>};
    # This pattern handles both single-line and multi-line arrays
    # Uses non-greedy matching to stop at the first closing brace-semicolon
    pattern_hr = r'real\s+heartrate\[\d+\]\s*=\s*\{.*?\};'
    replacement_hr = hr_array_str
    if not re.search(pattern_hr, content, flags=re.DOTALL):
        raise ValueError(f"Pattern 'real heartrate[...]' not found in {udf_path}")
    # Use DOTALL flag to allow . to match newlines across multiple lines
    content = re.sub(pattern_hr, replacement_hr, content, flags=re.DOTALL)
    
    # Pattern 4: Replace heartrateinitial with first value from HR array
    # Matches: #define heartrateinitial <any_value>
    hr_initial = hr_array[0]  # Get first value from HR array
    pattern_hr_initial = r'#define\s+heartrateinitial\s+[\d.]+'
    replacement_hr_initial = f'#define heartrateinitial {hr_initial:.2f}'
    if not re.search(pattern_hr_initial, content):
        raise ValueError(f"Pattern '#define heartrateinitial' not found in {udf_path}")
    content = re.sub(pattern_hr_initial, replacement_hr_initial, content)
    
    # Pattern 5: Replace inputtime array
    # Matches: real inputtime[<any_size>]={<any_values>};
    # This pattern handles both single-line and multi-line arrays
    pattern_inputtime = r'real\s+inputtime\[\d+\]\s*=\s*\{.*?\};'
    replacement_inputtime = time_array_str
    if not re.search(pattern_inputtime, content, flags=re.DOTALL):
        raise ValueError(f"Pattern 'real inputtime[...]' not found in {udf_path}")
    # Use DOTALL flag to allow . to match newlines across multiple lines
    content = re.sub(pattern_inputtime, replacement_inputtime, content, flags=re.DOTALL)

    #Pattern 6: Replace heattransfercoefficient_initial
    pattern_htc_initial = r'real\s+heattransfercoefficient_initial\s*=\s*[\d.]+;'
    replacement_htc_initial = f'real heattransfercoefficient_initial = {h_coefficient_initial};'
    if not re.search(pattern_htc_initial, content):
        raise ValueError(f"Pattern 'real heattransfercoefficient_initial' not found in {udf_path}")
    content = re.sub(pattern_htc_initial, replacement_htc_initial, content)
    
    #Pattern 7: Replace ambienttemperature_initial
    
    pattern_tamb_initial = r'real\s+ambienttemperature_initial\s*=\s*[\d.]+;'
    replacement_tamb_initial = f'real ambienttemperature_initial = {t_amb_initial};'
    if not re.search(pattern_tamb_initial, content):
        raise ValueError(f"Pattern 'real ambienttemperature_initial' not found in {udf_path}")
    content = re.sub(pattern_tamb_initial, replacement_tamb_initial, content)
    
    # Write modified file back
    with open(udf_path, 'w') as f:
        f.write(content)
    
    print(f"Modified UDF file: {udf_path}")
    print(f"  - Heat transfer coefficient: {h_coefficient} W/m^2/K")
    print(f"  - Ambient temperature: {t_amb} C")
    print(f"  - Initial heat transfer coefficient: {h_coefficient_initial} W/m^2/K")
    print(f"  - Initial ambient temperature: {t_amb_initial} C")
    print(f"  - HR profile array size: {len(hr_array)}")
    print(f"  - Time points array size: {len(time_points)}")
    print(f"  - Heart rate initial: {hr_initial:.2f} min^-1")


def set_transient_params(udf_path, scalar_params, initial_scalar_params, profile_params, initial_profile_params, max_waves=5, 
                         duration=None, step_size=None):
    """
    Main function to generate and set transient parameters in UDF.c.
    
    Args:
        udf_path (str): Path to UDF.c file
        scalar_params (dict): Dictionary of scalar parameter ranges
        initial_scalar_params (dict): Dictionary of initial scalar parameter ranges
        initial_profile_params (dict): Dictionary of initial profile parameter ranges
        profile_params (dict): Dictionary of profile parameter ranges
        max_waves (int): Maximum number of Fourier waves for HR profile
        duration (float, optional): Profile duration in seconds. Defaults to TRANSIENT_CONFIG.
        step_size (float, optional): Step size in seconds. Defaults to TRANSIENT_CONFIG.
        
    Returns:
        dict: Dictionary containing generated parameter values
    """
    # Use default values from config if not provided
    if duration is None:
        duration = PROFILE_CONFIG["profile_duration"]
    if step_size is None:
        step_size = PROFILE_CONFIG["step_size"]
    
    # Generate scalar parameters
    scalar_values, initial_scalar_values = generate_scalar_params(scalar_params, initial_scalar_params)
    
    # Generate time points (must match HR profile generation)
    time_points = np.arange(0, duration + step_size, step_size)
    
    # Generate HR profile
    hr_range = profile_params.get("HR", [60, 120])  # Default range if not specified
    initial_hr_range = initial_profile_params.get("HR", [40, 100])
    hr_array = generate_hr_profile(max_waves, duration, step_size, hr_range[:2], initial_hr_range[:2])  # Get [min, max]
    
    # Ensure time_points and hr_array have the same length
    if len(time_points) != len(hr_array):
        raise ValueError(f"Time points array length ({len(time_points)}) does not match HR array length ({len(hr_array)})")
    
    # Extract scalar values
    t_amb = scalar_values.get("T_amb", 25.0)  # Default if not found
    h_coefficient = scalar_values.get("h", 5.0)  # Default if not found
    h_coefficient_initial = initial_scalar_values.get("h", 5.0)
    t_amb_initial = initial_scalar_values.get("T_amb", 25.0)
    # Modify UDF file
    modify_udf_file(udf_path, hr_array, time_points, t_amb, h_coefficient, t_amb_initial, h_coefficient_initial)
    
    # Return generated values for logging/saving
    return {
        "scalar_values": scalar_values,
        "initial_scalar_values": initial_scalar_values,
        "hr_profile": hr_array,
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
    udf_path = SIMULATION_CONFIG["udf_file_general"]
    
    # Get parameters from input_parameters module
    scalar_params = input_parameters.scalar_params
    initial_scalar_params = input_parameters.initial_scalar_params
    profile_params = input_parameters.profile_params
    initial_profile_params = input_parameters.initial_profile_params
    
    # Set transient parameters
    result = set_transient_params(
        udf_path=udf_path,
        scalar_params=scalar_params,
        initial_scalar_params=initial_scalar_params,
        initial_profile_params=initial_profile_params,
        profile_params=profile_params,
        max_waves=PROFILE_CONFIG["max_fourier_waves"]
    )
    
    print("\nGenerated parameters:")
    print(f"  Scalar values: {result['scalar_values']}")
    print(f"  Initial scalar values: {result['initial_scalar_values']}")
    print(f"  HR profile length: {len(result['hr_profile'])}")
    print(f"  HR range: {result['hr_range']}")

    from matplotlib import pyplot as plt
    plt.plot(np.arange(0, len(result['hr_profile'])) * PROFILE_CONFIG["step_size"], result['hr_profile'], label='HR profile')
    plt.xlabel('Time (s)')
    plt.ylabel('HR (min^-1)')
    # plt.ylim((profile_params.get("HR")[0], profile_params.get("HR")[1]))
    plt.title('HR Profile')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()

