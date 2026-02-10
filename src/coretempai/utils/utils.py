from asyncio import BoundedSemaphore
import os
import torch
import re
import numpy as np
from pathlib import Path
import csv
import matplotlib.pyplot as plt

# Import config here to avoid circular imports
from coretempai.config import DIRECTORIES, PROFILE_CONFIG, FOURIER_CONFIG


# Function to save tensor data to a file
def save_tensor_to_file(tensor, file_name):
    if ".pt" in file_name:
        pass
    else:
        file_name = file_name + ".pt"

    # Check if file exists
    if os.path.exists(file_name):
        print(f"File {file_name} already exists.")
        user_input = input("Do you want to overwrite it? (y/n): ").strip().lower()
        if user_input == 'y':
            torch.save(tensor, file_name)
            print(f"File '{file_name}' has been overwritten.")
        else:
            # Optionally, create a new file with a unique name
            new_file_path = file_name
            counter = 1
            while os.path.exists(new_file_path):
                base, ext = os.path.splitext(file_name)
                new_file_path = f"{base}_{counter}{ext}"
                counter += 1
            torch.save(tensor, new_file_path)
            print(f"File '{new_file_path}' has been saved instead.")
    else:
        torch.save(tensor, file_name)
        print(f"File '{file_name}' has been saved.")


def min_max_normalize(tensor):

    # Min-Max Normalization
    min_vals = tensor.min(dim=0, keepdim=True).values  # Minimum value per column
    max_vals = tensor.max(dim=0, keepdim=True).values  # Maximum value per column

    normalized_tensor = (tensor - min_vals) / (max_vals - min_vals)

    return normalized_tensor

def z_score_normalize(tensor):

    # Compute mean and standard deviation along each feature (dim=0)
    mean = torch.mean(tensor, dim=0)
    std = torch.std(tensor, dim=0)

    # Prevent division by zero (if std is zero)
    std[std == 0] = 1e-8

    # Z-score normalization
    z_score_normalized = (tensor - mean) / std

    return z_score_normalized


# Function to extract numbers from the filename
def sort_case_file(file_path):
    # Extract the first sequence of digits from the filename
    match = re.search(r'\d+', os.path.basename(file_path))
    return int(match.group()) if match else 0  # Default to 0 if no number is found


def sort_core_temp_file(file_path):
    # Extract the first sequence of digits from the filename
    match = re.search(r'_(\d+)_1', os.path.basename(file_path))
    return int(match.group(1)) if match else 0  # Default to 0 if no number is found


class CustomLoss(torch.nn.Module):
    def __init__(self, avg_penalty=1):
        super(CustomLoss, self).__init__()
        self.avg_penalty = avg_penalty

    def forward(self, preds, targets):
        loss = torch.mean((preds - targets) ** 2) + 10*torch.sum((preds[:, :3] - targets[:, :3]) ** 2)

        return loss


class FunctionSampler:
    def __init__(self, x):
        self.x = x

    def sample(self, *args, **kwargs):
        pass


class Fourier(FunctionSampler):
    def __init__(self, x):
        super().__init__(x)
        self.nx = len(self.x)

    def sample(self, n, max_freq=None, range=None, initial_range=None, margin=0.95):
        """
        Sample random Fourier-series profiles.

        Notes:
        - `range` is treated as strict bounds [low, high] for all values, but samples do not need
          to hit the bounds exactly.
        - If `initial_range` is provided, only the first value of each profile is constrained to be
          within `initial_range` while the rest of the profile remains within `range`.
        """
        # Use default values if not provided
        if max_freq is None:

            max_freq = FOURIER_CONFIG["max_freq"]
        if range is None:

            range = FOURIER_CONFIG["nu_range"]

        if margin is None:
            margin = 0.95
        if not (0 < float(margin) <= 1.0):
            raise ValueError(f"margin must be in (0, 1], got {margin!r}")

        if range is None or len(range) != 2:
            raise ValueError(f"range must be a 2-length sequence [low, high], got {range!r}")
        low, high = float(range[0]), float(range[1])
        if not (low < high):
            raise ValueError(f"range must satisfy low < high, got {range!r}")

        if initial_range is not None:
            if len(initial_range) != 2:
                raise ValueError(
                    f"initial_range must be a 2-length sequence [low, high], got {initial_range!r}"
                )
            i_low, i_high = float(initial_range[0]), float(initial_range[1])
            if not (i_low < i_high):
                raise ValueError(f"initial_range must satisfy low < high, got {initial_range!r}")
            if not (low <= i_low and i_high <= high):
                raise ValueError(
                    f"initial_range must be within range; got initial_range={initial_range!r}, range={range!r}"
                )
        else:
            i_low, i_high = low, high

        # Functions (n , nx) = Sum of 'max_freq' frequency components (n , nx, max_freq)

        # num_terms = np.random.randint(1, max_freq+1,  size=(n, 1, 1)) 
        num_terms = np.random.randint(max_freq-5, max_freq+1,  size=(n, 1, 1)) 
        amp_cos = np.random.uniform(-5, 5, size=(n, 1, max_freq)) 
        amp_sin = np.random.uniform(-5, 5, size=(n, 1, max_freq)) 
        phase_cos = np.random.uniform(-np.pi, np.pi, size=(n, 1, max_freq)) 
        phase_sin = np.random.uniform(-np.pi, np.pi, size=(n, 1, max_freq)) 

        freq = np.arange(1, max_freq+1, 1).reshape(-1, 1) # (max_freq, 1)

        functions_cos = np.real(amp_cos*(np.e**(1j * (np.expand_dims((2 * np.pi * self.x * freq).T, 0) + phase_cos)))) # (n, nx, max_freq)
        functions_sin = np.real(amp_sin*(np.e**(1j * (np.expand_dims((2 * np.pi * self.x * freq).T, 0) + phase_sin)))) 
        functions = functions_cos + functions_sin 

        mask = np.broadcast_to(num_terms, (n, self.nx, max_freq)) >= np.broadcast_to(freq.reshape(1, 1, -1), (n, self.nx, max_freq)) 
        functions = np.sum(functions * mask, axis=-1)  # shape: (n, nx)

        # Enforce bounds by applying a per-sample affine transform about the first value.
        # This preserves the raw profile shape while ensuring all values are within [low, high],
        # and constraining the first value within [i_low, i_high].

        mean = np.mean(functions, axis=1)
        target_mean = (low + high)/2

        functions = functions + target_mean - mean



        # y0_targets = (low + high)/2
        
        # np.random.uniform(i_low, i_high, size=(n, 1))
        
        d = functions - target_mean # deltas from first point, shape: (n, nx)

        eps = 1e-12
        pos = d > eps
        neg = d < -eps

        # Upper bounds on scale s for each point, then take min across points.
        s_pos = np.full_like(d, np.inf, dtype=float)
        s_neg = np.full_like(d, np.inf, dtype=float)
        # If d > 0: y0 + s*d <= high  => s <= (high - y0)/d
        np.divide((high - target_mean), d, out=s_pos, where=pos)
        # If d < 0: y0 + s*d >= low   => s <= (y0 - low)/(-d)
        np.divide((target_mean - low), (-d), out=s_neg, where=neg)

        s_max = np.minimum(np.min(s_pos, axis=1), np.min(s_neg, axis=1)) # shape = (n,)
        # s_max_neg = np.min(s_neg, axis=1)

        s_max = np.maximum(s_max, 0.0)
        # s_max_neg = np.maximum(s_max_neg, 0.0)

        # Sample scale in [0, margin * s_max]. If s_max is 0, fall back to constant profile.
        s_max_scaled = (margin * s_max).reshape(n, 1)
        # s_max_scaled_neg = (margin* s_max_neg).reshape(n, 1)

        s = np.random.uniform(0.5, 1, size=(n, 1)) * s_max_scaled
        # s_neg = np.random.uniform(0.5, 1, size=(n, 1)) * s_max_scaled_neg

        bounded = s*d + target_mean
        # bounded[:, 1:] = bounded[:, 1:] + np.random.uniform(30, 50, size=(n,1))

        # For degenerate cases (essentially constant profile), bounded is already constant=y0_targets.
        return bounded

def create_run_directory():
    """
    Creates a new run directory with subdirectories for case_and_data, cbt, and profile.
    
    Returns:
        str: Path to the new run directory
    """
    # Get the raw directory path
    raw_dir = Path(DIRECTORIES["raw"])
    
    # Find the maximum run number using a simple pattern match
    run_pattern = re.compile(r"Run_(\d+)")
    existing_runs = [int(match.group(1)) for item in raw_dir.glob("Run_*") 
                    if item.is_dir() and (match := run_pattern.match(item.name))]
    
    # Determine the new run number
    new_n = 1 if not existing_runs else max(existing_runs) + 1
    
    # Create the new run directory with subdirectories
    new_run_dir = raw_dir / f"Run_{new_n}"
    for subdir in ["case_and_data", "output_profile", "input_profile"]:
        (new_run_dir / subdir).mkdir(parents=True, exist_ok=True)
    
    print(f"Created new run directory: {new_run_dir}")
    return new_run_dir, new_n


def generate_profile(profile_name, profile_range, run_dir, iter, profile_duration=None, step_size=None):
    """
    Generates a profile CSV file for use in Ansys Fluent.
    
    Args:
        profile_name (str): Name of the profile (e.g., "HR", "temperature")
        profile_range (list): Range of values [min, max] for the profile
        run_dir (str): Directory where profiles will be stored.
        profile_duration (int, optional): Duration of the profile in seconds. Defaults to config.PROFILE_DURATION.
        step_size (int, optional): Time step size in seconds. Defaults to config.STEP_SIZE.
        
    Returns:
        str: Filename of the generated profile
    """
    # Use default values from config if not provided
    if profile_duration is None:
        profile_duration = PROFILE_CONFIG["duration"]
    if step_size is None:
        step_size = PROFILE_CONFIG["step_size"]
    
    # Create profiles directory if it doesn't exist
    profiles_dir = os.path.join(run_dir, "input_profile")
    
    # Generate time points
    time_points = np.arange(0, profile_duration + step_size, step_size)
    
    # Normalize time points to [0, 1] for the Fourier sampler
    normalized_time = time_points / profile_duration
    
    # Create a Fourier sampler
    sampler = Fourier(normalized_time)
    
    # Generate a random profile within the specified range
    profile_values = sampler.sample(1, max_freq=5, range=profile_range)[0]
    
    # Create filename
    filename = f"random_{profile_name}_profile_{iter}.csv"
    filepath = os.path.join(profiles_dir, filename)
    
    # Write the profile to a CSV file
    with open(filepath, 'w', newline='') as csvfile:
        # Write header
        csvfile.write("[Name]\n")
        csvfile.write(f"random_{profile_name}_profile\n")
        csvfile.write("[Data]\n")
        csvfile.write(f"time,{profile_name}\n")
        
        # Write data
        for t, val in zip(time_points, profile_values):
            csvfile.write(f"{t},{val}\n")
    
    print(f"Generated profile: {filename}")
    return filepath


def delete_files_with_extension(directory, extension):
    """
    Deletes all files with the specified extension in the given directory.

    Args:
        directory (str or Path): The directory to search for files.
        extension (str): The file extension to delete (e.g., '.trn').

    Returns:
        None
    """
    # Ensure the directory is a Path object
    directory = Path(directory)

    # Iterate over all files in the directory with the given extension
    for file in directory.glob(f'*{extension}'):
        try:
            file.unlink()  # Delete the file
            print(f"Deleted: {file}")
        except Exception as e:
            print(f"Error deleting {file}: {e}")


def exit_solver(solver_session, cleanup=True, cleanup_dir=None):
    """
    Exits the solver session and cleans up temporary files (.log and .trn).
    
    Args:
        solver_session: The active PyFluent solver session
        cleanup: Boolean to determine if cleanup should be performed
        cleanup_dir (str or Path, optional): Directory to clean up. If None, uses current directory.
        
    Returns:
        None
    """
    # Exit the solver session
    solver_session.exit()

    if cleanup:
        # Get the directory to clean up
        if cleanup_dir is None:
            cleanup_dir = Path.cwd()
        else:
            cleanup_dir = Path(cleanup_dir)
    
        # Delete .log files
        delete_files_with_extension(cleanup_dir, '.log')
    
        # Delete .trn files
        delete_files_with_extension(cleanup_dir, '.trn')

        # Delete .bat files
        delete_files_with_extension(cleanup_dir, '.bat')
    else:
        pass


def safe_save(filepath, save_function, *args, **kwargs):
    """
    Save a file with overwrite protection.
    
    Args:
        filepath (str): Path where the file should be saved
        save_function (callable): Function to use for saving
        *args, **kwargs: Arguments to pass to save_function
        
    Returns:
        str: The actual filepath used for saving
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Check if file exists
    if os.path.exists(filepath):
        # Get file name and extension
        path = Path(filepath)
        name = path.stem
        suffix = path.suffix
        directory = path.parent
        
        # Find a new filename
        i = 1
        while True:
            new_filepath = os.path.join(directory, f"{name}_{i}{suffix}")
            if not os.path.exists(new_filepath):
                break
            i += 1
        
        # Save with new filename
        save_function(new_filepath, *args, **kwargs)
        return new_filepath
    else:
        # Save with original filename
        save_function(filepath, *args, **kwargs)
        return filepath


# Example usage for torch models
def save_model(model, filepath, overwrite_protection=True):
    """
    Save a PyTorch model with optional overwrite protection.
    
    Args:
        model: The PyTorch model to save
        filepath (str): Path where the model should be saved
        overwrite_protection (bool): Whether to use overwrite protection
        
    Returns:
        str: The actual filepath used for saving
    """
    if overwrite_protection:
        return safe_save(filepath, torch.save, model.state_dict())
    else:
        torch.save(model.state_dict(), filepath)
        return filepath


# Example usage for configurations
def save_config(config, filepath, overwrite_protection=True):
    """
    Save a configuration object with optional overwrite protection.
    
    Args:
        config: The configuration object to save
        filepath (str): Path where the config should be saved
        overwrite_protection (bool): Whether to use overwrite protection
        
    Returns:
        str: The actual filepath used for saving
    """
    if overwrite_protection:
        return safe_save(filepath, torch.save, config.to_dict())
    else:
        torch.save(config.to_dict(), filepath)
        return filepath


# Example usage for matplotlib figures
def save_figure(fig, filepath, overwrite_protection=True, **kwargs):
    """
    Save a matplotlib figure with optional overwrite protection.
    
    Args:
        fig: The matplotlib figure to save
        filepath (str): Path where the figure should be saved
        overwrite_protection (bool): Whether to use overwrite protection
        **kwargs: Additional arguments to pass to savefig
        
    Returns:
        str: The actual filepath used for saving
    """
    if overwrite_protection:
        def save_fig(path):
            fig.savefig(path, **kwargs)
        
        return safe_save(filepath, save_fig)
    else:
        fig.savefig(filepath, **kwargs)
        return filepath


def plot_comparison(y_true, y_pred, title="Comparison", save_path=None):
    """
    Plot a comparison between true and predicted values.
    
    Args:
        y_true (numpy.ndarray): True values
        y_pred (numpy.ndarray): Predicted values
        title (str): Plot title
        save_path (str, optional): Path to save the plot
        
    Returns:
        matplotlib.figure.Figure: The figure object
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot true values
    ax.plot(y_true, label="True", color="blue", linewidth=2)
    
    # Plot predicted values
    ax.plot(y_pred, label="Predicted", color="red", linewidth=2, linestyle="--")
    
    # Add labels and title
    ax.set_xlabel("Index")
    ax.set_ylabel("Value")
    ax.set_title(title)
    ax.legend()
    
    # Add grid
    ax.grid(True, linestyle="--", alpha=0.7)
    
    # Tight layout
    fig.tight_layout()
    
    # Save if path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    
    return fig

def calculate_metrics(y_true, y_pred):
    """
    Calculate metrics for model evaluation.
    
    Args:
        y_true (numpy.ndarray): True values
        y_pred (numpy.ndarray): Predicted values
        
    Returns:
        dict: Dictionary of metrics
    """
    # Convert to numpy arrays if they are tensors
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    
    # Calculate metrics
    mse = np.mean((y_true - y_pred) ** 2)
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(mse)
    
    # Calculate R^2 score
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_res = np.sum((y_true - y_pred) ** 2)
    r2 = 1 - (ss_res / (ss_tot + 1e-10))
    
    # Calculate normalized RMSE
    y_range = np.max(y_true) - np.min(y_true)
    nrmse = rmse / (y_range + 1e-10)
    
    return {
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "nrmse": nrmse
    }

def save_scalar_parameters(scalar_values, scalar_params, run_dir, iteration):
    """
    Save scalar parameters to a CSV file.
    
    Args:
        scalar_values (dict): Dictionary of parameter names and their values
        scalar_params (dict): Dictionary of parameter names and their ranges/units
        run_dir (str): Directory where the CSV file should be saved
        iteration (int): Current iteration number
        
    Returns:
        str: Path to the saved CSV file
    """
    # Create the directory if it doesn't exist
    scalar_dir = os.path.join(run_dir, "input_scalars")
    os.makedirs(scalar_dir, exist_ok=True)
    
    scalar_file_path = os.path.join(scalar_dir, "input_scalars.csv")
    
    # Check if file exists to determine if we need to write headers
    file_exists = os.path.isfile(scalar_file_path)
    
    with open(scalar_file_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Write header only if file is new
        if not file_exists:
            headers = ['Iteration'] + list(scalar_params.keys())
            writer.writerow(headers)
        
        # Write data row with iteration number and parameter values with units
        row = [iteration] + [scalar_values[param] for param in scalar_params.keys()]
        writer.writerow(row)
    
    print(f"Appended scalar parameters for iteration {iteration} to: {scalar_file_path}")
    return scalar_file_path



