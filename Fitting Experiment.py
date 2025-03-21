import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm
from astropy.modeling.models import Moffat2D
from skimage.transform import rotate
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit
from scipy.optimize import OptimizeWarning
import warnings
import time
import concurrent.futures

# --------------------------
# Global Settings and Invariants
# --------------------------

# Suppress warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message="overflow encountered in exp")
warnings.filterwarnings("ignore", category=OptimizeWarning, message="Covariance of the parameters could not be estimated")

def load_config(file_path):
    """Load configuration parameters from a text file."""
    config = {}
    with open(file_path, "r") as file:
        for line in file:
            if line.strip() and '=' in line:
                key, value = line.strip().split(" = ")
                config[key] = float(value)
    return config

# Star parameters
FLUX_MIN = 900
FLUX_MAX = 1e5
ALPHA_STAR = 2.35

# Load configuration parameters
config_file_path = "Config/image_config.txt"
params = load_config(config_file_path)
gamma = params.get("Gamma", 1.0)
alpha = params.get("Alpha", 2.0)
a = params.get("a", 3.0)
factor = params.get("Background factor", 4.0)
mu = params.get("Background mu", 5.0)

# Simulation parameters
image_size_full = 500
experiments = 1000

# Precompute the coordinate grid (invariant for every experiment)
grid_y, grid_x = np.mgrid[0:image_size_full, 0:image_size_full]

# --------------------------
# Helper Functions
# --------------------------

def back_transform(x, y, angle, original_shape, rotated_shape):
    """Transform coordinates from the rotated image back to the original image space."""
    angle_rad = -np.deg2rad(angle)
    original_center_x, original_center_y = original_shape[1] / 2, original_shape[0] / 2
    rotated_center_x, rotated_center_y = rotated_shape[1] / 2, rotated_shape[0] / 2
    x_shifted = x - rotated_center_x
    y_shifted = y - rotated_center_y
    x_original = x_shifted * np.cos(angle_rad) + y_shifted * np.sin(angle_rad) + original_center_x
    y_original = -x_shifted * np.sin(angle_rad) + y_shifted * np.cos(angle_rad) + original_center_y
    return x_original, y_original

def rotate_to_sigmoid_space(x, y, angle, original_shape, rotated_shape):
    """Rotate coordinates from the original image space into the rotated space used for sigmoid fitting."""
    angle_rad = np.deg2rad(angle)
    original_center_x, original_center_y = original_shape[1] / 2, original_shape[0] / 2
    rotated_center_x, rotated_center_y = rotated_shape[1] / 2, rotated_shape[0] / 2
    x_shifted = x - original_center_x
    y_shifted = y - original_center_y
    x_rotated = x_shifted * np.cos(angle_rad) + y_shifted * np.sin(angle_rad) + rotated_center_x
    y_rotated = -x_shifted * np.sin(angle_rad) + y_shifted * np.cos(angle_rad) + rotated_center_y
    return x_rotated, y_rotated

def double_sigmoid(x, A, B, c, s, f):
    """Double sigmoid function used for fitting the trail profile."""
    x = np.clip(x, -500, 500)
    return A * (1 / (1 + np.exp(c * (s - x))) + 1 / (1 + np.exp(c * (x - f)))) + B

temp_moffat = Moffat2D(amplitude=1, x_0=0, y_0=0, gamma=gamma, alpha=alpha)

# --------------------------
# Experiment Function (to run in parallel)
# --------------------------

def run_experiment(count):
    """
    One independent experiment:
      - Generates a simulated image with stars and an asteroid trail.
      - Rotates the image.
      - Extracts a trail profile.
      - Fits a double sigmoid to determine trail start and end.
      - Projects the true trail coordinates into the rotated space and compares along the 1D profile.
      
    Returns:
      A tuple (start_delta_rotated, end_delta_rotated)
    """
    # --------------------------
    # Create Background Image with Stars
    # --------------------------
    simulated_image = np.full((image_size_full, image_size_full), mu)
    '''
    n_sources = int(image_size_full / 3)
    r = np.random.uniform(0, 1, n_sources)
    fluxes = FLUX_MIN * (1 - r + r * (FLUX_MAX / FLUX_MIN) ** (1 - ALPHA_STAR)) ** (1 / (1 - ALPHA_STAR))
    x_coords = np.random.uniform(-0.5, image_size_full - 0.5, n_sources)
    y_coords = np.random.uniform(-0.5, image_size_full - 0.5, n_sources)
    
    data_psf = np.zeros((image_size_full, image_size_full))
    for i in range(n_sources):
        model = Moffat2D(amplitude=fluxes[i], x_0=x_coords[i], y_0=y_coords[i], gamma=gamma, alpha=alpha)
        model_image = model(grid_x, grid_y)
        data_psf += np.maximum(model_image - mu / 4, 0)
    
    def transformation_function(value, a=a):
        return -10**a * np.exp(-10**(-a) * value) + 10**a
    
    data_psf = transformation_function(data_psf)
    '''
    combined_image = simulated_image# + data_psf






    # --------------------------
    # Add Asteroid Trail
    # --------------------------
    trail_length = np.random.randint(int(image_size_full / 4), int(image_size_full / 3.5))
    spacing = 2.2
    base_flux = 1400
    flux_range = 0.1 * base_flux
    period = 0.01 * trail_length
    angle = np.random.uniform(0, 180)
    angle_rad = np.radians(angle)
    direction_vector = np.array([np.cos(angle_rad), np.sin(angle_rad)])
    center_of_image = np.array([image_size_full / 2, image_size_full / 2])
    start_position = center_of_image - (trail_length * spacing * direction_vector / 2)
    
    # Compute trial-specific true coordinates for the trail
    true_start = start_position
    true_end = start_position + (trail_length - 1) * spacing * direction_vector
    
    smoothing = 0.6

    data_trail = np.zeros((image_size_full, image_size_full))
    for i in range(trail_length):
        current_position = start_position + i * spacing * direction_vector
        x_coord, y_coord = current_position[0], current_position[1]
        fluctuating_flux = base_flux + flux_range * np.sin(2 * np.pi * i / period)
        adjusted_amplitude = fluctuating_flux / temp_moffat(0, 0)
        moffat_model = Moffat2D(amplitude=adjusted_amplitude, x_0=x_coord, y_0=y_coord, gamma=gamma, alpha=alpha)
        data_trail += np.maximum(moffat_model(grid_x, grid_y) - mu/2, 0)
        
    
    merged_data = data_trail + combined_image
    #merged_data = np.random.normal(loc=merged_data, scale=(1 / smoothing) * factor * np.sqrt(merged_data))
    #merged_data = gaussian_filter(merged_data, sigma=smoothing)
    
    
    
    
    

    # --------------------------
    # Rotate the Image
    # --------------------------
    rotated_image = rotate(merged_data, angle, resize=True, order=3)
    
    
    
    
    
    
    # --------------------------
    # Find the Row Containing the Trail and Fit Double Sigmoid
    # --------------------------
    row_sums = np.sum(rotated_image, axis=1)
    brightest_row = np.argmax(row_sums)
    rows_to_sum = np.arange(max(0, brightest_row - 2), min(rotated_image.shape[0], brightest_row + 2))
    smoothed_row_sums = np.sum(rotated_image[rows_to_sum, :], axis=0) / 4
    trail_profile = smoothed_row_sums

    background_threshold = 1200
    signal_indices = np.where(trail_profile > background_threshold)[0]
    
    if signal_indices.size > 0:
        start_signal = signal_indices[0]
        end_signal = signal_indices[-1]
        padding = int(0.05 * (end_signal - start_signal))
        start_fit = max(0, start_signal - padding)
        end_fit = min(len(trail_profile), end_signal + padding)
        x_data = np.arange(start_fit, end_fit)
        trail_profile_limited = trail_profile[start_fit:end_fit]
        initial_guess = [
            np.max(trail_profile_limited),   # A: Peak value
            np.median(trail_profile_limited),  # B: Baseline
            0.01,                              # c: Slope parameter
            start_fit,                         # s: Start of trail
            end_fit                            # f: End of trail
        ]
    else:
        raise ValueError("No signal detected above the background threshold.")
    
    try:
        popt, _ = curve_fit(double_sigmoid, x_data, trail_profile_limited, p0=initial_guess, maxfev=5000)
        start_pixel = popt[3]
        end_pixel = popt[4]
    except RuntimeError:
        start_pixel = image_size_full * 2
        end_pixel = image_size_full * 2
        
        
        
        
        
    
    # --------------------------
    # Project True Trail Coordinates onto the Fitted Trail Line in Rotated Space
    # --------------------------
    # Rotate the true trail coordinates into the rotated image space.
    rotated_true_start_x, rotated_true_start_y = rotate_to_sigmoid_space(true_start[0], true_start[1], angle, merged_data.shape, rotated_image.shape)
    rotated_true_end_x,   rotated_true_end_y   = rotate_to_sigmoid_space(true_end[0], true_end[1], angle, merged_data.shape, rotated_image.shape)
    
    # Since the fit was performed along the horizontal axis (at y = brightest_row),
    # we project the rotated true coordinates onto that horizontal line.
    # The projection onto a horizontal line keeps the x-coordinate unchanged.
    projected_true_start_x = rotated_true_start_x
    projected_true_end_x   = rotated_true_end_x
    
    # Calculate the 1D deltas (along the x-axis in rotated space) between the true and fitted positions.
    start_delta_rotated = projected_true_start_x - start_pixel
    end_delta_rotated   = projected_true_end_x - end_pixel
    
    return start_delta_rotated, end_delta_rotated

    '''
    # --------------------------
    # Back-Transform Coordinates
    # --------------------------
    original_shape = merged_data.shape
    rotated_shape = rotated_image.shape
    original_start_x, original_start_y = back_transform(start_pixel, brightest_row, angle, original_shape, rotated_shape)
    original_end_x, original_end_y = back_transform(end_pixel, brightest_row, angle, original_shape, rotated_shape)

    # --------------------------
    # Compare with Trial True Coordinates in Rotated Space
    # --------------------------
    rotated_true_start_x, rotated_true_start_y = rotate_to_sigmoid_space(true_start[0], true_start[1], angle, original_shape, rotated_shape)
    rotated_true_end_x, rotated_true_end_y = rotate_to_sigmoid_space(true_end[0], true_end[1], angle, original_shape, rotated_shape)
    
    start_delta_rotated = rotated_true_start_x - start_pixel
    end_delta_rotated = rotated_true_end_x - end_pixel
    
    return start_delta_rotated, end_delta_rotated
    '''


# --------------------------
# Main Execution: Parallelize the Experiments
# --------------------------
if __name__ == '__main__':
    overall_start_time = time.time()
    
    # Run experiments in parallel using ProcessPoolExecutor
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(executor.map(run_experiment, range(experiments)))
    
    # Unpack the results into separate lists
    stdv_s, stdv_e = zip(*results)
    stdv_s = list(stdv_s)
    stdv_e = list(stdv_e)

    
    # Save the results to .npy files
    np.save("stdv_s_clean.npy", stdv_s)
    np.save("stdv_e_clean.npy", stdv_e)
    
    # --------------------------
    # Plotting the Results
    # --------------------------
    filtered_stdv_s = [x for x in stdv_s if abs(x) < 2]
    filtered_stdv_e = [x for x in stdv_e if abs(x) < 2]

    overall_elapsed = time.time() - overall_start_time
    minutes, seconds = divmod(overall_elapsed, 60)
    print(f"Total elapsed time: {int(minutes)} min {int(seconds)} sec")
    
    plt.figure(figsize=(12, 6))
    
    # Subplot 1: Histogram for filtered_stdv_s
    plt.subplot(1, 2, 1)
    plt.hist(filtered_stdv_s, bins=20, density=True, alpha=0.6, color='blue')
    mean_s = np.mean(filtered_stdv_s) if filtered_stdv_s else 0
    std_s = np.std(filtered_stdv_s) if filtered_stdv_s else 0
    x_s = np.linspace(min(filtered_stdv_s) if filtered_stdv_s else -5, max(filtered_stdv_s) if filtered_stdv_s else 5, 100)
    plt.plot(x_s, norm.pdf(x_s, mean_s, std_s), color='red', 
             label=f'Gaussian\nMean={mean_s:.2f}\nStd={std_s:.2f}')
    plt.xlabel("Difference in Start (pixels)")
    plt.ylabel("Density")
    plt.title("Distribution of Differences (Start)")
    plt.legend()
    
    # Subplot 2: Histogram for filtered_stdv_e
    plt.subplot(1, 2, 2)
    plt.hist(filtered_stdv_e, bins=20, density=True, alpha=0.6, color='green')
    mean_e = np.mean(filtered_stdv_e) if filtered_stdv_e else 0
    std_e = np.std(filtered_stdv_e) if filtered_stdv_e else 0
    x_e = np.linspace(min(filtered_stdv_e) if filtered_stdv_e else -5, max(filtered_stdv_e) if filtered_stdv_e else 5, 100)
    plt.plot(x_e, norm.pdf(x_e, mean_e, std_e), color='red', 
             label=f'Gaussian\nMean={mean_e:.2f}\nStd={std_e:.2f}')
    plt.xlabel("Difference in End (pixels)")
    plt.ylabel("Density")
    plt.title("Distribution of Differences (End)")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"histogram_stdv_distributions_{image_size_full}x{image_size_full}_{experiments}.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    
