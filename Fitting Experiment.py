#!/usr/bin/env python3
"""
Script: parallel_trail_experiments.py
Description:
    This script runs a number of independent experiments to simulate asteroid trails
    and analyze the trail profile via a double sigmoid fit. Each experiment:
      - Generates a uniform background image.
      - Adds an asteroid trail simulated with a Moffat PSF.
      - Rotates the image by a random angle.
      - Extracts the trail profile and fits a double sigmoid function.
      - Computes the 1D difference (delta) between the true trail endpoints and the fitted endpoints.
    The experiments are run in parallel and the distributions of the differences are saved and plotted.
    
Author: Ivan
"""

# ===============================
# IMPORTS
# ===============================
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm
from astropy.modeling.models import Moffat2D
from skimage.transform import rotate
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit, OptimizeWarning
import warnings
import time
import concurrent.futures

# Suppress warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message="overflow encountered in exp")
warnings.filterwarnings("ignore", category=OptimizeWarning, message="Covariance of the parameters could not be estimated")

# ===============================
# CONFIGURATION & PARAMETERS
# ===============================
CONFIG_FILE_PATH = "Config/image_config.txt"
EXPERIMENTS = 1000
IMAGE_SIZE_FULL = 500

# Star parameters (for PSF simulation if needed)
FLUX_MIN = 900
FLUX_MAX = 1e5
ALPHA_STAR = 2.35

# Load configuration parameters from file
def load_config(file_path):
    """Load configuration parameters from a text file with format 'key = value'."""
    config = {}
    with open(file_path, "r") as file:
        for line in file:
            if line.strip() and '=' in line:
                key, value = line.strip().split(" = ")
                config[key] = float(value)
    return config

params = load_config(CONFIG_FILE_PATH)
gamma = params.get("Gamma", 1.0)
alpha = params.get("Alpha", 2.0)
a = params.get("a", 3.0)
factor = params.get("Background factor", 4.0)
mu = params.get("Background mu", 5.0)

# Precompute coordinate grid (invariant for all experiments)
grid_y, grid_x = np.mgrid[0:IMAGE_SIZE_FULL, 0:IMAGE_SIZE_FULL]

# ===============================
# HELPER FUNCTIONS
# ===============================
def back_transform(x, y, angle, original_shape, rotated_shape):
    """
    Transform coordinates from the rotated image back to the original image space.
    """
    angle_rad = -np.deg2rad(angle)
    original_center_x, original_center_y = original_shape[1] / 2, original_shape[0] / 2
    rotated_center_x, rotated_center_y = rotated_shape[1] / 2, rotated_shape[0] / 2
    x_shifted = x - rotated_center_x
    y_shifted = y - rotated_center_y
    x_original = x_shifted * np.cos(angle_rad) + y_shifted * np.sin(angle_rad) + original_center_x
    y_original = -x_shifted * np.sin(angle_rad) + y_shifted * np.cos(angle_rad) + original_center_y
    return x_original, y_original

def rotate_to_sigmoid_space(x, y, angle, original_shape, rotated_shape):
    """
    Rotate coordinates from the original image space into the rotated space used for sigmoid fitting.
    """
    angle_rad = np.deg2rad(angle)
    original_center_x, original_center_y = original_shape[1] / 2, original_shape[0] / 2
    rotated_center_x, rotated_center_y = rotated_shape[1] / 2, rotated_shape[0] / 2
    x_shifted = x - original_center_x
    y_shifted = y - original_center_y
    x_rotated = x_shifted * np.cos(angle_rad) + y_shifted * np.sin(angle_rad) + rotated_center_x
    y_rotated = -x_shifted * np.sin(angle_rad) + y_shifted * np.cos(angle_rad) + rotated_center_y
    return x_rotated, y_rotated

def double_sigmoid(x, A, B, c, s, f):
    """
    Double sigmoid function used for fitting the trail profile.
    """
    x = np.clip(x, -500, 500)
    return A * (1 / (1 + np.exp(c * (s - x))) + 1 / (1 + np.exp(c * (x - f)))) + B

# Precompute a temporary Moffat model (used for amplitude normalization)
temp_moffat = Moffat2D(amplitude=1, x_0=0, y_0=0, gamma=gamma, alpha=alpha)

# ===============================
# EXPERIMENT FUNCTION (Parallelizable)
# ===============================
def run_experiment(count):
    """
    Run one independent experiment:
      - Simulate a background image with an asteroid trail.
      - Rotate the image by a random angle.
      - Extract the trail profile and fit a double sigmoid.
      - Project the true trail endpoints into the rotated space.
      
    Returns:
        tuple: (start_delta_rotated, end_delta_rotated)
    """
    # --------------------------
    # Create Background Image with Stars
    # --------------------------
    simulated_image = np.full((IMAGE_SIZE_FULL, IMAGE_SIZE_FULL), mu)

    n_sources = int(IMAGE_SIZE_FULL / 3)
    r = np.random.uniform(0, 1, n_sources)
    fluxes = FLUX_MIN * (1 - r + r * (FLUX_MAX / FLUX_MIN) ** (1 - ALPHA_STAR)) ** (1 / (1 - ALPHA_STAR))
    x_coords = np.random.uniform(-0.5, IMAGE_SIZE_FULL - 0.5, n_sources)
    y_coords = np.random.uniform(-0.5, IMAGE_SIZE_FULL - 0.5, n_sources)
    
    data_psf = np.zeros((IMAGE_SIZE_FULL, IMAGE_SIZE_FULL))
    for i in range(n_sources):
        model = Moffat2D(amplitude=fluxes[i], x_0=x_coords[i], y_0=y_coords[i], gamma=gamma, alpha=alpha)
        model_image = model(grid_x, grid_y)
        data_psf += np.maximum(model_image - mu / 4, 0)
    
    def transformation_function(value, a=a):
        return -10**a * np.exp(-10**(-a) * value) + 10**a
    
    data_psf = transformation_function(data_psf)
    
    combined_image = simulated_image + data_psf


    
    # Add asteroid trail
    smoothing = 0.6
    trail_length = np.random.randint(int(IMAGE_SIZE_FULL/4), int(IMAGE_SIZE_FULL/3.5))
    spacing = 2.2
    base_flux = 1400
    flux_range = 0.1 * base_flux
    period = 0.01 * trail_length
    angle = np.random.uniform(0, 180)
    angle_rad = np.radians(angle)
    direction_vector = np.array([np.cos(angle_rad), np.sin(angle_rad)])
    center_of_image = np.array([IMAGE_SIZE_FULL/2, IMAGE_SIZE_FULL/2])
    start_position = center_of_image - (trail_length * spacing * direction_vector / 2)
    true_start = start_position
    true_end = start_position + (trail_length - 1) * spacing * direction_vector
    
    data_trail = np.zeros((IMAGE_SIZE_FULL, IMAGE_SIZE_FULL))
    for i in range(trail_length):
        current_position = start_position + i * spacing * direction_vector
        x_coord, y_coord = current_position
        fluctuating_flux = base_flux + flux_range * np.sin(2 * np.pi * i / period)
        adjusted_amplitude = fluctuating_flux / temp_moffat(0, 0)
        moffat_model = Moffat2D(amplitude=adjusted_amplitude, x_0=x_coord, y_0=y_coord,
                                gamma=gamma, alpha=alpha)
        data_trail += np.maximum(moffat_model(grid_x, grid_y) - mu/2, 0)
    
    merged_data = data_trail + combined_image
    merged_data = np.random.normal(loc=merged_data, scale=(1 / smoothing) * factor * np.sqrt(merged_data))
    merged_data = gaussian_filter(merged_data, sigma=smoothing)
    
    # Rotate the image by the trail angle
    rotated_image = rotate(merged_data, angle, resize=True, order=3)
    
    # Find the row with the brightest trail signal and extract a smoothed profile
    row_sums = np.sum(rotated_image, axis=1)
    brightest_row = np.argmax(row_sums)
    rows_to_sum = np.arange(max(0, brightest_row-2), min(rotated_image.shape[0], brightest_row+2))
    smoothed_row_sums = np.sum(rotated_image[rows_to_sum, :], axis=0) / 4
    trail_profile = smoothed_row_sums
    
    # Fit a double sigmoid to the trail profile
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
        initial_guess = [np.max(trail_profile_limited),
                         np.median(trail_profile_limited),
                         0.01,
                         start_fit,
                         end_fit]
    else:
        raise ValueError("No signal detected above the background threshold.")
    
    try:
        popt, _ = curve_fit(double_sigmoid, x_data, trail_profile_limited,
                              p0=initial_guess, maxfev=5000)
        start_pixel = popt[3]
        end_pixel = popt[4]
    except RuntimeError:
        start_pixel = IMAGE_SIZE_FULL * 2
        end_pixel = IMAGE_SIZE_FULL * 2
    
    # Project true trail endpoints into rotated (sigmoid) space
    rotated_true_start_x, _ = rotate_to_sigmoid_space(true_start[0], true_start[1],
                                                      angle, merged_data.shape, rotated_image.shape)
    rotated_true_end_x, _ = rotate_to_sigmoid_space(true_end[0], true_end[1],
                                                    angle, merged_data.shape, rotated_image.shape)
    # The fit was performed along the horizontal axis; compare x-coordinates
    start_delta_rotated = rotated_true_start_x - start_pixel
    end_delta_rotated = rotated_true_end_x - end_pixel
    
    return start_delta_rotated, end_delta_rotated

# ===============================
# MAIN EXECUTION: Parallel Experiments and Plotting
# ===============================
def main():
    overall_start_time = time.time()
    
    # Run experiments in parallel
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(executor.map(run_experiment, range(EXPERIMENTS)))
    
    # Unpack results
    stdv_s, stdv_e = zip(*results)
    stdv_s = list(stdv_s)
    stdv_e = list(stdv_e)
    
    # Save results to .npy files
    np.save("stdv_s_clean.npy", stdv_s)
    np.save("stdv_e_clean.npy", stdv_e)
    
    overall_elapsed = time.time() - overall_start_time
    minutes, seconds = divmod(overall_elapsed, 60)
    print(f"Total elapsed time: {int(minutes)} min {int(seconds)} sec")
    
    # Plot histograms of the differences (only keep differences with absolute value < 5)
    filtered_stdv_s = [x for x in stdv_s if abs(x) < 5]
    filtered_stdv_e = [x for x in stdv_e if abs(x) < 5]
    
    plt.figure(figsize=(12, 6))
    
    # Histogram: Start differences
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
    
    # Histogram: End differences
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
    plt.savefig(f"histogram_stdv_distributions_{IMAGE_SIZE_FULL}x{IMAGE_SIZE_FULL}_{EXPERIMENTS}.png",
                dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    main()
