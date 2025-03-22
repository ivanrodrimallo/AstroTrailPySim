#!/usr/bin/env python3
"""
Script: analyze_trail_image.py
Description:
    This script loads a simulated FITS image of an asteroid trail,
    refines the trail angle, rotates and clips the image,
    fits a double sigmoid to the trail profile to detect endpoints,
    computes the deltas between detected and true endpoints in the rotated space,
    and visualizes the results.
    
Author: Ivan
"""

# ===============================
# IMPORTS
# ===============================
import numpy as np
import pandas as pd
from astropy.io import fits
from skimage.transform import rotate
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from matplotlib.colors import LogNorm
import warnings

# Suppress overflow warnings
np.seterr(over='ignore')
warnings.filterwarnings("ignore", message="overflow encountered in exp")

# ===============================
# CONFIGURATION & PARAMETERS
# ===============================
CONFIG_FILE_PATH = "Config/comparison_config.txt"
FITS_FILENAME = 'Simulated Images/Merged_Trail.fits'
Trail_Output_Path = 'Plots/Trail_Output.png'
CLIP_SIZE = 2000  # Clipped image size (square)
BACKGROUND_THRESHOLD = 1200  # Threshold for background in trail profile fitting

# ===============================
# FUNCTION DEFINITIONS
# ===============================
def load_config(file_path):
    """
    Load configuration parameters from a file.
    Each line should be in the format 'key = value'.
    
    Parameters:
        file_path (str): Path to configuration file.
    
    Returns:
        dict: Dictionary with configuration keys and float values.
    """
    config = {}
    with open(file_path, "r") as file:
        for line in file:
            if " = " in line:
                key, value = line.strip().split(" = ")
                config[key] = float(value)
    return config

def refine_angle(data, init_angle, search_range=5, step=0.5):
    """
    Refine the rotation angle by searching within a range around init_angle.
    The best angle maximizes the sum of pixel intensities along rows.
    
    Parameters:
        data (np.ndarray): Input image data.
        init_angle (float): Initial angle guess.
        search_range (float): Range of angles to search (degrees).
        step (float): Step size for the search.
        
    Returns:
        float: Refined rotation angle.
    """
    best_angle = init_angle
    best_score = -np.inf
    angles = np.arange(init_angle - search_range, init_angle + search_range + step, step)
    for angle in angles:
        rotated_candidate = rotate(data, angle, resize=True, order=3)
        score = np.max(np.sum(rotated_candidate, axis=1))
        if score > best_score:
            best_score = score
            best_angle = angle
    return best_angle

def double_sigmoid(x, A, B, c, s, f):
    """
    Double sigmoid function used to fit the trail profile.
    
    Parameters:
        x (np.ndarray): Independent variable.
        A (float): Amplitude scaling factor.
        B (float): Baseline offset.
        c (float): Slope parameter.
        s (float): Start transition position.
        f (float): End transition position.
        
    Returns:
        np.ndarray: Computed double sigmoid values.
    """
    return A * (1 / (1 + np.exp(c * (s - x))) + 1 / (1 + np.exp(c * (x - f)))) + B

def rotate_to_sigmoid_space(x, y, angle, original_shape, rotated_shape):
    """
    Rotate coordinates from the original image space into the rotated (clipped) space.
    
    Parameters:
        x (float): Original x-coordinate.
        y (float): Original y-coordinate.
        angle (float): Rotation angle in degrees.
        original_shape (tuple): Shape of the original image.
        rotated_shape (tuple): Shape of the clipped, rotated image.
        
    Returns:
        tuple: (x_rotated, y_rotated) coordinates in the rotated space.
    """
    angle_rad = np.deg2rad(angle)
    original_center_x, original_center_y = original_shape[1] / 2, original_shape[0] / 2
    rotated_center_x, rotated_center_y = rotated_shape[1] / 2, rotated_shape[0] / 2
    x_shifted = x - original_center_x
    y_shifted = y - original_center_y
    x_rotated = x_shifted * np.cos(angle_rad) + y_shifted * np.sin(angle_rad) + rotated_center_x
    y_rotated = -x_shifted * np.sin(angle_rad) + y_shifted * np.cos(angle_rad) + rotated_center_y
    return x_rotated, y_rotated

# ===============================
# MAIN PROCESSING STEPS
# ===============================
def main():
    # Step 1: Load configuration parameters and FITS image
    params = load_config(CONFIG_FILE_PATH)
    init_angle = params["Angle"]
    true_start_x = params["Start_X"]
    true_start_y = params["Start_Y"]
    true_end_x   = params["End_X"]
    true_end_y   = params["End_Y"]
    print("Loaded trail parameters from configuration file")
    
    with fits.open(FITS_FILENAME) as hdul:
        data = hdul[0].data.astype(float)
    original_shape = data.shape
    
    # Step 2: Rotate the image using a refined angle
    # Optionally, you can refine the angle using the refine_angle function.
    # refined_angle = refine_angle(data, init_angle, search_range=1, step=0.05)
    refined_angle = init_angle
    print("Initial angle guess (deg):", init_angle)
    print("Refined angle (deg):", refined_angle)
    
    full_rotated_image = rotate(data, refined_angle, resize=True, order=3)
    
    # Clip the rotated image to a CLIP_SIZE x CLIP_SIZE region around its center.
    rotated_shape = full_rotated_image.shape
    center_y, center_x = rotated_shape[0] // 2, rotated_shape[1] // 2
    start_y = max(0, center_y - CLIP_SIZE // 2)
    end_y = start_y + CLIP_SIZE
    start_x = max(0, center_x - CLIP_SIZE // 2)
    end_x = start_x + CLIP_SIZE
    rotated_image = full_rotated_image[start_y:end_y, start_x:end_x]
    
    # Step 3: Identify the trail row and fit a double sigmoid to its profile
    row_sums = np.sum(rotated_image, axis=1)
    trail_row = np.argmax(row_sums)
    trail_profile = rotated_image[trail_row, :]
    
    # Isolate the signal region above the background threshold.
    signal_indices = np.where(trail_profile > BACKGROUND_THRESHOLD)[0]
    if signal_indices.size > 0:
        start_signal = signal_indices[0]
        end_signal = signal_indices[-1]
        padding = int(0.05 * (end_signal - start_signal))
        start_fit = max(0, start_signal - padding)
        end_fit = min(len(trail_profile), end_signal + padding)
        x_data = np.arange(start_fit, end_fit)
        trail_profile_limited = trail_profile[start_fit:end_fit]
        # Initial guess: [peak, baseline, slope, start, end]
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
    except RuntimeError as e:
        print(f"Curve fitting failed: {e}")
        popt = initial_guess
        start_pixel, end_pixel = 0.1 * CLIP_SIZE, 1.2 * CLIP_SIZE
    
    if start_pixel > end_pixel:
        start_pixel, end_pixel = end_pixel, start_pixel
    
    # Step 4: Compute deltas between detected and true endpoints in rotated space.
    rotated_true_start_x, _ = rotate_to_sigmoid_space(true_start_x, true_start_y,
                                                       refined_angle, original_shape,
                                                       rotated_image.shape)
    rotated_true_end_x, _ = rotate_to_sigmoid_space(true_end_x, true_end_y,
                                                     refined_angle, original_shape,
                                                     rotated_image.shape)
    start_delta_rotated = rotated_true_start_x - start_pixel
    end_delta_rotated = rotated_true_end_x - end_pixel
    print(f"Start Delta in Rotated Space: {start_delta_rotated:.2f} pixels")
    print(f"End Delta in Rotated Space: {end_delta_rotated:.2f} pixels")
    
    # Step 5: Visualization
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    
    # Left: Clipped rotated image with detected (red) and true (blue) endpoints.
    ax[0].imshow(rotated_image, origin='lower', cmap='gray', norm=LogNorm(vmin=600, vmax=3000))
    ax[0].scatter([start_pixel, end_pixel],
                  [trail_row, trail_row],
                  color='red', label='Detected')
    ax[0].scatter([rotated_true_start_x, rotated_true_end_x],
                  [trail_row, trail_row],
                  color='blue', label='True')
    ax[0].set_title('Clipped Rotated Image with Endpoints')
    ax[0].legend()
    
    # Right: Trail profile and double sigmoid fit.
    ax[1].plot(x_data, trail_profile_limited, label='Trail Profile')
    ax[1].plot(x_data, double_sigmoid(x_data, *popt), linestyle='--', label='Double Sigmoid Fit')
    ax[1].set_title('Double Sigmoid Fit')
    ax[1].legend()
    
    plt.tight_layout()
    plt.savefig(Trail_Output_Path)
    plt.show()

if __name__ == '__main__':
    main()
