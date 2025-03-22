#!/usr/bin/env python3
"""
Script: simulate_trail_image.py
Description:
    This script simulates a FITS image that includes:
      - A uniform background.
      - Simulated stars generated using a Moffat PSF.
      - An asteroid trail with sinusoidal flux variation.
      - Gaussian noise is added and the image is smoothed.
    The final merged image is saved as a FITS file along with supplementary
    configuration files for trail coordinates and star positions.
    
Author: Ivan
"""

# ===============================
# IMPORTS
# ===============================
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.modeling.models import Moffat2D
from matplotlib.colors import LogNorm
from scipy.ndimage import gaussian_filter

# ===============================
# CONFIGURATION & PARAMETERS
# ===============================
CONFIG_FILE_PATH = "Config/image_config.txt"
COMPARISON_CONFIG_PATH = "Config/comparison_config.txt"
STAR_COORDS_CSV = "CSV/Real_Star_Coordinates.csv"
FINAL_FITS_FILENAME = "Simulated Images/Merged_Trail.fits"

IMAGE_SIZE_FULL = 2500

# Star parameters
FLUX_MIN = 900
FLUX_MAX = 1e5
ALPHA_STAR = 2.35

# Trail parameters
TRAIL_BASE_FLUX = 1400
TRAIL_FLUX_RANGE_FACTOR = 0.1
TRAIL_PERIOD_FACTOR = 0.01
TRAIL_SMOOTHING = 0.6
TRAIL_SPACING = 2.2

# ===============================
# FUNCTION DEFINITIONS
# ===============================
def load_config(file_path):
    """
    Load configuration parameters from a file.
    
    The config file should have lines in the format "key = value".
    
    Parameters:
        file_path (str): Path to the configuration file.
        
    Returns:
        dict: Dictionary with config keys and float values.
    """
    config = {}
    with open(file_path, "r") as f:
        for line in f:
            if " = " in line:
                key, value = line.strip().split(" = ")
                config[key] = float(value)
    return config

def transformation_function(value, a):
    """
    Apply a non-linear transformation to the input value.
    
    Parameters:
        value (float or np.ndarray): Input value(s).
        a (float): Transformation parameter.
        
    Returns:
        np.ndarray: Transformed value(s).
    """
    return -10**a * np.exp(-10**(-a) * value) + 10**a

# ===============================
# MAIN PROCESSING STEPS
# ===============================
def main():
    start_time = time.time()
    
    # 1. Load configuration parameters
    params = load_config(CONFIG_FILE_PATH)
    gamma = params.get("Gamma", 1.0)
    alpha = params.get("Alpha", 2.0)
    a = params.get("a", 3.0)
    factor = params.get("Background factor", 4.0)
    mu = params.get("Background mu", 5.0)
    print("Loaded Analysis config")
    
    # 2. Precompute coordinate grid for simulation
    y_grid, x_grid = np.mgrid[0:IMAGE_SIZE_FULL, 0:IMAGE_SIZE_FULL]
    
    # 3. Create Uniform Background
    simulated_image = np.full((IMAGE_SIZE_FULL, IMAGE_SIZE_FULL), mu)
    
    # 4. Simulate Stars (PSF)
    n_sources = int(IMAGE_SIZE_FULL / 3)
    r = np.random.uniform(0, 1, n_sources)
    # Compute fluxes using a power-law distribution
    fluxes = FLUX_MIN * (1 - r + r * (FLUX_MAX / FLUX_MIN) ** (1 - ALPHA_STAR)) ** (1 / (1 - ALPHA_STAR))
    x_coords = np.random.uniform(-0.5, IMAGE_SIZE_FULL - 0.5, n_sources)
    y_coords = np.random.uniform(-0.5, IMAGE_SIZE_FULL - 0.5, n_sources)
    
    data_psf = np.zeros((IMAGE_SIZE_FULL, IMAGE_SIZE_FULL))
    for i in range(n_sources):
        # Create a Moffat2D model for each star
        model = Moffat2D(amplitude=fluxes[i], x_0=x_coords[i], y_0=y_coords[i],
                         gamma=gamma, alpha=alpha)
        model_image = model(x_grid, y_grid)
        # Subtract a fraction of the background level to simulate PSF features
        data_psf += np.maximum(model_image - mu / 4, 0)
    
    # Apply the transformation function to the PSF data
    data_psf = transformation_function(data_psf, a)
    combined_image = simulated_image + data_psf
    
    # 5. Add Asteroid Trail
    # Determine trail length and orientation
    trail_length = np.random.randint(int(IMAGE_SIZE_FULL / 4), int(IMAGE_SIZE_FULL / 3.5))
    angle = np.random.uniform(0, 180)
    angle_rad = np.radians(angle)
    direction_vector = np.array([np.cos(angle_rad), np.sin(angle_rad)])
    center = np.array([IMAGE_SIZE_FULL / 2, IMAGE_SIZE_FULL / 2])
    start_position = center - (trail_length * TRAIL_SPACING * direction_vector) / 2
    end_position = start_position + (trail_length - 1) * TRAIL_SPACING * direction_vector
    
    data_trail = np.zeros((IMAGE_SIZE_FULL, IMAGE_SIZE_FULL))
    # Compute peak value of the Moffat model for normalization
    peak_value = Moffat2D(amplitude=1, x_0=0, y_0=0, gamma=gamma, alpha=alpha)(0, 0)
    
    for i in range(trail_length):
        current_position = start_position + i * TRAIL_SPACING * direction_vector
        x_coord, y_coord = current_position
        # Sinusoidal flux variation along the trail
        fluctuating_flux = (TRAIL_BASE_FLUX +
                            TRAIL_FLUX_RANGE_FACTOR * TRAIL_BASE_FLUX *
                            np.sin(2 * np.pi * i / (TRAIL_PERIOD_FACTOR * trail_length)))
        adjusted_amplitude = fluctuating_flux / peak_value
        model = Moffat2D(amplitude=adjusted_amplitude, x_0=x_coord, y_0=y_coord,
                         gamma=gamma, alpha=alpha)
        data_trail += np.maximum(model(x_grid, y_grid) - mu / 2, 0)
    
    # 6. Merge Components and Add Noise
    merged_data = combined_image + data_trail
    # Add Gaussian noise and smooth the merged image
    merged_data = np.random.normal(loc=merged_data,
                                   scale=(1 / TRAIL_SMOOTHING) * factor * np.sqrt(merged_data))
    merged_data = gaussian_filter(merged_data, sigma=TRAIL_SMOOTHING)
    
    # 7. Save Outputs
    # Save merged image as a FITS file
    fits.writeto(FINAL_FITS_FILENAME, merged_data, overwrite=True)
    print("Saved Merged Trail Image.")
    
    # Save comparison configuration (trail angle and coordinates)
    with open(COMPARISON_CONFIG_PATH, "w") as f:
        f.write(f"Angle = {angle}\n")
        f.write(f"Start_X = {start_position[0]}\n")
        f.write(f"Start_Y = {start_position[1]}\n")
        f.write(f"End_X = {end_position[0]}\n")
        f.write(f"End_Y = {end_position[1]}\n")
    print("Saved Trail Coordinates.")
    
    # Save star coordinates for reference
    coordinates_df = pd.DataFrame({
        'x_coords': x_coords,
        'y_coords': y_coords,
        'fluxes': fluxes
    })
    coordinates_df.to_csv(STAR_COORDS_CSV, index=False)
    print("Saved Star Coordinates.")
    
    # 8. Print Processing Time
    elapsed = time.time() - start_time
    minutes, seconds = divmod(elapsed, 60)
    print(f"Processing time: {int(minutes)} min {int(seconds)} sec")
    
    # 9. Display the Result
    plt.figure(figsize=(10,7))
    plt.imshow(merged_data, cmap='gray', norm=LogNorm(vmin=600, vmax=3000), origin='lower')
    plt.title("Merged Asteroid Trail with Simulated PSF Image")
    plt.xlabel("X pixel")
    plt.ylabel("Y pixel")
    plt.show()

if __name__ == '__main__':
    main()
