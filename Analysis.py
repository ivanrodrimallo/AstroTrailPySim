#!/usr/bin/env python3
"""
Script: star_fit_analysis.py
Description:
    This script loads a FITS image, analyzes its background statistics over the entire image,
    detects stars using DAOStarFinder, fits a transformed Moffat model (via curve_fit)
    to the stars, saves the fitting results to a configuration file, and plots the results.

Author: Ivan
"""

# ===============================
# IMPORTS
# ===============================
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import curve_fit, OptimizeWarning
from astropy.io import fits
from photutils.detection import DAOStarFinder
from matplotlib.colors import LogNorm
import warnings

# Suppress overflow warnings in NumPy and OptimizeWarning from curve_fit
np.seterr(over='ignore', invalid='ignore')
warnings.filterwarnings("ignore", category=OptimizeWarning)

# ===============================
# CONFIGURATION & PARAMETERS
# ===============================
# File paths
FITS_FILE_PATH = 'FITS images original/J5523-1979XB.fits'
CONFIG_FILE_PATH = "Config/image_config.txt"

# Background analysis parameters
BACKGROUND_THRESHOLD = 1100  # Use only pixels with values <= 1100

# DAOStarFinder detection parameters
DAO_FWHM = 3.0
DAO_THRESHOLD = 3000
DAO_ROUNDLO = -1.0
DAO_ROUNDHI = 1.0
DAO_SHARPLO = 0.2
DAO_SHARPHI = 1.0

# Star fitting parameters
STAR_DETECTION_RADIUS = 5     # Half-size of subarray around detected star
STAR_FIT_MASK_RADIUS = 3      # Radius (in pixels) for circular mask during fitting
MIN_MASKED_POINTS = 10        # Minimum points needed for a reliable fit
MAX_STARS = 10000             # Maximum number of stars to process
CHI_SQUARED_THRESHOLD = 0.05  # Reduced chi-squared threshold for accepting a fit

# ===============================
# FUNCTION DEFINITIONS
# ===============================
def transformation_function(value, a):
    """
    Apply a non-linear transformation to the Moffat function output.
    
    Parameters:
        value (float or np.ndarray): Input value(s) to transform.
        a (float): Transformation parameter.
        
    Returns:
        Transformed value(s).
    """
    return -10**a * np.exp(-10**(-a) * value) + 10**a

def load_fits_image(file_path):
    """
    Load FITS image data from a file.
    
    Parameters:
        file_path (str): Path to the FITS file.
        
    Returns:
        np.ndarray: Image data.
    """
    with fits.open(file_path) as hdul:
        return hdul[0].data

def transformed_moffat2d(coordinates, amplitude, x0, y0, gamma, alpha, a):
    """
    Evaluate the transformed Moffat function for given coordinates.
    
    Parameters:
        coordinates (tuple): A tuple of (x, y) arrays.
        amplitude (float): Peak amplitude.
        x0 (float): X-coordinate of the center.
        y0 (float): Y-coordinate of the center.
        gamma (float): Width parameter.
        alpha (float): Moffat shape parameter.
        a (float): Transformation parameter.
        
    Returns:
        np.ndarray: Transformed function values (flattened).
    """
    x, y = coordinates
    r_squared = ((x - x0)**2 + (y - y0)**2) / gamma**2
    moffat = amplitude * (1 + r_squared)**(-alpha)
    return transformation_function(moffat, a)

# ===============================
# MAIN PROCESSING STEPS
# ===============================
def main():
    # 1. Load the FITS image data
    image_data = load_fits_image(FITS_FILE_PATH)
    n_rows, n_cols = image_data.shape

    # 2. Analyze background statistics over the entire image
    row_means = np.array([np.mean(row[row <= BACKGROUND_THRESHOLD]) for row in image_data])
    col_means = np.array([np.mean(image_data[:, col][image_data[:, col] <= BACKGROUND_THRESHOLD])
                          for col in range(n_cols)])
    
    # Use all image pixels that are below the threshold for background analysis
    filtered_values = image_data[image_data <= BACKGROUND_THRESHOLD]
    
    # Fit a normal distribution to the background pixel values
    mu, std = norm.fit(filtered_values)
    factor = std / np.sqrt(mu)
    print(f"Background Fit: Mean = {mu:.2f}, Std Dev = {std:.2f}, Factor = {factor:.2f}\n")

    # 3. Star detection using DAOStarFinder
    daofind = DAOStarFinder(fwhm=DAO_FWHM, threshold=DAO_THRESHOLD,
                            roundlo=DAO_ROUNDLO, roundhi=DAO_ROUNDHI,
                            sharplo=DAO_SHARPLO, sharphi=DAO_SHARPHI)
    sources = daofind(image_data)
    gammas, alphas, a_values = [], [], []

    if sources is not None and len(sources) > 0:
        num_stars = min(MAX_STARS, len(sources))
        random_indices = np.random.choice(len(sources), num_stars, replace=False)
        random_stars = sources[random_indices]

        # Process each detected star for model fitting
        for star in random_stars:
            try:
                x, y = star['xcentroid'], star['ycentroid']
                # Define a subarray around the star
                x_min, x_max = int(x) - STAR_DETECTION_RADIUS, int(x) + STAR_DETECTION_RADIUS + 1
                y_min, y_max = int(y) - STAR_DETECTION_RADIUS, int(y) + STAR_DETECTION_RADIUS + 1
                subarray = image_data[y_min:y_max, x_min:x_max]
                y_grid, x_grid = np.mgrid[y_min:y_max, x_min:x_max]
                
                # Create a circular mask centered on the star
                mask = (x_grid - x)**2 + (y_grid - y)**2 <= STAR_FIT_MASK_RADIUS**2
                subarray_masked = subarray[mask]
                x_masked = x_grid[mask]
                y_masked = y_grid[mask]
                
                # Ensure sufficient data points for fitting
                if subarray_masked.size < MIN_MASKED_POINTS:
                    continue

                # Initial parameter estimates: amplitude, x0, y0, gamma, alpha, a
                p0 = [subarray_masked.max(), x, y, 1, 2, 4]
                # Fit the model using curve_fit (flattening arrays as needed)
                popt, _ = curve_fit(
                    transformed_moffat2d,
                    (x_masked.flatten(), y_masked.flatten()),
                    subarray_masked.flatten(),
                    p0=p0
                )
                # Evaluate the fitted model and compute residuals
                model_vals = transformed_moffat2d((x_masked.flatten(), y_masked.flatten()), *popt)
                residual = subarray_masked.flatten() - model_vals
                chi_squared = np.sum((residual)**2 / (np.std(subarray_masked.flatten())**2))
                dof = subarray_masked.size - len(popt)
                reduced_chi_squared = chi_squared / dof

                # Accept the fit if reduced chi-squared is below threshold
                if reduced_chi_squared < CHI_SQUARED_THRESHOLD:
                    # popt returns: amplitude, x0, y0, gamma, alpha, a
                    gammas.append(popt[3])
                    alphas.append(popt[4])
                    a_values.append(popt[5])
            except Exception:
                # Skip this star if fitting fails
                continue

    # 4. Display fitted parameter statistics if available
    if gammas and alphas and a_values:
        print(f"Gamma: Mean = {np.mean(gammas):.2f}, Std Dev = {np.std(gammas):.2f}")
        print(f"Alpha: Mean = {np.mean(alphas):.2f}, Std Dev = {np.std(alphas):.2f}")
        print(f"a: Mean = {np.mean(a_values):.2f}, Std Dev = {np.std(a_values):.2f}\n")

    # 5. Save the fitting results to the configuration file
    if gammas and alphas and a_values and factor and mu:
        gamma_mean = np.mean(gammas)
        alpha_mean = np.mean(alphas)
        a_mean = np.mean(a_values)
        
        with open(CONFIG_FILE_PATH, "w") as f:
            f.write(f"Gamma = {gamma_mean:.4f}\n")
            f.write(f"Alpha = {alpha_mean:.4f}\n")
            f.write(f"a = {a_mean:.4f}\n")
            f.write(f"Background factor = {factor:.4f}\n")
            f.write(f"Background mu = {mu:.4f}\n")
        
        print("Fitting results saved to:", CONFIG_FILE_PATH)

    # 6. Plot the original FITS image
    plt.figure(figsize=(10,7))
    plt.imshow(image_data, cmap="gray", origin="lower", norm=LogNorm(vmin=600, vmax=3000))
    plt.colorbar(label="Pixel Value")
    plt.title("Original FITS Image")
    plt.show()

    # 7. Plot background distribution and row/column means
    plt.figure(figsize=(12, 6))
    
    # Plot histogram of background values
    plt.subplot(1, 2, 1)
    plt.hist(filtered_values, bins=100, alpha=0.7, color='blue', label="Filtered Values")
    plt.axvline(mu, color='r', linestyle='--', label=f'Mean = {mu:.2f}')
    plt.title("Background Value Distribution")
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")
    plt.legend()
    
    # Plot row and column means
    plt.subplot(1, 2, 2)
    plt.plot(row_means, label="Row Means")
    plt.plot(col_means, label="Column Means")
    plt.axhline(np.mean(row_means), color='red', linestyle='--', label=f'Row Mean = {np.mean(row_means):.2f}')
    plt.axhline(np.mean(col_means), color='blue', linestyle='--', label=f'Col Mean = {np.mean(col_means):.2f}')
    plt.title("Row and Column Means")
    plt.xlabel("Index")
    plt.ylabel("Mean Value")
    plt.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
