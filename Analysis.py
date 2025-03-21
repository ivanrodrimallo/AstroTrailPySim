# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 12:04:11 2024

@author: gigma
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from astropy.io import fits
from astropy.modeling import fitting, Fittable2DModel, Parameter
from photutils.detection import DAOStarFinder
from matplotlib.colors import LogNorm


# ---------------------------
# Necessary Functions
# ---------------------------
def transformation_function(value, a):
    """Apply the non-linear transformation."""
    return -10**a * np.exp(-10**(-a) * value) + 10**a

def load_fits_image(file_path):
    """Load the FITS image data from a file."""
    with fits.open(file_path) as hdul:
        return hdul[0].data

# ---------------------------
# File Paths and Parameters
# ---------------------------
fits_file_path = 'FITS images original/J5523-1979XB.fits'

#fits_file_path = 'Simulated Images/Merged_Trail_REPORT.fits'

config_file_path = "Config/image_config.txt"

# ---------------------------
# Load Image and Analyze Background
# ---------------------------
image_data = load_fits_image(fits_file_path)
n_rows, n_cols = image_data.shape

# Compute row and column means for pixels with values <= 3000
row_means = np.array([np.mean(row[row <= 1100]) for row in image_data])
col_means = np.array([np.mean(image_data[:, col][image_data[:, col] <= 1100])
                      for col in range(n_cols)])

# Select central 10 rows and columns for background distribution analysis
row_center, col_center = n_rows // 2, n_cols // 2
middle_rows = image_data[row_center-5:row_center+5, :]
middle_cols = image_data[:, col_center-5:col_center+5]
filtered_values = np.concatenate([middle_rows[middle_rows <= 1100],
                                  middle_cols[middle_cols <= 1100]])

# Fit a normal distribution to the filtered background values
mu, std = norm.fit(filtered_values)
factor = std / np.sqrt(mu)
print(f"Best Normal Fit - Mean: {mu:.2f}, Std Dev: {std:.2f}, fac = {factor:.2f}\n")

# ---------------------------
# Star Detection and Fitting with a Transformed Moffat Model
# ---------------------------
# Detect stars using DAOStarFinder
daofind = DAOStarFinder(fwhm=3.0, threshold=3000, roundlo=-1.0, roundhi=1.0,
                        sharplo=0.2, sharphi=1.0)
sources = daofind(image_data)

# Lists to hold fitted parameter values
gammas, alphas, a_values = [], [], []

if sources is not None and len(sources) > 0:
    num_stars = min(10000, len(sources))
    random_indices = np.random.choice(len(sources), num_stars, replace=False)
    random_stars = sources[random_indices]
    fitter = fitting.LevMarLSQFitter()

    class TransformedMoffat2D(Fittable2DModel):
        amplitude = Parameter(default=1)
        x_0 = Parameter(default=0)
        y_0 = Parameter(default=0)
        gamma = Parameter(default=1, bounds=(1e-3, None))
        alpha = Parameter(default=2, bounds=(0.1, 10))
        a = Parameter(default=4, bounds=(1, 10))

        @staticmethod
        def evaluate(x, y, amplitude, x_0, y_0, gamma, alpha, a):
            r_squared = ((x - x_0)**2 + (y - y_0)**2) / gamma**2
            moffat = amplitude * (1 + r_squared)**(-alpha)
            return transformation_function(moffat, a)

    for star in random_stars:
        try:
            x, y = star['xcentroid'], star['ycentroid']
            # Define a small subarray around the star
            x_min, x_max = int(x) - 5, int(x) + 6
            y_min, y_max = int(y) - 5, int(y) + 6
            subarray = image_data[y_min:y_max, x_min:x_max]
            y_grid, x_grid = np.mgrid[y_min:y_max, x_min:x_max]
            
            # Create a circular mask (radius = 3 pixels)
            mask = (x_grid - x)**2 + (y_grid - y)**2 <= 9
            subarray_masked = subarray[mask]
            x_masked = x_grid[mask]
            y_masked = y_grid[mask]
            
            # Skip if not enough data points for a reliable fit
            if subarray_masked.size < 10:
                continue

            initial_model = TransformedMoffat2D(
                amplitude=subarray_masked.max(),
                x_0=x, y_0=y, gamma=1, alpha=2, a=4
            )
            fit_model = fitter(initial_model, x_masked, y_masked, subarray_masked)
            model_vals = fit_model(x_masked, y_masked)
            residual = subarray_masked - model_vals
            chi_squared = np.sum((residual)**2 / (np.std(subarray_masked)**2))
            dof = subarray_masked.size - len(fit_model.parameters)
            reduced_chi_squared = chi_squared / dof

            if reduced_chi_squared < 0.05:
                gammas.append(fit_model.gamma.value)
                alphas.append(fit_model.alpha.value)
                a_values.append(fit_model.a.value)
        except Exception as e:
            # Optionally, print(e) for debugging
            continue

if gammas and alphas and a_values:
    print(f"\nGamma: Mean = {np.mean(gammas):.2f}, Std Dev = {np.std(gammas):.2f}")
    print(f"Alpha: Mean = {np.mean(alphas):.2f}, Std Dev = {np.std(alphas):.2f}")
    print(f"a: Mean = {np.mean(a_values):.2f}, Std Dev = {np.std(a_values):.2f}")

# ---------------------------
# Save Fitting Results to Configuration File
# ---------------------------
if gammas and alphas and a_values and factor and mu:
    gamma_mean = np.mean(gammas)
    alpha_mean = np.mean(alphas)
    a_mean = np.mean(a_values)
    
    with open(config_file_path, "w") as f:
        f.write(f"Gamma = {gamma_mean:.4f}\n")
        f.write(f"Alpha = {alpha_mean:.4f}\n")
        f.write(f"a = {a_mean:.4f}\n")
        f.write(f"Background factor = {factor:.4f}\n")
        f.write(f"Background mu = {mu:.4f}\n")
    
    print("\nFitting results saved to:", config_file_path)

# ---------------------------
# Plot the Original FITS Image and Background Stats
# ---------------------------
plt.figure(figsize=(10,7))
plt.imshow(image_data, cmap="gray", origin="lower", norm=LogNorm(vmin=600, vmax=3000))
#plt.title(f"Original FITS File Image: {fits_file_path[27:-5]}")
plt.colorbar(label="Pixel Value")
plt.show()

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.hist(filtered_values, bins=100, alpha=0.7, color='blue', label="Filtered Values")
plt.axvline(mu, color='r', linestyle='--', label=f'Mean = {mu:.2f}')
plt.title("Background Value Distribution")
plt.xlabel("Pixel Value")
plt.ylabel("Frequency")
plt.legend()

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
