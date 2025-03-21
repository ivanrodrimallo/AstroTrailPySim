# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 12:50:55 2025

@author: gigma
"""

import numpy as np
from astropy.io import fits
from skimage.transform import rotate
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from matplotlib.colors import LogNorm

# ---------------------
# Define the double sigmoid function
# ---------------------
def double_sigmoid(x, A, B, c, s, f):
    return A * (1/(1 + np.exp(c*(s - x))) + 1/(1 + np.exp(c*(x - f)))) + B

# ---------------------
# Function to refine the rotation angle using a grid search.
# For each candidate angle, the image is rotated by -angle,
# clipped to 2000, and the candidate with the highest max row sum is chosen.
# ---------------------
def refine_angle(data, init_angle, search_range=5, step=0.5):
    best_angle = init_angle
    best_score = -np.inf
    angles = np.arange(init_angle - search_range, init_angle + search_range + step, step)
    for angle in angles:
        rotated = rotate(data, -angle, resize=True, order=3)
        rotated = np.clip(rotated, 0, 2000)
        row_sums = np.sum(rotated, axis=1)
        current_score = np.max(row_sums)
        if current_score > best_score:
            best_score = current_score
            best_angle = angle
    return best_angle

# ---------------------
# User-defined initial angle guess (degrees anticlockwise from x-axis)
# ---------------------
init_angle = 125  # initial guess; not exact

# ---------------------
# Load the real FITS image
# ---------------------
fits_filename = '../../FITS images original/J5523-1979XB.fits'
with fits.open(fits_filename) as hdul:
    data = hdul[0].data.astype(float)

# ---------------------
# Refine the angle using grid search
# ---------------------
refined_angle = np.round(refine_angle(data, init_angle, search_range=1, step=0.05),4)
print("Initial angle guess (deg):", init_angle)
print("Refined angle (deg):", refined_angle)

# ---------------------
# Rotate the image by the negative of the refined angle to align the trail horizontally
# ---------------------
rotation_angle = -refined_angle
rotated_image = rotate(data, rotation_angle, resize=True, order=3)
rotated_image = np.clip(rotated_image, 0, 2000)
print("Rotated image by", rotation_angle, "degrees.")

# ---------------------
# Clip the rotated image to a central 700x700 square
# ---------------------
H, W = rotated_image.shape
clip_size = 700
row_start = (H - clip_size) // 2
col_start = (W - clip_size) // 2
clipped_image = rotated_image[row_start:row_start+clip_size, col_start:col_start+clip_size]

# ---------------------
# Identify the row in the clipped image with maximum summed intensity (assumed to contain the trail)
# ---------------------
clipped_row_sums = np.sum(clipped_image, axis=1)
trail_row_clipped = np.argmax(clipped_row_sums)

# Instead of using only that row, average a few rows around it (perpendicular to the trail)
row_offset = 1  # number of rows above and below to include
row_min = max(0, trail_row_clipped - row_offset)
row_max = min(clipped_image.shape[0]-1, trail_row_clipped + row_offset)
averaged_trail_profile = np.mean(clipped_image[row_min:row_max+1, :], axis=0)

# ---------------------
# Define a dynamic threshold for the averaged trail profile and isolate the signal region
# ---------------------
signal_indices = np.where(averaged_trail_profile > 0)[0]
if signal_indices.size > 0:
    start_signal = signal_indices[0]
    end_signal = signal_indices[-1]
    padding = int(0.05 * (end_signal - start_signal))
    start_fit = max(0, start_signal - padding)
    end_fit = min(len(averaged_trail_profile), end_signal + padding)
    
    x_data = np.arange(start_fit, end_fit)
    trail_profile_limited = averaged_trail_profile[start_fit:end_fit]
    
    # Initial guess for double sigmoid parameters: [peak, baseline, slope, start, end]
    initial_guess = [np.max(trail_profile_limited),
                     np.median(trail_profile_limited),
                     0.01,
                     start_fit,
                     end_fit]
else:
    raise ValueError("No signal detected above the background threshold.")

# ---------------------
# Fit the double sigmoid to the averaged trail profile
# ---------------------
try:
    popt, _ = curve_fit(double_sigmoid, x_data, trail_profile_limited,
                         p0=initial_guess, maxfev=5000)
    fit_start = int(popt[3])
    fit_end = int(popt[4])
except RuntimeError as e:
    print("Curve fitting failed:", e)
    popt = initial_guess
    fit_start, fit_end = int(0.1 * len(averaged_trail_profile)), int(0.9 * len(averaged_trail_profile))

# ---------------------
# For back-transformation, compute the corresponding coordinates in the rotated image.
# The clipped image is a subarray of the rotated image, so add offsets.
# ---------------------
rotated_trail_row = trail_row_clipped + row_start
rotated_fit_start = fit_start + col_start
rotated_fit_end = fit_end + col_start

# ---------------------
# Back-transform function (from rotated image to original image)
# ---------------------
def back_transform(x, y, angle, original_shape, rotated_shape, flip_y=False):
    a = np.deg2rad(angle)
    original_center = np.array([original_shape[1] / 2, original_shape[0] / 2])
    rotated_center  = np.array([rotated_shape[1] / 2, rotated_shape[0] / 2])
    x_r = x - rotated_center[0]
    y_r = y - rotated_center[1]
    x_orig = np.cos(a)*x_r + np.sin(a)*y_r
    y_orig = -np.sin(a)*x_r + np.cos(a)*y_r
    x_orig += original_center[0]
    y_orig += original_center[1]
    if flip_y:
        y_orig = original_shape[0] - y_orig
    return x_orig, y_orig

original_shape = data.shape  # original image shape
rotated_shape = rotated_image.shape  # shape of the full rotated image

# Back-transform the endpoints from the rotated image to the original image space.
original_start_x, original_start_y = back_transform(rotated_fit_start, rotated_trail_row,
                                                      rotation_angle,
                                                      original_shape, rotated_shape,
                                                      flip_y=True)
original_end_x, original_end_y = back_transform(rotated_fit_end, rotated_trail_row,
                                                  rotation_angle,
                                                  original_shape, rotated_shape,
                                                  flip_y=True)

print("Detected start point in original space: ({:.2f}, {:.2f})".format(original_start_x, original_start_y))
print("Detected end point in original space: ({:.2f}, {:.2f})".format(original_end_x, original_end_y))

# ---------------------
# Visualization
# ---------------------
# Create the figure with two subplots
fig, ax = plt.subplots(1, 2, figsize=(14, 6))

# Set desired min/max for the log scale
vmin = 600
vmax = 3000

# Plot the clipped image on a log scale
im0 = ax[0].imshow(
    clipped_image,
    origin='lower',
    cmap='gray',
    norm=LogNorm(vmin=vmin, vmax=vmax)  # <-- Log scale
)
ax[0].axhline(trail_row_clipped, color='red', linewidth=2, label='Detected Trail Row')
ax[0].plot([fit_start, fit_end], [trail_row_clipped, trail_row_clipped],
           color='blue', linewidth=2, label='Fitted Trail')
ax[0].set_title("Clipped 700x700 Image with Detected Trail (Log Scale)")
ax[0].legend()

# Add a colorbar that reflects the log scale
fig.colorbar(im0, ax=ax[0], label="Pixel Value (log scale)")

# Plot the averaged trail profile and the double sigmoid fit
ax[1].plot(x_data, trail_profile_limited, label='Averaged Trail Profile')
ax[1].plot(x_data, double_sigmoid(x_data, *popt), '--', label='Double Sigmoid Fit')
ax[1].set_title("Double Sigmoid Fit to Averaged Trail Profile")

# If you also want the second subplot on a log scale (y-axis):
ax[1].set_yscale('log')

ax[1].legend()
plt.tight_layout()
plt.show()