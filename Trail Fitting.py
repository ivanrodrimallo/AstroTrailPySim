import numpy as np
import pandas as pd
from astropy.io import fits
from skimage.transform import rotate
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from matplotlib.colors import LogNorm
np.seterr(over='ignore')
import warnings
warnings.filterwarnings("ignore", message="overflow encountered in exp")

# ---------------------
# User-defined parameters and configuration loading
# ---------------------
config_file_path = "Config/comparison_config.txt"

def load_config(file_path):
    config = {}
    with open(file_path, "r") as file:
        for line in file:
            if " = " in line:
                key, value = line.strip().split(" = ")
                config[key] = float(value)
    return config

# Load the parameters from the configuration file
params = load_config(config_file_path)

# Directly use values from the config file
init_angle = params["Angle"]
true_start_x = params["Start_X"]
true_start_y = params["Start_Y"]
true_end_x   = params["End_X"]
true_end_y   = params["End_Y"]
print("Loaded trail parameters from configuration file")

fits_filename = 'Simulated Images/Merged_Trail_REPORT.fits'

# ---------------------
# Step 1: Load the FITS file
# ---------------------
with fits.open(fits_filename) as hdul:
    data = hdul[0].data.astype(float)
length = data.shape[0]

# ---------------------
# Step 2: Rotate the Image Using a Refined Angle
# ---------------------
def refine_angle(data, init_angle, search_range=5, step=0.5):
    best_angle = init_angle
    best_score = -np.inf
    angles = np.arange(init_angle - search_range, init_angle + search_range + step, step)
    for angle in angles:
        # Rotate using the candidate angle.
        rotated_candidate = rotate(data, angle, resize=True, order=3)
        score = np.max(np.sum(rotated_candidate, axis=1))
        if score > best_score:
            best_score = score
            best_angle = angle
    return best_angle

# Use the config file's angle as an initial guess.
refined_angle = init_angle  # or: refine_angle(data, init_angle, search_range=1, step=0.05)
print("Initial angle guess (deg):", init_angle)
print("Refined angle (deg):", refined_angle)

# Rotate the full image by the refined angle.
full_rotated_image = rotate(data, refined_angle, resize=True, order=3)

# Clip the rotated image to a 1000x1000 region around its center.
clip_size = 1000
full_shape = full_rotated_image.shape  # (rows, cols)
center_y, center_x = full_shape[0]//2, full_shape[1]//2
start_y = max(0, center_y - clip_size//2)
end_y = start_y + clip_size
start_x = max(0, center_x - clip_size//2)
end_x = start_x + clip_size
rotated_image = full_rotated_image[start_y:end_y, start_x:end_x]  # now 1000x1000
# For further analysis, we use the clipped rotated image.

# ---------------------
# Step 3: Find the Row Containing the Trail and Fit a Double Sigmoid
# ---------------------
def double_sigmoid(x, A, B, c, s, f):
    return A * (1 / (1 + np.exp(c * (s - x))) + 1 / (1 + np.exp(c * (x - f)))) + B

# Identify the row with maximum summed intensity (brightest trail signal) in the clipped image.
row_sums = np.sum(rotated_image, axis=1)
trail_row = np.argmax(row_sums)
trail_profile = rotated_image[trail_row, :]

# Define a threshold for background noise and isolate the signal region.
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
    # Retain sub-pixel precision
    start_pixel = popt[3]
    end_pixel = popt[4]
except RuntimeError as e:
    print(f"Curve fitting failed: {e}")
    popt = initial_guess
    start_pixel, end_pixel = 0.1 * clip_size, 1.2 * clip_size

# Ensure start_pixel is less than end_pixel.
if start_pixel > end_pixel:
    start_pixel, end_pixel = end_pixel, start_pixel

# ---------------------
# Step 4: Compute Deltas in Rotated (Clipped) Space
# ---------------------
def rotate_to_sigmoid_space(x, y, angle, original_shape, rotated_shape):
    """
    Rotate coordinates from the original image space into the rotated (sigmoid) space.
    For our analysis, original_shape is the full data.shape and rotated_shape is the clipped image shape.
    """
    angle_rad = np.deg2rad(angle)
    original_center_x, original_center_y = original_shape[1] / 2, original_shape[0] / 2
    rotated_center_x, rotated_center_y = rotated_shape[1] / 2, rotated_shape[0] / 2
    x_shifted = x - original_center_x
    y_shifted = y - original_center_y
    x_rotated = x_shifted * np.cos(angle_rad) + y_shifted * np.sin(angle_rad) + rotated_center_x
    y_rotated = -x_shifted * np.sin(angle_rad) + y_shifted * np.cos(angle_rad) + rotated_center_y
    return x_rotated, y_rotated

# Transform true coordinates from original space into the rotated (clipped) space.
# Note: Use rotated_shape = rotated_image.shape (i.e. 1000x1000).
rotated_true_start_x, _ = rotate_to_sigmoid_space(true_start_x, true_start_y, refined_angle, data.shape, rotated_image.shape)
rotated_true_end_x, _   = rotate_to_sigmoid_space(true_end_x, true_end_y, refined_angle, data.shape, rotated_image.shape)

# Calculate deltas in the 1D (horizontal) clipped rotated space.
start_delta_rotated = rotated_true_start_x - start_pixel
end_delta_rotated = rotated_true_end_x - end_pixel

print(f"Start Delta in Rotated Space: {start_delta_rotated:.2f} pixels")
print(f"End Delta in Rotated Space: {end_delta_rotated:.2f} pixels")

# ---------------------
# Step 5: Visualization
# ---------------------
fig, ax = plt.subplots(1, 2, figsize=(12, 4))

# Left: Display the clipped rotated image with overlaid detected and true endpoints (in rotated space).
ax[0].imshow(rotated_image, origin='lower', cmap='gray', norm=LogNorm(vmin=600, vmax=3000))
# Plot the detected endpoints (from the double sigmoid fit) along the horizontal axis at trail_row.
ax[0].scatter([start_pixel, end_pixel],
              [trail_row, trail_row],
              color='red', label='Detected')
# Plot the true endpoints (projected into rotated space) along the horizontal axis.
ax[0].scatter([rotated_true_start_x, rotated_true_end_x],
              [trail_row, trail_row],
              color='blue', label='True')
ax[0].set_title('Clipped Rotated Image with Endpoints')
ax[0].legend()

# Right: Plot the trail profile and the double sigmoid fit.
ax[1].plot(x_data, trail_profile_limited, label='Trail Profile')
ax[1].plot(x_data, double_sigmoid(x_data, *popt), linestyle='--', label='Double Sigmoid Fit')
ax[1].set_title('Double Sigmoid Fit')
ax[1].legend()

plt.tight_layout()
plt.show()
