#!/usr/bin/env python3
"""
Script: star_detection_comparison.py
Description:
    This script performs two main tasks:
      1. Detect stars in a FITS image using DAOStarFinder and save the results to a CSV file.
      2. Compare synthetic star coordinates with the detected star coordinates,
         save the comparison to a CSV file, and generate a log–log plot of the
         position difference versus measured flux.
         
Author: Ivan
"""

# ===============================
# IMPORTS
# ===============================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.table import Table
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder

# ===============================
# CONFIGURATION & PARAMETERS
# ===============================
# File paths
FITS_FILE_PATH = "Simulated Images/Merged_Trail.fits"          # FITS image path
SYNTHETIC_CSV = "CSV/Real_Star_Coordinates.csv"                # Synthetic stars CSV file
DETECTED_CSV = "CSV/detected_stars.csv"                        # Output CSV for detected stars
COMPARISON_CSV = "CSV/comparison_star_coordinates.csv"         # Output CSV for comparison results
Star_Comparison_Path = "Plots/Star_Comparison.png"             # Output plot for comparison results

# Detection parameters
FWHM = 3.0
THRESHOLD_FACTOR = 5.0  # Multiplied by the background standard deviation for thresholding

# ===============================
# FUNCTION DEFINITIONS
# ===============================
def detect_stars(fits_file, output_csv):
    """
    Detect stars in a FITS image using DAOStarFinder and save the results to a CSV file.

    Parameters:
        fits_file (str): Path to the FITS image.
        output_csv (str): Path where detected star coordinates and fluxes will be saved.
        
    Returns:
        sources (astropy.table.Table): Detected star sources.
    """
    # Load the FITS image data
    with fits.open(fits_file) as hdul:
        data = hdul[0].data

    # Estimate background statistics using sigma clipping
    mean, median, std = sigma_clipped_stats(data, sigma=3.0)
    
    # Locate stars in the background-subtracted image
    daofind = DAOStarFinder(fwhm=FWHM, threshold=THRESHOLD_FACTOR * std)
    sources = daofind(data - median)
    
    if sources is not None and len(sources) > 0:
        # Convert the detected sources to a pandas DataFrame and save to CSV
        sources_df = Table(sources).to_pandas()
        sources_df.to_csv(output_csv, index=False)
        print(f"Star coordinates and fluxes saved to {output_csv}")
    else:
        print("No stars detected.")
    
    return sources

def compare_star_coordinates(synthetic_csv, detected_csv, output_comparison_csv, tolerance=5):
    """
    Compare synthetic star coordinates with those detected and save the comparison.

    For each synthetic star, the closest detected star is identified. If the distance
    between them is within the specified tolerance, the match is recorded.

    Parameters:
        synthetic_csv (str): Path to CSV file with synthetic star coordinates.
        detected_csv (str): Path to CSV file with detected star coordinates.
        output_comparison_csv (str): Path where the comparison results will be saved.
        tolerance (float): Maximum allowed distance (in pixels) for a match.
        
    Returns:
        comparison_df (pd.DataFrame): DataFrame containing the comparison results.
    """
    # Load the synthetic and detected star coordinates
    synthetic_df = pd.read_csv(synthetic_csv)
    detected_df = pd.read_csv(detected_csv)
    
    comparisons = []
    for _, synth_row in synthetic_df.iterrows():
        synthetic_x, synthetic_y = synth_row["x_coords"], synth_row["y_coords"]
        true_flux = synth_row["fluxes"]
        
        # Compute the distance from this synthetic star to each detected star
        distances = np.hypot(
            detected_df["xcentroid"] - synthetic_x,
            detected_df["ycentroid"] - synthetic_y
        )
        idx_min = distances.idxmin()
        min_distance = distances[idx_min]
        
        # Record match if within tolerance
        if min_distance <= tolerance:
            closest_star = detected_df.loc[idx_min]
            comparisons.append({
                "True_X": synthetic_x,
                "Detected_X": closest_star["xcentroid"],
                "True_Y": synthetic_y,
                "Detected_Y": closest_star["ycentroid"],
                "Distance": min_distance,
                "True Flux": true_flux,
                "Measured Flux": closest_star["flux"]  # from DAOStarFinder
            })
    
    # Convert comparisons to DataFrame and save to CSV
    comparison_df = pd.DataFrame(comparisons)
    comparison_df.to_csv(output_comparison_csv, index=False)
    print(f"Comparison results saved to {output_comparison_csv}")
    
    # Report average position difference if matches exist
    if not comparison_df.empty:
        distance_mean = np.mean(comparison_df["Distance"])
        print(f"\nAverage of star position differences: {distance_mean:.4f} pixels")
    else:
        print("\nNo matched stars found; cannot calculate average position difference.")

    # ---- PLOTTING SECTION ----
    if not comparison_df.empty:
        # Filter out non-positive values for log scaling
        plot_mask = (comparison_df["Distance"] > 0) & (comparison_df["Measured Flux"] > 0)
        df_plot = comparison_df.loc[plot_mask]
        
        if df_plot.empty:
            print("No valid data points for log-scale plotting (zero or negative values found).")
            return comparison_df
        
        x_vals = df_plot["Distance"].values
        y_vals = df_plot["Measured Flux"].values
        
        # Create a scatter plot in log-log space with a linear fit
        plt.figure(figsize=(8, 6))
        plt.scatter(x_vals, y_vals, alpha=0.7, label="Data Points")
        
        # Perform a linear fit in log space: log(y) = m*log(x) + b
        log_x = np.log10(x_vals)
        log_y = np.log10(y_vals)
        m, b = np.polyfit(log_x, log_y, 1)
        
        # Generate fit line and convert back to linear space
        fit_log_x = np.linspace(log_x.min(), log_x.max(), 100)
        fit_log_y = m * fit_log_x + b
        fit_x = 10**fit_log_x
        fit_y = 10**fit_log_y
        
        plt.plot(fit_x, fit_y, color="red", label="Best Fit")
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("Position Difference (True - Measured) [pixels, log scale]")
        plt.ylabel("Measured Flux (DAOStarFinder) [log scale]")
        plt.title("Position Error vs. Measured Flux (log-log scale)")
        plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.legend()
        plt.savefig(Star_Comparison_Path)
        plt.show()
    
    return comparison_df

# ===============================
# MAIN EXECUTION
# ===============================
def main():
    # Run star detection on the FITS image
    detected_sources = detect_stars(FITS_FILE_PATH, DETECTED_CSV)
    
    # Compare synthetic star coordinates with detected stars and plot the results
    compare_star_coordinates(SYNTHETIC_CSV, DETECTED_CSV, COMPARISON_CSV)

if __name__ == '__main__':
    main()
