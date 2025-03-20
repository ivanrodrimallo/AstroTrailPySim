# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 16:37:48 2024

@author: gigma
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.table import Table
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder


# File paths
fits_file = "Simulated Images/Merged_Trail_REPORT.fits"       # Replace with your FITS file path
synthetic_csv = "CSV/Real_Star_Coordinates.csv"            # Synthetic stars CSV file
detected_stars_csv = "CSV/detected_stars.csv"
comparison_csv = "CSV/comparison_star_coordinates.csv"


def detect_stars(fits_file, output_csv):
    """
    Detect stars in a FITS image using DAOStarFinder and save the results to a CSV file.
    """
    # Load the FITS image data
    with fits.open(fits_file) as hdul:
        data = hdul[0].data

    # Estimate background statistics using sigma clipping
    mean, median, std = sigma_clipped_stats(data, sigma=3.0)
    
    # Locate stars in the background-subtracted image
    daofind = DAOStarFinder(fwhm=3.0, threshold=5.0 * std)
    sources = daofind(data - median)
    
    if sources is not None and len(sources) > 0:
        # Convert the detected sources to a DataFrame and save to CSV
        sources_df = Table(sources).to_pandas()
        sources_df.to_csv(output_csv, index=False)
        print(f"Star coordinates and fluxes saved to {output_csv}")
    else:
        print("No stars detected.")
    
    return sources


def compare_star_coordinates(synthetic_csv, detected_csv, output_comparison_csv, tolerance=5):
    """
    Compare synthetic star coordinates with those detected and save the comparison.
    
    For each synthetic star, the closest detected star is identified.
    If the distance between them is within the specified tolerance,
    the match is recorded.
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
    
    # Convert comparisons to DataFrame and save
    comparison_df = pd.DataFrame(comparisons)
    comparison_df.to_csv(output_comparison_csv, index=False)
    print(f"Comparison results saved to {output_comparison_csv}")
    
    if not comparison_df.empty:
        distance_std = np.mean(comparison_df["Distance"])
        print(f"\nAverage of star position differences: {distance_std:.4f} pixels")
    else:
        print("\nNo matched stars found; cannot calculate standard deviation.")

    # ---- PLOTTING SECTION ----
    if not comparison_df.empty:
        # Filter out zero or negative values for log scaling
        plot_mask = (comparison_df["Distance"] > 0) & (comparison_df["Measured Flux"] > 0)
        df_plot = comparison_df.loc[plot_mask]
        
        if df_plot.empty:
            print("No valid data points for log-scale plotting (zero or negative values found).")
            return comparison_df
        
        x_vals = df_plot["Distance"].values
        y_vals = df_plot["Measured Flux"].values
        
        # Create figure
        plt.figure(figsize=(8, 6))
        
        # Scatter plot in log space
        plt.scatter(x_vals, y_vals, alpha=0.7, label="Data Points")
        
        # Perform a linear fit in log space
        log_x = np.log10(x_vals)
        log_y = np.log10(y_vals)
        
        # Fit a line: log_y = m*log_x + b
        m, b = np.polyfit(log_x, log_y, 1)
        
        # Generate fit line in log space
        fit_log_x = np.linspace(log_x.min(), log_x.max(), 100)
        fit_log_y = m * fit_log_x + b
        
        # Convert fit line back to linear space
        fit_x = 10**fit_log_x
        fit_y = 10**fit_log_y
        
        # Plot the best-fit line
        plt.plot(fit_x, fit_y, color="red",
                 label=f"Best Fit")
        
        # Set axes to log scale
        plt.xscale("log")
        plt.yscale("log")
        
        # Axis labels and title
        plt.xlabel("Position Difference (True - Measured) [pixels, log scale]")
        plt.ylabel("Measured Flux (DAOStarFinder) [log scale]")
        plt.title("Position Error vs. Measured Flux (log-log scale)")
        plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.legend()
        plt.savefig('comparison_plot.png')
        plt.show()
    
    return comparison_df


# Run detection and comparison
detected_sources = detect_stars(fits_file, detected_stars_csv)
comparison_results = compare_star_coordinates(synthetic_csv, detected_stars_csv, comparison_csv)
