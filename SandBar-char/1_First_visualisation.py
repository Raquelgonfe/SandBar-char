# -*- title: First visualisation of the data -*-
"""
Created on Thursday 1 March 2025

@author: Raquelgonfe

Description: 
First visualisation of bathimetry profiles for different years showing the sandbar's crest
Width, slope, height, and depth are calculated to select only sandbars with a minimum height.
The parameter calculations here are first approximations, in 2.Dataset_extraction some parameters are recalculated.
"""

#%% === Imports ===
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx
from shapely.geometry import LineString, Polygon
from scipy.interpolate import CubicSpline
import matplotlib.colors as mcolors
import imageio
import seaborn as sns
from sklearn.linear_model import LinearRegression
import os
import cv2
import math

#%% === Jarkus dataset Configuration ===
# Load the data and filter it
#nc_file_path = os.path.expanduser("~/Downloads/transect_r20240405.nc")
nc_file_path = os.path.expanduser("~/Downloads/transect.nc")                    # Load the dataset

xrds = xr.open_dataset(nc_file_path)
ameland_data = xrds.sel(alongshore=xrds['areacode'] == 3)                       # Filter for Ameland by areacode (Ameland's areacode = 3)

# Variables: time, cross-shore, transect, and altitude
xRSP = ameland_data['cross_shore'][:]                                           # Cross-shore coordinates
time = ameland_data["time"]                                                     # Time 
year = time.dt.year                                                             # Array of years


# === Example plot: Cross-shore bathymetry evolution ===
transect_index = 77                                                             # Transect index

plt.figure(figsize=(20, 12))
for i, y in enumerate(range(1997, 2004)):                                       # Loop through the years and plot
    if y in year.values:
        year_index = np.where(year == y)[0]                                     # Get the year index
        altitude = ameland_data['altitude'][year_index, transect_index, :].values.flatten()
        xRSP_clean = xRSP.values.flatten()                                      # Clean NaN values
        altitude_clean = altitude[~np.isnan(altitude)]
        xRSP_clean = xRSP_clean[~np.isnan(altitude)]

        # === Savitzky-Golay Smoothing the profiles ===
        window_length = 11                                                      # Define the window length (must be odd)
        if len(altitude_clean) < window_length:                                 # Skip a year if data length is insufficient for filtering
            print(f"Year {y} - Skipping due to insufficient data for filtering")
            continue

        polyorder = 2                                                           # Polynomial order
        smoothed_altitude = savgol_filter(altitude_clean, window_length=window_length, polyorder=polyorder)

        # === Finding sandbar creast and troughs from 1st and 2nd derivative ===
        first_derivative = np.gradient(smoothed_altitude)
        second_derivative = np.gradient(first_derivative)

        crests = []
        for j in range(1, len(first_derivative) - 1):
            if first_derivative[j] < 0 and first_derivative[j - 1] >= 0:
                if second_derivative[j] < 0 and smoothed_altitude[j] < 0:
                    crests.append((xRSP_clean[j], smoothed_altitude[j]))

        troughs = []
        for j in range(1, len(first_derivative) - 1):
            if first_derivative[j] > 0 and first_derivative[j - 1] <= 0:
                if smoothed_altitude[j] < 0:
                    troughs.append((xRSP_clean[j], smoothed_altitude[j]))

        
        # === Calculate sandbar parameters and filter significant crests ===
        significant_crests = []
        for crest in crests:
            position, crest_height = crest
            left_trough, right_trough = None, None                              # Find nearest left and right troughs

            for trough in reversed(troughs):
                if trough[0] < position:
                    left_trough = trough
                    break

            for trough in troughs:
                if trough[0] > position:
                    right_trough = trough
                    break

            if right_trough is None:
                right_trough = (xRSP_clean[-1], smoothed_altitude[-1])

            if left_trough is not None and right_trough is not None:
                width = right_trough[0] - left_trough[0]
                slope = (right_trough[1] - left_trough[1]) / width
                intersection_altitude = left_trough[1] + slope * (position - left_trough[0])
                height = intersection_altitude - crest_height
                depth = crest_height

                if height < -0.5:
                    significant_crests.append((position, crest_height))
                    print(f"Year {y} - Significant crest at {position} m with height {height} m")

        # === Plot visualisation ===
        altitude_offset = i * 3                                                 # Offset the altitude for each year to prevent overlap
        plt.plot(xRSP_clean, smoothed_altitude + altitude_offset, color='black', label=f'Year {y}')

        if significant_crests:
            crest_x, crest_y = zip(*significant_crests)
            plt.scatter(crest_x, np.array(crest_y) + altitude_offset, color='red', marker='o', label=f'Significant Crests Year {y}', s=40)

        plt.text(1220, smoothed_altitude[-1] + altitude_offset, f'{y}', fontsize=20, color='black')

# === Plot details ===
plt.title(f'Transect {transect_index}', fontsize=20)
plt.xlabel('Cross-Shore (m)', fontsize=20)
plt.ylabel('Elevation (m)', fontsize=20)
plt.xlim([-200, 1200])
plt.ylim([-8, 17])
plt.grid(True)
plt.tight_layout()
plt.show()




