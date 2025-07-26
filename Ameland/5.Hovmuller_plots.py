# -*- title: Hovmuller plots (sandbar volume and profile volume) -*-
"""
Created on Thursday 1 March 2025

@author: Raquelgonfe

Description:
Hovmuller plots for sandbar and profile volume to analyse alongshore transport.

The script includes the following components:
(References to the corresponding thesis figures.)

1. Hovmöller plot showing elevation vs. alongshore distance for a single transect.
   Useful for visualizing cross-shore trends across the dataset for different transects (sections).
2.1. Hovmöller plot of total sandbar volume (Figure 37)
2.2. Hovmöller plot of sandbar volume filtered by different cross-shore positions
3. Hovmöller plot of profile volume (Figure 37)

Note:
No Jarkus data is available for the years 1972 and 1973.
"""

#%% === Imports ===
import matplotlib.pyplot as plt 
import numpy as np 
import os 
from scipy.signal import savgol_filter 
import cv2
import pandas as pd
import xarray as xr
import geopandas as gpd
import contextily as ctx
from shapely.geometry import LineString, Polygon
from scipy.interpolate import CubicSpline
import matplotlib.ticker as ticker
import math
import matplotlib.colors as mcolors
import imageio
import seaborn as sns
from scipy import stats
from scipy.interpolate import interp1d
from scipy.interpolate import griddata
from matplotlib.colors import LinearSegmentedColormap



#%% === 1. Altitude Hovmuller plot for 1 transect ===

nc_file_path = os.path.expanduser("~/Downloads/transect_r20240405.nc")                                                                      # Load the dataset and filter data for Ameland (areacode = 3)
xrds = xr.open_dataset(nc_file_path)
ameland_data = xrds.sel(alongshore=xrds['areacode'] == 3)

xRSP = ameland_data['cross_shore'][:]                                                                                                       # Cross-shore coordinates
time = ameland_data['time']
lat = ameland_data['lat']                                                                                                                   # Latitude for each cross-shore point
lon = ameland_data['lon']                                                                                                                   # Longitude for each cross-shore point
year = time.dt.year                                                                                                                         # Extract the year

transect_index = 53                                                                                                                         # The transect index 

altitude_data = []
cross_shore_data = []
year_data = []

altitude = ameland_data['altitude']

crest_positions = []                                                                                                                        # Store the crest positions and their corresponding years for plotting
crest_years = []

for y in range(1965, 2024):                                                                                                                 # Loop through years to gather data
    if y in year.values:
        year_index = np.where(year == y)[0][0]                                                                                              # Get index for the specific year
        altitude = ameland_data['altitude'][year_index, transect_index, :].values.flatten()
        
        xRSP_clean = xRSP.values.flatten()                                                                                                  # Clean NaN values
        altitude_clean = altitude[~np.isnan(altitude)]
        xRSP_clean = xRSP_clean[~np.isnan(altitude)]
        
        for i in range(len(altitude_clean)):                                                                                                # Store altitude and corresponding cross-shore and year data
            altitude_data.append(altitude_clean[i])
            cross_shore_data.append(xRSP_clean[i])
            year_data.append(y)
        
        if len(altitude_clean) >= 11:                                                                                                       # Detect crests (use the logic from the crest-detection code)
            window_length = 11                                                                                                              # Apply Savitzky-Golay filter for smoothing. Define the window length
            polyorder = 2                                                                                                                   # Polynomial order
            smoothed_altitude = savgol_filter(altitude_clean, window_length=window_length, polyorder=polyorder)

            first_derivative = np.gradient(smoothed_altitude)                                                                               # Calculate the first and second derivatives
            second_derivative = np.gradient(first_derivative)

            for j in range(1, len(first_derivative) - 1):
                if first_derivative[j] < 0 and first_derivative[j - 1] >= 0:                                                                # Downward zero-crossing
                    if second_derivative[j] < 0 and smoothed_altitude[j] < 0:                                                               # Below sea level
                        crest_positions.append(xRSP_clean[j])                                                                               # Store position of the crest
                        crest_years.append(y)                                                                                               # Store corresponding year

altitude_data = np.array(altitude_data)                                                                                                     # Convert to numpy arrays for plotting
cross_shore_data = np.array(cross_shore_data)
year_data = np.array(year_data)

# Interpolating altitude data to create a grid
cross_shore_min = np.min(cross_shore_data)                                                                                                  # Define the grid limits for cross-shore distance and years
cross_shore_max = np.max(cross_shore_data)
year_min = np.min(year_data)
year_max = np.max(year_data)

grid_cross_shore, grid_year = np.mgrid[cross_shore_min:cross_shore_max:150j, year_min:year_max:150j]                                        # Create a grid for interpolation

grid_altitude = griddata((cross_shore_data, year_data), altitude_data, (grid_cross_shore, grid_year), method='linear')                      # Perform interpolation using 'linear' method

grid_altitude = np.clip(grid_altitude, -7, 0)                                                                                               # Clip the interpolated altitude values between -7 and 0 to ensure consistency

plt.figure(figsize=(6, 10))                                                                                                                 # Adjust the figure size if needed

colors = ['darkblue', 'cyan', 'green', 'yellow', 'orange', 'red']                                                                           # Create a custom continuous colormap
positions = [0, 0.25, 0.3, 0.5, 0.6, 1] 
cmap = LinearSegmentedColormap.from_list('custom_cmap', list(zip(positions, colors)))

heatmap = plt.pcolormesh(grid_cross_shore, grid_year, grid_altitude, cmap=cmap, shading='auto', vmin=-7, vmax=0)                            # Create a pcolormesh plot where year is on the y-axis and cross-shore distance is on the x-axis

cbar = plt.colorbar(heatmap, extend='both')                                                                                                 # Adding colorbar with extend option
cbar.set_label('Altitude (m)')

plt.scatter(crest_positions, crest_years, color='black', marker='o', s=20, label='Crests')
plt.title(f'Sandbar migration with crests, Transect {transect_index}')
plt.xlabel('Cross-shore distance (m)')
plt.ylabel('Year')
plt.ylim([1965, 2023])
plt.xlim([-200, 1000])
plt.grid(True)

plt.tight_layout()
plt.show()

# === Plotting without the crests ===
grid_altitude = np.clip(grid_altitude, -7, 0)                                                                                               # Clip the interpolated altitude values between -7 and 0 to ensure consistency

plt.figure(figsize=(6, 10))                                                                                                                 # Adjust the figure size if needed
colors = ['darkblue', 'cyan', 'green', 'yellow', 'orange', 'red']
positions = [0, 0.25, 0.3, 0.5, 0.6, 1] 
cmap = LinearSegmentedColormap.from_list('custom_cmap', list(zip(positions, colors)))

heatmap = plt.pcolormesh(grid_cross_shore, grid_year, grid_altitude, cmap=cmap, shading='auto', vmin=-7, vmax=0)                            # Create a pcolormesh plot where year is on the y-axis and cross-shore distance is on the x-axis
cbar = plt.colorbar(heatmap, extend='both')         
cbar.set_label('Elevation (m)')

plt.xlabel('Cross-shore distance (m)')
plt.ylim([1965, 2023])
plt.xlim([-200, 1000])
plt.grid(True)
plt.tight_layout()
plt.show()



#%% === 2.1. Hovmuller plot for sandbar volume ===

years_to_process = range(1985, 2024)                                                                                                        # Years to analyze
transect_min, transect_max = 40, 120                                                                                                        # Transect range
transect_spacing_km = 0.2                                                                                                                   # Spacing between transects in kilometers

transect_to_alongshore = lambda transect: (transect - transect_min) * transect_spacing_km                                                   # Function to convert transect number to alongshore distance

sandbar_volume_data = []                                                                                                                    # Storage for processed data
alongshore_data = []
year_data = []

for year in years_to_process:                                                                                                               # Loop through each year to process the data
    print(f"Processing year: {year}")
    
    sandbar_csv_file = f'updated_sandbar_data_{year}.csv'                                                                                   # Load the sandbar volume data for the current year
    
    try:
        sandbar_df = pd.read_csv(sandbar_csv_file)
    except FileNotFoundError:
        print(f"File not found: {sandbar_csv_file}. Skipping this year.")
        continue
    
    if 'transect' not in sandbar_df.columns or 'volume' not in sandbar_df.columns:
        print(f"Missing 'transect' or 'volume' column in {sandbar_csv_file}. Skipping this year.")
        continue
    
    for transect in range(transect_min, transect_max + 1):                                                                                  # Loop through transects in the range
        transect_data = sandbar_df[sandbar_df['transect'] == transect]
        
        total_sandbar_volume = transect_data['volume'].sum() if not transect_data.empty else 0                                              # Sum the volumes of all sandbars for this transect
        
        sandbar_volume_data.append(total_sandbar_volume)                                                                                    # Append the data
        alongshore_data.append(transect_to_alongshore(transect))
        year_data.append(year)

sandbar_volume_data = np.array(sandbar_volume_data)                                                                                         # Convert to numpy arrays
alongshore_data = np.array(alongshore_data)
year_data = np.array(year_data)

if len(sandbar_volume_data) == 0:                                                                                                           # Check if any data was processed
    print("No valid data found for the specified years and transect range.")
else:
    grid_alongshore, grid_year = np.mgrid[                                                                                                  # Define grid for interpolation
        min(alongshore_data):max(alongshore_data):50j,
        min(years_to_process):max(years_to_process):50j
    ]

    grid_volume = griddata((alongshore_data, year_data), sandbar_volume_data, (grid_alongshore, grid_year), method='linear')                # Perform interpolation using 'linear' method

    plt.figure(figsize=(10, 10))                                                                                                            # Plot the Hovmöller diagram
    
    colors = ['darkblue', 'cyan', 'green', 'yellow', 'orange', 'red']                                                                       # Define a colormap
    positions = [0, 0.25, 0.5, 0.75, 1]
    cmap = LinearSegmentedColormap.from_list('volume_cmap', list(zip(positions, colors)))
    
    heatmap = plt.pcolormesh(grid_alongshore, grid_year, grid_volume, cmap=cmap, shading='auto', vmin=0, vmax=np.nanmax(grid_volume))
    
    cbar = plt.colorbar(heatmap, extend='both')                                                                                             # Add color bar and labels
    cbar.set_label('Total sandbar volume (m³/m)', fontsize=16)                                                                              # Increased font size
    cbar.ax.tick_params(labelsize=12)                                                                                                       # Increase color bar tick labels

    plt.xlabel('Alongshore distance (km)', fontsize=16)                                                                                     # Increased font size
    plt.xticks(fontsize=12)                                                                                                                 # Increase tick label size
    plt.yticks(fontsize=12)                                                                                                                 # Increase tick label size
    plt.ylim([min(years_to_process), max(years_to_process)])
    plt.xlim(0, transect_to_alongshore(transect_max))
    plt.grid(True)
    plt.tight_layout()
    plt.show()



#%% === 2.2. Multiple sandbar volume hovmuller plot for different cross-shore positions ===

years_to_process = range(1965, 2024)                                                                                                        # Adjust the year range as necessary
fixed_cross_shores = [200, 400, 600, 800]                                                                                                   # Fixed cross-shore distances to analyze
transect_min, transect_max = 40, 120                                                                                                        # Approximate range of transects
tolerance = 50
transect_spacing_km = 0.2                                                                                                                   # Spacing between transects in kilometers

# Calculate alongshore distances for transects
transect_to_alongshore = lambda transect: (transect - 40) * transect_spacing_km                                                             # Transect 40 = 0 km

all_grid_volumes = []                                                                                                                       # Storage for processed data for each cross-shore

for fixed_cross_shore in fixed_cross_shores:
    volume_data = []
    alongshore_data = []
    year_data = []

    for year in years_to_process:                                                                                                           # Process each year's data
        csv_file = f'updated_sandbar_data_{year}.csv'
        try:
            df = pd.read_csv(csv_file)
        except FileNotFoundError:
            print(f"File not found: {csv_file}. Skipping this year.")
            continue

        df['relative_label'] = df['relative_label'].astype(str)                                                                             # Ensure 'relative_label' is treated as string 

        df_fixed_cs = df[np.abs(df['cross_shore'] - fixed_cross_shore) <= tolerance]                                                        # Filter the data for the fixed cross-shore distance 

        valid_transects = df_fixed_cs['transect'].unique()                                                                                  # Identify valid transects with sandbar data

        for transect in range(transect_min, transect_max + 1):                                                                              # Loop through all transects in the range, sum volumes for each transect and year
            if transect in valid_transects:                                                                                                 # Check if transect has valid data
                total_volume = df_fixed_cs[df_fixed_cs['transect'] == transect]['volume'].sum()                                             # Sum the volumes for each year at the given transect
                volume_data.append(total_volume)
                alongshore_data.append(transect_to_alongshore(transect))
                year_data.append(year)
            else:
                volume_data.append(0)                                                                                                       # If no sandbar data for this transect, assume volume = 0
                alongshore_data.append(transect_to_alongshore(transect))
                year_data.append(year)

    volume_data = np.array(volume_data)                                                                                                     # Convert to numpy arrays
    alongshore_data = np.array(alongshore_data)
    year_data = np.array(year_data)

    # Interpolating volume data to create a grid
    grid_alongshore, grid_year = np.mgrid[                                                                                                  # Define the grid limits and create a grid for interpolation year
        min(alongshore_data):max(alongshore_data):50j,
        min(years_to_process):max(years_to_process):50j
    ]

    grid_volume = griddata((alongshore_data, year_data), volume_data, (grid_alongshore, grid_year), method='linear')                        # Perform interpolation using 'linear' method
    all_grid_volumes.append(grid_volume)

global_vmin = min(np.nanmin(grid) for grid in all_grid_volumes)                                                                             # Determine global vmin and vmax for colorbar consistency
global_vmax = max(np.nanmax(grid) for grid in all_grid_volumes)

fig = plt.figure(figsize=(10, 8))                                                                                                           # Plotting the subplots
gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.05], height_ratios=[1, 1], wspace=0.1, hspace=0.1)
axes = []

for i, fixed_cross_shore in enumerate(fixed_cross_shores):
    row, col = divmod(i, 2)
    ax = fig.add_subplot(gs[row, col])
    
    heatmap = ax.pcolormesh(                                                                                                                # Plot the data for this cross-shore distance
        grid_alongshore, grid_year, all_grid_volumes[i], 
        cmap='turbo', shading='auto', vmin=global_vmin, vmax=global_vmax
    )
    ax.set_title(f'Cross-shore-distance = {fixed_cross_shore} m', fontsize=10)                                                              # Titles and labels
    if row == 1:
        ax.set_xlabel('Alongshore distance (km)', fontsize=10)
    else:
        ax.set_xticks([])
        ax.set_xticklabels(ax.get_xticks(), fontsize=9)

    if col == 0:
        ax.set_ylabel(' ', fontsize=12)
    else:
        ax.set_yticks([])
        ax.set_xticklabels(ax.get_xticks(), fontsize=9)

    axes.append(ax)
cbar_ax = fig.add_subplot(gs[:, 2])                                                                                                         # Add a single colorbar
cbar = fig.colorbar(heatmap, cax=cbar_ax, extend='both')
cbar.set_label('Total volume (m³/m)', fontsize=10)
plt.tight_layout()
plt.show()



#%% === 3. Hovmuller plot of profile volume ===

years_to_process = range(1985, 2024)                                                                                                        # Years to analyze
transect_min, transect_max = 40, 120                                                                                                        # Transect range
transect_spacing_km = 0.2                                                                                                                   # Spacing between transects in kilometers

transect_to_alongshore = lambda transect: (transect - 40) * transect_spacing_km                                                             # Transect 40 = 0 km

profile_volume_data = []
alongshore_data = []
year_data = []

for year in years_to_process:                                                                                                               # Loop through each year to process the data
    print(f"Processing year: {year}")
    profile_csv_file = f'{year}_profile_volume.csv'                                                                                         # Load the profile volume data for the current year
    
    try:
        profile_df = pd.read_csv(profile_csv_file)
        if profile_df.empty:
            raise pd.errors.EmptyDataError("File exists but is empty.")
    except FileNotFoundError:
        print(f"File not found: {profile_csv_file}. Skipping this year.")
        continue
    except pd.errors.EmptyDataError:
        print(f"File is empty: {profile_csv_file}. Skipping this year.")
        continue

    if 'Transect' not in profile_df.columns or 'Volume' not in profile_df.columns:                                                          # Check if the required columns exist
        print(f"Missing 'Transect' or 'Volume' column in {profile_csv_file}. Skipping this year.")
        continue

    for transect in range(transect_min, transect_max + 1):                                                                                  # Loop through transects in the range
        if transect in profile_df['Transect'].values:                                                                                       # Check if the transect exists in the profile data
            profile_volume = profile_df[profile_df['Transect'] == transect]['Volume'].values[0]
        else:
            profile_volume = 0                                                                                                              # If no data, assume profile volume = 0

        profile_volume_data.append(profile_volume)                                                                                          # Append the data
        alongshore_data.append(transect_to_alongshore(transect))
        year_data.append(year)

profile_volume_data = np.array(profile_volume_data)                                                                                         # Convert to numpy arrays
alongshore_data = np.array(alongshore_data)
year_data = np.array(year_data)

if len(profile_volume_data) == 0:                                                                                                           # Check if any data was processed
    print("No valid data found for the specified years and transect range.")
else:
    # Interpolating profile volume data to create a grid
    grid_alongshore, grid_year = np.mgrid[                                                                                                  # Define the grid limits and create a grid for interpolation
        min(alongshore_data):max(alongshore_data):50j,
        min(years_to_process):max(years_to_process):50j
    ]

    grid_volume = griddata((alongshore_data, year_data), profile_volume_data, (grid_alongshore, grid_year), method='linear')                # Perform interpolation using 'linear' method

    plt.figure(figsize=(10, 10))                                                                                                            # Plot the Hovmöller diagram

    colors = ['darkblue', 'cyan', 'green', 'yellow', 'orange', 'red']                                                                       # Define a colormap
    positions = [0, 0.25, 0.5, 0.75, 1]
    cmap = LinearSegmentedColormap.from_list('volume_cmap', list(zip(positions, colors)))

    heatmap = plt.pcolormesh(grid_alongshore, grid_year, grid_volume, cmap=cmap, shading='auto', vmin=0, vmax=np.nanmax(grid_volume))       # Create the plot

    cbar = plt.colorbar(heatmap, extend='both')     # Add color bar and labels
    cbar.set_label('Profile volume (m³/m)')

    plt.xlabel('Alongshore distance (km)')  
    plt.ylim([min(years_to_process), max(years_to_process)])
    plt.xlim(0, 15.75)
    plt.grid(True)
    plt.tight_layout()
    plt.show()