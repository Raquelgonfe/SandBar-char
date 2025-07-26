
# -*- title: Plots for Volumes (Sandbar and Profile) -*-
"""
Created on Thursday 1 March 2025

@author: Raquelgonfe

Description:
Calculation and plotting of sandbar volume, profile volume, ratio between these two volumes, and number of sandbars.

The script includes the following components:
(References to the corresponding thesis figures.)

1.1 Plot average total sandbar volume for the whole study area (Figure 33)
1.2 Plot average total sandbar volume per section (Figure 34)
2.1. Calculate profile volume and stores it in a csv file for each year
2.2  Plot average profile volume for the whole area (Figure 35)
3.1  Plot average sandbar volume, profile volume and ratio between those two
3.2  Plot number of sandbars, average sandbar volume, profile volume, and ratio between those two (Figure 36)
     Only difference from 3.2 is the addition of the n. of sandbars. 

Note:
The transect range can be modified to plot the different sections in each step but specially interesting in step 1.2, 3.1 and 3.2. 
The sections are defined in the thesis as follows:
    Section 1: 40-60
    Section 2: 60-110
    Section 3: 110-120
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
import matplotlib.pyplot as plt
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



#%% === 1.1 Plot average total sandbar volume ===

years_to_process = range(1965, 2023)
transect_range = range(40, 120)                                                                         # whole study area

yearly_sandbar_volumes = []

for year in years_to_process:                                                                           # Loop through years to process the data
    print(f'Processing year: {year}')
    
    sandbar_csv_file = f'updated_sandbar_data_{year}.csv'                                               # Load the sandbar volume data
    
    try:
        sandbar_df = pd.read_csv(sandbar_csv_file)
    except FileNotFoundError:
        print(f"File not found: {sandbar_csv_file}. Skipping this year.")
        continue

    if {'transect', 'volume'}.issubset(sandbar_df.columns):                                             # Validate columns
        year_sandbar_volume = []                                                                        # Initialize a list to store total sandbar volume for this year

        for transect in transect_range:                                                                 # Process each transect in the range
            transect_data = sandbar_df[sandbar_df['transect'] == transect]                              # Filter for the specific transect in the sandbar data
            total_sandbar_volume = transect_data['volume'].sum()                                        # Sum the volumes of all sandbars for this transect
            year_sandbar_volume.append(total_sandbar_volume)                                            # Append the total volume for this transect
        if year_sandbar_volume:                                                                         # Store the year and calculated total sandbar volume
            yearly_sandbar_volumes.append({'Year': year, 'TotalVolume': year_sandbar_volume})
    else:
        print(f"Missing required columns in {sandbar_csv_file}. Skipping this year.")

if yearly_sandbar_volumes:                                                                              # Prepare data for plotting
    stats = []                                                                                          # Calculate mean, 25th, and 75th percentiles for each year
    for entry in yearly_sandbar_volumes:
        year = entry['Year']
        sandbar_volumes = entry['TotalVolume']
        mean_sandbar_volume = np.mean(sandbar_volumes)
        q25_sandbar_volume = np.percentile(sandbar_volumes, 25)
        q75_sandbar_volume = np.percentile(sandbar_volumes, 75)

        stats.append({'Year': year, 'Mean': mean_sandbar_volume, 'Q25': q25_sandbar_volume, 'Q75': q75_sandbar_volume})

    stats_df = pd.DataFrame(stats)                                                                      # Convert stats to a DataFrame

    stats_df['Mean_MA'] = stats_df['Mean'].rolling(window=3, min_periods=1).mean()                      # Apply moving average (3-year window) to the mean sandbar volume

    stats_df['Q25_MA'] = stats_df['Q25'].rolling(window=3, min_periods=1).mean()                        # Apply moving average (3-year window) to the 25th and 75th percentiles 
    stats_df['Q75_MA'] = stats_df['Q75'].rolling(window=3, min_periods=1).mean()

    plt.figure(figsize=(12, 6))
    plt.plot(stats_df['Year'], stats_df['Mean_MA'], color='saddlebrown', linewidth=2, label='Moving average (3 years)')
    plt.fill_between(stats_df['Year'], stats_df['Q25_MA'], stats_df['Q75_MA'], 
                     color='sandybrown', alpha=0.5, label='25th - 75th Percentile')

    plt.xlabel('Year')
    plt.ylabel('Total sandbar volume (m3/m)')
    plt.grid(True)
    plt.ylim(0, 1200)  
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()
else:
    print("No valid data found for the specified years and transect range.")


# === Plot adding stars representing nourishments with their volume ===

nourishments = [                                                                                        # Define nourishments
    {'Year': 1998, 'Volume': 254},
    {'Year': 2003, 'Volume': 333},
    {'Year': 2006, 'Volume': 300},
    {'Year': 2011, 'Volume': 539.3},
    {'Year': 2015, 'Volume': 434.8},
    {'Year': 2019, 'Volume': 446}
]

if yearly_sandbar_volumes:
    plt.figure(figsize=(12, 6))
    plt.plot(stats_df['Year'], stats_df['Mean_MA'], color='saddlebrown', linewidth=2, label='Moving average (3 years)')
    plt.fill_between(stats_df['Year'], stats_df['Q25_MA'], stats_df['Q75_MA'], 
                     color='sandybrown', alpha=0.5, label='25th - 75th Percentile')
    nourishment_years = [n['Year'] for n in nourishments]
    nourishment_volumes = [n['Volume'] for n in nourishments]
    plt.scatter(nourishment_years, nourishment_volumes, color='sienna', marker='*', s=100, label='Nourishments', zorder=5)
    plt.xlabel('Year')
    plt.ylabel('Total sandbar volume (m3/m)')
    plt.grid(True)
    plt.ylim(0, 1200)  
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()
else:
    print("No valid data found for the specified years and transect range.")



#%% === 1.2. Plot average sandbar volume per section ===

years_to_process = range(1965, 2024)                                                                    # Define years and transect range of interest
transect_range = range(40, 61)  

yearly_sandbar_volumes = []

for year in years_to_process:                                                                           # Loop through years to process the data
    print(f'Processing year: {year}')
    
    sandbar_csv_file = f'updated_sandbar_data_{year}.csv'                                               # Load the sandbar volume data
    
    try:
        sandbar_df = pd.read_csv(sandbar_csv_file)
    except FileNotFoundError:
        print(f"File not found: {sandbar_csv_file}. Skipping this year.")
        continue

    if 'transect' not in sandbar_df.columns or 'volume' not in sandbar_df.columns:                      # Check if 'transect' and 'volume' columns exist
        print(f"Missing 'transect' or 'volume' column in {sandbar_csv_file}. Skipping this year.")
        continue

    year_sandbar_volumes = []                                                                           # Initialize list for this year's sandbar volumes

    for transect in transect_range:                                                                     # Process each transect in the range
        transect_data = sandbar_df[sandbar_df['transect'] == transect]                                  # Filter for the specific transect in the sandbar data
        total_sandbar_volume = transect_data['volume'].sum()                                            # Sum the volumes of all sandbars for this transect
        year_sandbar_volumes.append(total_sandbar_volume)

    if year_sandbar_volumes:                                                                            # Store the year and calculated data if we have any data for this year
        yearly_sandbar_volumes.append({'Year': [year] * len(year_sandbar_volumes), 'SandbarVolumes': year_sandbar_volumes})

if yearly_sandbar_volumes:                                                                              # Prepare data for plotting
    sandbar_volumes = []
    years = []

    for entry in yearly_sandbar_volumes:
        sandbar_volumes.extend(entry['SandbarVolumes'])
        years.extend(entry['Year'])

    df = pd.DataFrame({                                                                                 # Convert data to a DataFrame
        'Year': years,
        'SandbarVolume': sandbar_volumes
    })

    stats = []                                                                                          # Calculate the mean, 25th and 75th percentiles for each year
    for year in df['Year'].unique():
        year_data = df[df['Year'] == year]
        mean_sandbar = year_data['SandbarVolume'].mean()
        q25_sandbar = np.percentile(year_data['SandbarVolume'], 25)
        q75_sandbar = np.percentile(year_data['SandbarVolume'], 75)

        stats.append({
            'Year': year,
            'SandbarMean': mean_sandbar,
            'SandbarQ25': q25_sandbar,
            'SandbarQ75': q75_sandbar
        })

    stats_df = pd.DataFrame(stats)                                                                      # Convert stats to DataFrame

    stats_df['SandbarMean_MA'] = stats_df['SandbarMean'].rolling(window=3, min_periods=1).mean()        # Apply 3-year moving average consistently to means and percentiles
    stats_df['SandbarQ25_MA'] = stats_df['SandbarQ25'].rolling(window=3, min_periods=1).mean()
    stats_df['SandbarQ75_MA'] = stats_df['SandbarQ75'].rolling(window=3, min_periods=1).mean()

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(stats_df['Year'], stats_df['SandbarMean_MA'], color='darkblue', label='Mean Sandbar Volume (MA)')
    ax.fill_between(stats_df['Year'], stats_df['SandbarQ25_MA'], stats_df['SandbarQ75_MA'], color='lightblue', alpha=0.5, label='25th-75th Percentile')
    ax.set_ylabel('Sandbar Volume', fontsize=16)
    ax.grid(True)
    ax.tick_params(axis='both', labelsize=14)
    ax.set_ylim(0, 1200)
    ax.legend(fontsize=14)

    plt.title('Section 1', fontsize=18)
    plt.tight_layout()
    plt.show()

else:
    print("No valid data found for the specified years and transect range.")



#%% === 2.1. Calculate profile volume ===

nc_file_path = os.path.expanduser("~/Downloads/transect_r20240405.nc")                                  # Load the dataset and filter data for Ameland (areacode = 3)
xrds = xr.open_dataset(nc_file_path)
ameland_data = xrds.sel(alongshore=xrds['areacode'] == 3)

xRSP = ameland_data['cross_shore'][:]                                                                   # Cross-shore coordinates
time = ameland_data['time']
lat = ameland_data['lat']                                                                               # Latitude for each cross-shore point
lon = ameland_data['lon']                                                                               # Longitude for each cross-shore point
depth_min, depth_max = 0, -8                                                                            # Integration bounds

results = {}

years_to_process = range(1965, 2024)                                                                    # From 1997 to 2000 inclusive
window_length = 11                                                                                      # Define the window length and polynomial order for smoothing
polyorder = 2  
stable_zone_distance = 350                                                                              # Stable zone threshold in meters
threshold_percent = 0.01                                                                                # Threshold for second derivative

for target_year in years_to_process:                                                                    # Main loop for each year
    print(f'Processing year: {target_year}')
    year_indices = np.where(ameland_data['time'].dt.year == target_year)[0]
    year_results = []
    if len(year_indices) == 0:
        print(f"No data for year {target_year}. Skipping.")
        continue

    for transect_index in range(39, 120):                                                               # Adjust range based on your transects
        for time_idx in year_indices:
            altitude = ameland_data['altitude'].isel(time=time_idx, alongshore=transect_index).values.flatten()
        xRSP_clean = xRSP.values                                                                        # Ensure NumPy array
        valid_indices = ~np.isnan(altitude)
        altitude_clean = altitude[valid_indices]
        xRSP_clean = xRSP_clean[valid_indices]

        if len(altitude_clean) < window_length:
            print(f"Insufficient data for transect {transect_index + 1}, year {target_year}. Skipping.")
            continue

        smoothed_altitude = savgol_filter(altitude_clean, window_length, polyorder)                     # Smooth the profile

        depth_mask = (smoothed_altitude <= depth_min) & (smoothed_altitude >= depth_max)                # Select only the depth range of interest
        if np.sum(depth_mask) == 0:
            print(f"Transect {transect_index + 1} does not span the required depth range. Skipping.")
            continue
        xRSP_depth = xRSP_clean[depth_mask]                                                             # Filter data for integration
        altitude_depth = smoothed_altitude[depth_mask]

        volume = np.abs(np.trapz(altitude_depth, xRSP_depth))                                           # Integrate to calculate volume (use absolute value)
        year_results.append({'Transect': transect_index, 'Volume': volume})

    year_df = pd.DataFrame(year_results)                                                                # Convert year_results to a DataFrame
    
    file_name = f"{target_year}_profile_volume.csv"
    
    year_df.to_csv(file_name, index=False)                                                              # Save to CSV in the current directory
    print(f"Saved: {file_name}")
    


#%% === 2.2. Plot average profile volume for the whole area ===

years_to_process = range(1992, 2024)                                                                    # Define years and transect range of interest
transect_range = range(40, 121)  

yearly_profile_volumes = []

for year in years_to_process:                                                                           # Loop through years to process the data
    print(f'Processing year: {year}')
    
    profile_csv_file = f'{year}_profile_volume.csv'                                                     # Load the profile volume data
    try:
        profile_df = pd.read_csv(profile_csv_file)
    except FileNotFoundError:
        print(f"File not found: {profile_csv_file}. Skipping this year.")
        continue
    if {'Transect', 'Volume'}.issubset(profile_df.columns):                                             # Validate columns
        year_profile_volume = []                                                                        # Initialize a list to store total profile volume for this year

        for transect in transect_range:                                                                 # Process each transect in the range
            if transect in profile_df['Transect'].values:                                               # Filter for the specific transect in the profile volume data
                profile_volume = profile_df[profile_df['Transect'] == transect]['Volume'].values[0]
                year_profile_volume.append(profile_volume)

        if year_profile_volume:                                                                         # Store the year and calculated total profile volume
            yearly_profile_volumes.append({'Year': year, 'TotalVolume': year_profile_volume})
    else:
        print(f"Missing required columns in {profile_csv_file}. Skipping this year.")

if yearly_profile_volumes:                                                                              # Prepare data for plotting
    stats = []                                                                                          # Calculate mean, 25th, and 75th percentiles for each year
    for entry in yearly_profile_volumes:
        year = entry['Year']
        profile_volumes = entry['TotalVolume']
        mean_profile_volume = np.mean(profile_volumes)
        q25_profile_volume = np.percentile(profile_volumes, 25)
        q75_profile_volume = np.percentile(profile_volumes, 75)

        stats.append({'Year': year, 'Mean': mean_profile_volume, 'Q25': q25_profile_volume, 'Q75': q75_profile_volume})

    stats_df = pd.DataFrame(stats)                                                                      # Convert stats to a DataFrame
    stats_df['Mean_MA'] = stats_df['Mean'].rolling(window=3, min_periods=1).mean()                      # Apply moving average (3-year window) to the mean profile volume
    stats_df['Q25_MA'] = stats_df['Q25'].rolling(window=3, min_periods=1).mean()                        # Apply moving average (3-year window) to the 25th and 75th percentiles
    stats_df['Q75_MA'] = stats_df['Q75'].rolling(window=3, min_periods=1).mean()

    plt.figure(figsize=(12, 6))
    plt.plot(stats_df['Year'], stats_df['Mean_MA'], color='darkgreen', linewidth=2, label='Moving average (3 years)')
    plt.fill_between(stats_df['Year'], stats_df['Q25_MA'], stats_df['Q75_MA'], 
                     color='lightgreen', alpha=0.5, label='25th - 75th Percentile')
    plt.axvline(x=2009, color='black', linestyle='--', label='Shoal attachment')
    plt.xlabel('Year')
    plt.ylabel('Profile volume (m3/m)')
    plt.grid(True)
    plt.ylim(7000, 11000) 
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()
else:
    print("No valid data found for the specified years and transect range.")



#%% === 3.1. Plot average sandbar volume, profile volume and ratio ===

years_to_process = range(1992, 2024)                                                                    # Define years and transect range of interest
transect_range = range(40, 61)                                                                          # Define the transects depending on the Section of interest

yearly_ratios = []
yearly_profile_volumes = []
yearly_sandbar_volumes = []
sandbar_counts = []                                                                                     # Track sandbar counts
years_with_counts = []                                                                                  # rack years for which counts are valid

for year in years_to_process:
    print(f'Processing year: {year}')
    
    sandbar_csv_file = f'updated_sandbar_data_{year}.csv'                                               # Load the sandbar volume data
    try:
        sandbar_df = pd.read_csv(sandbar_csv_file)
    except FileNotFoundError:
        print(f"File not found: {sandbar_csv_file}. Skipping this year.")
        continue
    if 'transect' not in sandbar_df.columns or 'volume' not in sandbar_df.columns:                      # Check required columns
        print(f"Missing columns in {sandbar_csv_file}. Skipping this year.")
        continue

    profile_csv_file = f'{year}_profile_volume.csv'                                                     # Load the profile volume for the current year
    try:
        profile_df = pd.read_csv(profile_csv_file)
    except FileNotFoundError:
        print(f"File not found: {profile_csv_file}. Skipping this year.")
        continue

    if 'Transect' not in profile_df.columns or 'Volume' not in profile_df.columns:
        print(f"Missing columns in {profile_csv_file}. Skipping this year.")
        continue

    year_ratios = []                                                                                    # Initialize lists for this year's data
    year_profile_volumes = []
    year_sandbar_volumes = []
    year_sandbar_count = 0                                                                              # Initialize counter

    for transect in transect_range:
        transect_data = sandbar_df[sandbar_df['transect'] == transect]
        total_sandbar_volume = transect_data['volume'].sum()

        if transect in profile_df['Transect'].values:
            profile_volume = profile_df[profile_df['Transect'] == transect]['Volume'].values[0]
        else:
            continue

        if profile_volume != 0:
            volume_ratio = total_sandbar_volume / profile_volume
            if volume_ratio <= 1:
                year_ratios.append(volume_ratio)
                year_profile_volumes.append(profile_volume)
                year_sandbar_volumes.append(total_sandbar_volume)
                year_sandbar_count += 1                                                                 # Count this valid transect

    if year_ratios:                                                                                     # Store the year and calculated data
        yearly_ratios.append({'Year': [year] * len(year_ratios), 'Ratios': year_ratios})
        yearly_profile_volumes.append({'Year': [year] * len(year_profile_volumes), 'ProfileVolumes': year_profile_volumes})
        yearly_sandbar_volumes.append({'Year': [year] * len(year_sandbar_volumes), 'SandbarVolumes': year_sandbar_volumes})

        sandbar_counts.append(year_sandbar_count)                                                       # Store count
        years_with_counts.append(year)                                                                  # Store corresponding year

if yearly_ratios:
    sandbar_volumes = []
    profile_volumes = []
    ratios = []
    years = []

    for entry in yearly_sandbar_volumes:
        sandbar_volumes.extend(entry['SandbarVolumes'])
        years.extend(entry['Year'])
    for entry in yearly_profile_volumes:
        profile_volumes.extend(entry['ProfileVolumes'])
    for entry in yearly_ratios:
        ratios.extend(entry['Ratios'])

    df = pd.DataFrame({
        'Year': years,
        'SandbarVolume': sandbar_volumes,
        'ProfileVolume': profile_volumes,
        'Ratio': ratios
    })

    sandbar_count_df = pd.DataFrame({                                                                   # Safe creation of count DataFrame
        'Year': years_with_counts,
        'SandbarCount': sandbar_counts
    })

    stats = []                                                                                          # Calculate stats
    for year in df['Year'].unique():
        year_data = df[df['Year'] == year]
        stats.append({
            'Year': year,
            'SandbarMean': year_data['SandbarVolume'].mean(),
            'SandbarQ25': np.percentile(year_data['SandbarVolume'], 25),
            'SandbarQ75': np.percentile(year_data['SandbarVolume'], 75),
            'ProfileMean': year_data['ProfileVolume'].mean(),
            'ProfileQ25': np.percentile(year_data['ProfileVolume'], 25),
            'ProfileQ75': np.percentile(year_data['ProfileVolume'], 75),
            'RatioMean': year_data['Ratio'].mean(),
            'RatioQ25': np.percentile(year_data['Ratio'], 25),
            'RatioQ75': np.percentile(year_data['Ratio'], 75)
        })

    stats_df = pd.DataFrame(stats)

    for var in ['SandbarMean', 'SandbarQ25', 'SandbarQ75',                                              # Rolling averages
                'ProfileMean', 'ProfileQ25', 'ProfileQ75',
                'RatioMean', 'RatioQ25', 'RatioQ75']:
        stats_df[f'{var}_MA'] = stats_df[var].rolling(window=3, min_periods=1).mean()

    # Plotting
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    axs[0].plot(stats_df['Year'], stats_df['SandbarMean_MA'], color='darkblue', label='Mean Sandbar Volume (MA)')
    axs[0].fill_between(stats_df['Year'], stats_df['SandbarQ25_MA'], stats_df['SandbarQ75_MA'], color='lightblue', alpha=0.5)
    axs[0].set_xlabel('Year')
    axs[0].set_ylabel('Sandbar Volume')
    axs[0].grid(True)
    axs[0].set_ylim(0, 1200)
    axs[0].legend()

    axs[1].plot(stats_df['Year'], stats_df['ProfileMean_MA'], color='darkgreen', label='Mean Profile Volume (MA)')
    axs[1].fill_between(stats_df['Year'], stats_df['ProfileQ25_MA'], stats_df['ProfileQ75_MA'], color='lightgreen', alpha=0.5)
    axs[1].set_xlabel('Year')
    axs[1].set_ylabel('Profile Volume')
    axs[1].grid(True)
    axs[1].set_ylim(6000, 16000)
    axs[1].legend()

    axs[2].plot(stats_df['Year'], stats_df['RatioMean_MA'], color='darkred', label='Mean Ratio (MA)')
    axs[2].fill_between(stats_df['Year'], stats_df['RatioQ25_MA'], stats_df['RatioQ75_MA'], color='lightcoral', alpha=0.5)
    axs[2].set_xlabel('Year')
    axs[2].set_ylabel('Vsandbar / Vprofile Ratio')
    axs[2].grid(True)
    axs[2].set_ylim(0, 0.3)
    axs[2].legend()

    fig.suptitle(f"Transects {transect_range.start}-{transect_range.stop - 1}", fontsize=16)
    plt.tight_layout()
    plt.show()

else:
    print("No valid data found for the specified years and transect range.")



#%% === 3.2. Plot n. sandbars, average sandbar volume, profile volume and ratio ===

if yearly_ratios:
    sandbar_volumes = []
    profile_volumes = []
    ratios = []
    years = []

    for entry in yearly_sandbar_volumes:                                                                # Extract flat lists from yearly data
        sandbar_volumes.extend(entry['SandbarVolumes'])
        years.extend(entry['Year'])
    for entry in yearly_profile_volumes:
        profile_volumes.extend(entry['ProfileVolumes'])
    for entry in yearly_ratios:
        ratios.extend(entry['Ratios'])

    sandbar_counts = []                                                                                 # Track years where sandbar counts are valid
    valid_sandbar_years = []

    for year in years_to_process:
        try:
            sandbar_data_for_year = pd.read_csv(f'updated_sandbar_data_{year}.csv')
            if 'sandbar_label' in sandbar_data_for_year.columns:
                filtered_sandbar_data = sandbar_data_for_year[
                    sandbar_data_for_year['transect'].between(transect_range.start, transect_range.stop - 1)
                ]
                num_sandbars = filtered_sandbar_data['sandbar_label'].nunique()
                sandbar_counts.append(num_sandbars)
                valid_sandbar_years.append(year)                                                        # Store only valid years here
        except FileNotFoundError:
            print(f"File not found for year {year}, skipping this year.")
            continue

    sandbar_count_df = pd.DataFrame({                                                                   # Create DataFrame with only valid years and counts
        'Year': valid_sandbar_years,
        'SandbarCount': sandbar_counts
    })

    df = pd.DataFrame({                                                                                 # Create main data DataFrame
        'Year': years,
        'SandbarVolume': sandbar_volumes,
        'ProfileVolume': profile_volumes,
        'Ratio': ratios,
    })

    stats = []                                                                                          # Calculate yearly stats as before
    for year in df['Year'].unique():
        year_data = df[df['Year'] == year]
        stats.append({
            'Year': year,
            'SandbarMean': year_data['SandbarVolume'].mean(),
            'SandbarQ25': np.percentile(year_data['SandbarVolume'], 25),
            'SandbarQ75': np.percentile(year_data['SandbarVolume'], 75),
            'ProfileMean': year_data['ProfileVolume'].mean(),
            'ProfileQ25': np.percentile(year_data['ProfileVolume'], 25),
            'ProfileQ75': np.percentile(year_data['ProfileVolume'], 75),
            'RatioMean': year_data['Ratio'].mean(),
            'RatioQ25': np.percentile(year_data['Ratio'], 25),
            'RatioQ75': np.percentile(year_data['Ratio'], 75)
        })

    stats_df = pd.DataFrame(stats)

    for col in ['SandbarMean', 'SandbarQ25', 'SandbarQ75',                                              # Moving averages
                'ProfileMean', 'ProfileQ25', 'ProfileQ75',
                'RatioMean', 'RatioQ25', 'RatioQ75']:
        stats_df[f'{col}_MA'] = stats_df[col].rolling(window=3, min_periods=1).mean()

    # Plotting
    fig, axs = plt.subplots(1, 4, figsize=(24, 6))

    axs[0].plot(sandbar_count_df['Year'], sandbar_count_df['SandbarCount'], color='purple', label='Number of Sandbars')
    axs[0].set_title('N. Sandbars', fontsize=18)
    axs[0].grid(True)
    axs[0].set_ylim(0, max(sandbar_count_df['SandbarCount']) + 1 if len(sandbar_count_df) > 0 else 8)
    axs[0].set_ylim(0, 8)
    axs[0].xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    axs[0].yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    axs[0].tick_params(axis='both', which='major', labelsize=14)

    axs[1].plot(stats_df['Year'], stats_df['SandbarMean_MA'], color='darkblue', label='Mean Sandbar Volume (MA)')
    axs[1].fill_between(stats_df['Year'], stats_df['SandbarQ25_MA'], stats_df['SandbarQ75_MA'], color='lightblue', alpha=0.5, label='25th-75th Percentile')
    axs[1].set_title('Sandbar volume (m3/m)', fontsize=18)
    axs[1].grid(True)
    axs[1].set_ylim(0, 1200)
    axs[1].legend(fontsize=14)
    axs[1].xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    axs[1].yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    axs[1].tick_params(axis='both', which='major', labelsize=14)

    axs[2].plot(stats_df['Year'], stats_df['ProfileMean_MA'], color='darkgreen', label='Mean Profile Volume (MA)')
    axs[2].fill_between(stats_df['Year'], stats_df['ProfileQ25_MA'], stats_df['ProfileQ75_MA'], color='lightgreen', alpha=0.5, label='25th-75th Percentile')
    axs[2].set_title('Profile volume (m3/m)', fontsize=18)
    axs[2].grid(True)
    axs[2].set_ylim(6000, 16000)
    axs[2].legend(fontsize=14)
    axs[2].xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    axs[2].yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    axs[2].tick_params(axis='both', which='major', labelsize=14)

    axs[3].plot(stats_df['Year'], stats_df['RatioMean_MA'], color='darkred', label='Mean Ratio (MA)')
    axs[3].fill_between(stats_df['Year'], stats_df['RatioQ25_MA'], stats_df['RatioQ75_MA'], color='lightcoral', alpha=0.5, label='25th-75th Percentile')
    axs[3].set_title('V sandbar/ V profile ratio', fontsize=18)
    axs[3].grid(True)
    axs[3].set_ylim(0, 0.3)
    axs[3].legend(fontsize=14)
    axs[3].xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    axs[3].yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    axs[3].tick_params(axis='both', which='major', labelsize=14)

    plt.tight_layout()
    plt.show()

else:
    print("No valid data found for the specified years and transect range.")

# %%
