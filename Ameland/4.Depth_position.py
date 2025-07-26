# -*- title: Alongshore slope variability analysis and depth-position calculation -*-
"""
Created on Thursday 1 March 2025

@author: Raquelgonfe

Description:
Calculation and plotting of sandbar volume, profile volume, ratio between these two volumes, and number of sandbars.

The script includes the following components:
(References to the corresponding thesis figures.)

1. Plot altitude vs cross-shore distance profile and calculates depth-cross-shore conversion saving it into a csv file named "depth_cross_shore_T{transect_index}.csv"
2. Plotting the depth-position  and generating csv file with depth-position column added

Note:
Depth-position parameter combines both the depth (vertical distance from the 0-meter altitude) and the cross-shore distance, representing the depth contour line at a specific location along the study area. 
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



#%% === 1. Plot altitude vs cross-shore distance profile ===

nc_file_path = os.path.expanduser("~/Downloads/transect_r20240405.nc")                                                                      # Load the dataset and filter data for Ameland (areacode = 3)
xrds = xr.open_dataset(nc_file_path)
ameland_data = xrds.sel(alongshore=xrds['areacode'] == 3)

xRSP = ameland_data['cross_shore'][:]                                                                                                       # Cross-shore coordinates
time = ameland_data['time']
lat = ameland_data['lat']                                                                                                                   # Latitude for each cross-shore point
lon = ameland_data['lon']                                                                                                                   # Longitude for each cross-shore point

depth_min, depth_max = -8, 0                                                                                                                # Depth bounds for profile analysis (invert for proper range)
transect1,transect2 = 30,121

years_to_process = range(1985, 2024)                                                                                                        # From 2000 to 2015 inclusive
window_length = 11                                                                                                                          # Define the window length and polynomial order for smoothing
polyorder = 2                                                                                                                               # Polynomial order for smoothing

stable_zone_distance = 350                                                                                                                  # Stable zone threshold in meters
threshold_percent = 0.075                                                                                                                   # Threshold for second derivative

for transect_index in range(transect1,transect2):                                                                                           # Loop through transects
    print(f'Processing Transect {transect_index}...')
    
    all_profiles = []                                                                                                                       # Collect altitude profiles for the current transect
    all_xRSP = []
    
    for target_year in years_to_process:
        year_index = np.where(ameland_data['time'].dt.year == target_year)[0]
        
        altitude = ameland_data['altitude'].isel(time=year_index[0], alongshore=transect_index).values                                      # Extract altitude data for the current transect and year
        xRSP_clean = xRSP.values  # Ensure NumPy array
        valid_indices = ~np.isnan(altitude)
        altitude_clean = altitude[valid_indices]
        xRSP_clean = xRSP_clean[valid_indices]

        if len(altitude_clean) < window_length:
            print(f"Insufficient data for transect {transect_index + 1}, year {target_year}. Skipping.")
            continue

        smoothed_altitude = savgol_filter(altitude_clean, window_length, polyorder)                                                         # Smooth the profile

        depth_mask = (smoothed_altitude >= depth_min) & (smoothed_altitude <= depth_max)                                                    # Select only the depth range of interest (between -8 and 0 meters)
        if np.sum(depth_mask) == 0:
            print(f"Transect {transect_index + 1} does not span the required depth range. Skipping.")
            continue

        xRSP_depth = xRSP_clean[depth_mask]                                                                                                 # Filter data for depth range
        altitude_depth = smoothed_altitude[depth_mask]
        
        if len(all_xRSP) == 0:                                                                                                              # Interpolate to align the profiles (using a common xRSP range for all years)
            all_xRSP = xRSP_depth                                                                                                           # Set the common xRSP values (using the first valid profile)
        else:
            interp_func = interp1d(xRSP_depth, altitude_depth, kind='linear', fill_value='extrapolate')                                     # Interpolate current year's profile to match the common xRSP range
            altitude_depth_interpolated = interp_func(all_xRSP)
            all_profiles.append(altitude_depth_interpolated)
            continue

        all_profiles.append(altitude_depth)                                                                                                 # For the first profile, just store it directly

    if all_profiles:                                                                                                                        # Calculate mean profile for the current transect across 2000-2015
        all_profiles = np.array(all_profiles)
        mean_profile = np.mean(all_profiles, axis=0)
    
    if all_profiles is None or len(all_profiles) == 0:                                                                                      # Check if all_profiles is empty or None
        print(f"No valid data for Transect {transect_index}")
        continue
    
    all_profiles = np.array(all_profiles)                                                                                                   # Convert to numpy array and calculate the mean profile
    mean_profile = np.mean(all_profiles, axis=0)
    
    window_length = 11                                                                                                                      #  window length
    polyorder = 2                                                                                                                           # Polynomial order for smoothing
    smoothed_mean_profile = savgol_filter(mean_profile, window_length, polyorder)
    
    if len(all_xRSP) != len(smoothed_mean_profile):                                                                                         # Ensure all_xRSP matches the length of smoothed_mean_profile
        print(f"Dimension mismatch: all_xRSP ({len(all_xRSP)}) vs smoothed_mean_profile ({len(smoothed_mean_profile)})")
        continue
    
    first_derivative = np.gradient(smoothed_mean_profile)                                                                                   # Calculate first and second derivatives for sandbar detection
    second_derivative = np.gradient(first_derivative)
       
    crests = []                                                                                                                             # Detect crests and troughs
    troughs = []
    for i in range(1, len(first_derivative) - 1):
        if i >= len(all_xRSP):                                                                                                              # Ensure indices are valid for all_xRSP
            break                                                                                                                           # Stop if index exceeds bounds of all_xRSP
        
        if first_derivative[i] < 0 and first_derivative[i - 1] >= 0 and second_derivative[i] < 0:                                           # Crest detection
            crests.append((all_xRSP[i], smoothed_mean_profile[i]))
        
        if first_derivative[i] > 0 and first_derivative[i - 1] <= 0:                                                                        # Trough detection
            troughs.append((all_xRSP[i], smoothed_mean_profile[i]))
    
    sandbar_params = []                                                                                                                     # Calculate sandbar parameters and identify left/right troughs
    for crest in crests:
        crest_position, crest_height = crest
        left_trough = next((t for t in reversed(troughs) if t[0] < crest_position), None)
        right_trough = next((t for t in troughs if t[0] > crest_position), None)
        
        if left_trough and right_trough:                                                                                                    # Width, height, and depth calculations
            troughline = right_trough[0] - left_trough[0]
            height = crest_height - ((left_trough[1] + right_trough[1]) / 2)
            depth = crest_height
           
            
            # Add parameters to the list if valid
            if height > 0:                                                                                                                  # Threshold for a valid sandbar
                sandbar_params.append({
                    'crest_position': crest_position,
                    'crest_height': crest_height,
                    'left_trough': left_trough,
                    'right_trough': right_trough,
                    'width': troughline,
                    'height': height,
                    'depth': depth
                })
    
    plt.figure(figsize=(12, 8))
    plt.plot(all_xRSP, mean_profile, label='Mean Profile', color='black', alpha=0.7)
    plt.axhline(0, color='gray', linestyle='--', label='Sea Level')
    
    for param in sandbar_params:
        plt.scatter(param['crest_position'], param['crest_height'], color='red', label='Crest')
        plt.text(param['crest_position'], param['crest_height'] + 0.2, f"{param['crest_position']:.1f}", color='red', fontsize=9)
        
        left_trough, right_trough = param['left_trough'], param['right_trough']
        plt.scatter(left_trough[0], left_trough[1], color='green', label='Left Trough')
        plt.scatter(right_trough[0], right_trough[1], color='purple', label='Right Trough')
        
        plt.plot(
            [left_trough[0], param['crest_position'], right_trough[0]],
            [left_trough[1], param['crest_height'], right_trough[1]],
            color='orange', linestyle='--', alpha=0.6
        )
    
    plt.title(f'Sandbar Detection for Transect {transect_index}')
    plt.xlabel('Cross-shore Distance (m)')
    plt.ylabel('Altitude (m)')
    plt.legend()
    plt.grid(True)
    plt.show()

    second_derivative_min, second_derivative_max = np.min(second_derivative), np.max(second_derivative)                                     # Calculate the threshold for stable zones
    second_derivative_range = second_derivative_max - second_derivative_min
    threshold = threshold_percent * second_derivative_range

    stable_zones = []                                                                                                                       # Identify stable zones
    zone_start = None
    for i in range(len(all_xRSP) - 1):
        if abs(second_derivative[i]) < threshold:
            if zone_start is None:
                zone_start = all_xRSP[i]
        else:
            if zone_start is not None:
                zone_length = all_xRSP[i] - zone_start
                if zone_length >= stable_zone_distance:
                    stable_zones.append((zone_start, all_xRSP[i]))
                zone_start = None

    if zone_start is not None and (all_xRSP[-1] - zone_start) >= stable_zone_distance:                                                      # Handle the last zone if it ends at the edge
        stable_zones.append((zone_start, all_xRSP[-1]))

    offshore_boundaries = []                                                                                                                # Find offshore boundaries
    for start, end in stable_zones:
        start_idx = np.where(all_xRSP >= start)[0][0]
        end_idx = np.where(all_xRSP <= end)[0][-1]
        for i in range(start_idx, end_idx):
            if second_derivative[i] < 0 and second_derivative[i + 1] >= 0:
                zero_crossing = all_xRSP[i + 1]
                offshore_boundary = zero_crossing + 250                                                                                     # Adjusted buffer
                offshore_boundaries.append((zero_crossing, offshore_boundary))
                break

    fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    axs[0].plot(all_xRSP, smoothed_mean_profile, color='blue', label='Smoothed Mean Profile')
    axs[0].set_xlim(0, 2000)  
    axs[0].set_ylim(-10, 2.5)
    axs[0].axhline(0, color='black', linestyle='--', linewidth=0.8)  
    axs[0].set_ylabel("Altitude (m)")
    for zero_crossing, offshore_boundary in offshore_boundaries:
        axs[0].axvline(zero_crossing, color='gray', linestyle=':', label='Zero Crossing')
        axs[0].axvline(offshore_boundary, color='black', linestyle=':', label='Offshore Boundary')
    axs[0].legend()

    axs[1].plot(all_xRSP, first_derivative, color='green', label="First Derivative")
    axs[1].set_xlim(0, 2000)
    axs[1].set_ylim(-0.4, 0.4)
    axs[1].axhline(0, color='black', linestyle='--', linewidth=0.8)  
    axs[1].set_ylabel("dz/dx (m/m)")
    axs[1].legend()

    axs[2].plot(all_xRSP, second_derivative, color='red', label="Second Derivative")
    for start, end in stable_zones:
        axs[2].fill_betweenx([-0.1, 0.1], start, end, color='yellow', alpha=0.3, label="Stable Zone" if start == stable_zones[0][0] else "")
    for zero_crossing, offshore_boundary in offshore_boundaries:
        axs[2].axvline(zero_crossing, color='gray', linestyle=':', label='Zero Crossing' if zero_crossing == offshore_boundaries[0][0] else "")
        axs[2].axvline(offshore_boundary, color='black', linestyle=':', label='Offshore Boundary' if offshore_boundary == offshore_boundaries[0][1] else "")
    axs[2].set_xlim(0, 2000)
    axs[2].set_ylim(-0.1, 0.1)
    axs[2].axhline(0, color='black', linestyle='--', linewidth=0.8)
    axs[2].set_xlabel("Cross-shore Distance (m)")
    axs[2].set_ylabel("d²z/dx² (m²/m²)")
    axs[2].legend()

    plt.tight_layout()
    plt.show()

    print(crest_position)
    print(left_trough)
    print(right_trough)
    print(offshore_boundary)


    # === Extract left trough and offshore boundary positions ===
    left_trough_position = left_trough[0]                                                                                                   # x-coordinate of the left trough
    offshore_boundary_position = offshore_boundary                                                                                          # x-coordinate of the offshore boundary

    left_trough_idx = np.abs(all_xRSP - left_trough_position).argmin()                                                                      # Find offshore boundaries
    offshore_boundary_idx = np.abs(all_xRSP - offshore_boundary_position).argmin()

    # Create the replacement straight line between left trough and offshore boundary
    left_trough_altitude = mean_profile[left_trough_idx]                                                                                    # Extract the corresponding altitude values at these points
    offshore_boundary_altitude = mean_profile[offshore_boundary_idx]

    x_values_between = all_xRSP[left_trough_idx:offshore_boundary_idx + 1]                                                                  # Generate a linear interpolation between the two points
    y_values_between = np.linspace(left_trough_altitude, offshore_boundary_altitude, len(x_values_between))

    mean_profile_modified = mean_profile.copy()                                                                                             # Replace the section of mean_profile with the interpolated straight line
    mean_profile_modified[left_trough_idx:offshore_boundary_idx + 1] = y_values_between

    plt.figure(figsize=(12, 6))

    plt.plot(all_xRSP, mean_profile_modified, label='Modified mean profile', color='darkblue', linewidth=2)
    plt.plot(all_xRSP, mean_profile, label='Original mean profile', color='black', linestyle='--', linewidth=2)
    plt.title(f'Transect {transect_index}', fontsize=22)
    plt.xlabel('Cross-shore Distance (m)', fontsize=20)
    plt.ylabel('Altitude (m)', fontsize=20)

    plt.ylim(-8.2, 0)
    plt.xlim(0, 2000)

    plt.tick_params(axis='x', labelsize=18)
    plt.tick_params(axis='y', labelsize=18)
    plt.legend(fontsize=18)
    plt.grid(True, linestyle='--', linewidth=0.8)
    plt.show()

    print(f'Left Trough: {left_trough_position}, Offshore Boundary: {offshore_boundary_position}')


    # === Depth-cross-shore conversion csv file for depth-position calculation ===

    output_csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'depth_cross_shore_T{transect_index}.csv')                  # Define the output CSV file path (same folder as input file)

    depth_cross_shore_data = []                                                                                                             # Create an empty list to hold the data
        
    all_profiles = np.array(all_profiles)                                                                                                   # Convert to numpy array and calculate the mean profile
    mean_profile = np.mean(all_profiles, axis=0)
        
    window_length = 11                                                                                                                      # window length
    polyorder = 2                                                                                                                           # Polynomial order for smoothing
    smoothed_mean_profile = savgol_filter(mean_profile, window_length, polyorder)
        
    if len(all_xRSP) != len(smoothed_mean_profile):                                                                                         # Ensure all_xRSP matches the length of smoothed_mean_profile
        print(f"Dimension mismatch: all_xRSP ({len(all_xRSP)}) vs smoothed_mean_profile ({len(smoothed_mean_profile)})")
        continue
        
    first_derivative = np.gradient(smoothed_mean_profile)                                                                                   # Calculate first and second derivatives for sandbar detection
    second_derivative = np.gradient(first_derivative)
        
    crests = []                                                                                                                             # Detect crests and troughs
    troughs = []
    for i in range(1, len(first_derivative) - 1):
        if first_derivative[i] < 0 and first_derivative[i - 1] >= 0 and second_derivative[i] < 0:                                           # Crest detection
            crests.append((all_xRSP[i], smoothed_mean_profile[i]))
            
        if first_derivative[i] > 0 and first_derivative[i - 1] <= 0:                                                                        # Trough detection
            troughs.append((all_xRSP[i], smoothed_mean_profile[i]))
        
    sandbar_params = []                                                                                                                     # Calculate sandbar parameters and identify left/right troughs
    for crest in crests:
        crest_position, crest_height = crest
        left_trough = next((t for t in reversed(troughs) if t[0] < crest_position), None)
        right_trough = next((t for t in troughs if t[0] > crest_position), None)
            
        if left_trough and right_trough:
            troughline = right_trough[0] - left_trough[0]
            height = crest_height - ((left_trough[1] + right_trough[1]) / 2)
            depth = crest_height
                
            if height > 0:                                                                                                                  # Threshold for a valid sandbar
                sandbar_params.append({
                    'crest_position': crest_position,
                    'crest_height': crest_height,
                    'left_trough': left_trough,
                    'right_trough': right_trough,
                    'width': troughline,
                    'height': height,
                    'depth': depth
                })
        
    left_trough_position = left_trough[0]                                                                                                   # x-coordinate of the left trough
    offshore_boundary_position = offshore_boundary                                                                                          # x-coordinate of the offshore boundary

    left_trough_idx = np.abs(all_xRSP - left_trough_position).argmin()                                                                      # Find the indices of these points in the mean profile
    offshore_boundary_idx = np.abs(all_xRSP - offshore_boundary_position).argmin()

    # Create the replacement straight line between left trough and offshore boundary
    left_trough_altitude = smoothed_mean_profile[left_trough_idx]                                                                           # Extract the corresponding altitude values at these points
    offshore_boundary_altitude = smoothed_mean_profile[offshore_boundary_idx]

    x_values_between = all_xRSP[left_trough_idx:offshore_boundary_idx + 1]                                                                  # Generate a linear interpolation between the two points
    y_values_between = np.linspace(left_trough_altitude, offshore_boundary_altitude, len(x_values_between))

    mean_profile_modified = smoothed_mean_profile.copy()                                                                                    # Replace the section of smoothed_mean_profile with the interpolated straight line
    mean_profile_modified[left_trough_idx:offshore_boundary_idx + 1] = y_values_between

    for i in range(len(mean_profile_modified)):                                                                                             # Now extract the depths (from mean_profile_modified) and the corresponding cross-shore distances
        depth = mean_profile_modified[i]
        cross_shore_distance = all_xRSP[i]
            
        depth_cross_shore_data.append({                                                                                                     # Append to the data list
            'Transect': transect_index,
            'Depth (m)': depth,
            'Cross-Shore Distance (m)': cross_shore_distance
        })

    depth_cross_shore_df = pd.DataFrame(depth_cross_shore_data)                                                                             # Convert the list of dictionaries to a DataFrame

    depth_cross_shore_df.to_csv(output_csv_path, index=False)                                                                               # Save the DataFrame to a CSV file in the same folder as the input file

    print(f"CSV file saved to: {output_csv_path}")



#%% === 2. Using the depth-position dataset and generating csv file with depth-position column added ===

sections = {                                                                                                                                # Define sections as lists of transects
    'Section_1': list(range(30, 61)),                                                                                                       # Transects 30-60
    'Section_2': list(range(60, 111)),                                                                                                      # Transects 60-110
    'Section_3': list(range(110, 121))                                                                                                      # Transects 110-120
}

color_map = {
    '1': 'orange', '2': 'salmon', '3': 'peru', '4': 'slateblue', '5': 'teal', '6': 'gray', '4a': 'cadetblue',
    '7': 'royalblue', '5a': 'saddlebrown', '8': 'darkorange', '9': 'forestgreen', '7a': 'dodgerblue', '7b': 'slateblue', '8': 'lightcoral',
    '8a': 'orangered', '10': 'indigo', '10a': 'mediumvioletred', 'N11': 'chocolate', 'N15': 'darkred', '11a': 'seagreen',
    '11': 'cadetblue', '12': 'peru', '13': 'olivedrab', 'N21': 'rosybrown'                                                                  # Define labels and color map
}
years_to_process = range(1965, 2024)                                                                                                        # Covers 1965 to 2024

section_data = {
    'Section_1': [],
    'Section_2': [],
    'Section_3': []
}

def load_depth_data(transect):                                                                                                              # Function to load depth data for each transect
    depth_file = f'depth_cross_shore_T{transect}.csv'                                                                                       # Construct the filename for the depth data CSV based on the transect number
    try:
        depth_df = pd.read_csv(depth_file)
        depth_df = depth_df.rename(columns={"Depth (m)": "depth", "Cross-Shore Distance (m)": "cross_shore"})                               # Rename columns for easier reference
        return depth_df
    except FileNotFoundError:
        print(f"Depth data file not found: {depth_file}. Skipping this transect.")
        return None

for year in years_to_process:                                                                                                               # Process each year's data
    csv_file = f'updated_sandbar_data_{year}.csv'
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"File not found: {csv_file}. Skipping this year.")
        continue

    df['sandbar_label'] = df['sandbar_label'].astype(str)                                                                                   # Ensure 'sandbar_label' is treated as string to match color_map

    for section_name, transects in sections.items():
        section_df = df[df['transect'].isin(transects)].copy()

        section_df['depth_position'] = None                                                                                                 # Initialize depth_position column explicitly. Ensure the column exists

        for transect in transects:                                                                                                          # For each transect in the section, convert the crest position to depth
            depth_data = load_depth_data(transect)                                                                                          # Load the corresponding depth data for the transect
            if depth_data is not None:
                for idx, row in section_df[section_df['transect'] == transect].iterrows():                                                  # Find the closest depth value for the given crest position (cross-shore)
                    crest_position = row['crest_position']                                                                                  # This is the cross-shore position
                    closest_depth = depth_data.iloc[(depth_data['cross_shore'] - crest_position).abs().argmin()]['depth']                   # Find the closest depth value at the cross-shore position
                    section_df.at[idx, 'depth_position'] = closest_depth                                                                    # Assign the closest depth to the 'depth_position' column

        if section_df['depth_position'].isnull().any():                                                                                     # Check if the 'depth_position' column has been properly analysed
            print(f"Warning: There are still missing 'depth_position' values in section {section_name} for year {year}.")
        else:
            print(f"Depth position for section {section_name} in year {year} has been analysed.")

        # Save to new file without altering the original columns
        updated_filename = f'updated_sandbar_data_with_depth_{year}.csv'
        df.to_csv(updated_filename, index=False)
        print(f"Saved: {updated_filename}")

        sandbar_groups = section_df.groupby('sandbar_label').agg({                                                                          # Group the data by sandbar label and calculate mean values
            'height': 'mean',
            'width2': 'mean',
            'width3': 'mean',
            'depth': 'mean',
            'cross_shore': 'mean',
            'crest_position': 'mean',   
            'depth_position': 'mean',                                                                                                       # Now using the corrected depth position
            'volume': 'mean'
        }).reset_index()

        sandbar_groups['Year'] = year
        section_data[section_name].append(sandbar_groups)

section_1_df = pd.concat(section_data['Section_1'], ignore_index=True)                                                                      # When combining all data and calculating, ensure sandbar_label remains as strings
section_2_df = pd.concat(section_data['Section_2'], ignore_index=True)
section_3_df = pd.concat(section_data['Section_3'], ignore_index=True)

section_1_df['sandbar_label'] = section_1_df['sandbar_label'].astype(str)                                                                   # Ensure 'sandbar_label' is treated as strings for plot functions
section_2_df['sandbar_label'] = section_2_df['sandbar_label'].astype(str)
section_3_df['sandbar_label'] = section_3_df['sandbar_label'].astype(str)

def calculate_avg_and_std(section_data, parameter, absolute=False):                                                                         # Function to calculate average and standard deviation per sandbar label
    combined_df = pd.concat(section_data.values(), ignore_index=True)
    if absolute and parameter == 'height':
        combined_df[parameter] = combined_df[parameter].abs()                                                                               # Take the absolute value of the height before calculating mean and std
    
    grouped = combined_df.groupby(['Year', 'sandbar_label']).agg(
        mean_value=(parameter, 'mean'),
        std_value=(parameter, 'std')
    ).reset_index()
    return grouped

def plot_avg_parameter_with_std(grouped_data, parameter, ylabel):                                                                           # Function to plot average parameter with standard deviation
    plt.figure(figsize=(14, 8))

    for label in grouped_data['sandbar_label'].unique():                                                                                    # Plot each sandbar label's mean and standard deviation
        subset = grouped_data[grouped_data['sandbar_label'] == label]
        color = color_map.get(label, '#000000')                                                                                           # Default to black if label is not in color_map

        marker = '*' if label.startswith('N') else 'o'                                                                                      # Use a star marker if the sandbar label starts with "N", otherwise use a circle
        markersize = 10 if marker == '*' else 6

        plt.errorbar(subset['Year'], subset['mean_value'], yerr=subset['std_value'], 
                     fmt=f'-{marker}', label=f'Sandbar {label}', color=color, capsize=5, markersize=markersize)

    plt.title(f'Average {parameter} per sandbar')
    plt.xlabel('Year')
    plt.ylabel(ylabel)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), ncol=2)
    plt.tight_layout()
    plt.show()

depth_position_data = calculate_avg_and_std({'Section_1': section_1_df, 'Section_2': section_2_df, 'Section_3': section_3_df}, 'depth_position')    # Calculate and plot the average and std for each parameter, including the corrected depth position
plot_avg_parameter_with_std(depth_position_data, 'depth_position', 'Depth Position (m)')



# %%
