# -*- title: Ameland dataset visualisation -*-
"""
Created on Thursday 1 March 2025

@author: Raquelgonfe

Description:
Visualisation of the csv (updated_sandbar_data_{year}.csv) data generated from 2.Dataset_extraction for Ameland and after manuual modification. 

The script includes the following components:
(References to the corresponding thesis figures.)

1.1. Example of crescentic index calculation with high-pass filtering
1.2. Calculate crescentic index for the whole dataset. Stores the output in a csv file
2.   Plot sandbar on depth-position instead of cross-shore (Figure 27)
3.   Plot crescentic index against time (Figure 28)
4.   Statistics test: Mann-Whitney U test for independent data and Wilcoxon signed-rank test for paired data (Figure 28)


Note:
There is no Jarkus data for 1972 and 1973.
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
import math
import matplotlib.colors as mcolors
import imageio
import seaborn as sns
from sklearn.linear_model import LinearRegression
import csv  
from scipy.stats import shapiro
from scipy.stats import mannwhitneyu
from scipy.stats import wilcoxon



#%% === 1.1. Example of crescentic index calculation with high-pass filtering ===

def calculate_crescentic_index(df, window_length=11, polyorder=1):                                                                              # Function to calculate the crescentic index (standard deviation) for each sandbar
    """
    Calculate the crescentic index (standard deviation) for each sandbar.
    
    Parameters:
    - df: DataFrame containing sandbar data
    - window_length: Length of the smoothing window for Savitzky-Golay filter (must be odd)
    - polyorder: Polynomial order for Savitzky-Golay filter
    
    Returns:
    - crescentic_indices: Dictionary with standard deviations of detrended positions for each sandbar label
    - trends: Detailed data about trends and detrended values for plotting
    """
    crescentic_indices = {}
    trends = {}

    for label in df['sandbar_label'].unique():
        subset = df[df['sandbar_label'] == label].sort_values(by='x')                                                                           # Sort by alongshore distance (x)

        if len(subset) > 1:                                                                                                                     # At least two points needed for detrending
            x_values = subset['x'].values
            y_values = subset['y'].values

            trend = savgol_filter(y_values, window_length=window_length, polyorder=polyorder)                                                   # Smooth the data to extract the long-wavelength trend
            detrended = y_values - trend                                                                                                        # Detrend the cross-shore positions

            std_detrended = np.std(detrended)                                                                                                   # Calculate the standard deviation of detrended positions
            crescentic_indices[label] = std_detrended
            trends[label] = {'x': x_values, 'y': y_values, 'trend': trend, 'detrended': detrended}
        else:
            crescentic_indices[label] = np.nan
            trends[label] = None

    return crescentic_indices, trends


years_to_visualize = range(2020, 2021)                                                                                                          # Specify the range of years to visualize

output_folder = 'output'                                                                                                                        # Replace with your output folder
os.makedirs(output_folder, exist_ok=True)

color_map = {                                                                                                                                   # Define the labels and color map
    '1': 'orange', '2': 'salmon', '2a': 'darksalmon', '3': 'peru', '4': 'slateblue', '5': 'teal', '6': 'gray', '6a': 'silver', '4a': 'cadetblue',
    '7': 'royalblue', '5a': 'saddlebrown', '8': 'darkorange', '9': 'forestgreen', '7a': 'dodgerblue',
    '7b': 'slateblue', '7c': 'skyblue', '7d': 'paleturquoise','8a': 'orangered', '8b': 'coral', '10': 'indigo', '10a': 'mediumvioletred', '10b': 'orchid', 
    'N11': 'chocolate', 'N15': 'darkred', '11a': 'seagreen', '11': 'cadetblue', '11b': 'thistle', '12': 'peru',
    '13': 'olivedrab', 'N21': 'rosybrown'
}

for year in years_to_visualize:
    csv_file = f'updated_sandbar_data_{year}.csv'

    if not os.path.exists(csv_file):
        print(f"Skipping year {year}: input file {csv_file} is missing.")
        continue

    crests_df = pd.read_csv(csv_file)                                                                                                           # Load the CSV file
    crests_df['sandbar_label'] = crests_df['sandbar_label'].astype(str)

    crests_df['x'] = (crests_df['transect'] - 40) * 200 / 1000                                                                                  # Alongshore distance in kilometers
    crests_df['y'] = crests_df['crest_position']                                                                                                # Cross-shore distance in meters

    crescentic_indices, trends = calculate_crescentic_index(crests_df)                                                                          # Calculate crescentic indices with high-pass filtering

    print(f"Crescentic indices for year {year}:")
    for label, index in crescentic_indices.items():
        print(f"  Sandbar {label}: {index:.4f}" if not np.isnan(index) else f"  Sandbar {label}: Insufficient data")

    for label, data in trends.items():                                                                                                          # Plot sandbar crests and trends
        if data is None:
            continue

        x = data['x']
        y = data['y']
        trend = data['trend']
        detrended = data['detrended']
        color = color_map.get(label, 'black')                                                                                                   # Use color from the color map

        fig, axes = plt.subplots(2, 1, figsize=(15, 10), sharex=True)                                                                           # Create subplots

        # Subplot 1: Original positions with trend
        axes[0].scatter(x, y, color=color, label='Original positions')
        axes[0].plot(x, trend, color=color, linestyle='--', label='Trend line (smoothed)')
        axes[0].set_ylabel('Cross-shore distance (m)', fontsize=18)                                                                             # Increased font size
        axes[0].set_title(f'Sandbar {label} - Original positions and trend', fontsize=20)                                                       # Increased font size
        axes[0].legend(fontsize=16)                                                                                                             # Increased font size
        axes[0].grid(True)

        # Subplot 2: Detrended positions
        axes[1].scatter(x, detrended, color=color, label='Detrended position')
        axes[1].axhline(0, color='red', linestyle='--', label='Zero mean line')
        axes[1].set_xlabel('Alongshore distance (km)', fontsize=18)                                                                             # Increased font size
        axes[1].set_ylabel('Detrended distance (m)', fontsize=18)                                                                               # Increased font size
        axes[1].set_title(f'Sandbar {label} - Detrended positions', fontsize=20)                                                                # Increased font size
        axes[1].legend(fontsize=16)                                                                                                             # Increased font size
        axes[1].grid(True)

        for ax in axes:
            ax.tick_params(axis='both', labelsize=14)                                                                                           # Set tick label size

        output_image_path = os.path.join(output_folder, f"sandbar_{year}_label_{label}.png")
        plt.tight_layout()
        plt.savefig(output_image_path)
        plt.show()
        plt.close()

        print(f"Saved plot for sandbar {label} as {output_image_path}")

print("\nCrescentic Indices:")                                                                                                                  # Display crescentic indices
for label, index in crescentic_indices.items():
    print(f"  Sandbar {label}: {index:.4f}" if not np.isnan(index) else f"  Sandbar {label}: Insufficient data")



#%% === 1.2. Calculate crescentic index for the whole dataset ===

def calculate_crescentic_index(df, window_length=11, polyorder=1):                                                                              # Function to calculate the crescentic index (standard deviation) for each sandbar
    """
    Calculate the crescentic index (standard deviation) for each sandbar.
    
    Parameters:
    - df: DataFrame containing sandbar data
    - window_length: Length of the smoothing window for Savitzky-Golay filter (must be odd)
    - polyorder: Polynomial order for Savitzky-Golay filter
    
    Returns:
    - crescentic_indices: Dictionary with standard deviations of detrended positions for each sandbar label
    """
    crescentic_indices = {}

    for label in df['sandbar_label'].unique():
        subset = df[df['sandbar_label'] == label].sort_values(by='x')                                                                           # Sort by alongshore distance (x)

        if len(subset) >= window_length:                                                                                                        # Check if the subset has enough data points for the given window length
            x_values = subset['x'].values
            y_values = subset['y'].values

            trend = savgol_filter(y_values, window_length=window_length, polyorder=polyorder)                                                   # Smooth the data to extract the long-wavelength trend
            detrended = y_values - trend                                                                                                        # Detrend the cross-shore positions
            std_detrended = np.std(detrended)                                                                                                   # Calculate the standard deviation of detrended positions
            crescentic_indices[label] = std_detrended
        else:
            crescentic_indices[label] = np.nan                                                                                                  # Not enough data points for this sandbar

    return crescentic_indices


years_to_process = range(1965, 2024)                                                                                                            # From 1965 to 2024

all_results = []                                                                                                                                # Initialize an empty list to collect results

for year in years_to_process:                                                                                                                   # Loop through each year
    csv_file = f'updated_sandbar_data_{year}.csv'

    if not os.path.exists(csv_file):
        print(f"Skipping year {year}: input file {csv_file} is missing.")
        continue

    crests_df = pd.read_csv(csv_file)                                                                                                           # Load the CSV file
    
    if crests_df.empty:                                                                                                                         # Skip year if the DataFrame is empty
        print(f"Skipping year {year}: no data available in the file.")
        continue

    crests_df['sandbar_label'] = crests_df['sandbar_label'].astype(str)

    crests_df['x'] = (crests_df['transect'] - 40) * 200 / 1000                                                                                  # Alongshore distance in kilometers
    crests_df['y'] = crests_df['crest_position']                                                                                                # Cross-shore distance in meters

    crescentic_indices = calculate_crescentic_index(crests_df)                                                                                  # Calculate crescentic indices with high-pass filtering

    for label, index in crescentic_indices.items():                                                                                             # Store results with year and sandbar labels
        all_results.append({
            'Year': year,
            'Sandbar': label,
            'Crescentic_Index': index
        })

    print(f"Crescentic indices for year {year}:")
    for label, index in crescentic_indices.items():
        print(f"  Sandbar {label}: {index:.4f}" if not np.isnan(index) else f"  Sandbar {label}: Insufficient data")

output_csv = 'crescentic_indices.csv'                                                                                                           # Write results to a CSV file in the same folder as the script

header = ['Year', 'Sandbar', 'Crescentic_Index']                                                                                                # Define the CSV header

with open(output_csv, mode='w', newline='') as file:                                                                                            # Write data to the CSV
    writer = csv.DictWriter(file, fieldnames=header)
    writer.writeheader()
    writer.writerows(all_results)

print(f"\nCrescentic indices saved to {output_csv}")



#%% === 2. Plot sandbar on depth-position instead of cross-shore ===

output_folder = os.path.join(os.getcwd(), "sandbar_frames_depth")                                                                               # Set the output folder
os.makedirs(output_folder, exist_ok=True)

years_to_visualize = range(2020, 2021)                                                                                                          # Specify the range of years to visualize

color_map = {                                                                                                                                   # Define the labels and color map
    '1': 'orange', '2': 'salmon', '3': 'peru', '4': 'slateblue', '5': 'teal', '6': 'gray', '4a': 'cadetblue',
    '7': 'royalblue', '5a': 'saddlebrown', '8': 'darkorange', '9': 'forestgreen', '7a': 'dodgerblue',
    '7b': 'slateblue', '8': 'lightcoral', '8a': 'orangered', '10': 'indigo', '10a': 'mediumvioletred',
    'N11': 'chocolate', 'N15': 'darkred', '11a': 'seagreen', '11': 'cadetblue', '12': 'peru',
    '13': 'olivedrab', 'N21': 'rosybrown'
}

for year in years_to_visualize:                                                                                                                 # Loop through each year to generate plots
    csv_file = f'updated_sandbar_data_{year}.csv'
    
    if not os.path.exists(csv_file):
        print(f"Skipping year {year}: input file {csv_file} is missing.")
        continue

    crests_df = pd.read_csv(csv_file)                                                                                                           # Load the CSV file
    crests_df['sandbar_label'] = crests_df['sandbar_label'].astype(str)

    print(f"Processing year {year} with {len(crests_df)} rows of data.")

    crests_df['x'] = (crests_df['transect'] - 40) * 200 / 1000                                                                                  # Alongshore distance in kilometers
    depths = []
    for idx, row in crests_df.iterrows():
        transect = row['transect']
        crest_position = row['crest_position']

        depth_file = os.path.join(os.getcwd(), f'depth_cross_shore_T{transect}.csv')                                                            # Load the corresponding depth file for the transect
        if not os.path.exists(depth_file):
            print(f"Missing depth file for transect {transect}. Skipping row {idx}.")
            depths.append(None)
            continue

        depth_data = pd.read_csv(depth_file)
        cross_shore_distances = depth_data['Cross-Shore Distance (m)']
        depths_values = depth_data['Depth (m)']

        try:                                                                                                                                    # Interpolate depth for the given crest position
            depth_interp = np.interp(crest_position, cross_shore_distances, depths_values)
            depths.append(depth_interp)
        except Exception as e:
            print(f"Error interpolating depth for transect {transect}, crest position {crest_position}: {e}")
            depths.append(None)

    crests_df['depth'] = depths                                                                                                                 # Add depth column to DataFrame

    print(f"First few rows after adding depths:\n{crests_df[['transect', 'x', 'crest_position', 'depth']].head()}")

    fig, ax = plt.subplots(figsize=(20, 10))                                                                                                    # Plot sandbar crests with depths
    for label in crests_df['sandbar_label'].unique():
        subset = crests_df[crests_df['sandbar_label'] == label].sort_values(by='x')
        color = color_map.get(label, 'black')
        
        if len(subset) > 0:                                                                                                                     # Check if there are any points for this label
            ax.scatter(subset['x'], subset['depth'], color=color, s=100, label=f'Sandbar Label {label}')                                        # Plot the sandbars (markers for crests)
            ax.plot(subset['x'], subset['depth'], linestyle='-', color=color, lw=2)                                                             # Connect the sandbar crests with lines

    ax.set_xlabel("Alongshore distance (km)", fontsize=14)                                                                                      # Set axis labels and title
    ax.set_ylabel("Depth (m)", fontsize=14)
    ax.set_title(f"{year} Sandbar Depths", fontsize=16)
    ax.invert_yaxis()                                                                                                                           # Depth increases downwards

    ax.legend(title='Sandbar Labels', loc='upper right', bbox_to_anchor=(1.15, 1), fontsize=10)                                                 # Add legend

    output_image_path = os.path.join(output_folder, f"sandbar_{year}_depths.png")
    plt.tight_layout()
    plt.savefig(output_image_path)
    plt.show()
    plt.close()
    print(f"Saved plot for year {year} as {output_image_path}")



#%% === 3. Plot crescentic index against time ===

results_file = 'crescentic_indices.csv'                                                                                                         # Load the results CSV file
if not os.path.exists(results_file):
    raise FileNotFoundError(f"Results file '{results_file}' not found.")

results_df = pd.read_csv(results_file)                                                                                                          # Read the CSV file into a DataFrame

filtered_df = results_df[                                                                                                                       # Filter out sandbars with labels starting with 'N', and filter outliers
    (~results_df['Sandbar'].str.startswith('N')) & 
    (results_df['Crescentic_Index'] < 80)
]

color_map = {                                                                                                                                   # Color map for sandbar labels
    '1': 'orange', '2': 'salmon', '2a': 'darksalmon', '3': 'peru', '4': 'slateblue', '5': 'teal',
    '6': 'gray', '6a': 'silver', '4a': 'cadetblue', '7': 'royalblue', '5a': 'saddlebrown',
    '8': 'darkorange', '9': 'forestgreen', '7a': 'dodgerblue', '7b': 'slateblue', '7c': 'skyblue',
    '7d': 'paleturquoise', '8a': 'orangered', '8b': 'coral', '10': 'indigo', '10a': 'mediumvioletred',
    '10b': 'orchid', '11a': 'seagreen', '11': 'cadetblue', '11b': 'thistle', '12': 'peru', '13': 'olivedrab'
}

plt.figure(figsize=(14, 8))

unique_sandbars = filtered_df['Sandbar'].unique()                                                                                               # Get the unique sandbar labels

for sandbar in unique_sandbars:                                                                                                                 # Loop through each sandbar and plot its crescentic index over time
    sandbar_data = filtered_df[filtered_df['Sandbar'] == sandbar]
    color = color_map.get(sandbar, 'black')                                                                                                     # Default to 'black' if the sandbar is not in the color map
    
    plt.plot(sandbar_data['Year'], sandbar_data['Crescentic_Index'], marker='o', linestyle='-', color=color, label=f'Sandbar {sandbar}')

plt.title(f'Crescentic index per sandbar', fontsize=14)
plt.xlabel('Year', fontsize=14)
plt.ylabel('Crescentic index (m)', fontsize=14)
plt.legend(bbox_to_anchor=(1, 1), loc='upper left', fontsize=10)
plt.grid(True)
plt.tight_layout()
plt.show()



#%% === 4. Statistics test ===

# === Check normality ===
results_df = pd.read_csv(results_file)                                                                                                          # Read the CSV file into a DataFrame

filtered_df = results_df[~results_df['Sandbar'].str.startswith('N')]
filtered_df_no_outliers = filtered_df[filtered_df['Crescentic_Index'] <= 60]
before_1998 = filtered_df_no_outliers[filtered_df_no_outliers['Year'] < 1998]['Crescentic_Index']
after_1998 = filtered_df_no_outliers[filtered_df_no_outliers['Year'] >= 1998]['Crescentic_Index']


stat_before, p_before = shapiro(before_1998)                                                                                                    # Check normality for before and after datasets
stat_after, p_after = shapiro(after_1998)

print(f"Shapiro-Wilk test for normality (Before 1998): p-value = {p_before:.4f}")
print(f"Shapiro-Wilk test for normality (After 1998): p-value = {p_after:.4f}")

if p_before < 0.05:
    print("Data before 1998 is not normally distributed.")
else:
    print("Data before 1998 is normally distributed.")

if p_after < 0.05:
    print("Data after 1998 is not normally distributed.")
else:
    print("Data after 1998 is normally distributed.")

########################### Mann-Whitney U test for independent data (no normal distributon) #################################
results_file = 'crescentic_indices.csv'                                                                                                         # Load the results CSV file
if not os.path.exists(results_file):
    raise FileNotFoundError(f"Results file '{results_file}' not found.")

results_df = pd.read_csv(results_file)                                                                                                          # Read the CSV file into a DataFrame

filtered_df = results_df[~results_df['Sandbar'].str.startswith('N')]
filtered_df_no_outliers = filtered_df[filtered_df['Crescentic_Index'] <= 60]

before_1998 = filtered_df_no_outliers[filtered_df_no_outliers['Year'] < 1998]['Crescentic_Index']                                               # Group data before and after 1998
after_1998 = filtered_df_no_outliers[filtered_df_no_outliers['Year'] >= 1998]['Crescentic_Index']

mean_before = before_1998.mean()                                                                                                                # Calculate the means for before and after 1998 (after removing outliers)
mean_after = after_1998.mean()

u_stat, p_value = mannwhitneyu(before_1998, after_1998, alternative='two-sided')                                                                # Perform the Mann-Whitney U test

print("  ")
print("Mann-Whitney U test for independent data:")
print(f"Mean before 1998 (after outlier removal): {mean_before:.2f}")
print(f"Mean after 1998 (after outlier removal): {mean_after:.2f}")
print(f"U-statistic: {u_stat:.2f}")
print(f"P-value: {p_value:.4f}")

if p_value < 0.05:
    print("The difference in means is statistically significant (p < 0.05).")
else:
    print("The difference in means is not statistically significant (p >= 0.05).")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))                                                                                                 # Create a figure with two subplots: one for the scatter plot, one for the boxplot


# === Scatter Plot with Means ===
axes[0].scatter(filtered_df_no_outliers['Year'], filtered_df_no_outliers['Crescentic_Index'], color='gray', alpha=0.5, label='Data Points')
axes[0].axhline(mean_before, color='blue', linestyle='--', label=f'Before 1998: {mean_before:.2f}')
axes[0].axhline(mean_after, color='red', linestyle='--', label=f'After 1998: {mean_after:.2f}')

axes[0].set_xlabel('Year', fontsize=14)
axes[0].set_ylabel('Crescentic index', fontsize=14)
axes[0].set_title('Crescentic index mean before and after 1998', fontsize=16)
axes[0].legend(fontsize=10)
axes[0].grid(True)

# === Boxplot of Two Periods ===
axes[1].boxplot([before_1998, after_1998], labels=['1965-1998', '1998-2023'], patch_artist=True)
axes[1].set_xlabel('Time period', fontsize=14)
axes[1].set_ylabel('Crescentic index', fontsize=14)
axes[1].set_title('Boxplot of crescentic index before and after 1998', fontsize=16)

plt.tight_layout()
plt.show()


########################### Wilcoxon signed-rank test for paired data (no normal distributon) #################################
results_file = 'crescentic_indices.csv'                                                                                                         # Load the results CSV file
if not os.path.exists(results_file):
    raise FileNotFoundError(f"Results file '{results_file}' not found.")

results_df = pd.read_csv(results_file)                                                                                                          # Read the CSV file into a DataFrame
filtered_df = results_df[~results_df['Sandbar'].str.startswith('N')]                                                                            # Filter out sandbars with labels starting with 'N'
filtered_df_no_outliers = filtered_df[filtered_df['Crescentic_Index'] <= 60]                                                                    # Remove outliers where Crescentic_Index > 60

before_1998 = filtered_df_no_outliers[filtered_df_no_outliers['Year'] < 1998]['Crescentic_Index']                                               # Group data before and after 1998
after_1998 = filtered_df_no_outliers[filtered_df_no_outliers['Year'] >= 1998]['Crescentic_Index']

min_length = min(len(before_1998), len(after_1998))                                                                                             # Ensure the datasets are of equal length for paired comparison
paired_before = before_1998[:min_length].reset_index(drop=True)
paired_after = after_1998[:min_length].reset_index(drop=True)

mean_before = paired_before.mean()                                                                                                              # Calculate the means for before and after 1998 (after removing outliers)
mean_after = paired_after.mean()

w_stat, p_value = wilcoxon(paired_before, paired_after)                                                                                         # Perform the Wilcoxon signed-rank test

print("Wilcoxon signed-rank test for paired data:")                                                                                             # Print the test results
print(f"Mean before 1998 (after outlier removal): {mean_before:.2f}")
print(f"Mean after 1998 (after outlier removal): {mean_after:.2f}")
print(f"Wilcoxon signed-rank statistic: {w_stat:.2f}")
print(f"P-value: {p_value:.4f}")

if p_value < 0.05:
    print("The difference in means is statistically significant (p < 0.05).")
else:
    print("The difference in means is not statistically significant (p >= 0.05).")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))                                                                                                 # Create a figure with two subplots: one for the scatter plot, one for the boxplot

# === Scatter Plot with Means ===
axes[0].scatter(filtered_df_no_outliers['Year'], filtered_df_no_outliers['Crescentic_Index'], color='gray', alpha=0.5, label='Data Points')
axes[0].axhline(mean_before, color='blue', linestyle='--', label=f'Before 1998: {mean_before:.2f}')
axes[0].axhline(mean_after, color='red', linestyle='--', label=f'After 1998: {mean_after:.2f}')

# Add details to the scatter plot
axes[0].set_xlabel('Year', fontsize=14)
axes[0].set_ylabel('Crescentic index', fontsize=14)
axes[0].set_title('Crescentic index mean before and after 1998', fontsize=16)
axes[0].legend(fontsize=10)
axes[0].grid(True)

# === Boxplot of Two Periods ===
axes[1].boxplot([paired_before, paired_after], labels=['1965-1998', '1998-2023'], patch_artist=True)
axes[1].set_xlabel('Time period', fontsize=14)
axes[1].set_ylabel('Crescentic index', fontsize=14)
axes[1].set_title('Boxplot of crescentic index before and after 1998', fontsize=16)
axes[1].grid(True)
plt.tight_layout()
plt.show()




