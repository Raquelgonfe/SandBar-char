# -*- title: Plotting the data -*-
"""
Created on Thursday 1 March 2025

@author: Raquelgonfe

Description:
Examples of plots from the data generated in the previous code files.
Average height, width, depth, crest position, cross-shore position, and volume are plotted for each sandbar.

These metrics can be used to derive other parameters such as migration rates or total sandbar volume.

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

#%% === Plot sanbar parameters per sandbar label ===

sections = {                                                                            # Define sections as lists of transects
    'Section_1': list(range(30, 61)),                                                   # Transects 30-60
    'Section_2': list(range(60, 91)),                                                   # Transects 60-90
    'Section_3': list(range(100, 121))                                                  # Transects 100-120
}

years_to_process = range(2000, 2002)                                                    # Define the years to process

section_data = {
    'Section_1': [],
    'Section_2': [],
    'Section_3': []
}

for year in years_to_process:
    csv_file = f'sandbar_data_{year}.csv'                                               # Load the CSV file for the current year into a pandas DataFrame
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"File not found: {csv_file}. Skipping this year.")
        continue

    for section_name, transects in sections.items():
        section_df = df[df['transect'].isin(transects)].copy()

        label_mapping = {3: 1, 6: '1a', 2: 2, 4: '2a', 1: 3, 5: 4, 11: '4a',            # Change sandbar labels according to specified mapping
                          8: 5, 12: '5a', 9:6, 13: 7, 10: 'N1', 14:'N2'}
        section_df['sandbar_label'] = section_df['sandbar_label'].replace(label_mapping)

        sandbar_groups = section_df.groupby('sandbar_label').agg({                      # Group the data by sandbar label
            'height': 'mean',       
            'width': 'mean',
            'depth': 'mean',
            'cross_shore': 'mean',  
            'crest_position' : 'mean',
            'volume': 'mean'
        }).reset_index()

        sandbar_groups['Year'] = year
        section_data[section_name].append(sandbar_groups)

section_1_df = pd.concat(section_data['Section_1'], ignore_index=True)                  # Group the data by sandbar label
section_2_df = pd.concat(section_data['Section_2'], ignore_index=True)
section_3_df = pd.concat(section_data['Section_3'], ignore_index=True) 

color_map = {
    1: 'royalblue',  '1a': 'cornflowerblue',  2: 'darkorange',  '2a': 'orangered',  3: 'forestgreen',  4: 'mediumorchid',  '4a': 'palevioletred',  
    5: 'mediumaquamarine',  '5a': 'saddlebrown',  6: 'gray', 7: 'gold',  'N1': 'darkkhaki',  'N2': 'peru' 
}

def calculate_avg_and_std(section_data, parameter):                                     # Function to calculate mean and standard deviation per sandbar label
    combined_df = pd.concat(section_data.values(), ignore_index=True)
    grouped = combined_df.groupby(['Year', 'sandbar_label']).agg(
        mean_value=(parameter, 'mean'),
        std_value=(parameter, 'std')
    ).reset_index()
    return grouped

def plot_avg_parameter_with_std(grouped_data, parameter, ylabel):                       # Function to plot average parameter with standard deviation
    plt.figure(figsize=(14, 8))

    for label in grouped_data['sandbar_label'].unique():                                # Function to plot average parameter with standard deviation
        subset = grouped_data[grouped_data['sandbar_label'] == label]
        color = color_map.get(label, '#000000')                                       # Default to black if label is not in color_map

        plt.errorbar(subset['Year'], subset['mean_value'], yerr=subset['std_value'], 
                     fmt='-o', label=f'Sandbar {label}', color=color, capsize=5, markersize=8)

    plt.title(f'Average {parameter} per sandbar')
    plt.xlabel('Year')
    plt.ylabel(ylabel)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), ncol=2)
    plt.tight_layout()
    plt.show()

# Calculate and plot the average and std for each parameter
height_data = calculate_avg_and_std({'Section_1': section_1_df, 'Section_2': section_2_df, 'Section_3': section_3_df}, 'height')
plot_avg_parameter_with_std(height_data, 'height', 'Height (m)')

width_data = calculate_avg_and_std({'Section_1': section_1_df, 'Section_2': section_2_df, 'Section_3': section_3_df}, 'width')
plot_avg_parameter_with_std(width_data, 'width', 'Width (m)')

depth_data = calculate_avg_and_std({'Section_1': section_1_df, 'Section_2': section_2_df, 'Section_3': section_3_df}, 'depth')
plot_avg_parameter_with_std(depth_data, 'depth', 'Depth (m)')

crest_position_data = calculate_avg_and_std({'Section_1': section_1_df, 'Section_2': section_2_df, 'Section_3': section_3_df}, 'crest_position')
plot_avg_parameter_with_std(crest_position_data, 'crest_position', 'Crest Position (m)')

cross_shore_data = calculate_avg_and_std({'Section_1': section_1_df, 'Section_2': section_2_df, 'Section_3': section_3_df}, 'cross_shore')
plot_avg_parameter_with_std(cross_shore_data, 'cross_shore', 'Cross Shore Position (m)')

volume = calculate_avg_and_std({'Section_1': section_1_df, 'Section_2': section_2_df, 'Section_3': section_3_df}, 'volume')
plot_avg_parameter_with_std(cross_shore_data, 'volume', 'Cross Shore Position (m)')
