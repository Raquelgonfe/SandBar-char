
# -*- title: Plotting Ameland's sandbar parameters and migration rates -*-
"""
Created on Thursday 1 March 2025

@author: Raquelgonfe

Description:
Plotting the parameters of the csv (updated_sandbar_data_{year}.csv) data generated from 2.Dataset_extraction for Ameland and after manuual modification. 

The script includes the following components:
(References to the corresponding thesis figures.)

1. Save plots and a video for dataset visualization 
   similar to 'Ameland_dataset_visualisation.py' (Figure 19)
2.1. Plot sandbar parameters grouped by sandbar label (Figure 16)
2.2. Calculate and plot sandbar migration rates (Figure 30)
3. Plot parameters and migration rates by coastal section (Figure 31)
4. Plot sandbar parameters as a function of distance from the shoreline (Figure 18)
5. Plot width and height for individual sandbars (Figure 17)

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
import matplotlib.pyplot as plt
import contextily as ctx
from shapely.geometry import LineString, Polygon
from scipy.interpolate import CubicSpline
import math
import matplotlib.colors as mcolors
import imageio
import seaborn as sns
import re



#%% === 1. Save plots and video of dataset visualisation ===

output_folder = os.path.join(os.getcwd(), "sandbar_frames")                                 # Set output folder for images
os.makedirs(output_folder, exist_ok=True)

years_to_visualize = range(1965, 20024)                                                     # Define range of years to visualize

labels = ['1', '2', '3', '4', '5', '6', '4a', '7', '5a', '8', '9', '7a', '7', '8', '9', '8a', '10', '7b', '11', '12', '10a', 'N11', 'N15', '11a', '13', 'N21']
color_map = {
    '1': 'orange', '2': 'salmon', '3': 'peru', '4': 'slateblue', '5': 'teal', '6': 'gray', '4a': 'cadetblue',
    '7': 'royalblue', '5a': 'saddlebrown', '8': 'darkorange', '9': 'forestgreen', '7a': 'dodgerblue', '7b': 'slateblue', '8': 'lightcoral',
    '8a': 'orangered', '10': 'indigo', '10a': 'mediumvioletred', 'N11': 'chocolate', 'N15': 'darkred', '11a': 'seagreen',
    '11': 'cadetblue', '12': 'peru', '13': 'olivedrab', 'N21': 'rosybrown'                  # Define labels and color map
}


boundary_csv_file = 'updated_sandbar_data_2006.csv'                                         # Load the 1965 data to set boundaries
if os.path.exists(boundary_csv_file):
    boundary_df = pd.read_csv(boundary_csv_file)
    boundary_gdf = gpd.GeoDataFrame(
        boundary_df,
        geometry=gpd.points_from_xy(boundary_df['longitude'], boundary_df['latitude']),
        crs="EPSG:4326"
    ).to_crs(epsg=3857)
    
    x_min, y_min, x_max, y_max = boundary_gdf.total_bounds
    x_min -= 2000                                                                           # Add buffer to longitude boundaries
    x_max += 2000
    y_min -= 0.001                                                                          # Add buffer to latitude boundaries
    y_max += 0.0015
else:
    raise FileNotFoundError("Boundary file for 1965 not found. Please ensure the file exists.")


for year in years_to_visualize:                                                             # Loop through each year to generate images with consistent boundaries
    csv_file = f'updated_sandbar_data_{year}.csv'
    
    if not os.path.exists(csv_file):
        print(f"Skipping year {year}: input file {csv_file} is missing.")
        continue
    
    crests_df = pd.read_csv(csv_file)
    crests_df['sandbar_label'] = crests_df['sandbar_label'].astype(str)

    gdf = gpd.GeoDataFrame(
        crests_df, 
        geometry=gpd.points_from_xy(crests_df['longitude'], crests_df['latitude']),
        crs="EPSG:4326"
    ).to_crs(epsg=3857)                                                                     # Reproject to Mercator 
    
    fig, ax = plt.subplots(figsize=(25, 25))
    for label in gdf['sandbar_label'].unique():
        subset = gdf[gdf['sandbar_label'] == label].sort_values(by='longitude')
        color = color_map.get(label, 'black')
        
        subset.plot(ax=ax, marker='o', color=color, markersize=25, label=f'Sandbar Label {label}')
        plt.plot(subset['geometry'].x, subset['geometry'].y, linestyle='-', color=color, lw=2)
    
    ax.set_xlim(x_min, x_max)                                                               # Set fixed boundaries based on 1965 data
    ax.set_ylim(y_min, y_max)

    ctx.add_basemap(ax, crs=gdf.crs.to_string(), source=ctx.providers.CartoDB.Positron)     # Add basemap 

    ax.set_title(f"{year}", fontsize=18)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    plt.legend(title='Sandbar Labels', loc='upper left', bbox_to_anchor=(1, 1), ncol=2)
    
    output_image_path = os.path.join(output_folder, f"sandbar_{year}.png")                  # Save image for each year
    plt.tight_layout()
    plt.savefig(output_image_path)
    plt.close()
    print(f"Saved image for {year} as {output_image_path}")

video_output_path = "sandbar_animation.mp4"                                                 # Create a video from saved images
frames_per_second = 1

image_files = [os.path.join(output_folder, f"sandbar_{year}.png") for year in years_to_visualize if os.path.exists(os.path.join(output_folder, f"sandbar_{year}.png"))]
with imageio.get_writer(video_output_path, format='FFMPEG', fps=frames_per_second) as writer:
    for image_file in image_files:
        image = imageio.imread(image_file)
        writer.append_data(image)
    print(f"Video saved as {video_output_path}")



#%% === 2.1 Plot sandbar parameters per sandbar label ===
import matplotlib.ticker as ticker

sections = {                                                                                # Define sections as lists of transects
    'Section 1': list(range(40, 61)),                                                       # Transects 30-60
    'Section 2': list(range(60, 111)),                                                      # Transects 60-110
    'Section 3': list(range(110, 121))                                                      # Transects 110-120
}

years_to_process = range(1965, 2024)  
section_data = {
    'Section 1': [],
    'Section 2': [],
    'Section 3': []
}

color_map = {                                                                               # Define specific colors for each sandbar label (as strings)
    '1': 'orange', '2': 'salmon', '2a': 'darksalmon', '3': 'peru', '4': 'slateblue', '5': 'teal',
    '6': 'gray', '6a': 'silver', '4a': 'cadetblue', '7': 'royalblue', '5a': 'saddlebrown',
    '8': 'darkorange', '9': 'forestgreen', '7a': 'dodgerblue', '7b': 'slateblue', '7c': 'skyblue',
    '7d': 'paleturquoise', '8a': 'orangered', '8b': 'coral', '10': 'indigo', '10a': 'mediumvioletred',
    '10b': 'orchid', '11a': 'seagreen', '11': 'cadetblue', '11b': 'thistle', '12': 'peru', '13': 'olivedrab'
}

for year in years_to_process:                                                               # Process each year's data
    csv_file = f'updated_sandbar_data_{year}.csv'
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"File not found: {csv_file}. Skipping this year.")
        continue

    df['sandbar_label'] = df['sandbar_label'].astype(str)                                   # Ensure 'sandbar_label' is treated as string to match color_map

    for section_name, transects in sections.items():
        section_df = df[df['transect'].isin(transects)].copy()
        
        sandbar_groups = section_df.groupby('sandbar_label').agg({                          # Group the data by sandbar label
            'height': 'mean',
            'width2': 'mean',
            'width3': 'mean',
            'depth': 'mean',
            'cross_shore': 'mean',
            'crest_position': 'mean',
            'volume' : 'mean',
            'depth_position' : 'mean',
            'centroid_x' : 'mean'
        }).reset_index()

        sandbar_groups['Year'] = year
        section_data[section_name].append(sandbar_groups)

section_1_df = pd.concat(section_data['Section 1'], ignore_index=True)                      # When combining all data and calculating, ensure sandbar_label remains as strings
section_2_df = pd.concat(section_data['Section 2'], ignore_index=True)
section_3_df = pd.concat(section_data['Section 3'], ignore_index=True)

section_1_df['sandbar_label'] = section_1_df['sandbar_label'].astype(str)                   # Ensure 'sandbar_label' is treated as strings for plot functions
section_2_df['sandbar_label'] = section_2_df['sandbar_label'].astype(str)
section_3_df['sandbar_label'] = section_3_df['sandbar_label'].astype(str)

def calculate_avg_and_std(section_data, parameter, absolute=False):                         # Function to calculate mean and standard deviation per sandbar label
    combined_df = pd.concat(section_data.values(), ignore_index=True)
    if absolute and parameter == 'height':
        combined_df[parameter] = combined_df[parameter].abs()                               # Take the absolute value of the height before calculating mean and std
    
    grouped = combined_df.groupby(['Year', 'sandbar_label']).agg(
        mean_value=(parameter, 'mean'),
        std_value=(parameter, 'std')
    ).reset_index()
    return grouped

def plot_avg_parameter_with_std(grouped_data, parameter, ylabel, reverse_y=False):          # Function to plot average parameter with standard deviation
    plt.figure(figsize=(12, 9))                                                             # Slightly larger figure for clarity

    for label in grouped_data['sandbar_label'].unique():
        subset = grouped_data[grouped_data['sandbar_label'] == label]
        color = color_map.get(label, '#000000')                                           # Default to black if label is not in color_map

        marker = '*' if label.startswith('N') else 'o'                                      # Use a star marker if the sandbar label starts with "N", otherwise use a circle
        markersize = 12 if marker == '*' else 8                                             # Slightly larger markers

        plt.errorbar(subset['Year'], subset['mean_value'], yerr=subset['std_value'], 
                     fmt=f'-{marker}', label=f'Sandbar {label}', color=color, capsize=5, markersize=markersize)

    plt.title(f'Average {parameter} per sandbar', fontsize=20)  
    plt.xlabel('Year', fontsize=18)
    plt.ylabel(ylabel, fontsize=18)

    if reverse_y:
        plt.gca().invert_yaxis()
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x)}"))
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=4, fontsize=16)
    plt.tight_layout()
    plt.show()

height_data = calculate_avg_and_std({'Section 1': section_1_df, 'Section 2': section_2_df, 'Section 3': section_3_df}, 'height', absolute=True)
plot_avg_parameter_with_std(height_data, 'height', 'Height (m)')

width2_data = calculate_avg_and_std({'Section 1': section_1_df, 'Section 2': section_2_df, 'Section 3': section_3_df}, 'width2')
plot_avg_parameter_with_std(width2_data, 'width2', 'Width2 (m)')

width3_data = calculate_avg_and_std({'Section 1': section_1_df, 'Section 2': section_2_df, 'Section 3': section_3_df}, 'width3')
plot_avg_parameter_with_std(width3_data, 'width3', 'Width3 (m)')

depth_data = calculate_avg_and_std({'Section 1': section_1_df, 'Section 2': section_2_df, 'Section 3': section_3_df}, 'depth')
plot_avg_parameter_with_std(depth_data, 'depth', 'Depth (m)')

centroid_x = calculate_avg_and_std({'Section 1': section_1_df, 'Section 2': section_2_df, 'Section 3': section_3_df}, 'centroid_x')
plot_avg_parameter_with_std(centroid_x, 'centroid', 'Centroid-position (m)')

crest_position_data = calculate_avg_and_std({'Section 1': section_1_df, 'Section 2': section_2_df, 'Section 3': section_3_df}, 'crest_position')
plot_avg_parameter_with_std(crest_position_data, 'crest-position', 'Crest position (m)')

cross_shore_data = calculate_avg_and_std({'Section 1': section_1_df, 'Section 2': section_2_df, 'Section 3': section_3_df}, 'cross_shore')
plot_avg_parameter_with_std(cross_shore_data, 'cross-shore-distance', 'Cross shore distance (m)')

volume_data = calculate_avg_and_std({'Section 1': section_1_df, 'Section 2': section_2_df, 'Section 3': section_3_df}, 'volume')
plot_avg_parameter_with_std(volume_data, 'volume', 'Volume (m³)')

depth_position_data = calculate_avg_and_std({'Section 1': section_1_df, 'Section 2': section_2_df, 'Section 3': section_3_df}, 'depth_position')
plot_avg_parameter_with_std(depth_position_data, 'depth-position', 'Depth position (m)', reverse_y=True)



#%% === 2.2. Calculate and plot migration rates ===

def filter_sandbars_with_min_years(data, min_consecutive_years=3):                          # Function to filter sandbars with more than X consecutive years
    valid_sandbars = []
    for label in data['sandbar_label'].unique():
        sandbar_data = data[data['sandbar_label'] == label].sort_values('Year')
        years = sandbar_data['Year'].values

        consecutive_count = 1
        for i in range(1, len(years)):
            if years[i] == years[i - 1] + 1:
                consecutive_count += 1
                if consecutive_count >= min_consecutive_years:
                    valid_sandbars.append(label)
                    break
            else:
                consecutive_count = 1

    return data[data['sandbar_label'].isin(valid_sandbars)]

filtered_crest_position_data = filter_sandbars_with_min_years(cross_shore_data)             # Filter crest_position_data to include only valid sandbars

def calculate_yearly_migration_rate(crest_position_data):                                   # Function to calculate year-by-year migration rate for each sandbar
    migration_rates = []
    crest_position_data = crest_position_data.sort_values(['sandbar_label', 'Year'])

    for label in crest_position_data['sandbar_label'].unique():
        sandbar_data = crest_position_data[crest_position_data['sandbar_label'] == label]
        
        for i in range(1, len(sandbar_data)):
            previous_year = sandbar_data.iloc[i - 1]
            current_year = sandbar_data.iloc[i]
            migration_rate = current_year['mean_value'] - previous_year['mean_value']
            
            migration_rates.append({
                'sandbar_label': label,
                'year': current_year['Year'],
                'migration_rate_per_year': migration_rate
            })

    return pd.DataFrame(migration_rates)

yearly_migration_rates = calculate_yearly_migration_rate(filtered_crest_position_data)      # Calculate yearly migration rates for filtered crest position data

def apply_moving_average(migration_rate_df, window_size=3):                                 # Function to apply moving average to migration rates
    migration_rate_df['smoothed_migration_rate'] = migration_rate_df.groupby('sandbar_label')['migration_rate_per_year'].transform(lambda x: x.rolling(window=window_size, min_periods=1).mean())
    return migration_rate_df

smoothed_migration_rates = apply_moving_average(yearly_migration_rates)                     # Apply moving average to migration rates

def plot_migration_rate_evolution(migration_rate_df):                                       # Function to plot migration rate evolution over the years for each sandbar
    plt.figure(figsize=(10, 8))
    for label in migration_rate_df['sandbar_label'].unique():
        sandbar_data = migration_rate_df[migration_rate_df['sandbar_label'] == label].sort_values('year')
        color = color_map.get(label, '#000000')

        plt.plot(sandbar_data['year'], sandbar_data['smoothed_migration_rate'], label=f'Sandbar {label}', color=color, marker='o')
    plt.axhline(0, color='grey', linestyle='--', linewidth=1)
    plt.xlabel('Year', fontsize=18)
    plt.ylabel('Migration rate (m/year)', fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x)}"))
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=4)
    plt.tight_layout()
    plt.show()

plot_migration_rate_evolution(smoothed_migration_rates)



#%% === 3. Plot parameters and migration rates per section ===

for section_name, transects in sections.items():                                                # Iterate over sections to process and plot data
    print(f"Processing {section_name}...")

    section_data = []                                                                           # Collect data for the current section
    for year in years_to_process:
        csv_file = f'updated_sandbar_data_{year}.csv'
        try:
            df = pd.read_csv(csv_file)
        except FileNotFoundError:
            print(f"File not found: {csv_file}. Skipping this year.")
            continue

        # Filter the section's transects
        section_df = df[df['transect'].isin(transects)].copy()

        sandbar_groups = section_df.groupby('sandbar_label').agg({                              # Group by sandbar label and aggregate the necessary parameters
            'height': 'mean',
            'width2': 'mean',
            'width3': 'mean',
            'depth': 'mean',
            'cross_shore': 'mean',
            'crest_position': 'mean',
            'depth_position': 'mean',                                                           # Add depth_position to aggregation
            'volume': 'mean',
            'depth_position' : 'mean'
        }).reset_index()

        sandbar_groups['Year'] = year
        section_data.append(sandbar_groups)

    section_df = pd.concat(section_data, ignore_index=True)                                     # Combine yearly data for the current section
    section_df['sandbar_label'] = section_df['sandbar_label'].astype(str)                       # Ensure labels are strings

    def calculate_avg_and_std(section_df, parameter, absolute=False):                           # Function to calculate mean and std per sandbar label
        if absolute and parameter == 'height':
            section_df[parameter] = section_df[parameter].abs()
        grouped = section_df.groupby(['Year', 'sandbar_label']).agg(
            mean_value=(parameter, 'mean'),
            std_value=(parameter, 'std')
        ).reset_index()
        return grouped

    def plot_avg_parameter_with_std(grouped_data, parameter, ylabel, section_name):             # Function to plot average parameter with standard deviation
        plt.figure(figsize=(10, 8))
        for label in grouped_data['sandbar_label'].unique():
            subset = grouped_data[grouped_data['sandbar_label'] == label]
            color = color_map.get(label, '#000000')                                         # Default to black if label is not in color_map

            marker = '*' if label.startswith('N') else 'o'                                      # Use a star marker if the sandbar label starts with "N", otherwise use a circle
            markersize = 10 if marker == '*' else 6

            plt.errorbar(subset['Year'], subset['mean_value'], yerr=subset['std_value'], 
                         fmt=f'-{marker}', label=f'Sandbar {label}', color=color, capsize=5, markersize=markersize)

        plt.title(f'{section_name}: Average {parameter} per sandbar')
        plt.xlabel('Year')
        plt.ylabel(ylabel)
        
        if parameter == 'depth_position':                                                       # If the parameter is "depth_position", invert the y-axis
            plt.gca().invert_yaxis()
        
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4)
        plt.tight_layout()
        plt.show()

    for parameter, ylabel in [
        ('height', 'Height (m)'),
        ('width2', 'Width2 (m)'),
        ('width3', 'Width3 (m)'),
        ('depth', 'Depth (m)'),
        ('crest_position', 'Crest Position (m)'),
        ('depth_position', 'Depth Position (m)'),                                               # Add depth_position plot
        ('cross_shore', 'Cross Shore Position (m)'),
        ('volume', 'Volume (m³)'),
        ('depth_position', 'Depth Position (m)')                                                # Add this one as well for depth_position
    ]:
        grouped_data = calculate_avg_and_std(section_df, parameter, absolute=(parameter == 'height'))
        plot_avg_parameter_with_std(grouped_data, parameter, ylabel, section_name)

    #******************************* CALCULATE AND PLOT MIGRATION RATES PER SECTION ************************

    print(f"Calculating migration rates for {section_name}...")

    def process_and_plot_migration_rate(section_df, parameter, ylabel, section_name):           # Function to calculate and plot migration rates
        section_parameter_data = calculate_avg_and_std(section_df, parameter)
        filtered_section_data = filter_sandbars_with_min_years(section_parameter_data)
        yearly_migration_rates = calculate_yearly_migration_rate(filtered_section_data)
        smoothed_migration_rates = apply_moving_average(yearly_migration_rates)

        plt.figure(figsize=(10, 8))
        for label in smoothed_migration_rates['sandbar_label'].unique():
            sandbar_data = smoothed_migration_rates[smoothed_migration_rates['sandbar_label'] == label].sort_values('year')
            color = color_map.get(label, '#000000')

            plt.plot(sandbar_data['year'], sandbar_data['smoothed_migration_rate'], label=f'Sandbar {label}', color=color, marker='o')

        plt.title(f'{section_name}: Evolution of Sandbar Migration ({ylabel}, Smoothed)')
        plt.xlabel('Year')
        plt.ylabel(f'{ylabel} Migration Rate (m/year)')
        plt.ylim(-70, 200)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4)
        plt.tight_layout()
        plt.show()

    process_and_plot_migration_rate(section_df, 'crest_position', 'Crest Position', section_name)



#%% === 4. Parameters against distance from shoreline ===

def calculate_avg_and_std_cross_shore(section_data, parameter):                                 # Function to calculate mean and standard deviation per sandbar label over cross-shore position
    combined_df = pd.concat(section_data.values(), ignore_index=True)
    if parameter == 'height':
        combined_df[parameter] = combined_df[parameter].abs()                                   # Ensure height values are positive
    grouped = combined_df.groupby(['cross_shore', 'sandbar_label']).agg(
        mean_value=(parameter, 'mean'),
        std_value=(parameter, 'std')
    ).reset_index()
    return grouped

def plot_composite_figure(height_data, width_data, depth_data, volume_data):                    # Function to plot parameters against cross-shore position
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True)
    parameters = [
        ('Height', height_data, 'Height (m)'),
        ('Width', width_data, 'Width (m)'),
        ('Depth', depth_data, 'Depth (m)'),
        ('Volume', volume_data, 'Volume (m³/m)')
    ]
    
    for ax, (_, data, ylabel) in zip(axes.flatten(), parameters):
        for label in data['sandbar_label'].unique():
            subset = data[data['sandbar_label'] == label]
            color = color_map.get(label, '#000000')                                           # Default to black if label is not in color_map
            marker = '*' if label.startswith('N') else 'o'
            markersize = 10 if marker == '*' else 6
            
            ax.errorbar(subset['cross_shore'], subset['mean_value'], yerr=subset['std_value'], 
                        fmt=f'-{marker}', label=f'Sandbar {label}', color=color, capsize=5, markersize=markersize)
        
        ax.set_ylabel(ylabel)
    
    for ax in axes[1]:
        ax.set_xlabel('Distance to shoreline (m)')
    
    for ax in axes.flatten():
        ax.yaxis.set_tick_params(labelleft=True)
    
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0), ncol=4)
    plt.tight_layout()
    plt.show()

# Calculate and plot the composite figure
height_cross_shore = calculate_avg_and_std_cross_shore({'Section 1': section_1_df, 'Section 2': section_2_df, 'Section 3': section_3_df}, 'height')
width_cross_shore = calculate_avg_and_std_cross_shore({'Section 1': section_1_df, 'Section 2': section_2_df, 'Section 3': section_3_df}, 'width2')
depth_cross_shore = calculate_avg_and_std_cross_shore({'Section 1': section_1_df, 'Section 2': section_2_df, 'Section 3': section_3_df}, 'depth')
volume_cross_shore = calculate_avg_and_std_cross_shore({'Section 1': section_1_df, 'Section 2': section_2_df, 'Section 3': section_3_df}, 'volume')

plot_composite_figure(height_cross_shore, width_cross_shore, depth_cross_shore, volume_cross_shore)



#%% === 5. Plots for width and height per individual sandbar ===


def filter_sandbars_with_consecutive_years(grouped_data, min_consecutive_years=5):                  # Function to plot parameters against cross-shore position
    valid_labels = []
    
    for label in grouped_data['sandbar_label'].unique():
        sandbar_data = grouped_data[grouped_data['sandbar_label'] == label].sort_values('Year')
        years = sandbar_data['Year'].values
        
        consecutive_count = 1                                                                       #  Check for consecutive years
        for i in range(1, len(years)):
            if years[i] == years[i - 1] + 1:
                consecutive_count += 1
                if consecutive_count >= min_consecutive_years:
                    valid_labels.append(label)
                    break
            else:
                consecutive_count = 1

    return grouped_data[grouped_data['sandbar_label'].isin(valid_labels)]                           # Filter the grouped_data to include only valid labels

def custom_sort(labels):                                                                            # Function to custom sort labels, numeric first, then alphabetic
    def sort_key(label):
        match = re.match(r'(\d+)([A-Za-z]*)', label)
        if match:
            numeric_part = int(match.group(1))
            non_numeric_part = match.group(2)
            return (numeric_part, non_numeric_part)
        return (float('inf'), label)

    return sorted(labels, key=sort_key)

def plot_avg_parameter_individual_subplots(grouped_data, parameter, ylabel):                        # Function to plot the data with individual subplots
    grouped_data['sandbar_label'] = grouped_data['sandbar_label'].astype(str)
    
    unique_labels = custom_sort(grouped_data['sandbar_label'].unique())
    num_labels = len(unique_labels)

    num_cols = 3
    num_rows = math.ceil(num_labels / num_cols)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(16, 6 * num_rows), sharex=True, sharey=True)

    axes = axes.flatten()

    min_year = grouped_data['Year'].min()
    max_year = grouped_data['Year'].max()
    min_y = grouped_data['mean_value'].min() - grouped_data['std_value'].max()
    max_y = grouped_data['mean_value'].max() + grouped_data['std_value'].max()

    for idx, ax in enumerate(axes):
        if idx >= num_labels:
            ax.axis('off')
            continue

        label = unique_labels[idx]
        subset = grouped_data[grouped_data['sandbar_label'] == label]
        color = color_map.get(label, '#000000')
        marker = '*' if label.startswith('N') else 'o'
        markersize = 12 if marker == '*' else 6

        ax.errorbar(
            subset['Year'], 
            subset['mean_value'], 
            yerr=subset['std_value'], 
            fmt=f'-{marker}', 
            color=color, 
            capsize=5, 
            markersize=markersize
        )

        ax.set_title(f'Sandbar {label}', fontsize=22, fontweight='bold')                                # Increased title size
        ax.set_xlim(min_year, max_year)
        ax.set_ylim(min_y, max_y)
        ax.tick_params(axis='both', which='major', labelsize=18)                                        # Increased tick label font size

    fig.text(0.06, 0.5, ylabel, va='center', rotation='vertical', fontsize=28, fontweight='bold')       # Adjusted Y-axis label position and size
    plt.tight_layout(rect=[0.1, 0.04, 1, 0.96])
    plt.show()

filtered_width3_data = filter_sandbars_with_consecutive_years(width3_data)                              # Apply the filter to the width data
plot_avg_parameter_individual_subplots(filtered_width3_data, 'width', 'Width (m)')  

filtered_height_data = filter_sandbars_with_consecutive_years(height_data)                              # Apply the filter to the height data
plot_avg_parameter_individual_subplots(filtered_height_data, 'height', 'Height (m)')