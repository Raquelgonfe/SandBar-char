# -*- title: Ameland dataset visualisation -*-
"""
Created on Thursday 1 March 2025

@author: Raquelgonfe

Description:
Visualisation of the csv (updated_sandbar_data_{year}.csv) data generated from 2.Dataset_extraction for Ameland and after manuual modification. 

The script includes the following components:
(References to the corresponding thesis figures.)

1. Visualisation of crests
2. Visualisation of crests with labels for manual modification
3. Visualisation of crests with lines between crests
4. Plot sandbars for several years (Figure 19)


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
import matplotlib.cm as cm



#%% === 1. Visualisation of crests ===

years_to_visualize = range(1965, 2023)                                                                                      # Define the range of years to visualize. 
marker_dict = {1: 'o', 2: 'x', 3: '^', 4: 's', 5: '*', 6: 'P', 7: 'D'}                                                      # Define a marker dictionary for different Sandbar Labels

for year in years_to_visualize:
    csv_file = f'updated_sandbar_data_{year}.csv'
    try:
        crests_df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"File for year {year} not found. Skipping.")
        continue

    gdf = gpd.GeoDataFrame(
        crests_df, 
        geometry=gpd.points_from_xy(crests_df['longitude'], crests_df['latitude']),
        crs="EPSG:4326"                                                                                                     # WGS 84 (latitude, longitude)
    )

    gdf = gdf.to_crs(epsg=3857)                                                                                             # Reproject to a Mercator projection (for web mapping)

    fig, ax = plt.subplots(figsize=(25, 25))  

    for label in gdf['sandbar_label'].unique():                                                                             # Loop through the unique sandbar labels and plot each subset with a different marker
        subset = gdf[gdf['sandbar_label'] == label]
        marker = marker_dict.get(label, 'o')                                                                                # Default to 'o' if label is not in the dictionary
        subset.plot(ax=ax, marker=marker, color='black', markersize=25, label=f'Sandbar Label {label}')

    ctx.add_basemap(ax, crs=gdf.crs.to_string(), source=ctx.providers.CartoDB.Positron)

    ax.set_title(f"Sandbar Crests for {year} (Ameland)")
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    plt.legend()
    plt.tight_layout()
    plt.show()

#%% === 2. Visualisation of crests with labels for manual modification. And distances in meters ===

years_to_visualize = range(2000, 2001)                                                                                      # Define the range of years to visualize
marker_dict = {1: 'o', 2: 'x', 3: '^', 4: 's', 5: '*', 6: 'P', 7: 'D'}                                                      # Define a marker dictionary for different Sandbar Labels

for year in years_to_visualize:
    csv_file = f'updated_sandbar_data_{year}.csv'
    try:
        crests_df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"File for year {year} not found. Skipping.")
        continue
    gdf = gpd.GeoDataFrame(
        crests_df, 
        geometry=gpd.points_from_xy(crests_df['longitude'], crests_df['latitude']),
        crs="EPSG:4326"                                                                                                     # WGS 84 (latitude, longitude)
    )

    gdf = gdf.to_crs(epsg=32631)                                                                                            # Reproject to UTM (distance in meters) UTM zone 31N for Northern Hemisphere

    min_x = gdf.geometry.x.min()                                                                                            # Get the minimum X and Y values to shift the coordinates to start from (0,0)
    min_y = gdf.geometry.y.min()

    # Shift all X and Y values so they start from (0,0)
    gdf['shifted_x'] = gdf.geometry.x - min_x
    gdf['shifted_y'] = gdf.geometry.y - min_y

    exaggeration_factor = 4                                                                                                 # Apply Y-axis exaggeration for better visualisation (scale by 4)
    gdf['exaggerated_y'] = gdf['shifted_y'] * exaggeration_factor

    fig, ax = plt.subplots(figsize=(25, 25)) 

    for label in gdf['sandbar_label'].unique():                                                                             # Loop through the unique sandbar labels and plot each subset with a different marker
        subset = gdf[gdf['sandbar_label'] == label]
        marker = marker_dict.get(label, 'o')                                                                                # Default to 'o' if label is not in the dictionary
        
        ax.scatter(subset['shifted_x'], subset['exaggerated_y'], marker=marker, color='black', s=100, label=f'Sandbar Label {label}')

        for x, y, transect in zip(subset['shifted_x'], subset['exaggerated_y'], subset['transect']):                        # Add the transect numbers as text labels on the map (with shifted coordinates)
            ax.text(x, y, str(transect), fontsize=9, color='blue', weight='bold')

    ax.set_title(f"Sandbar Crests for {year} (Ameland)")
    ax.set_xlabel("X (meters from reference point)")  
    ax.set_ylabel("Y (meters from reference point, exaggerated)")  

    plt.legend()
    ax.set_aspect(aspect=1 / exaggeration_factor)                                                                           # Set equal scaling on X-axis and exaggerated Y-axis
    plt.tight_layout()
    plt.show()

#%% === 3. Visualisation of crests with lines between crests ===

output_folder = os.path.join(os.getcwd(), "sandbar_frames")
os.makedirs(output_folder, exist_ok=True)

years_to_visualize = range(1965, 2023)                                                                                      # Define range of years to visualize

all_labels = ['1', '2', '3', '4', '5', '6', '4a', '7', '5a', '8', '9', '7a', '7', '8', 
              '9', '8a', '10', '7b', '11', '12', '10a', 'N11', 'N15', '11a', '13', 'N21']    # Define sandbar label mappings and color map


# Step 1: Collect all unique sandbar labels across all years
all_labels = set()

for year in years_to_visualize:
    csv_file = f'updated_sandbar_data_{year}.csv'
    try:
        df = pd.read_csv(csv_file)
        all_labels.update(df['sandbar_label'].astype(str).unique())
    except FileNotFoundError:
        continue

all_labels = sorted(map(str, all_labels))

# Step 2: Create a color map marker map
colormap = cm.get_cmap('tab20', len(all_labels))  # Use any colormap you like
label_color_map = {label: mcolors.to_hex(colormap(i)) for i, label in enumerate(all_labels)}
marker_types = ['o', 'x', '^', 's', '*', 'P', 'D', 'v', '<', '>', 'h', '+', '1', '2']
label_marker_map = {label: marker_types[i % len(marker_types)] for i, label in enumerate(all_labels)}

# Step 3: Plot year-by-year with consistent colors
for year in years_to_visualize:
    csv_file = f'updated_sandbar_data_{year}.csv'

    try:
        crests_df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"File for year {year} not found. Skipping.")
        continue

    crests_df['sandbar_label'] = crests_df['sandbar_label'].astype(str)

    gdf = gpd.GeoDataFrame(
        crests_df,
        geometry=gpd.points_from_xy(crests_df['longitude'], crests_df['latitude']),
        crs="EPSG:4326"
    )

    gdf = gdf.to_crs(epsg=3857)

    fig, ax = plt.subplots(figsize=(25, 25))

    for label in gdf['sandbar_label'].unique():
        subset = gdf[gdf['sandbar_label'] == label]
        subset_sorted = subset.sort_values(by='longitude')

        color = label_color_map.get(label, 'black')
        marker = label_marker_map.get(label, 'o')

        subset_sorted.plot(ax=ax, marker=marker, color=color, markersize=25, label=f'Sandbar Label {label}')
        plt.plot(subset_sorted['geometry'].x, subset_sorted['geometry'].y, linestyle='-', color=color, lw=2)

    ctx.add_basemap(ax, crs=gdf.crs.to_string(), source=ctx.providers.CartoDB.Positron)

    ax.set_title(f"{year}", fontsize=18)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    plt.legend(title='Sandbar Labels', loc='upper left', bbox_to_anchor=(1, 1), ncol=2)

    plt.tight_layout()
    plt.show()



#%% === 4. Figure: Plot sandbars for several years ===

years_to_visualize = range(1997, 2004)  # Specify the range of years to visualize

color_map = {
    '1': 'orange', '2': 'salmon', '2a': 'darksalmon', '3': 'peru', '4': 'slateblue', '5': 'teal', '6': 'gray', # Define the labels and color map
    '6a': 'silver', '4a': 'cadetblue', '7': 'royalblue', '5a': 'saddlebrown', '8': 'darkorange', '9': 'forestgreen', 
    '7a': 'dodgerblue', '7b': 'slateblue', '7c': 'skyblue', '7d': 'paleturquoise', '8a': 'orangered', '8b': 'coral', 
    '10': 'indigo', '10a': 'mediumvioletred', '10b': 'orchid', 'N11': 'chocolate', 'N15': 'darkred', '11a': 'seagreen', 
    '11': 'cadetblue', '11b': 'thistle', '12': 'peru', '13': 'olivedrab', 'N21': 'rosybrown'
}

num_years = len(years_to_visualize) # Number of rows and columns for subplots
cols = 1  # Number of columns in the grid
rows = (num_years + cols - 1) // cols  # Calculate number of rows

fig, axes = plt.subplots(rows, cols, figsize=(10, 2.5 * rows), sharex=True, sharey=True)
axes = axes.flatten()  # Flatten the 2D array of axes for easy iteration

global_handles = {}

for idx, year in enumerate(years_to_visualize): # Loop through each year to generate subplots
    csv_file = f'updated_sandbar_data_{year}.csv'
    
    if not os.path.exists(csv_file):
        print(f"Skipping year {year}: input file {csv_file} is missing.")
        continue

    crests_df = pd.read_csv(csv_file)       # Load the CSV file
    crests_df['sandbar_label'] = crests_df['sandbar_label'].astype(str)

    crests_df['x'] = (crests_df['transect'] - 40) * 200 / 1000  # Alongshore distance in kilometers
    crests_df['y'] = crests_df['crest_position']  # Cross-shore distance in meters

    ax = axes[idx]

    for label in crests_df['sandbar_label'].unique():       # Plot sandbar crests
        subset = crests_df[crests_df['sandbar_label'] == label].sort_values(by='x')
        color = color_map.get(label, 'black')
        
        if len(subset) > 0:
            scatter = ax.scatter(subset['x'], subset['y'], color=color, s=25, alpha=0.7, label=label)
            ax.plot(subset['x'], subset['y'], linestyle='-', color=color, lw=1)

            if label not in global_handles:
                global_handles[label] = scatter

    ax.set_title(f"{year}", fontsize=12)
    ax.set_ylabel("Cross-shore (m)", fontsize=10)
    if idx == len(years_to_visualize) - 1:
        ax.set_xlabel("Alongshore (km)", fontsize=10)
    ax.legend().set_visible(False)

for ax in axes:
    ax.tick_params(axis='both', which='major', labelsize=8)

for ax in axes[num_years:]:
    ax.axis('off')

fig.legend(
    handles=global_handles.values(),
    labels=global_handles.keys(),
    title="Sandbar labels",
    loc="upper right",
    bbox_to_anchor=(0.5, 0.98),
    ncol=5,
    fontsize=10
)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
