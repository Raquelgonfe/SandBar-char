# -*- title: Dataset visualisation for manual modification -*-
"""
Created on Thursday 1 March 2025

@author: Raquelgonfe

Description:
Visualisation of the csv (sandbar_data_{year}.csv) data generated in 2.Dataset_extraction. 

The objective of this code is to visualise the crests with their transect number so it can be manually modified:
    1. Delete those points that you don't consider sandbars.
    2. Relable those points that have the wrong lable.
After manual modification Section 3 "Visualisation of crests with lines between crests", should show clear straight lines representing the sandbar's crest

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

#%% === 1. Visualisation of crests ===

years_to_visualize = range(2000, 2001)                                                                                      # Define the range of years to visualize. 
marker_dict = {1: 'o', 2: 'x', 3: '^', 4: 's', 5: '*', 6: 'P', 7: 'D'}                                                      # Define a marker dictionary for different Sandbar Labels

for year in years_to_visualize:                                                                                             # Loop through each year and visualize the sandbar crests
    csv_file = f'sandbar_data_{year}.csv'                                                                                   # Load the CSV file for the current year into a pandas DataFrame
    crests_df = pd.read_csv(csv_file)

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

for year in years_to_visualize:                                                                                             # Loop through each year and visualize the sandbar crests
    csv_file = f'sandbar_data_{year}.csv'
    crests_df = pd.read_csv(csv_file)

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

years_to_visualize = range(2000, 2001)                                                                                      # Define range of years to visualize

label_mapping = {3: 1, 6: '1a', 2: 2, 4: '2a', 1: 3, 5: 4, 11: '4a', 8: 5, 12: '5a', 9: 6, 13: 7, 10: 'N1', 14: 'N2'}       # Define sandbar label mappings and color map
color_map = {
    1: 'royalblue',  '1a': 'cornflowerblue',  2: 'darkorange',  '2a': 'orangered',  3: 'forestgreen',  
    4: 'mediumorchid',  '4a': 'palevioletred',  5: 'teal',  '5a': 'saddlebrown',  6: 'gray', 
    7: 'gold',  'N1': 'darkkhaki',  'N2': 'peru' }

marker_dict = {1: 'o', 2: 'x', 3: '^', 4: 's', 5: '*', 6: 'P', 7: 'D'}                                                      # Define marker types for each label

for year in years_to_visualize:                                                                                             # Visualize each year
    csv_file = f'sandbar_data_{year}.csv'
    crests_df = pd.read_csv(csv_file)

    crests_df['sandbar_label'] = crests_df['sandbar_label'].replace(label_mapping)                                          # Replace sandbar labels according to the label_mapping

    gdf = gpd.GeoDataFrame(
        crests_df, 
        geometry=gpd.points_from_xy(crests_df['longitude'], crests_df['latitude']),
        crs="EPSG:4326"
    )
    
    gdf = gdf.to_crs(epsg=3857)

    fig, ax = plt.subplots(figsize=(25, 25))

    for label in gdf['sandbar_label'].unique():                                                                             # Plot each sandbar label subset with distinct markers and colors
        subset = gdf[gdf['sandbar_label'] == label]

        subset_sorted = subset.sort_values(by='longitude')                                                                  # Sort by longitude for consistent line plotting

        # Set color and marker based on mappings
        color = color_map.get(label, 'black')                                                                               # Use 'black' if label not in color_map
        marker = marker_dict.get(label, 'o')                                                                                # Default to 'o' if label not in marker_dict

        subset_sorted.plot(ax=ax, marker=marker, color=color, markersize=25, label=f'Sandbar Label {label}')

        plt.plot(subset_sorted['geometry'].x, subset_sorted['geometry'].y, linestyle='-', color=color, lw=2)

    ctx.add_basemap(ax, crs=gdf.crs.to_string(), source=ctx.providers.CartoDB.Positron)

    ax.set_title(f"{year}", fontsize=18)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    plt.legend(title='Sandbar Labels', loc='upper left', bbox_to_anchor=(1, 1), ncol=2)

    plt.tight_layout()
    plt.show()
# %%
