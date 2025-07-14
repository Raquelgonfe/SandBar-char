# -*- title: Sandbar Parameters definition -*-
"""
Created on Thursday 1 March 2025

@author: Raquelgonfe

Description: 
Explanatory figures of the sandbar parameter calculations performed in 2.Dataset_extraction

Two Sections:
    1) Initial parameter definition. Without correction for offshore boundary, computing offshore sandbars too wide
        1.2) Plot
    2) 1st and 2nd derovative analysis, stable zone calculation
    3) Second, coorrect parameter definition. With correction for offshore boundary through the stable zone
        3.2) Plot

The parameters calculated in Dataset_extraction correspond to those shown in Section 3 of this code, so the correct parameters
"""

#%% === Imports ===
import xarray as xr 
import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd
from shapely.geometry import LineString
import os 
from scipy.signal import savgol_filter 
from scipy.interpolate import CubicSpline
import math
import geopandas as gpd
from shapely.geometry import LineString, Polygon



#%% === Dataset configuration ===

nc_file_path = os.path.expanduser("~/Downloads/transect.nc")                                                                # Load the dataset
xrds = xr.open_dataset(nc_file_path) 

ameland_data = xrds.sel(alongshore=xrds['areacode'] == 3)                                                                   # Filter for Ameland by areacode (Ameland's areacode = 3) 
time = ameland_data["time"] 
year = time.dt.year                                                                                                         # Array of the years 

year_filter = ameland_data['time'].dt.year == 2010                                                                          # Filter data for the year 2010
data_2010 = ameland_data.where(year_filter, drop=True)                                                                      # Apply the mask to filter data

xRSP = data_2010['cross_shore'][:]                                                                                          # Cross-shore coordinates
transect_index = 111                                                                                                        # The transect index to analyze
altitude_2010 = data_2010['altitude'][:, transect_index, :].values.squeeze()                                                # Get altitude for 2010

valid_indices = ~np.isnan(altitude_2010)                                                                                    # Indices where altitude is valid
altitude_clean_2010 = altitude_2010[valid_indices]                                                                          # Clean NaN values from the altitude 
xRSP_clean_2010 = xRSP[valid_indices]                                                                                       # Ensure cross-shore coordinates match cleaned altitude 

cross_shore = xRSP_clean_2010                                                                                               # Cross-shore coordinates (already cleaned)
altitude = altitude_clean_2010                                                                                              # Altitude values (already cleaned)

window_length = 12                                                                                                          # Adjust window size as needed
polyorder = 2                                                                                                               # Polynomial order for smoothing
smoothed_altitude = savgol_filter(altitude, window_length=window_length, polyorder=polyorder)                               # Savitzky-Golay filter for smoothing



#%% === 1. Initial Parameter definition ===

first_derivative = np.gradient(smoothed_altitude)                                                                           # Calculate first and second derivatives for crest and trough detection
second_derivative = np.gradient(first_derivative)

crests = []
troughs = []
baseline = 0  # Sea level (altitude = 0)
#max_cross_shore_value = 1000  # Limiting cross-shore distance for analysis

for i in range(1, len(first_derivative) - 1):                                                                               # Identify crests (downward zero-crossings in first derivative and negative second derivative)
    if first_derivative[i] < 0 and first_derivative[i - 1] >= 0 and second_derivative[i] :
        if smoothed_altitude[i] < baseline:
            crests.append((cross_shore[i], smoothed_altitude[i]))

for i in range(1, len(first_derivative) - 1):                                                                               # Identify troughs (upward zero-crossings in first derivative)
    if first_derivative[i] > 0 and first_derivative[i - 1] <= 0 and smoothed_altitude[i] < baseline:
        troughs.append((cross_shore[i], smoothed_altitude[i]))

sandbar_params = []                                                                                                         # Calculate sandbar parameters
peak_label = 1
for crest in crests:
    crest_position, crest_height = crest

    left_trough = next((t for t in reversed(troughs) if t[0] < crest_position), None)                                       # Search for nearest left and right troughs
    right_trough = next((t for t in troughs if t[0] > crest_position), None)

    if right_trough is None:
        right_trough = (cross_shore[-1], smoothed_altitude[-1])                                                             # Use last point as right trough

    if left_trough is not None and right_trough is not None:
        troughline = right_trough[0] - left_trough[0]                                                                       # Calculate troughline (original width between left and right troughs)

        slope = (right_trough[1] - left_trough[1]) / (right_trough[0] - left_trough[0])                                     # Calculate slope of the line between the troughs

        intersection_altitude = left_trough[1] + slope * (crest_position - left_trough[0])                                  # Calculate the altitude of the intersection point along the slope line

        height = intersection_altitude - crest_height                                                                       # Calculate height as the difference between intersection altitude and crest height

        depth = crest_height - baseline                                                                                     # Depth is the distance from the crest to the sea level (baseline)

        width_line = LineString([(left_trough[0], left_trough[1]), (right_trough[0], right_trough[1])])                     # Create the width line (between left trough and right trough)
        profile_line = LineString(zip(cross_shore, smoothed_altitude))                                                      # Create the smoothed profile line

        intersection_point = profile_line.intersection(width_line)                                                          # Find intersection point

        valid_intersection_x = []                                                                                           # Check if intersection points are valid and filter them based on troughs
        intersection_y = None                                                                                               # Initialize intersection_y

        if not intersection_point.is_empty:                                                                                 # Check if intersection is not empty
            if intersection_point.geom_type == 'Point':
                intersection_points = [intersection_point]                                                                  # Make it a list for consistency
            elif intersection_point.geom_type == 'MultiPoint':
                intersection_points = list(intersection_point.geoms)                                                        # Convert to list

            for point in intersection_points:                                                                               # Filter intersection points that lie between the left and right troughs
                if left_trough[0] <= point.x <= right_trough[0]:
                    valid_intersection_x.append(point)

            if valid_intersection_x:                                                                                        # If we have valid intersection points, choose the second or closest one
                intersection_x = valid_intersection_x[1].x                                                                  # Choose the second valid point
                intersection_y = valid_intersection_x[1].y                                                                  # Choose the corresponding y value

        if intersection_x is not None and intersection_y is not None:                                                       # Calculate the new width as the Euclidean distance between left trough and intersection point
            width2 = math.sqrt((intersection_x - left_trough[0]) ** 2 + (intersection_y - left_trough[1]) ** 2)
        else:
            width2 = None

        # Calculate the volume by integrating the smoothed profile and the trough line
        x_trough = [left_trough[0], intersection_x]                                                                         # x-coordinates of the left trough and intersection point
        y_trough = [left_trough[1], intersection_y]                                                                         # y-coordinates based on their heights

        # Create cubic splines for the smoothed profile and the trough line
        spline_profile = CubicSpline(cross_shore, smoothed_altitude)                                                        # Smoothed profile spline
        spline_trough = CubicSpline(x_trough, y_trough)                                                                     # Trough line spline

        # Calculate volume using definite integral
        volume = (spline_profile.integrate(left_trough[0], intersection_x) -
                   spline_trough.integrate(left_trough[0], intersection_x))
      
        # Define points for polygon creation
        x_fill = np.linspace(left_trough[0], intersection_x, 100)                                                           # Generate points between left and right trough
        y_profile_fill = spline_profile(x_fill)                                                                             # Get heights from the smoothed profile
        
        # Construct the polygon vertices using left trough and right trough heights
        y_trough_fill = np.full_like(x_fill, intersection_y)                                                                # Use the left trough height as baseline for the polygon
        
        for i, x in enumerate(x_fill):                                                                                      # Calculate the y values of the trough line based on the x_fill range
            if left_trough[0] <= x <= right_trough[0]:
                y_trough_fill[i] = left_trough[1] + slope * (x - left_trough[0])                                            # Interpolate the trough height

        polygon_vertices = list(zip(x_fill, y_profile_fill)) + list(zip(reversed(x_fill), reversed(y_trough_fill)))         # Create vertices for the polygon

        sandbar_polygon = Polygon(polygon_vertices)                                                                         # Create polygon and calculate centroid
        centroid = sandbar_polygon.centroid
        
        # Store sandbar parameters if height is significant
        if height < -0.3:                                                                                                   # Only consider sandbars with significant height
            if volume > 10:
                sandbar_params.append({
                    'crest_position': crest_position,
                    'crest_height': crest_height,
                    'troughline': troughline,                                                                               # Store original width as troughline
                    'width2': width2,                                                                                       # Store new width
                    'depth': depth,
                    'height': height,
                    'left_trough': left_trough,
                    'right_trough': right_trough,
                    'intersection_altitude': intersection_altitude,
                    'sandbar_label': peak_label,
                    'intersection_x': intersection_x if valid_intersection_x else None,                                     # Store x coordinate of intersection
                    'intersection_y': intersection_y if valid_intersection_x else None,                                     # Store y coordinate of intersection
                    'volume': volume,                                                                                       # Store the calculated volume
                    'centroid': centroid                                                                                    # Store the centroid coordinates
                })

                peak_label += 1                                                                                             # Increment peak label for the next sandbar


gdf = gpd.GeoDataFrame(sandbar_params)

# Output the sandbar parameters, including centroids
#print(gdf[['sandbar_label', 'centroid']])



#%% === 1.2. Plotting Initial Parameter definition ===
plt.figure(figsize=(10, 8))
plt.plot(cross_shore, altitude, label='Profile', color='gray', linewidth=2, alpha=0.6)
plt.plot(cross_shore, smoothed_altitude, label='Smoothed Profile', color='black', linewidth=2)
plt.axhline(y=baseline, color='gray', linestyle='--', label='Sea Level')

for param in sandbar_params:                                                                                                # Iterate through all detected sandbar parameters to plot
    crest_pos = param['crest_position']
    crest_height = param['crest_height']
    
    troughline = param.get('troughline')                                                                                    # Access the original troughline
    depth = param['depth']
    height = param['height']
    left_trough = param['left_trough']
    right_trough = param['right_trough']
    intersection_x = param.get('intersection_x')  
    intersection_y = param.get('intersection_y')  
    intersection_altitude = param.get('intersection_altitude')
    width2 = param.get('width2') 

    plt.plot(crest_pos, crest_height, 'ro', label='Crest' if 'Crest' not in plt.gca().get_legend_handles_labels()[1] else "")
    plt.text(crest_pos, crest_height + 0.3, f'Crest Pos: {crest_pos:.2f}\nHeight: {height:.2f}', color='purple', fontsize=9)
    
    plt.plot(left_trough[0], left_trough[1], 'bo', label='Left Trough' if 'Left Trough' not in plt.gca().get_legend_handles_labels()[1] else "")
    plt.plot(right_trough[0], right_trough[1], 'bo', label='Right Trough' if 'Right Trough' not in plt.gca().get_legend_handles_labels()[1] else "")
    
    if intersection_x is not None and intersection_y is not None:
        plt.plot(intersection_x, intersection_y, 'go', label='Intersection Point' if 'Intersection Point' not in plt.gca().get_legend_handles_labels()[1] else "")
        
        plt.plot([crest_pos, crest_pos], [crest_height, intersection_altitude], 'purple', linestyle='--', label='Height' if 'Height' not in plt.gca().get_legend_handles_labels()[1] else "")
        
    plt.plot([crest_pos, crest_pos], [crest_height, baseline], 'r--', label='Depth' if 'Depth' not in plt.gca().get_legend_handles_labels()[1] else "")
    plt.text(crest_pos + 20, crest_height - 0.5, f'Depth: {depth:.2f}', color='red', fontsize=9)

    if troughline is not None:
        plt.plot([left_trough[0], right_trough[0]], [left_trough[1], right_trough[1]], 'g--', label='Troughline' if 'Troughline' not in plt.gca().get_legend_handles_labels()[1] else "")
    
    if width2 is not None and intersection_x is not None:
        plt.plot([left_trough[0], intersection_x], [left_trough[1], intersection_y], 'orange', linestyle='--', label='Width' if 'Width' not in plt.gca().get_legend_handles_labels()[1] else "")
        plt.text(intersection_x - 300, intersection_y - 0.05, f'Width: {width2:.2f}', color='orange', fontsize=9)

    x_trough = [left_trough[0], intersection_x]                                                                             # x-coordinates of the left trough and intersection point
    y_trough = [left_trough[1], intersection_y]                                                                             # y-coordinates of the trough line

    # Create cubic splines for the smoothed profile and the trough line
    spline_profile = CubicSpline(cross_shore, smoothed_altitude)                                                            # Smoothed profile spline
    spline_trough = CubicSpline(x_trough, y_trough)                                                                         # Trough line spline

    # Calculate the volume between the smoothed profile and the trough line using definite integrals
    volume = (spline_profile.integrate(left_trough[0], intersection_x) -
              spline_trough.integrate(left_trough[0], intersection_x))

    # Shade the area between the profile and the trough line
    x_fill = np.linspace(left_trough[0], intersection_x, 100)
    y_profile_fill = spline_profile(x_fill)
    y_trough_fill = spline_trough(x_fill)
    
    plt.fill_between(x_fill, y_profile_fill, y_trough_fill, color='gray', alpha=0.5, label='Sandbar Volume' if 'Sandbar Volume' not in plt.gca().get_legend_handles_labels()[1] else "")
    
    # Label the volume on the plot
    mid_x = (left_trough[0] + intersection_x) / 2
    mid_y = (np.max(y_profile_fill) + np.min(y_trough_fill)) / 2
    plt.text(mid_x, mid_y, f'Volume: {volume:.2f}', color='blue', fontsize=10)

    # Plot the centroid
    centroid = param['centroid']
    plt.plot(centroid.x, centroid.y, 'ms', markersize=8, label='Centroid' if 'Centroid' not in plt.gca().get_legend_handles_labels()[1] else "")
    plt.text(centroid.x, centroid.y + 0.3, f'Centroid: ({centroid.x:.2f}, {centroid.y:.2f})', color='magenta', fontsize=9)

plt.xlim(0, 2500)                                                                                                           # Adjust according to your data
plt.ylim(-10, baseline + 0.5)                                                                                               # Adjust altitude limits

plt.title('Sandbar parameters', fontsize=16)
plt.xlabel('Cross-shore Distance (m)', fontsize=14)
plt.ylabel('Altitude (m)', fontsize=14)
plt.legend(fontsize=12, loc='lower left')
plt.grid(True)

plt.tight_layout()
plt.show()



#%% === 2. 1st and 2nd derovative analysis, stable zone calculation ===

stable_zone_distance = 350                                                                                                  # Distance in meters to consider for stability
threshold_percent = 0.01                                                                                                    # Percentage threshold

second_derivative_min, second_derivative_max = np.min(second_derivative), np.max(second_derivative)                         # Calculate threshold based on percentage of the second derivative range
second_derivative_range = second_derivative_max - second_derivative_min
threshold = threshold_percent * second_derivative_range

stable_zones = []
zone_start = None

for i in range(len(cross_shore) - 1):
    if abs(second_derivative[i]) < threshold:
        if zone_start is None:
            zone_start = cross_shore[i]                                                                                     # Start a new stable zone
    else:
        if zone_start is not None:
            zone_length = cross_shore[i] - zone_start                                                                       # Calculate threshold based on percentage of the second derivative range
            if zone_length >= stable_zone_distance:
                stable_zones.append((zone_start, cross_shore[i]))                                                           # Append the current stable zone
            zone_start = None                                                                                               # Reset zone start

if zone_start is not None:                                                                                                  # Calculate threshold based on percentage of the second derivative range
    zone_length = cross_shore[-1] - zone_start
    if zone_length >= stable_zone_distance:
        stable_zones.append((zone_start, cross_shore[-1]))                                                                  # Use the last cross_shore position

offshore_boundaries = []                                                                                                    # Find offshore boundary for the last stable zone
for start, end in stable_zones:
    # Get indices for the start and end of the stable zone
    start_idx = np.where(cross_shore >= start)[0][0]
    end_idx = np.where(cross_shore <= end)[0][-1]
    
    for i in range(start_idx, end_idx):                                                                                     # Find the first zero crossing in the second derivative
        if second_derivative[i] < 0 and second_derivative[i + 1] >= 0:                                                      # Downward to upward crossing
            zero_crossing = cross_shore[i + 1]                                                                              # This is the x position of the zero crossing
            offshore_boundary = zero_crossing + 200                                                                         # Add 250m offshore
            offshore_boundaries.append((zero_crossing, offshore_boundary))
            break                                                                                                           # Stop after finding the first crossing

fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

axs[0].plot(cross_shore, smoothed_altitude, color='blue', label='Smoothed original profile')
axs[0].set_xlim(0, 2000)  
axs[0].set_ylim(-10, 2.5)
axs[0].axhline(0, color='black', linestyle='--', linewidth=0.8)  
axs[0].set_ylabel("Altitude (m)")
for zero_crossing, offshore_boundary in offshore_boundaries:
    axs[0].axvline(zero_crossing, color='gray', linestyle=':', label='Zero crossing in d2z/d2x')
    axs[0].axvline(offshore_boundary, color='black', linestyle=':', label='Offshore boundary')
axs[0].legend()

axs[1].plot(cross_shore, first_derivative, color='green', label="First derivative")
axs[1].set_xlim(0, 2000)
axs[1].set_ylim(-0.4, 0.4)
axs[1].axhline(0, color='black', linestyle='--', linewidth=0.8)  
axs[1].set_ylabel("dz/dx (m/m)")
axs[1].legend()

axs[2].plot(cross_shore, second_derivative, color='red', label="Second derivative")
for start, end in stable_zones:
    axs[2].fill_betweenx([-0.1, 0.1], start, end, color='yellow', alpha=0.3, label="Stable Zone" if start == stable_zones[0][0] else "")
for zero_crossing, offshore_boundary in offshore_boundaries:
    axs[2].axvline(zero_crossing, color='gray', linestyle=':', label='Zero crossing' if zero_crossing == offshore_boundaries[0][0] else "")
    axs[2].axvline(offshore_boundary, color='black', linestyle=':', label='Offshore boundary' if offshore_boundary == offshore_boundaries[0][1] else "")
axs[2].set_xlim(0, 2500)
axs[2].set_ylim(-0.1, 0.1)
axs[2].axhline(0, color='black', linestyle='--', linewidth=0.8)
axs[2].set_xlabel("Cross-shore distance (m)")
axs[2].set_ylabel("d²z/dx² (m²/m²)")
axs[2].legend()

plt.tight_layout()
plt.show()



#%% === 3. Second, coorrect parameter definition ===

stable_zone_distance = 350                                                                                                  # Distance in meters to consider for stability
threshold_percent = 0.01                                                                                                    # Percentage threshold

second_derivative_min, second_derivative_max = np.min(second_derivative), np.max(second_derivative)                         # Calculate threshold based on percentage of the second derivative range
second_derivative_range = second_derivative_max - second_derivative_min
threshold = threshold_percent * second_derivative_range

stable_zones = []                                                                                                           # Identify stable zones in the second derivative where values are within the threshold over the specified distance
zone_start = None

for i in range(len(cross_shore) - 1):
    if abs(second_derivative[i]) < threshold:
        if zone_start is None:
            zone_start = cross_shore[i]                                                                                     # Start a new stable zone
    else:
        if zone_start is not None:
            zone_length = cross_shore[i] - zone_start                                                                       # Calculate the length of the stable zone
            if zone_length >= stable_zone_distance:
                stable_zones.append((zone_start, cross_shore[i]))                                                           # Append the current stable zone
            zone_start = None                                                                                               # Reset zone start

if zone_start is not None:                                                                                                  # If there is a zone still open at the end of the loop, check its length
    zone_length = cross_shore[-1] - zone_start
    if zone_length >= stable_zone_distance:
        stable_zones.append((zone_start, cross_shore[-1]))                                                                  # Use the last cross_shore position

offshore_boundaries = []                                                                                                    # Find offshore boundary for the last stable zone
for start, end in stable_zones:
    start_idx = np.where(cross_shore >= start)[0][0]                                                                        # Get indices for the start and end of the stable zone
    end_idx = np.where(cross_shore <= end)[0][-1]
    
    for i in range(start_idx, end_idx):                                                                                     # Find the first zero crossing in the second derivative
        if second_derivative[i] < 0 and second_derivative[i + 1] >= 0:                                                      # Downward to upward crossing
            zero_crossing = cross_shore[i + 1]                                                                              # This is the x position of the zero crossing
            offshore_boundary = zero_crossing + 250                                                                         # Correctly add 250m offshore
            offshore_boundaries.append((zero_crossing, offshore_boundary))
            break                                                                                                           # Stop after finding the first crossing

sandbar_params = []                                                                                                         # Calculate sandbar parameters
peak_label = 1
for crest in crests:
    crest_position, crest_height = crest

    left_trough = next((t for t in reversed(troughs) if t[0] < crest_position), None)                                       # Search for nearest left and right troughs
    right_trough = next((t for t in troughs if t[0] > crest_position), None)

    if right_trough is None:
        right_trough = (cross_shore[-1], smoothed_altitude[-1])                                                             # Use last point as right trough

    if left_trough is not None and right_trough is not None:
        troughline = right_trough[0] - left_trough[0]                                                                       # Calculate troughline (original width between left and right troughs)
        slope = (right_trough[1] - left_trough[1]) / (right_trough[0] - left_trough[0])                                     # Calculate slope of the line between the troughs
        intersection_altitude = left_trough[1] + slope * (crest_position - left_trough[0])                                  # Calculate the altitude of the intersection point along the slope line
        height = intersection_altitude - crest_height                                                                       # Calculate height as the difference between intersection altitude and crest height
        depth = crest_height - baseline                                                                                     # Depth is the distance from the crest to the sea level (baseline)
        width_line = LineString([(left_trough[0], left_trough[1]), (right_trough[0], right_trough[1])])                     # Create the width line (between left trough and right trough)
        profile_line = LineString(zip(cross_shore, smoothed_altitude))                                                      # Create the smoothed profile line
        
        # Calculate the cross-shore distance relative to the 0 altitude point
        zero_altitude_line = LineString([(cross_shore[0], 0), (cross_shore[-1], 0)])                                        # Horizontal line at altitude 0
        zero_intersection = profile_line.intersection(zero_altitude_line)
        zero_intersection_x = None
        if not zero_intersection.is_empty:
            if zero_intersection.geom_type == 'Point':
                zero_intersection_x = zero_intersection.x                                                                   # Get the x coordinate of the intersection point
            elif zero_intersection.geom_type == 'MultiPoint':
                zero_intersections = list(zero_intersection.geoms)
                zero_intersection_x = zero_intersections[-1].x                                                              # Get the x coordinate of the last (most offshore) intersection point

        if zero_intersection_x is not None:
            cross_shore_distance = crest_position - zero_intersection_x

        intersection_point = profile_line.intersection(width_line)                                                          # Find intersection point

        valid_intersection_x = []                                                                                           # Check if intersection points are valid and filter them based on troughs
        intersection_y = None                                                                                               # Initialize intersection_y

        if not intersection_point.is_empty:                                                                                 # Check if intersection is not empty
            if intersection_point.geom_type == 'Point':
                intersection_points = [intersection_point]                                                                  # Make it a list for consistency
            elif intersection_point.geom_type == 'MultiPoint':
                intersection_points = list(intersection_point.geoms)                                                        # Convert to list

            for point in intersection_points:                                                                               # Filter intersection points that lie between the left and right troughs
                if left_trough[0] <= point.x <= right_trough[0]:
                    valid_intersection_x.append(point)

            if valid_intersection_x:                                                                                        # If we have valid intersection points, choose the second or closest one
                intersection_x = valid_intersection_x[1].x                                                                  # Choose the second valid point
                intersection_y = valid_intersection_x[1].y                                                                  # Choose the corresponding y value

        if offshore_boundaries:                                                                                             # Determine whether to use the intersection point or the offshore boundary
            zero_crossing_x, offshore_boundary_x = offshore_boundaries[-1]                                                  # Get the last offshore boundary calculated
            
            smoothed_value_at_offshore = CubicSpline(cross_shore, smoothed_altitude)(offshore_boundary_x)                   # Calculate the smoothed profile value at the offshore boundary's x position
            
            # Compare distances to decide which point to use
            if offshore_boundary_x is not None and intersection_x is not None:
                if offshore_boundary_x < intersection_x:
                    intersection_x = offshore_boundary_x                                                                    # Use the offshore boundary's x position and the corresponding smoothed profile value
                    intersection_y = smoothed_value_at_offshore
                    slope_offshore = (intersection_y - left_trough[1]) / (intersection_x - left_trough[0])                  # Recalculate height as distance from the crest to the new intersection point (width line)
                    intersection_altitude_width = left_trough[1] + slope_offshore * (intersection_x - left_trough[0])
                    height = intersection_altitude_width - crest_height                                                     # Recalculate based on the width line intersection
        else:
            offshore_boundary_x = None 
        if intersection_x is not None and intersection_y is not None:                                                       # Calculate the new width as the Euclidean distance between left trough and intersection point
            #width2 = math.sqrt((intersection_x - left_trough[0]) ** 2 + (intersection_y - left_trough[1]) ** 2)
            width2 = abs(intersection_x - left_trough[0])
        else:
            width2 = None

        # Calculate the volume by integrating the smoothed profile and the trough line
        x_trough = [left_trough[0], intersection_x]                                                                         # x-coordinates of the left trough and intersection point
        y_trough = [left_trough[1], intersection_y]                                                                         # y-coordinates based on their heights

        if x_trough[0] < x_trough[1]:
            # Create cubic splines for the smoothed profile and the trough line
            spline_profile = CubicSpline(cross_shore, smoothed_altitude)                                                    # Smoothed profile spline
            spline_trough = CubicSpline(x_trough, y_trough)                                                                 # Trough line spline

            volume = (spline_profile.integrate(left_trough[0], intersection_x) -                                            # Calculate volume using definite integral
                    spline_trough.integrate(left_trough[0], intersection_x))

        max_height = abs(crest_height)                                                                                      # Calculate width3 from volume and crest_height
        fraction_k = 10
        if crest_height != 0:                                                                                               # Ensure we don't divide by zero in case of flat profiles
            width3 = (2 * volume * math.sqrt(math.log(fraction_k))) / (max_height * math.sqrt(math.pi))
        else:
            width3 = None                                                                                                   # Assign None if crest height is zero
        # Define points for polygon creation
        x_fill = np.linspace(left_trough[0], intersection_x, 100)                                                           # Generate points between left and right trough
        y_profile_fill = spline_profile(x_fill)                                                                             # Get heights from the smoothed profile
        
        # Construct the polygon vertices using left trough and right trough heights
        y_trough_fill = np.full_like(x_fill, intersection_y)                                                                # Use the left trough height as baseline for the polygon
        
        for i, x in enumerate(x_fill):                                                                                      # Calculate the y values of the trough line based on the x_fill range
            if left_trough[0] <= x <= right_trough[0]:
                # Interpolate the trough height
                y_trough_fill[i] = left_trough[1] + slope * (x - left_trough[0])

        polygon_vertices = list(zip(x_fill, y_profile_fill)) + list(zip(reversed(x_fill), reversed(y_trough_fill)))         # Create vertices for the polygon

        # Create polygon and calculate centroid
        sandbar_polygon = Polygon(polygon_vertices)
        centroid = sandbar_polygon.centroid
        
        # Store sandbar parameters if height is significant
        if height < -0.3:                                                                                                   # Only consider sandbars with significant height
            sandbar_data = {
                'crest_position': crest_position,
                'crest_height': crest_height,
                'troughline': troughline,                                                                                   # Store original width as troughline
                'width2': width2,                                                                                           # Store new width
                'width3': width3,                                                                                           # Store new width3
                'depth': depth,
                'height': height,
                'left_trough': left_trough,
                'right_trough': right_trough,
                'intersection_altitude': intersection_altitude,
                'sandbar_label': peak_label,
                'intersection_x': intersection_x if valid_intersection_x else None,                                         #  Store x coordinate of intersection
                'intersection_y': intersection_y if valid_intersection_x else None,                                         # Store y coordinate of intersection
                'volume': volume,                                                                                           # Store the calculated volume
                'centroid': centroid,                                                                                       # Store the centroid coordinates
                'cross_shore_distance': cross_shore_distance
            }

            if offshore_boundary_x is not None and intersection_x is not None:                                              # Add 'intersection_altitude_width' if it's not None
                if offshore_boundary_x < intersection_x:
                    sandbar_data['intersection_altitude_width'] = intersection_altitude_width

            sandbar_params.append(sandbar_data)                                                                             # Append the data to sandbar_params
            peak_label += 1                                                                                                 # Increment peak label for the next sandbar

gdf = gpd.GeoDataFrame(sandbar_params)
#print(cross_shore_distance)



#%% === 3.1. Plotting coorrect parameter definition ===

plt.figure(figsize=(10, 8))
plt.plot(cross_shore, altitude, label='Profile', color='gray', linewidth=2, alpha=0.6)
plt.plot(cross_shore, smoothed_altitude, label='Smoothed Profile', color='black', linewidth=2)
plt.axhline(y=baseline, color='gray', linestyle='--', label='Sea Level')
for param in sandbar_params:
    crest_pos = param['crest_position']
    crest_height = param['crest_height']
    
    troughline = param.get('troughline')                                                                                    # Access the original troughline
    depth = param['depth']
    height = param['height']
    left_trough = param['left_trough']
    right_trough = param['right_trough']
    intersection_x = param.get('intersection_x')                                                                            # Use .get() to avoid KeyError
    intersection_y = param.get('intersection_y')                                                                            # Use .get() to avoid KeyError
    intersection_altitude = param.get('intersection_altitude')
    width2 = param.get('width2')                                                                                            # Access the stored width2
    width3 = param.get('width3') 

    print(width2)
    print(width3)
    plt.plot(crest_pos, crest_height, 'ro', label='Crest' if 'Crest' not in plt.gca().get_legend_handles_labels()[1] else "")
    plt.text(crest_pos, crest_height + 0.3, f'Crest Pos: {crest_pos:.2f}\nHeight: {height:.2f}', color='purple', fontsize=9)
    
    plt.plot(left_trough[0], left_trough[1], 'bo', label='Left Trough' if 'Left Trough' not in plt.gca().get_legend_handles_labels()[1] else "")
    plt.plot(right_trough[0], right_trough[1], 'bo', label='Right Trough' if 'Right Trough' not in plt.gca().get_legend_handles_labels()[1] else "")
    
    if intersection_x is not None and intersection_y is not None:
        plt.plot(intersection_x, intersection_y, 'go', label='Intersection Point' if 'Intersection Point' not in plt.gca().get_legend_handles_labels()[1] else "")
        plt.plot([crest_pos, crest_pos], [crest_height, intersection_altitude], 'purple', linestyle='--', label='Height' if 'Height' not in plt.gca().get_legend_handles_labels()[1] else "")

    plt.plot([crest_pos, crest_pos], [crest_height, baseline], 'r--', label='Depth' if 'Depth' not in plt.gca().get_legend_handles_labels()[1] else "")
    plt.text(crest_pos + 20, crest_height - 0.5, f'Depth: {depth:.2f}', color='red', fontsize=9)

    if troughline is not None:
        plt.plot([left_trough[0], right_trough[0]], [left_trough[1], right_trough[1]], 'g--', label='Troughline' if 'Troughline' not in plt.gca().get_legend_handles_labels()[1] else "")
    
    if width2 is not None and intersection_x is not None:
        plt.plot([left_trough[0], intersection_x], [intersection_y, intersection_y], 'orange', linestyle='--', label='Width' if 'Width' not in plt.gca().get_legend_handles_labels()[1] else "")
        plt.text(intersection_x - 300, intersection_y - 0.25, f'Width: {width2:.2f}', color='orange', fontsize=9)
    
    # Calculate and display the volume
    x_trough = [left_trough[0], intersection_x]                                                                             # x-coordinates of the left trough and intersection point
    y_trough = [left_trough[1], intersection_y]                                                                             # y-coordinates of the trough line

    # Create cubic splines for the smoothed profile and the trough line
    spline_profile = CubicSpline(cross_shore, smoothed_altitude)                                                            # Smoothed profile spline
    spline_trough = CubicSpline(x_trough, y_trough)                                                                         # Trough line spline

    volume = (spline_profile.integrate(left_trough[0], intersection_x) -                                                    # Calculate the volume between the smoothed profile and the trough line using definite integrals
              spline_trough.integrate(left_trough[0], intersection_x))

    x_fill = np.linspace(left_trough[0], intersection_x, 100)                                                               # Shade the area between the profile and the trough line
    y_profile_fill = spline_profile(x_fill)
    y_trough_fill = spline_trough(x_fill)
    
    plt.fill_between(x_fill, y_profile_fill, y_trough_fill, color='gray', alpha=0.5, label='Sandbar Volume' if 'Sandbar Volume' not in plt.gca().get_legend_handles_labels()[1] else "")
    mid_x = (left_trough[0] + intersection_x) / 2                                                                           # Label the volume on the plot
    mid_y = (np.max(y_profile_fill) + np.min(y_trough_fill)) / 2
    plt.text(mid_x, mid_y, f'Volume: {volume:.2f}', color='blue', fontsize=10)

    centroid = param['centroid']
    plt.plot(centroid.x, centroid.y, 'ms', markersize=8, label='Centroid' if 'Centroid' not in plt.gca().get_legend_handles_labels()[1] else "")
    plt.text(centroid.x, centroid.y + 0.3, f'Centroid: ({centroid.x:.2f}, {centroid.y:.2f})', color='magenta', fontsize=9)

plt.xlim(0, 2500)                                                                                                           # Adjust according to your data
plt.ylim(-10, baseline + 0.5)                                                                                               # Adjust altitude limits

plt.title('Sandbar Parameters', fontsize=16)
plt.xlabel('Cross-shore Distance (m)', fontsize=14)
plt.ylabel('Altitude (m)', fontsize=14)
plt.legend(fontsize=12, loc='upper right')
plt.grid(True)

plt.tight_layout()
plt.show()



# %%
