# -*- title: Dataset extraction: Sandbar position and characteristics -*-
"""
Created on Thursday 1 March 2025

@author: Raquelgonfe

Description:

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
import math
import matplotlib.colors as mcolors
import imageio
import seaborn as sns
from sklearn.linear_model import LinearRegression


#%% === Jarkus Dataset Configuration ===
nc_file_path = os.path.expanduser("~/Downloads/transect.nc")                                                                                                # Load the dataset
xrds = xr.open_dataset(nc_file_path)
ameland_data = xrds.sel(alongshore=xrds['areacode'] == 3)                                                                                                   # Filter data for Ameland (areacode = 3)

# Extract relevant dimensions
xRSP = ameland_data['cross_shore'][:]                                                                                                                       # Cross-shore coordinates
time = ameland_data['time']
lat = ameland_data['lat']                                                                                                                                   # Latitude for each cross-shore point
lon = ameland_data['lon']                                                                                                                                   # Longitude for each cross-shore point



# === Processing Configuration === 
years_to_process = range(2000, 2001)                                                                                                                        # Define the years to process
window_length = 11                                                                                                                                          # Define the window length 
polyorder = 2                                                                                                                                               # Define polynomial order for smoothing
stable_zone_distance = 350                                                                                                                                  # Stable zone threshold in meters
threshold_percent = 0.01                                                                                                                                    # Threshold for second derivative



# === Function Definition === 
def calculate_stable_zones(cross_shore, second_derivative, threshold, stable_distance):                                                                     # Function to calculate stable zones in the second derivative
    stable_zones = []
    zone_start = None
    for i in range(len(cross_shore) - 1):
        if abs(second_derivative[i]) < threshold:
            if zone_start is None:
                zone_start = cross_shore[i]                                                                                                                 # Start a new stable zone
        else:
            if zone_start is not None:
                zone_length = cross_shore[i] - zone_start
                if zone_length >= stable_distance:
                    stable_zones.append((zone_start, cross_shore[i]))
                zone_start = None
    if zone_start is not None and (cross_shore[-1] - zone_start) >= stable_distance:
        stable_zones.append((zone_start, cross_shore[-1]))
    return stable_zones

def calculate_offshore_boundaries(cross_shore, second_derivative, stable_zones):                                                                            # Function to calculate offshore boundaries based on zero crossings
    offshore_boundaries = []
    for start, end in stable_zones:
        start_idx = np.where(cross_shore >= start)[0][0]
        end_idx = np.where(cross_shore <= end)[0][-1]
        for i in range(start_idx, end_idx):
            if second_derivative[i] < 0 and second_derivative[i + 1] >= 0:
                zero_crossing = cross_shore[i + 1]
                offshore_boundary = zero_crossing + 250
                offshore_boundaries.append((zero_crossing, offshore_boundary))
                break
    return offshore_boundaries



# === Computation: Sandbar crest identification and location of parameters === 
for target_year in years_to_process:
    print(f'Processing year: {target_year}')
    year_index = np.where(ameland_data['time'].dt.year == target_year)[0]
    sandbar_params = []
    
    for transect_index in range(39, 120):                                                                                                                   # Define the study area
        altitude = ameland_data['altitude'][year_index, transect_index, :].values.flatten()
        latitude = lat[transect_index, :].values.flatten()
        longitude = lon[transect_index, :].values.flatten()
        
        valid_indices = ~np.isnan(altitude)
        altitude_clean = altitude[valid_indices]
        xRSP_clean = xRSP[valid_indices].values  
        latitude_clean = latitude[valid_indices]
        longitude_clean = longitude[valid_indices]
        
        if len(altitude_clean) < window_length:
            print(f"Insufficient data for transect {transect_index + 1}, year {target_year}. Skipping.")
            continue
        
        smoothed_altitude = savgol_filter(altitude_clean, window_length, polyorder)
        first_derivative = np.gradient(smoothed_altitude)
        second_derivative = np.gradient(first_derivative)
        
        # === 1. Preconfiguration ===
        # Calculate threshold for stable zones
        second_derivative_range = np.max(second_derivative) - np.min(second_derivative)
        threshold = threshold_percent * second_derivative_range
        
        # Detect stable zones and calculate offshore boundaries
        stable_zones = calculate_stable_zones(xRSP_clean, second_derivative, threshold, stable_zone_distance)
        offshore_boundaries = calculate_offshore_boundaries(xRSP_clean, second_derivative, stable_zones)
        
        # === 2. Detect crest and troughs ===
        crests = []
        for i in range(1, len(first_derivative) - 1):
            if first_derivative[i] < 0 and first_derivative[i - 1] >= 0:                                                                                    # Downward zero-crossing
                if second_derivative[i] < 0 and smoothed_altitude[i] < 0:                                                                                   # Below sea level
                            crests.append((xRSP_clean[i], smoothed_altitude[i]))                                                                            # Store crest position and height
        troughs = []
        for i in range(1, len(first_derivative) - 1):
            if first_derivative[i] > 0 and first_derivative[i - 1] <= 0:                                                                                    # Upward zero-crossing
                if smoothed_altitude[i] < 0:  # Below sea level
                    troughs.append((xRSP_clean[i], smoothed_altitude[i]))
        
        # === 3. Calculate sandbar parameters ===
        peak_label = 1
        for crest in crests:
            crest_position, crest_height = crest

            # Search for nearest left and right troughs
            left_trough = next((t for t in reversed(troughs) if t[0] < crest_position), None)
            right_trough = next((t for t in troughs if t[0] > crest_position), None)

            if right_trough is None:
                right_trough = (float(xRSP_clean[-1]), float(smoothed_altitude[-1]))

            if left_trough and right_trough:
                troughline = right_trough[0] - left_trough[0]
                slope = (right_trough[1] - left_trough[1]) / (right_trough[0] - left_trough[0])
                intersection_altitude = left_trough[1] + slope * (crest_position - left_trough[0])
                height = intersection_altitude - crest_height                                                                                               # PARAMETER 1: HEIGHT (INITIAL CALCULATION, NOT FINAL)
                depth = crest_height                                                                                                                        # PARAMETER 2: DEPTH

                width_line = LineString([(left_trough[0], left_trough[1]), (right_trough[0], right_trough[1])])
                profile_line = LineString(zip(xRSP_clean, smoothed_altitude))
                zero_altitude_line = LineString([(xRSP_clean[0], 0), (xRSP_clean[-1], 0)])                                                                  # Horizontal line at altitude 0
                zero_intersection = profile_line.intersection(zero_altitude_line)
                zero_intersection_x = None
                if not zero_intersection.is_empty:
                    if zero_intersection.geom_type == 'Point':
                        zero_intersection_x = zero_intersection.x                                                                                           # Get the x coordinate of the intersection point
                    elif zero_intersection.geom_type == 'MultiPoint':
                        zero_intersections = list(zero_intersection.geoms)
                        zero_intersection_x = zero_intersections[-1].x                                                                                      # Get the x coordinate of the last (most offshore) intersection point

                if zero_intersection_x is not None:
                    cross_shore_distance = crest_position - zero_intersection_x
                
                intersection_point = profile_line.intersection(width_line)

                valid_intersection_x = []
                intersection_y = None

                # Main section within the loop where intersections are processed
                if not intersection_point.is_empty:                                                                                                         # Check if intersection is not empty
                    if intersection_point.geom_type == 'Point':
                        intersection_points = [intersection_point]                                                                                          # Make it a list for consistency
                    elif intersection_point.geom_type == 'MultiPoint':
                        intersection_points = list(intersection_point.geoms)                                                                                # Convert to list

                    # Collect valid intersection points within the trough bounds
                    valid_intersection_x = []
                    intersection_y = None
                    for point in intersection_points:
                        if left_trough[0] <= point.x <= right_trough[0]:
                            valid_intersection_x.append(point)

                    # Only proceed if there are enough intersection points
                    if len(valid_intersection_x) >= 2:
                        intersection_x = float(valid_intersection_x[1].x)
                        intersection_y = float(valid_intersection_x[1].y)
                    else:
                        intersection_x = None
                        intersection_y = None

                if offshore_boundaries:
                    zero_crossing_x, offshore_boundary_x = offshore_boundaries[-1]
                    smoothed_value_at_offshore = CubicSpline(xRSP_clean, smoothed_altitude)(offshore_boundary_x)

                    # Adjust intersection if the offshore boundary is closer
                    if offshore_boundary_x < intersection_x:
                        intersection_x = offshore_boundary_x
                        intersection_y = smoothed_value_at_offshore
                        slope_offshore = (intersection_y - left_trough[1]) / (intersection_x - left_trough[0])
                        intersection_altitude_width = left_trough[1] + slope_offshore * (intersection_x - left_trough[0])
                        height = intersection_altitude_width - crest_height                                                                                 # PARAMETER 1: HEIGHT (RECALCULATION, FINAL)

                width = abs(intersection_x - left_trough[0]) if intersection_x else None                                                                    # PARAMETER 3: WIDTH (INITIAL CALCULATION, NOT FINAL)

                x_trough = [left_trough[0], intersection_x]
                y_trough = [left_trough[1], intersection_y]

                if x_trough[0] is not None and x_trough[1] is not None and x_trough[0] < x_trough[1]:
                    spline_profile = CubicSpline(xRSP_clean, smoothed_altitude)
                    spline_trough = CubicSpline(x_trough, y_trough)
                    volume = float(spline_profile.integrate(left_trough[0], intersection_x) - spline_trough.integrate(left_trough[0], intersection_x))      # PARAMETER 4: VOLUME

                if intersection_x is not None and left_trough[0] is not None:
                    x_fill = np.linspace(left_trough[0], intersection_x, 100)
                    y_profile_fill = spline_profile(x_fill)
                    y_trough_fill = np.full_like(x_fill, intersection_y)

                for i, x in enumerate(x_fill):
                    if left_trough[0] <= x <= right_trough[0]:
                        y_trough_fill[i] = left_trough[1] + slope * (x - left_trough[0])

                polygon_vertices = list(zip(x_fill, y_profile_fill)) + list(zip(reversed(x_fill), reversed(y_trough_fill)))
                sandbar_polygon = Polygon(polygon_vertices)
                centroid = sandbar_polygon.centroid                                                                                                         # PARAMETER 5: CENTROID
                
                if intersection_y is None or left_trough is None:
                    continue
                
                # Vertical distance (y-direction)
                vertical_distance = abs(intersection_y - left_trough[1])

                max_height = abs(height)
                if height != 0:
                    width2 = (2 * volume * math.sqrt(math.log(10))) / (max_height * math.sqrt(math.pi))                                                     # PARAMETER 3: WIDTH (RECALCULATION, NOT FINAL)
                else:
                    width2 = None
                
                # Calculate the horizontal width (width3) from width2 and vertical distance 
                if width2 is not None and vertical_distance is not None and width2 > vertical_distance:
                    width3 = math.sqrt(width2**2 - vertical_distance**2)                                                                                    # PARAMETER 3: WIDTH (RECALCULATION, FINAL)
                else:
                    width3 = None
                                
                if height < -0.3:
                    crest_index = np.where(xRSP_clean == crest_position)[0][0]
                    sandbar_data = {
                        'year': target_year,
                        'transect': transect_index + 1,
                        'crest_position': float(crest_position),
                        'cross_shore' : float(cross_shore_distance),
                        'height': float(crest_height),
                        'width': float(width) if width is not None else None,
                        'width2' : float(width2) if width2 is not None else None,
                        'width3' : float(width3) if width3 is not None else None,
                        'depth': float(depth),
                        'height': float(height),
                        'volume': float(volume),
                        'centroid_x': float(centroid.x),
                        'centroid_y': float(centroid.y),
                        'latitude': latitude_clean[crest_index].item(),
                        'longitude': longitude_clean[crest_index].item(),
                        'sandbar_label': peak_label,
                        'relative_label': peak_label
                    }

                    if offshore_boundaries and offshore_boundary_x < intersection_x:
                        sandbar_data['intersection_altitude_width'] = float(intersection_altitude_width)

                    sandbar_params.append(sandbar_data)
                    peak_label += 1

    gdf = pd.DataFrame(sandbar_params)
    output_path = f"sandbar_data_{target_year}.csv"                                                                                                         # Saves a csv file for each year with location, transect and parameters
    gdf.to_csv(output_path, index=False)                
    print(f"Sandbar data saved for year {target_year} at {output_path}")


# %%
