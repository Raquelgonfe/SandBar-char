# SandBar-char
Sandbar characterization from bathymetry profiles

This repository contains Python scripts for identifying and characterizing sandbars from coastal bathymetry profiles using data from the JarKus dataset (publicly available via Deltares OPeNDAP).

JarKus bathymetry data can be accessed through Deltares OPeNDAP:  
https://opendap.deltares.nl/thredds/catalog/opendap/rijkswaterstaat/jarkus/profiles/catalog.html?dataset=varopendap/rijkswaterstaat/jarkus/profiles/transect.nc

## Objective

- Identify sandbars within bathymetry profiles.  
- Track and connect sandbars spatially and temporally (with some manual adjustments).  
- Compute and store key geometric and positional parameters of each sandbar.  
- Generate time-series data for individual sandbars.  
- Visualize the 3D development of sandbars over time.

## Parameter description

1. `year`            = Year of measurement  
2. `transect`        = Transect index (from 0 to number of transects for Ameland)  
3. `crest_position`  = Distance from crest to beach pole (horizontal)  
4. `cross_shore`     = Distance from crest to 0-depth water line (horizontal)  
5. `height`          = Crest and the sandbar's bottom/width line vertical height  
6. `width`           = Distance between sandbar's onshore and offshore boundary (horizontal), before offshore correction  
7. `width2`          = Width estimated from volume assuming a normal shape  
8. `width3`          = Width from corrected volume (adjusted offshore boundary)  
9. `depth`           = Elevation of crest relative to 0-depth line  
10. `volume`         = Integrated volume between onshore and offshore boundaries  
11. `centroid_x`     = X-coordinate of sandbar polygon centroid  
12. `centroid_y`     = Y-coordinate of sandbar polygon centroid  
13. `latitude`       = Latitude of the crest  
14. `longitude`      = Longitude of the crest  
15. `sandbar_label`  = Unique ID of the sandbar in the dataset  
16. `relative_label` = Relative position: 1 = onshore, 2 = middle, 3 = offshore sandbar  

A visual reference is included in the repository under the image file **SandbarParameters.png**.

## Code structure

1. `1_First_visualisation.py`  
2. `2_Dataset_extraction.py`  
3. `3_Parameter_definition.py`  
4. `4_Data_visualisation.py`  
5. `5_Plotting.py`

## Manual step required

While sandbar detection and parameter computation are automated, manual adjustments on the "sandbar_data_{year}.csv" (2_Dataset_extraction output) are needed to verify and correct sandbar labels across years and profiles.
