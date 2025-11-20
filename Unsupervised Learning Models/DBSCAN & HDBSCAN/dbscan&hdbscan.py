"""
========================================================
Comparing DBSCAN and HDBSCAN clustering
========================================================
"""

"""
===================================
Introduction
===================================

This code creates two clustering models, DBSCAN and HDBSCAN using the data curated by StatCan containing names, types and locations of
cultural and art facilities across Canada.

Data Source: The Open Database of Cultural and Art Facilities (ODCAF)
"""

"""
===================================
Import the required libraries
===================================
"""
import numpy as np;
import pandas as pd;
import matplotlib.pyplot as plt;
from sklearn.cluster import DBSCAN;
import hdbscan;
from sklearn.preprocessing import StandardScaler;

## Geographical libraries
import geopandas as gpd; # pandas dataframe like geodataframes for geographical data
import contextily as ctx; # for basic map tiles
from shapely.geometry import Point;
import requests;
import zipfile;
import io;
import os;

import warnings;
warnings.filterwarnings("ignore");

"""
===================================
Download the Canada Map
===================================

To get a proper context of the final output, we will need a reference map of Canada.
"""
# URL source for the zip file on the cloud server
zip_file_url: str = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/YcUk-ytgrPkmvZAh5bf7zA/Canada.zip";

# Directory to save the extracted map files from the zip file
output_directory: str = "./Unsupervised Learning Models/DBSCAN & HDBSCAN/canada map/";
os.makedirs(output_directory, exist_ok=True);

# Download the zip file
response = requests.get(zip_file_url);
response.raise_for_status() # Ensure that the request was successful

# Open the zip file in memory
with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
    # Iterate through each file in the zip
    for file_name in zip_ref.namelist():
        # Check if the file is as TTF format file
        if file_name.endswith(".tif"):
            # Extract the file to the output directory we specified.
            zip_ref.extract(file_name, output_directory);
            print(f"Downloaded, extracted and saved in {output_directory}: {file_name}");

"""
===================================
Include a plotting function
===================================

The code for a helper function is provided to help you plot your results. Although you don't need to worry about the details, 
it's quite instructive as it uses a geopandas dataframe and a basemap to plot coloured cluster points on a map of Canada
"""
def plot_clustered_locations(df: pd.DataFrame, title: str="Museums Clustered by Proximity"):
    """
    Plots clustered locations and overlays on a basemap.
    
    Parameters:
    - df: DataFrame containing 'Latitude', 'Longitude', and 'Cluster' columns
    - title: str, title of the plot
    """

    # Load the coordinates into a GeoDataFrame
    gdf: gpd.GeoDataFrame = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["Longitude"], df["Latitude"]), crs="EPSG:4326");

    # Convert to Web Mercator for plotting with contextily
    gdf = gdf.to_crs(epsg=3857);

    # Create the plot
    figure, ax = plt.subplots(figsize=(15, 10));

    # Separate non-noise, or clustered points from noise points or unclustered points
    non_noise = gdf[gdf["Cluster"] != -1];
    noise = gdf[gdf["Cluster"] != -1];

    # Plot the noise points
    noise.plot(ax=ax, color="k", markersize=30, ec="r", alpha=1, label="Noise");

    # Plot the clustered points, coloured by `Cluster` column
    non_noise.plot(ax=ax, column="Cluster", cmap="tab20", markersize=30, ec="k", legend=False, alpha=0.7);

    # Add basemap of Canada
    ctx.add_basemap(ax, source="./Unsupervised Learning Models/DBSCAN & HDBSCAN/canada map/Canada.tif", zoom="4");

    # Format plot
    plt.title(title);
    plt.xlabel("Longitude");
    plt.ylabel("Latitude");
    ax.set_xticks([]);
    ax.set_yticks([]);
    plt.tight_layout();

    # Show the plot
    plt.show();


"""
===================================
Explore the data and extract what you need from it
===================================

Start by loading the data set into a Pandas DataFrame and displaying the first few rows.
"""
url: str = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/r-maSj5Yegvw2sJraT15FA/ODCAF-v1-0.csv";
df: pd.DataFrame = pd.read_csv(url, encoding="ISO-8859-1");
# print("Data view:\n", df.head());

"""
===================
Exercise 1. Explore the table. What do missing values look like in this data?
===================
"""
# print("Missing data values:\n", df.isnull());
# print("Number of missing values in each column:\n", df.isnull().sum());

"""
==================
Exercise 2. Display the facility types and their counts
==================
"""
print("Facility Types:\n", df["ODCAF_Facility_Type"]);
print("Get facility type counts:\n", df["ODCAF_Facility_Type"].value_counts());