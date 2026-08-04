"""
====================================
Interactive Maps with GeoPandas
====================================

This example demonstrates how to create interactive maps using
the ``.explore()`` method in GeoPandas, which builds on top of Folium.
"""

import geopandas as gpd
import geodatasets

# Load New York City boroughs dataset using geodatasets
nybb = gpd.read_file(geodatasets.get_path("nybb"))

# Create an interactive map visualization
m = nybb.explore(column="BoroName", legend=True, cmap="Set1")

# Display the map object
m