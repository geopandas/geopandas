"""
Clip Vector Data with GeoPandas
==================================================================

Learn how to clip point, line, or polygon geometries to the boundary
of a polygon geometry using GeoPandas.

"""

###############################################################################
# Clip Vector Data in Python Using GeoPandas
# ---------------------------------------------
#
# The example below shows you how to clip a set of vector geometries
# to the spatial extent / shape of another vector object. Both sets of geometries
# must be opened with GeoPandas as GeoDataFrames and be in the same Coordinate
# Reference System (CRS) for the ``clip()`` function in EarthPy to work.
#
# This example uses Polygons, a Line, and Points made with shapely and then
# turned into GeoDataframes.
#
# .. note::
#    The object to be clipped will be clipped to the full extent of the clip
#    object. If there are multiple polygons in clip object, the input data will
#    be clipped to the total boundary of all polygons in clip object.

###############################################################################
# Import Packages
# ---------------
#
# To begin, import the needed packages.

import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Polygon, LineString, Point
import geopandas.clip as gc

###############################################################################
# Create Example Data
# -------------------
#
# Below, some point, line and polygon geometries are created and coerced into
# GeoDataFrames to demonstrate the use of clip.

polygon1 = Polygon([(0, 0), (0, 10), (10, 10), (10, 0), (0, 0)])
polygon2 = Polygon([(-5, -5), (-5, 5), (5, 5), (5, -5), (-5, -5)])
line = LineString([(3, 4), (5, 7), (12, 2), (10, 5), (9, 7.5)])
pts = np.array([[2, 2], [3, 4], [9, 8]])

poly_gdf1 = gpd.GeoDataFrame(
    [1], geometry=[polygon1], crs={"init": "epsg:4326"}
)
poly_gdf2 = gpd.GeoDataFrame(
    [1], geometry=[polygon2], crs={"init": "epsg:4326"}
)
line_gdf = gpd.GeoDataFrame([1], geometry=[line], crs={"init": "epsg:4326"})
points_gdf = gpd.GeoDataFrame(
    [Point(xy) for xy in pts], columns=["geometry"], crs={"init": "epsg:4326"}
)

###############################################################################
# Plot the Unclipped Data
# -----------------------

fig, ax = plt.subplots(figsize=(12, 8))
poly_gdf1.boundary.plot(ax=ax)
poly_gdf2.boundary.plot(ax=ax, color="red")
line_gdf.plot(ax=ax, color="green")
points_gdf.plot(ax=ax, color="purple")
ax.set_title("All Unclipped Data", fontsize=20)
ax.set_axis_off()
plt.show()

###############################################################################
# Clip the Data
# --------------
#
# When you call ``clip()``, the first object called is the object that will
# be clipped. The second object called is the clip extent. The returned output
# will be a new clipped GeoDataframe. All of the attributes for each returned
# geometry will be retained when you clip.
#
# .. note::
#    Recall that the data must be in the same CRS in order to use the
#    ``clip()`` function. If the data are not in the same CRS, be sure to use
#    the GeoPandas ``to_crs()`` method to ensure both datasets are in the
#    same CRS.

###############################################################################
# Clip the Polygon Data
# ---------------------

polys_clipped = gc.clip(poly_gdf1, poly_gdf2)

# Plot the clipped data
# The plot below shows the results of the clip function applied to the polygons
fig, ax = plt.subplots(figsize=(12, 8))
polys_clipped.plot(ax=ax, color="purple")
poly_gdf1.boundary.plot(ax=ax)
poly_gdf2.boundary.plot(ax=ax, color="red")
ax.set_title("Polygons Clipped", fontsize=20)
ax.set_axis_off()
plt.show()

###############################################################################
# Clip the Line Data
# ---------------------

line_clip = gc.clip(poly_gdf1, line_gdf)

# Plot the clipped data
# The plot below shows the results of the clip function applied to the lines
# sphinx_gallery_thumbnail_number = 3
fig, (ax1, ax2) = plt.subplots(1, 2)
line_gdf.plot(ax=ax1, color="green")
poly_gdf1.boundary.plot(ax=ax1)
line_clip.plot(ax=ax2, color="green")
poly_gdf1.boundary.plot(ax=ax2)
plt.show()

###############################################################################
# Clip the Point Data
# ---------------------

points_clip = gc.clip(poly_gdf2, points_gdf)

# Plot the clipped data
# The plot below shows the results of the clip function applied to the points
fig, (ax1, ax2) = plt.subplots(1, 2)
points_gdf.plot(ax=ax1, color="purple")
poly_gdf2.boundary.plot(ax=ax1, color="red")
points_clip.plot(ax=ax2, color="purple")
poly_gdf2.boundary.plot(ax=ax2, color="red")
plt.show()
