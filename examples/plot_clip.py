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
# .. note::
#    The example below will show you how to use the clip() function to clip
#    vector data such as points, lines and polygons to a vector boundary.
#
# The example below walks you through a typical workflow for clipping one
# vector data file to the shape of another. Both vector data files must be
# opened with GeoPandas as GeoDataFrames and be in the same Coordinate
# Reference System (CRS) for the ``clip_shp()`` function in EarthPy to work.
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
# To begin, import the needed packages. You will primarily use EarthPy's clip
# utility alongside GeoPandas.

import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Polygon, LineString, Point
import geopandas.clip as gc

###############################################################################
# Create Example Data
# -------------------
#
# Once the packages have been imported, you need to create the shapes to clip.
# You need to make two polygons, one line, and one point feature with shapely,
# and then open those shapes up with GeoPandas.

polygon1 = Polygon([(0, 0), (0, 10), (10, 10), (10, 0), (0, 0)])
polygon2 = Polygon([(-5, -5), (-5, 5), (5, 5), (5, -5), (-5, -5)])
line = LineString([(3, 4), (5, 7), (12, 2), (10, 5), (9, 7.5)])
pts = np.array([[2, 2], [3, 4], [9, 8]])

###############################################################################
# Open Files with GeoPandas and Reproject the Data
# -------------------------------------------------
#
# Open the data files to as GeoDataFrames using GeoPandas.
#
# .. note::
#    Recall that the data must be in the same CRS in order to use the
#    ``clip()`` function. If the data are not in the same CRS, be sure to use
#    the ``to_crs()`` function from GeoPandas to match the projects between the
#    two objects, as shown below. In this example, since you make all of the data,
#    you don't have to change the CRS.

# Now that since you have all of the shapes created, you can open them with GeoPandas
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
# The plot below shows all of the data before it has been clipped. Notice that
# the ``.boundary`` method for a GeoPandas object is used to plot the
# boundary rather than the filled polygon. This allows for other data, such as
# the line and point data, to be overlayed on top of the polygon boundary.

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
# Now that the data are opened as GeoDataFrame objects and in the same
# projection, the data can be clipped! In this example we clip a polygon,
# a line, and points to a created polygon.
#
# To clip the data, make
# sure you put the object to be clipped as the first argument in
# ``clip()``, followed by the vector object (boundary) to which you want
# the first object clipped. The function will return the clipped GeoDataFrame
# of the object that is being clipped (e.g. points).

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
