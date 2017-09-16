"""
Plotting with CartoPy and GeoPandas
-----------------------------------

Converting between GeoPandas and CartoPy for visualizing data.

`CartoPy <http://scitools.org.uk/cartopy/>`_ is a Python library
that specializes in creating geospatial
visualizations. It has a slightly different way of representing
Coordinate Reference Systems (CRS) as well as constructing plots.
This example steps through a round-trip transfer of data
between GeoPandas and CartoPy.

First we'll load in the data using GeoPandas.
"""
# sphinx_gallery_thumbnail_number = 7
import matplotlib.pyplot as plt
import geopandas as gpd
from cartopy import crs as ccrs

path = gpd.datasets.get_path('naturalearth_lowres')
df = gpd.read_file(path)
# Add a column we'll use later
df['gdp_pp'] = df['gdp_md_est'] / df['pop_est']

####################################################################
# First we'll visualize the map using GeoPandas
df.plot()

###############################################################################
# Plotting with CartoPy
# =====================
#
# Cartopy also handles Shapely objects well, but it uses a different system for
# CRS. To plot this data with CartoPy, we'll first need to project it into a
# new CRS. We'll use a CRS defined within CartoPy and use the GeoPandas
# ``to_crs`` method to make the transformation.

# Define the CartoPy CRS object.
crs = ccrs.AzimuthalEquidistant()

# This can be converted into a `proj4` string/dict compatible with GeoPandas
crs_proj4 = crs.proj4_init
df_ae = df.to_crs(crs_proj4)

# Here's what the plot looks like in GeoPandas
df_ae.plot()

###############################################################################
# Now that our data is in a CRS based off of CartoPy, we can easily
# plot it.

fig, ax = plt.subplots(subplot_kw={'projection': crs})
ax.add_geometries(df_ae['geometry'], crs=crs)

###############################################################################
# Note that we could have easily done this with an EPSG code like so:
crs_epsg = ccrs.epsg('3857')
df_epsg = df.to_crs(epsg='3857')

# Generate a figure with two axes, one for CartoPy, one for GeoPandas
fig, axs = plt.subplots(1, 2, subplot_kw={'projection': crs_epsg},
                        figsize=(10, 5))
# Make the CartoPy plot
axs[0].add_geometries(df_epsg['geometry'], crs=crs_epsg,
                      facecolor='white', edgecolor='black')
# Make the GeoPandas plot
df_epsg.plot(ax=axs[1], color='white', edgecolor='black')

###############################################################################
# CartoPy to GeoPandas
# ====================
#
# Next we'll perform a CRS projection in CartoPy, and then convert it
# back into a GeoPandas object.

crs_new = ccrs.AlbersEqualArea()
new_geometries = [crs_new.project_geometry(ii, src_crs=crs)
                  for ii in df_ae['geometry'].values]

fig, ax = plt.subplots(subplot_kw={'projection': crs_new})
ax.add_geometries(new_geometries, crs=crs_new)

###############################################################################
# Now that we've created new Shapely objects with the CartoPy CRS,
# we can use this to create a GeoDataFrame.

df_aea = gpd.GeoDataFrame(df['gdp_pp'], geometry=new_geometries,
                          crs=crs_new.proj4_init)
df_aea.plot()

###############################################################################
# We can even combine these into the same figure. Here we'll plot the
# shapes of the countries with CartoPy. We'll then calculate the centroid
# of each with GeoPandas and plot it on top.

# Generate a CartoPy figure and add the countries to it
fig, ax = plt.subplots(subplot_kw={'projection': crs_new})
ax.add_geometries(new_geometries, crs=crs_new)

# Calculate centroids and plot
df_aea_centroids = df_aea.geometry.centroid
df_aea_centroids.plot(ax=ax, markersize=5, color='r')

plt.show()
