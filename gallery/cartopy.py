"""
Plotting with CartoPy and GeoPandas
-----------------------------------

Converting between GeoPandas and CartoPy for visualizing data.

CartoPy is a Python library that specializes in creating geospatial
visualizations. It has a slightly different way of representing CRS
coordinates as well as constructing plots. This example steps through a
round-trip transfer of data between GeoPandas and CartoPy.

First we'll load in the data using GeoPandas.
"""
# sphinx_gallery_thumbnail_number = 7
import geopandas as gpd
import matplotlib.pyplot as plt
import cartopy as cp

path = gpd.datasets.get_path('naturalearth_lowres')
df = gpd.read_file(path)

####################################################################
# First we'll visualize the map using GeoPandas
df.plot()

###############################################################################
# Plotting with CartoPy
# ---------------------
#
# Cartopy also handles Shapely objects well, but it uses a different system for
# CRS. To plot this data with CartoPy, we'll first need to project it into a
# new CRS. We'll use a CRS defined within shapely and use the GeoPandas
# ``to_crs`` method to make the transformation.

# Define the CartoPy CRS object.
crs = cp.crs.AzimuthalEquidistant()

# This can be converted into a `proj4` string/dict compatible with GeoPandas
crs_proj4 = crs.proj4_init
df = df.to_crs(crs_proj4)

# Here's what the plot looks like in GeoPandas
df.plot()

###############################################################################
# Now that our data is in a CRS based off of CartoPy, we can easily
# plot it.

fig, ax = plt.subplots(subplot_kw={'projection': crs})
ax.add_geometries(df['geometry'], crs=crs)

###############################################################################
# Note that we could have easily done this with an EPSG code like so:
crs_epsg = cp.crs.epsg('2039')
df_epsg = df.to_crs(epsg='2039')
df_epsg.plot()

###############################################################################
# CartoPy to GeoPandas
# --------------------
#
# Next we'll perform a CRS projection in CartoPy, and then convert it
# back into a GeoPandas object.

crs_new = cp.crs.AlbersEqualArea()
new_geometries = [crs_new.project_geometry(ii, src_crs=crs)
                  for ii in df['geometry'].values]

fig, ax = plt.subplots(subplot_kw={'projection': crs_new})
ax.add_geometries(new_geometries, crs=crs_new)

###############################################################################
# Now that we've created new Shapely objects with the CartoPy CRS,
# we can use this to create a GeoDataFrame.

df_new = gpd.GeoDataFrame(crs=crs_new.proj4_init, geometry=new_geometries)
df_new.plot()

###############################################################################
# We can even combine these into the same figure.

fig, axs = plt.subplots(1, 2, subplot_kw={'projection': crs_new},
                        figsize=(10, 5))
axs[0].add_geometries(new_geometries, crs=crs_new)
df_new.plot(ax=axs[1])

plt.show()
