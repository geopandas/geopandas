"""
Creating a GeoDataFrame from a DataFrame with coordinates
---------------------------------------------------------

This example shows how to create a ``GeoDataFrame`` when starting from
a *regular* ``DataFrame`` that has coordinates either WKT
(`well-known text <https://en.wikipedia.org/wiki/Well-known_text>`_)
format, or in
two columns.

"""
import pandas as pd
import geopandas
import matplotlib.pyplot as plt

###############################################################################
# From longitudes and latitudes
# =============================
#
# First, let's consider a ``DataFrame`` containing cities and their respective
# longitudes and latitudes.

df = pd.DataFrame(
    {'City': ['Buenos Aires', 'Brasilia', 'Santiago', 'Bogota', 'Caracas'],
     'Country': ['Argentina', 'Brazil', 'Chile', 'Colombia', 'Venezuela'],
     'Latitude': [-34.58, -15.78, -33.45, 4.60, 10.48],
     'Longitude': [-58.66, -47.91, -70.66, -74.08, -66.86]})

###############################################################################
# A ``GeoDataFrame`` needs a ``shapely`` object. We use geopandas
# ``points_from_xy()`` to transform **Longitude** and **Latitude** into a list
# of ``shapely.Point`` objects and set it as a ``geometry`` while creating the
# ``GeoDataFrame``. (note that ``points_from_xy()`` is an enhanced wrapper for
# ``[Point(x, y) for x, y in zip(df.Longitude, df.Latitude)]``)

gdf = geopandas.GeoDataFrame(
    df, geometry=geopandas.points_from_xy(df.Longitude, df.Latitude))


###############################################################################
# ``gdf`` looks like this :

print(gdf.head())

###############################################################################
# Finally, we plot the coordinates over a country-level map.

world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))

# We restrict to South America.
ax = world[world.continent == 'South America'].plot(
    color='white', edgecolor='black')

# We can now plot our ``GeoDataFrame``.
gdf.plot(ax=ax, color='red')

plt.show()

###############################################################################
# From WKT format
# ===============
# Here, we consider a ``DataFrame`` having coordinates in WKT format.

df = pd.DataFrame(
    {'City': ['Buenos Aires', 'Brasilia', 'Santiago', 'Bogota', 'Caracas'],
     'Country': ['Argentina', 'Brazil', 'Chile', 'Colombia', 'Venezuela'],
     'Coordinates': ['POINT(-58.66 -34.58)', 'POINT(-47.91 -15.78)',
                     'POINT(-70.66 -33.45)', 'POINT(-74.08 4.60)',
                     'POINT(-66.86 10.48)']})

###############################################################################
# We use ``shapely.wkt`` sub-module to parse wkt format:
from shapely import wkt

df['Coordinates'] = df['Coordinates'].apply(wkt.loads)

###############################################################################
#  The ``GeoDataFrame`` is constructed as follows :

gdf = geopandas.GeoDataFrame(df, geometry='Coordinates')

print(gdf.head())

#################################################################################
# Again, we can plot our ``GeoDataFrame``.
ax = world[world.continent == 'South America'].plot(
    color='white', edgecolor='black')

gdf.plot(ax=ax, color='red')

plt.show()
