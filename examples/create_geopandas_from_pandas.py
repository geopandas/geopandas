"""
Creating a GeoDataFrame from a DataFrame
----------------------------------------

This example shows how to create a ``GeoDataFrame`` when starting from
a *regular* ``DataFrame`` that has coordinates stored in two columns.

"""
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

###############################################################################
# First, let's consider a ``DataFrame`` containing cities and their respective
# longitudes and latitudes.

df = pd.DataFrame(
    {'City': ['Buenos Aires', 'Brasilia', 'Santiago', 'Bogota', 'Caracas'],
     'Country': ['Argentina', 'Brazil', 'Chile', 'Colombia', 'Venezuela'],
     'Latitude': [-34.583333, -15.783333, -33.450000, 4.600000, 10.483333],
     'Longitude': [-58.666667, -47.916667, -70.666667, -74.083333, -66.866667]})

###############################################################################
# A ``GeoDataFrame`` needs a ``shapely`` object, so we create a new column
# **Coordinates** as a tuple of **Longitude** and **Latitude** :

df['Coordinates']  = list(zip(df.Longitude, df.Latitude))

###############################################################################
# Then, we transform tuples to ``Point`` :

df['Coordinates'] = df['Coordinates'].apply(Point)

###############################################################################
# Now, we can create the ``GeoDataFrame`` by setting ``geometry`` with the
# coordinates created previously.

gdf = gpd.GeoDataFrame(df, geometry='Coordinates')


###############################################################################
# Finally, we plot the coordinates over a country-level map.

world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

# We restrict to South America.
ax = world[world.continent == 'South America'].plot(
    color='white', edgecolor='black')

# We can now plot our GeoDataFrame.
gdf.plot(ax=ax, color='red')

plt.show()
