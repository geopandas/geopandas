"""
Visualizing NYC Boroughs
------------------------

Visualize the Boroughs of New York City with Geopandas.

This example generates many images that are used in the documentation. See
the `Geometric Manipulations <geometric_manipulations>` example for more
details.

First we'll import a dataset containing each borough in New York City. We'll
use the ``datasets`` module to handle this quickly.
"""

import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point
from geopandas import GeoSeries, GeoDataFrame
import geopandas as gpd
import geodatasets

np.random.seed(1)
DPI = 100

path_nybb = geodatasets.get_path("nybb")
boros = GeoDataFrame.from_file(path_nybb)
boros = boros.set_index("BoroCode")
boros

##############################################################################
# Next, we'll plot the raw data
ax = boros.plot()
plt.xticks(rotation=90)
plt.savefig("nyc.png", dpi=DPI, bbox_inches="tight")

##############################################################################
# We can easily retrieve the convex hull of each shape. This corresponds to
# the outer edge of the shapes.
boros.geometry.convex_hull.plot()
plt.xticks(rotation=90)

# Grab the limits which we'll use later
xmin, xmax = plt.gca().get_xlim()
ymin, ymax = plt.gca().get_ylim()

plt.savefig("nyc_hull.png", dpi=DPI, bbox_inches="tight")

##############################################################################
# We'll generate some random dots scattered throughout our data, and will
# use them to perform some set operations with our boroughs. We can use
# GeoPandas to perform unions, intersections, etc.

N = 2000  # number of random points
R = 2000  # radius of buffer in feet

# xmin, xmax, ymin, ymax = 900000, 1080000, 120000, 280000
xc = (xmax - xmin) * np.random.random(N) + xmin
yc = (ymax - ymin) * np.random.random(N) + ymin
pts = GeoSeries([Point(x, y) for x, y in zip(xc, yc)])
mp = pts.buffer(R).union_all()
boros_with_holes = boros.geometry - mp
boros_with_holes.plot()
plt.xticks(rotation=90)
plt.savefig("boros_with_holes.png", dpi=DPI, bbox_inches="tight")

##############################################################################
# Finally, we'll show the holes that were taken out of our boroughs.

holes = boros.geometry & mp
holes.plot()
plt.xticks(rotation=90)
plt.savefig("holes.png", dpi=DPI, bbox_inches="tight")
plt.show()
