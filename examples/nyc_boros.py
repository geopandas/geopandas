"""
Generate example images for GeoPandas documentation.

TODO: autogenerate these from docs themselves

Kelsey Jordahl
Time-stamp: <Tue May  6 12:17:29 EDT 2014>
"""
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point
from geopandas import GeoSeries, GeoDataFrame

np.random.seed(1)
DPI = 100

# http://www1.nyc.gov/assets/planning/download/zip/data-maps/open-data/nybb_16a.zip
boros = GeoDataFrame.from_file('nybb.shp')
boros.set_index('BoroCode', inplace=True)
boros.sort()
boros.plot()
plt.xticks(rotation=90)
plt.savefig('nyc.png', dpi=DPI, bbox_inches='tight')
#plt.show()
boros.geometry.convex_hull.plot()
plt.xticks(rotation=90)
plt.savefig('nyc_hull.png', dpi=DPI, bbox_inches='tight')
#plt.show()

N = 2000  # number of random points
R = 2000  # radius of buffer in feet
xmin, xmax = plt.gca().get_xlim()
ymin, ymax = plt.gca().get_ylim()
#xmin, xmax, ymin, ymax = 900000, 1080000, 120000, 280000
xc = (xmax - xmin) * np.random.random(N) + xmin
yc = (ymax - ymin) * np.random.random(N) + ymin
pts = GeoSeries([Point(x, y) for x, y in zip(xc, yc)])
mp = pts.buffer(R).unary_union
boros_with_holes = boros.geometry - mp
boros_with_holes.plot()
plt.xticks(rotation=90)
plt.savefig('boros_with_holes.png', dpi=DPI, bbox_inches='tight')
plt.show()
holes = boros.geometry & mp
holes.plot()
plt.xticks(rotation=90)
plt.savefig('holes.png', dpi=DPI, bbox_inches='tight')
plt.show()
