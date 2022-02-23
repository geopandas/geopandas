import geopandas


# types
multipolygons = geopandas.read_file(geopandas.datasets.get_path("nybb"))
polygons = multipolygons.explode()
multilinestrings = multipolygons.boundary
linestrings = polygons.boundary

multipolygon = multipolygons.geometry.values.data[0]
polygon = polygons.geometry.values.data[0]
multilinestring = multilinestrings.geometry.values.data[0]
linestring = linestrings.geometry.values.data[0]

from geopandas.tools.grids import make_grid
from geopandas.tools._random import uniform
from matplotlib import pyplot as plt

make_grid(polygons)

overall = multipolygons.geometry.iloc[[0]].sample_points(
    10, method="random", by_parts=False
)
by_parts = multipolygons.geometry.iloc[[0]].sample_points(
    10, method="random", by_parts=True
)


overall = multipolygons.geometry.sample_points(10, method="random", by_parts=False)
by_parts = multipolygons.geometry.sample_points(10, method="random", by_parts=True)

overall_multi = multipolygons.geometry.sample_points(
    (10, 2), method="random", by_parts=False
)
by_parts_multi = multipolygons.geometry.sample_points(
    (10, 2), method="random", by_parts=True
)
