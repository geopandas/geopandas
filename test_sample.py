import geopandas


# types
multipolygons = geopandas.read_file(geopandas.datasets.get_path("nybb")).geometry
polygons = multipolygons.explode().geometry
multilinestrings = multipolygons.boundary
linestrings = polygons.boundary
multipoints = None
points = multipolygons.centroid
mixed = geopandas.pd.concat(
    (multipolygons.geometry.head(), multilinestrings.geometry.head(), points.head())
)


multipolygon = multipolygons.geometry.values.data[0]
polygon = polygons.geometry.values.data[0]
multilinestring = multilinestrings.geometry.values.data[0]
linestring = linestrings.geometry.values.data[0]


from geopandas.tools.grids import make_grid
from geopandas.tools._random import uniform
from matplotlib import pyplot as plt

make_grid(polygons)


overall = multipolygons.geometry.sample_points(10, method="random", by_parts=False)
assert isinstance(overall, geopandas.GeoSeries)
by_parts = multipolygons.geometry.sample_points(10, method="random", by_parts=True)
assert isinstance(by_parts, geopandas.GeoSeries)
assert by_parts.geometry.apply(lambda x: len(x.geoms) > 10).all()

overall_multi = multipolygons.geometry.sample_points(
    (10, 2), method="random", by_parts=False
)
assert isinstance(overall_multi, geopandas.GeoDataFrame)

line_overall = multilinestrings.geometry.sample_points(
    10, method="random", by_parts=False
)
assert isinstance(line_overall, geopandas.GeoSeries)


line_overall_multi = multilinestrings.geometry.sample_points(
    (10, 2), method="random", by_parts=False
)
assert isinstance(line_overall_multi, geopandas.GeoDataFrame)


point_overall = points.geometry.sample_points(10, method="random", by_parts=False)
assert isinstance(point_overall, geopandas.GeoSeries)

point_overall_multi = points.geometry.sample_points(
    (10, 2), method="random", by_parts=False
)
assert isinstance(point_overall_multi, geopandas.GeoSeries)

point_by_parts_multi = points.geometry.sample_points(
    (10, 2), method="random", by_parts=True
)
assert isinstance(point_by_parts_multi, geopandas.GeoDataFrame)

mixed_overall = mixed.geometry.sample_points((10, 2), method="random", by_parts=False)
assert isinstance(line_overall_multi, geopandas.GeoDataFrame)

f, ax = plt.subplots(4, 2, figsize=(16, 4))
for i in range(8):
    ax_ = ax.flatten()[i]

    df = (
        overall,
        by_parts,
        overall_multi,
        by_parts_multi,
        line_overall,
        line_by_parts,
        line_overall_multi,
        line_by_parts_multi,
    )[i]
    df.set_crs(polygons.crs)
    target = multipolygons if i < 4 else multilinestrings
    target.plot(ax=ax_)
    df.plot(ax=ax_, color="k")
f.tight_layout()
plt.show()
