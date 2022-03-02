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

for method in ("random", "grid"):
    for tile in (None, "square", "hex"):
        print(method, tile)
        if method == "grid":
            size = (10, 10)
        else:
            size = 10
        print(method, tile, "polygons")
        overall = multipolygons.geometry.sample_points(
            size=size, method=method, tile=tile
        )
        assert isinstance(overall, geopandas.GeoSeries)

        print(method, tile, "lines")

        line_overall = multilinestrings.geometry.sample_points(
            size=size, method=method, tile=tile
        )
        assert isinstance(line_overall, geopandas.GeoSeries)

        print(method, tile, "points")

        point_overall = points.geometry.sample_points(
            size=size, method=method, tile=tile
        )
        assert isinstance(point_overall, geopandas.GeoSeries)

        print(method, tile, "mixed")

        mixed_overall = mixed.geometry.sample_points(
            size=size, method=method, tile=tile
        )
        assert isinstance(mixed_overall, geopandas.GeoSeries)

# f, ax = plt.subplots(2, 4, figsize=(16, 4))
# for i in range(8):
#    ax_ = ax.flatten()[i]
#
#    df = (
#        overall,
#        overall_multi,
#        line_overall,
#        line_overall_multi,
#        point_overall,
#        point_overall_multi,
#        mixed_overall,
#        mixed_overall_multi,
#    )[i]
#    df.set_crs(polygons.crs)
#    target = multipolygons if i < 4 else multilinestrings
#    target.plot(ax=ax_)
#    df.plot(ax=ax_, color="k")
# f.tight_layout()
# plt.show()
