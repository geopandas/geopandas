import geopandas


# types
multipolygons = geopandas.read_file(geopandas.datasets.get_path("nybb")).geometry
polygons = multipolygons.explode().geometry
multilinestrings = multipolygons.boundary
linestrings = polygons.boundary
multipoints = None
points = multipolygons.centroid
mixed = geopandas.pd.concat(
    (multipolygons.geometry.head(2), multilinestrings.geometry.tail(2), points)
)


multipolygon = multipolygons.geometry.values.data[0]
polygon = polygons.geometry.values.data[0]
multilinestring = multilinestrings.geometry.values.data[0]
linestring = linestrings.geometry.values.data[0]


from geopandas.tools.grids import make_grid
from geopandas.tools._random import uniform
from matplotlib import pyplot as plt

make_grid(polygons)

f, ax = plt.subplots(6, 4, figsize=(10, 15))

for ji, method in enumerate(("random", "grid")):
    for jj, tile in enumerate((None, "square", "hex")):
        if method == "grid":
            size = (10, 10)
        else:
            size = 10
        overall = multipolygons.geometry.sample_points(
            size=size, method=method, tile=tile
        )
        assert isinstance(overall, geopandas.GeoSeries)

        line_overall = multilinestrings.geometry.sample_points(
            size=size, method=method, tile=tile
        )
        assert isinstance(line_overall, geopandas.GeoSeries)

        point_overall = points.geometry.sample_points(
            size=size, method=method, tile=tile
        )
        assert isinstance(point_overall, geopandas.GeoSeries)

        mixed_overall = mixed.geometry.sample_points(
            size=size, method=method, tile=tile
        )
        assert isinstance(mixed_overall, geopandas.GeoSeries)

        j = ji * 3 + jj
        for i in range(4):
            sample, data = (
                (overall, multipolygons),
                (line_overall, multilinestrings),
                (point_overall, points),
                (mixed_overall, mixed),
            )[i]
            data.plot(ax=ax[j, i], zorder=-1)
            sample.plot(ax=ax[j, i], color="r", marker=".", markersize=10)

            if i == 0:
                ax[j, i].set_ylabel(f"{method}\n{tile}")
            if j == 0:
                ax[j, i].set_title(("polygons", "lines", "points", "mixed")[i])
            ax[j, i].set_xticks([])
            ax[j, i].set_xticklabels([])
            ax[j, i].set_yticks([])
            ax[j, i].set_yticklabels([])

f.tight_layout()
plt.show()


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
