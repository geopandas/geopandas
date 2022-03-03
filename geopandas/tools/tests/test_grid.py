import geopandas, pygeos, numpy
from matplotlib import pyplot as plt

from geopandas.tools.grids import make_grid
from geopandas.tools._random import grid as random_grid
from geopandas.tools._random import uniform
from shapely import geometry

size = (10, 12)
bounds = (1, 1, 10, 12)


def show_grid(size=None, method="hex", shape=None, bounds=None, spacing=None, ax=None):
    if bounds is None:
        if shape is None:
            bounds = (0, 0, 1, 1)
        else:
            try:
                bounds = shape.bounds
            except AttributeError:
                shape = pygeos.to_shapely(shape)
                bounds = shape.bounds
    else:
        shape = pygeos.to_shapely(pygeos.box(*bounds))

    ax = geopandas.tools.grids.make_grid(
        shape, size=size, method=method, spacing=spacing
    ).plot(color="k", ax=ax, markersize=4)
    geopandas.GeoDataFrame(geometry=[shape]).plot(ax=ax, zorder=-1)
    return ax


nybb = geopandas.read_file(geopandas.datasets.get_path("nybb"))
bboxes = geopandas.GeoSeries(nybb.bounds.apply(lambda x: geometry.box(*x), axis=1))

staten = nybb.geometry.iloc[0]

f, ax = plt.subplots(3, 4, figsize=(24, 8))
ax = ax.flatten()
show_grid((10, 12), method="hex", bounds=(1, 1, 10, 2), ax=ax[0])
show_grid((10, 12), method="hex", bounds=(1, 1, 10, 12), ax=ax[1])
show_grid(spacing=0.5, method="hex", bounds=(1, 1, 10, 2), ax=ax[2])
show_grid(spacing=0.5, method="hex", bounds=(1, 1, 10, 12), ax=ax[3])
show_grid((10, 12), method="square", bounds=(1, 1, 10, 2), ax=ax[4])
show_grid((10, 12), method="square", bounds=(1, 1, 10, 12), ax=ax[5])
show_grid(spacing=0.5, method="square", bounds=(1, 1, 10, 2), ax=ax[6])
show_grid(spacing=0.5, method="square", bounds=(1, 1, 10, 12), ax=ax[7])
show_grid((10, 12), method="hex", shape=staten, ax=ax[8])
show_grid((10, 12), method="square", shape=staten, ax=ax[9])
show_grid(spacing=10000, method="hex", shape=staten, ax=ax[10])
show_grid(spacing=10000, method="square", shape=staten, ax=ax[11])
for i in range(12):
    ax[i].set_title(
        (
            "hex 10x12 in longboi",
            "hex 10x12 in rectangle",
            "hex spaced in longboi",
            "hex spaced in rectangle",
            "square 10x12 in longboi",
            "square 10x12 in rectangle",
            "square spaced in longboi",
            "square spaced in rectangle",
            "hex 10x12 in staten",
            "square 10x12 in staten",
            "hex spaced in staten",
            "square spaced in staten",
        )[i]
    )
f.tight_layout()
plt.show()
plt.close()


nybb = geopandas.read_file(geopandas.datasets.get_path("nybb"))
shapes = (
    pygeos.to_shapely(pygeos.box(0, 0, 1, 1)),
    pygeos.to_shapely(pygeos.box(1, 1, 10, 12)),
    nybb.geometry.iloc[0],
)
f, ax = plt.subplots(3, 5, sharex=False, sharey=False)
n_reps = 5
for j in range(3):
    shape = shapes[j]
    for k in range(2):
        for i in range(n_reps):
            geopandas.GeoSeries(random_grid(shape, method=["square", "hex"][k])).plot(
                ax=ax[j, k], alpha=0.5, markersize=2
            )
    for i in range(n_reps):
        geopandas.GeoSeries(uniform(shape, size=100)).plot(
            ax=ax[j, k + 1], alpha=0.5, markersize=2
        )

    for i in range(2):
        make_grid(shape, method=("square", "hex")[i]).plot(
            ax=ax[j, k + 2 + i], markersize=2, color="k"
        )
    ax[j, 0].set_ylabel(("square", "rect", "staten")[j])

    for k in range(5):
        geopandas.GeoSeries([shape]).plot(ax=ax[j, k], zorder=-1, alpha=0.5)
        ax[j, k].set_title(
            [
                "random\nsquaregrid",
                "random\nhexgrid",
                "uniform",
                "squaregrid",
                "hexgrid",
            ][k]
        )
        ax[j, k].set_xticks([])
        ax[j, k].set_xticklabels([])
        ax[j, k].set_yticks([])
        ax[j, k].set_yticklabels([])
f.tight_layout()
plt.savefig("/Users/lw17329/Downloads/sample_points.png", dpi=300, bbox_inches="tight")
plt.show()
