import geopandas
import numpy
import pytest

from geopandas import _compat as compat
from scipy.spatial import distance
from geopandas.tools._random import uniform, grid

multipolygons = geopandas.read_file(geopandas.datasets.get_path("nybb")).geometry
polygons = multipolygons.explode().geometry
multilinestrings = multipolygons.boundary
linestrings = polygons.boundary
points = multipolygons.centroid


multipolygon = multipolygons.geometry.iloc[0]
polygon = polygons.geometry.iloc[2]
multilinestring = multilinestrings.geometry.iloc[0]
linestring = linestrings.geometry.iloc[0]
point = points.geometry.iloc[0]


def find_spacing(input, return_distances=False):
    if not (isinstance(input, (geopandas.GeoSeries, geopandas.GeoDataFrame))):
        input = geopandas.GeoSeries([g for g in input.geoms])
    x, y = input.x, input.y
    first = numpy.array([[x[0], y[0]]])
    rest = numpy.column_stack((x[1:], y[1:]))
    distances = distance.cdist(first, rest)
    if return_distances:
        return distances.min(), distances
    else:
        return distances.min()


@pytest.mark.parametrize(
    ["data", "size", "batch_size"],
    [
        (data, size, batch_size)
        for data in (multipolygon, polygon)
        for size in (None, 10)
        for batch_size in (None, 2)
    ],
)
def test_uniform(data, size, batch_size):
    if size is None:
        sample = uniform(data, batch_size=batch_size)
    else:
        sample = uniform(data, batch_size=batch_size, size=size)
    size = 1 if size is None else size
    sample_series = geopandas.GeoSeries(sample).explode().reset_index(drop=True)
    assert (
        len(sample_series) == size
    ), f"{size} points were requested, but {len(sample_series)} were drawn."
    sample_in_geom = sample_series.sindex.query(data, predicate="intersects")
    assert len(sample_in_geom) == size


@pytest.mark.parametrize(
    ["data", "size", "spacing", "tile", "seed"],
    [
        (data, size, spacing, tile, seed)
        for data in (multipolygon, polygon, multilinestring, linestring)
        for size in (None, 10, (5, 3))
        for spacing in (None, 500)
        for tile in ("hex", "square")
        for seed in (112112, 123456)
        if ((not ((spacing is not None) and (size is not None))))
    ],
)
@pytest.mark.skipif(
    not compat.HAS_PYGEOS, reason="Need pygeos to sample random grids from shapes..."
)
def test_grid(data, size, spacing, tile, seed):
    numpy.random.seed(seed)
    sample = grid(data, size=size, spacing=spacing, tile=tile)
    if spacing is not None:
        implied_spacing = find_spacing(sample)
        assert implied_spacing <= spacing, (
            f"spacing is wrong for size={size}, "
            f"spacing={spacing}, tile={tile} on {data.type}"
        )
    else:
        if isinstance(size, tuple):
            n_specified = size[0] * size[1]
        elif size is None:
            n_specified = 100
        elif data.type in ("Polygon", "MultiPolygon"):
            n_specified = size ** 2
        else:
            n_specified = size
        assert (len(sample.geoms) - n_specified) <= 1, (
            f"Sampled {len(sample.geoms)} points instead of "
            f"{n_specified} for size={size}, spacing={spacing}, "
            f"tile={tile} on {data.type}"
        )  # This is an artefact of the way lines are sampled;
        # check numpy.arange docstring for the details.

    points_in_grid = geopandas.GeoSeries(sample).explode()
    intersects = points_in_grid.sindex.query(data.buffer(1e-7), predicate="intersects")
    assert len(intersects) == len(points_in_grid)


def test_size_and_spacing_failures():
    for bad_size in [(10, 10, 2), 0.1, (0.1, 0.2)]:
        with pytest.raises(TypeError):
            grid(size=bad_size)
        with pytest.raises(TypeError):
            uniform(size=bad_size)
    with pytest.raises(ValueError):
        grid(size=10, spacing=1)
