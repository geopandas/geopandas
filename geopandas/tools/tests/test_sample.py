from sklearn import random_projection
import geopandas

import pytest
from shapely import geometry
from scipy.spatial import distance
from geopandas.tools._random import uniform, grid

multipolygons = geopandas.read_file(geopandas.datasets.get_path("nybb")).geometry
polygons = multipolygons.explode().geometry
multilinestrings = multipolygons.boundary
linestrings = polygons.boundary
points = multipolygons.centroid


multipolygon = multipolygons.geometry.iloc[0]
polygon = polygons.geometry.iloc[0]
multilinestring = multilinestrings.geometry.iloc[0]
linestring = linestrings.geometry.iloc[0]
point = points.geometry.iloc[0]


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
    sample = uniform(data, size=size, batch_size=batch_size)

    assert (
        len(sample) == size
    ), f"{size} points were requested, but {len(sample)} were drawn."
    sample_in_geom = (
        geopandas.GeoSeries(sample).explode().sindex.query(data, predicate="intersects")
    )
    assert len(sample_in_geom) == size


@pytest.mark.parametrize(
    ["data", "size", "spacing", "method"],
    [
        (data, size, spacing, method)
        for data in (multipolygon, polygon, multilinestring, linestring)
        for size in (None, 10, (5, 3))
        for spacing in (None, 500)
        for method in ("hex", "square")
        if ((not ((spacing is not None) and (size is not None))))
    ],
)
def test_grid(data, size, spacing, method):

    sample = grid(data, size=size, spacing=spacing, method=method)
    if spacing is not None:
        a, b, *rest = sample.geoms
        assert a.distance(b) == spacing
    else:
        if isinstance(size, tuple):
            n_specified = size[0] * size[1]
        elif size is None:
            n_specified = 100
        else:
            n_specified = size
        assert (
            len(sample.geoms) <= n_specified
        ), f"Sampled {len(sample.geoms)} points instead of {n_specified} for {data.type}"

    # check all points intersect the input geom


def test_size_and_spacing_failures():
    ...
