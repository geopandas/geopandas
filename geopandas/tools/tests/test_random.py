import numpy

import geopandas
from geopandas.tools._random import uniform

import pytest


@pytest.fixture
def multipolygons(nybb_filename):
    return geopandas.read_file(nybb_filename).geometry


@pytest.fixture
def polygons(multipolygons):
    return multipolygons.explode(ignore_index=True).geometry


@pytest.fixture
def multilinestrings(multipolygons):
    return multipolygons.boundary


@pytest.fixture
def linestrings(polygons):
    return polygons.boundary


@pytest.fixture
def points(multipolygons):
    return multipolygons.centroid


@pytest.mark.parametrize("size", [10, 100])
@pytest.mark.parametrize(
    "geom_fixture", ["multipolygons", "polygons", "multilinestrings", "linestrings"]
)
def test_uniform(geom_fixture, size, request):
    geom = request.getfixturevalue(geom_fixture)[0]
    sample = uniform(geom, size=size, rng=1)
    sample_series = (
        geopandas.GeoSeries(sample).explode(index_parts=True).reset_index(drop=True)
    )
    assert len(sample_series) == size
    sample_in_geom = sample_series.buffer(0.00000001).sindex.query(
        geom, predicate="intersects"
    )
    assert len(sample_in_geom) == size


def test_uniform_unsupported(points):
    with pytest.warns(UserWarning, match="Sampling is not supported"):
        sample = uniform(points[0], size=10, rng=1)
    assert sample.is_empty


def test_uniform_generator(polygons):
    sample = uniform(polygons[0], size=10, rng=1)
    sample2 = uniform(polygons[0], size=10, rng=1)
    assert sample.equals(sample2)

    generator = numpy.random.default_rng(seed=1)
    gen_sample = uniform(polygons[0], size=10, rng=generator)
    gen_sample2 = uniform(polygons[0], size=10, rng=generator)

    assert sample.equals(gen_sample)
    assert not sample.equals(gen_sample2)
