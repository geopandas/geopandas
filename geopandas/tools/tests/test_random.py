import pytest
import numpy
import geopandas
import geopandas._compat as compat

from geopandas.tools._random import uniform

multipolygons = geopandas.read_file(geopandas.datasets.get_path("nybb")).geometry
polygons = multipolygons.explode(ignore_index=True).geometry
multilinestrings = multipolygons.boundary
linestrings = polygons.boundary
points = multipolygons.centroid


@pytest.mark.skipif(
    not (compat.USE_PYGEOS or compat.USE_SHAPELY_20),
    reason="array input in interpolate not implemented for shapely<2",
)
@pytest.mark.parametrize("size", [10, 100])
@pytest.mark.parametrize(
    "geom", [multipolygons[0], polygons[0], multilinestrings[0], linestrings[0]]
)
def test_uniform(geom, size):
    sample = uniform(geom, size=size, seed=1)
    sample_series = geopandas.GeoSeries(sample).explode().reset_index(drop=True)
    assert len(sample_series) == size
    sample_in_geom = sample_series.buffer(0.00000001).sindex.query(
        geom, predicate="intersects"
    )
    assert len(sample_in_geom) == size


@pytest.mark.skipif(
    not (compat.USE_PYGEOS or compat.USE_SHAPELY_20),
    reason="array input in interpolate not implemented for shapely<2",
)
def test_uniform_unsupported():
    with pytest.warns(UserWarning, match="Sampling is not supported"):
        sample = uniform(points[0], size=10, seed=1)
    assert sample.is_empty


@pytest.mark.skipif(
    not (compat.USE_PYGEOS or compat.USE_SHAPELY_20),
    reason="array input in interpolate not implemented for shapely<2",
)
def test_uniform_generator():
    sample = uniform(polygons[0], size=10, seed=1)
    sample2 = uniform(polygons[0], size=10, seed=1)
    assert sample.equals(sample2)

    generator = numpy.random.default_rng(seed=1)
    gen_sample = uniform(polygons[0], size=10, seed=generator)
    gen_sample2 = uniform(polygons[0], size=10, seed=generator)

    assert sample.equals(gen_sample)
    assert not sample.equals(gen_sample2)
