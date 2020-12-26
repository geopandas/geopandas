"""
See generate_legacy_storage_files.py for the creation of the legacy files.

"""
from distutils.version import LooseVersion
import glob
import os
import pathlib

import pandas as pd

import pyproj

import pytest
from geopandas.testing import assert_geodataframe_equal
from geopandas import _compat as compat
import geopandas
from shapely.geometry import Point

DATA_PATH = pathlib.Path(os.path.dirname(__file__)) / "data"


@pytest.fixture(scope="module")
def current_pickle_data():
    # our current version pickle data
    from .generate_legacy_storage_files import create_pickle_data

    return create_pickle_data()


files = glob.glob(str(DATA_PATH / "pickle" / "*.pickle"))


@pytest.fixture(params=files, ids=[p.split("/")[-1] for p in files])
def legacy_pickle(request):
    return request.param


@pytest.fixture
def with_use_pygeos_false():
    orig = geopandas.options.use_pygeos
    geopandas.options.use_pygeos = not orig
    yield
    geopandas.options.use_pygeos = orig


@pytest.mark.skipif(
    compat.USE_PYGEOS or (str(pyproj.__version__) < LooseVersion("2.4")),
    reason=(
        "pygeos-based unpickling currently only works for pygeos-written files; "
        "old pyproj versions can't read pickles from newer pyproj versions"
    ),
)
def test_legacy_pickles(current_pickle_data, legacy_pickle):
    result = pd.read_pickle(legacy_pickle)

    for name, value in result.items():
        expected = current_pickle_data[name]
        assert_geodataframe_equal(value, expected)


def test_round_trip_current(tmpdir, current_pickle_data):
    data = current_pickle_data

    for name, value in data.items():
        path = str(tmpdir / "{}.pickle".format(name))
        value.to_pickle(path)
        result = pd.read_pickle(path)
        assert_geodataframe_equal(result, value)
        assert isinstance(result.has_sindex, bool)


@pytest.mark.skipif(not compat.HAS_PYGEOS, reason="requires pygeos to test #1745")
def test_pygeos_switch(tmpdir, with_use_pygeos_false):
    gdf_crs = geopandas.GeoDataFrame(
        {"a": [0.1, 0.2, 0.3], "geometry": [Point(1, 1), Point(2, 2), Point(3, 3)]},
        crs="EPSG:4326",
    )
    path = str(tmpdir / "gdf_crs.pickle")
    gdf_crs.to_pickle(path)
    result = pd.read_pickle(path)
    assert_geodataframe_equal(result, gdf_crs)
