"""
See generate_legacy_storage_files.py for the creation of the legacy files.

"""
from contextlib import contextmanager
import glob
import os
import pathlib

import pandas as pd

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


@contextmanager
def with_use_pygeos(option):
    orig = geopandas.options.use_pygeos
    geopandas.options.use_pygeos = option
    try:
        yield
    finally:
        geopandas.options.use_pygeos = orig


@pytest.mark.skipif(
    compat.USE_SHAPELY_20 or compat.USE_PYGEOS,
    reason=(
        "shapely 2.0/pygeos-based unpickling currently only works for "
        "shapely-2.0/pygeos-written files"
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


def _create_gdf():
    return geopandas.GeoDataFrame(
        {"a": [0.1, 0.2, 0.3], "geometry": [Point(1, 1), Point(2, 2), Point(3, 3)]},
        crs="EPSG:4326",
    )


@pytest.mark.skipif(not compat.HAS_PYGEOS, reason="requires pygeos to test #1745")
def test_pygeos_switch(tmpdir):
    # writing and reading with pygeos disabled
    with with_use_pygeos(False):
        gdf = _create_gdf()
        path = str(tmpdir / "gdf_crs1.pickle")
        gdf.to_pickle(path)
        result = pd.read_pickle(path)
        assert_geodataframe_equal(result, gdf)

    # writing without pygeos, reading with pygeos
    with with_use_pygeos(False):
        gdf = _create_gdf()
        path = str(tmpdir / "gdf_crs1.pickle")
        gdf.to_pickle(path)

    with with_use_pygeos(True):
        result = pd.read_pickle(path)
        gdf = _create_gdf()
        assert_geodataframe_equal(result, gdf)

    # writing with pygeos, reading without pygeos
    with with_use_pygeos(True):
        gdf = _create_gdf()
        path = str(tmpdir / "gdf_crs1.pickle")
        gdf.to_pickle(path)

    with with_use_pygeos(False):
        result = pd.read_pickle(path)
        gdf = _create_gdf()
        assert_geodataframe_equal(result, gdf)
