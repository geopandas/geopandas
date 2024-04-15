"""
See generate_legacy_storage_files.py for the creation of the legacy files.

"""

import glob
import os
import pathlib

import pandas as pd

import pytest
from geopandas.testing import assert_geodataframe_equal

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


@pytest.mark.skip(
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
