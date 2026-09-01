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
        path = str(tmpdir / f"{name}.pickle")
        value.to_pickle(path)
        result = pd.read_pickle(path)
        assert_geodataframe_equal(result, value)
        assert isinstance(result.has_sindex, bool)


def test_pickle_linear_ring():
    import pickle

    from shapely.geometry import LinearRing, LineString

    import geopandas as gpd

    lr = LinearRing([(0, 0), (1, 1), (1, 0)])
    ls = LineString([(0, 0), (1, 1)])

    # GeoSeries
    s = gpd.GeoSeries([lr, ls])
    unpickled_s = pickle.loads(pickle.dumps(s))
    assert isinstance(unpickled_s[0], LinearRing)
    assert isinstance(unpickled_s[1], LineString)

    # GeoDataFrame
    df = gpd.GeoDataFrame({"geometry": s})
    unpickled_df = pickle.loads(pickle.dumps(df))
    assert isinstance(unpickled_df.geometry[0], LinearRing)
    assert isinstance(unpickled_df.geometry[1], LineString)


def test_pickle_linear_ring_edge_cases():
    import pickle
    import numpy as np
    import shapely
    from shapely.geometry import LinearRing, LineString, Point, GeometryCollection

    from geopandas.array import GeometryArray, from_shapely

    lr = LinearRing([(0, 0), (1, 1), (1, 0)])
    lr_3d = LinearRing([(0, 0, 1), (1, 1, 2), (1, 0, 3)])
    lr_empty = LinearRing()
    ls = LineString([(0, 0), (1, 1)])
    gc = GeometryCollection([lr, Point(0, 0)])

    # 1. Array with no LinearRing
    ga_no_ring = from_shapely([ls, Point(0, 0)])
    unp_no_ring = pickle.loads(pickle.dumps(ga_no_ring))
    assert isinstance(unp_no_ring[0], LineString)

    # 2. Array with various LinearRings (2D, 3D, empty, multi-geom collection)
    ga_rings = from_shapely([lr, lr_3d, lr_empty, ls, gc])
    unp_rings = pickle.loads(pickle.dumps(ga_rings))
    assert isinstance(unp_rings[0], LinearRing)
    assert isinstance(unp_rings[1], LinearRing)
    assert isinstance(unp_rings[2], LinearRing)
    assert unp_rings[1].has_z
    assert unp_rings[2].is_empty
    assert isinstance(unp_rings[3], LineString)
    assert isinstance(unp_rings[4], GeometryCollection)

    # 3. Backwards compatibility / dict state handling in __setstate__
    # Case A: dict state with data key and without _crs key
    ga_dict0 = GeometryArray.__new__(GeometryArray)
    ga_dict0.__setstate__({
        "data": np.array([ls, lr], dtype=object),
        "_ring_indices": np.array([1]),
    })
    assert ga_dict0._crs is None
    assert isinstance(ga_dict0[1], LinearRing)

    # Case B: dict state with raw object array data
    ga_dict1 = GeometryArray.__new__(GeometryArray)
    ga_dict1.__setstate__({
        "_data": np.array([ls, lr], dtype=object),
        "_crs": None,
        "_ring_indices": np.array([1]),
    })
    assert isinstance(ga_dict1[1], LinearRing)

    # Case C: dict state with WKB-encoded data
    ga_dict2 = GeometryArray.__new__(GeometryArray)
    ga_dict2.__setstate__({
        "_data": shapely.to_wkb(np.array([ls, lr], dtype=object)),
        "_crs": None,
        "_ring_indices": np.array([1]),
    })
    assert isinstance(ga_dict2[1], LinearRing)

    # Case D: dict state with empty _ring_indices
    ga_dict3 = GeometryArray.__new__(GeometryArray)
    ga_dict3.__setstate__({
        "_data": np.array([ls, lr], dtype=object),
        "_crs": None,
        "_ring_indices": np.array([]),
    })
    assert isinstance(ga_dict3[0], LineString)
