from __future__ import absolute_import
import os
import pytest

from geopandas import GeoDataFrame, GeoSeries, read_file, read_parquet
from geopandas.datasets import get_path
from geopandas.io.parquet import (
    get_engine,
    PyArrowImpl,
    _wkb_name,
    _encode_crs,
    _decode_crs,
)
from geopandas.testing import assert_geodataframe_equal, assert_geoseries_equal


def test_wkb_name():
    assert _wkb_name("foo") == "__wkb_foo"


def test_encode_crs():
    crs = {"init": "EPSG:4326"}
    assert _encode_crs(crs) == crs

    crs = "+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs"
    assert _encode_crs(crs) == {"proj4": crs}


def test_decode_crs():
    crs = {"init": "EPSG:4326"}
    assert _decode_crs(crs) == crs

    crs_str = "+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs"
    crs = {"proj4": crs_str}
    assert _decode_crs(crs) == crs_str


def test_parquet_get_engine():
    assert isinstance(get_engine("pyarrow"), PyArrowImpl)


@pytest.mark.parametrize(
    "test_dataset", ["naturalearth_lowres", "naturalearth_cities", "nybb"]
)
def test_to_parquet(test_dataset, tmpdir):
    filename = os.path.join(tmpdir, "test.pq")
    df = read_file(get_path(test_dataset))

    df.to_parquet(filename)

    assert os.path.exists(filename)


def test_to_parquet_multiple_geom_cols(tmpdir):
    test_dataset = "naturalearth_lowres"
    filename = os.path.join(tmpdir, "test.pq")
    df = read_file(get_path(test_dataset))

    df["geom2"] = df.geometry.copy()

    df.to_parquet(filename)

    assert os.path.exists(filename)


@pytest.mark.parametrize(
    "test_dataset", ["naturalearth_lowres", "naturalearth_cities", "nybb"]
)
def test_from_parquet(test_dataset, tmpdir):
    filename = os.path.join(tmpdir, "test.pq")
    df = read_file(get_path(test_dataset))

    df.to_parquet(filename)
    pq_df = read_parquet(filename)

    assert isinstance(pq_df, GeoDataFrame)
    assert_geodataframe_equal(df, pq_df, check_like=True)


def test_from_parquet_multiple_geom_cols(tmpdir):
    test_dataset = "naturalearth_lowres"
    filename = os.path.join(tmpdir, "test.pq")
    df = read_file(get_path(test_dataset))

    df["geom2"] = df.geometry.copy()

    df.to_parquet(filename)
    pq_df = read_parquet(filename)

    assert isinstance(pq_df, GeoDataFrame)
    assert_geodataframe_equal(df, pq_df, check_like=True)

    assert_geoseries_equal(
        GeoSeries(df.geom2), GeoSeries(pq_df.geom2), check_geom_type=True
    )
