from __future__ import absolute_import
import os
import pytest

from geopandas import GeoDataFrame, GeoSeries, read_file, read_parquet
from geopandas.datasets import get_path
from geopandas.io.parquet import _encode_crs, _decode_crs
from geopandas.testing import assert_geodataframe_equal, assert_geoseries_equal


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


@pytest.mark.parametrize(
    "test_dataset", ["naturalearth_lowres", "naturalearth_cities", "nybb"]
)
def test_to_parquet(test_dataset, tmpdir):
    filename = os.path.join(str(tmpdir), "test.pq")
    df = read_file(get_path(test_dataset))
    orig = df.copy()

    df.to_parquet(filename)

    assert os.path.exists(filename)

    # make sure that the original data frame is unaltered
    assert_geodataframe_equal(df, orig)


@pytest.mark.parametrize("compression", ["snappy", "gzip", "brotli", None])
def test_to_parquet_compression(compression, tmpdir):
    test_dataset = "naturalearth_lowres"
    filename = os.path.join(str(tmpdir), "test.pq")
    df = read_file(get_path(test_dataset))

    df.to_parquet(filename, compression=compression)

    assert os.path.exists(filename)

    pq_df = read_parquet(filename)

    assert isinstance(pq_df, GeoDataFrame)
    assert_geodataframe_equal(df, pq_df, check_like=True)


def test_to_parquet_multiple_geom_cols(tmpdir):
    test_dataset = "naturalearth_lowres"
    filename = os.path.join(str(tmpdir), "test.pq")
    df = read_file(get_path(test_dataset))

    df["geom2"] = df.geometry.copy()

    df.to_parquet(filename)

    assert os.path.exists(filename)


@pytest.mark.parametrize(
    "test_dataset", ["naturalearth_lowres", "naturalearth_cities", "nybb"]
)
def test_from_parquet(test_dataset, tmpdir):
    filename = os.path.join(str(tmpdir), "test.pq")
    df = read_file(get_path(test_dataset))

    df.to_parquet(filename)
    pq_df = read_parquet(filename)

    assert isinstance(pq_df, GeoDataFrame)
    assert_geodataframe_equal(df, pq_df, check_like=True)


def test_from_parquet_multiple_geom_cols(tmpdir):
    test_dataset = "naturalearth_lowres"
    filename = os.path.join(str(tmpdir), "test.pq")
    df = read_file(get_path(test_dataset))

    df["geom2"] = df.geometry.copy()

    df.to_parquet(filename)
    pq_df = read_parquet(filename)

    assert isinstance(pq_df, GeoDataFrame)
    assert_geodataframe_equal(df, pq_df, check_like=True)

    assert_geoseries_equal(
        GeoSeries(df.geom2), GeoSeries(pq_df.geom2), check_geom_type=True
    )


def test_from_parquet_columns(tmpdir):
    """Reading a subset of columns should correctly decode selected geometry
    columns"""

    test_dataset = "naturalearth_lowres"
    filename = os.path.join(str(tmpdir), "test.pq")
    df = read_file(get_path(test_dataset))
    df.to_parquet(filename)

    pq_df = read_parquet(filename, columns=["name", "geometry"])

    assert_geodataframe_equal(df[["name", "geometry"]], pq_df, check_like=True)


def test_from_parquet_columns_promote_secondary_geometry(tmpdir):
    """Reading a subset of columns that does not include the primary geometry
    column should promote the first geometry column present"""

    test_dataset = "naturalearth_lowres"
    filename = os.path.join(str(tmpdir), "test.pq")
    df = read_file(get_path(test_dataset))
    df["geom2"] = df.geometry.copy()
    df.to_parquet(filename)

    pq_df = read_parquet(filename, columns=["name", "geom2"])

    assert_geodataframe_equal(
        df.set_geometry("geom2")[["name", "geom2"]], pq_df, check_like=True
    )


def test_from_parquet_columns_no_geometry(tmpdir):
    """Reading a parquet file that is missing all of the geometry columns
    should raise a warning"""

    test_dataset = "naturalearth_lowres"
    filename = os.path.join(str(tmpdir), "test.pq")
    df = read_file(get_path(test_dataset))
    df.to_parquet(filename)

    with pytest.warns(UserWarning):
        read_parquet(filename, columns=["name"])


def test_parquet_text_crs(tmpdir):
    test_dataset = "naturalearth_lowres"
    filename = os.path.join(str(tmpdir), "test.pq")
    df = read_file(get_path(test_dataset))

    # Construct a GeoDataFrame with a PROJ4 version of crs
    df = GeoDataFrame(df, crs="+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs")
    df.to_parquet(filename)

    pq_df = read_parquet(filename)

    assert_geodataframe_equal(df, pq_df, check_crs=True)
