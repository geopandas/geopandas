from __future__ import absolute_import
import os
import pytest
from pandas import DataFrame
import numpy as np

from geopandas import GeoDataFrame, GeoSeries, read_file, read_parquet
from geopandas.array import to_wkb
from geopandas.datasets import get_path
from geopandas.io.parquet import (
    _encode_crs,
    _decode_crs,
    _encode_metadata,
    _decode_metadata,
    _validate_dataframe,
)
from geopandas.testing import assert_geodataframe_equal, assert_geoseries_equal


def test_encode_crs():
    crs = {"init": "EPSG:4326"}
    assert _encode_crs(crs) == crs

    crs = "+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs"
    assert _encode_crs(crs) == {"proj4": crs}

    crs = None
    assert _encode_crs(crs) is None


def test_decode_crs():
    crs = {"init": "EPSG:4326"}
    assert _decode_crs(crs) == crs

    crs_str = "+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs"
    crs = {"proj4": crs_str}
    assert _decode_crs(crs) == crs_str

    crs = None
    assert _decode_crs(crs) == crs


def test_encode_metadata():
    metadata = {"a": "b"}

    expected = b'{"a": "b"}'
    assert _encode_metadata(metadata) == expected


def test_decode_metadata():
    metadata_str = b'{"a": "b"}'

    expected = {"a": "b"}
    assert _decode_metadata(metadata_str) == expected


def test_validate_dataframe():
    test_dataset = "naturalearth_lowres"
    df = read_file(get_path(test_dataset))

    # valid: should not raise ValueError
    _validate_dataframe(df)
    _validate_dataframe(df.set_index("iso_a3"))

    # add column with non-string type
    df[0] = 1

    # invalid: should raise ValueError
    with pytest.raises(ValueError):
        _validate_dataframe(df)

    with pytest.raises(ValueError):
        _validate_dataframe(df.set_index(0))

    # not a DataFrame: should raise ValueError
    with pytest.raises(ValueError):
        _validate_dataframe("not a dataframe")


@pytest.mark.parametrize(
    "test_dataset", ["naturalearth_lowres"]  # FIXME: , "naturalearth_cities", "nybb"
)
def test_parquet_io(test_dataset, tmpdir):
    """Writing to parquet should not raise errors, and should not alter original
    GeoDataFrame
    """

    df = read_file(get_path(test_dataset))
    orig = df.copy()

    filename = os.path.join(str(tmpdir), "test.pq")
    df.to_parquet(filename)

    assert os.path.exists(filename)

    # make sure that the original data frame is unaltered
    assert_geodataframe_equal(df, orig)

    # make sure that we can roundtrip the data frame
    pq_df = read_parquet(filename)

    assert isinstance(pq_df, GeoDataFrame)
    assert_geodataframe_equal(df, pq_df, check_like=True)


def test_parquet_index(tmpdir):
    """Setting index=`True` should preserve index in output, and
    setting index=`False` should drop index from output.
    """

    test_dataset = "naturalearth_lowres"
    df = read_file(get_path(test_dataset)).set_index("iso_a3")

    filename = os.path.join(str(tmpdir), "test_with_index.pq")
    df.to_parquet(filename, index=True)
    pq_df = read_parquet(filename)
    assert_geodataframe_equal(df, pq_df, check_like=True)

    filename = os.path.join(str(tmpdir), "drop_index.pq")
    df.to_parquet(filename, index=False)
    pq_df = read_parquet(filename)
    assert_geodataframe_equal(df.reset_index(drop=True), pq_df, check_like=True)


@pytest.mark.parametrize("compression", ["snappy", "gzip", "brotli", None])
def test_parquet_compression(compression, tmpdir):
    """Using compression options should not raise errors, and should
    return identical GeoDataFrame.
    """

    test_dataset = "naturalearth_lowres"
    df = read_file(get_path(test_dataset))

    filename = os.path.join(str(tmpdir), "test.pq")
    df.to_parquet(filename, compression=compression)
    pq_df = read_parquet(filename)

    assert isinstance(pq_df, GeoDataFrame)
    assert_geodataframe_equal(df, pq_df, check_like=True)


def test_parquet_multiple_geom_cols(tmpdir):
    """If multiple geometry columns are present when written to parquet,
    they should all be returned as such when read from parquet.
    """

    test_dataset = "naturalearth_lowres"
    df = read_file(get_path(test_dataset))
    df["geom2"] = df.geometry.copy()

    filename = os.path.join(str(tmpdir), "test.pq")
    df.to_parquet(filename)

    assert os.path.exists(filename)

    pq_df = read_parquet(filename)

    assert isinstance(pq_df, GeoDataFrame)
    assert_geodataframe_equal(df, pq_df, check_like=True)

    assert_geoseries_equal(
        GeoSeries(df.geom2), GeoSeries(pq_df.geom2), check_geom_type=True
    )


def test_parquet_missing_metadata(tmpdir):
    """Missing geo metadata, such as from a parquet file created
    from a pandas DataFrame, will require geometry_columns.
    """

    test_dataset = "naturalearth_lowres"
    df = read_file(get_path(test_dataset))
    orig = df.copy()

    # convert to DataFrame
    df = DataFrame(df)

    # convert the geometry column so we can extract later
    df["geometry"] = to_wkb(df["geometry"].values)

    filename = os.path.join(str(tmpdir), "test.pq")

    # use pandas to_parquet (no geo metadata)
    df.to_parquet(filename)

    # missing geometry_columns will raise ValueError
    with pytest.raises(ValueError):
        read_parquet(filename)

    # parquet file should be read if geometry_columns is provided;
    # CRS will be None (default).
    # A warning should be raised for missing metadata.
    with pytest.warns(UserWarning):
        pq_df = read_parquet(filename, geometry_columns=["geometry"])

    assert_geodataframe_equal(orig, pq_df, check_like=True, check_crs=False)
    assert pq_df.crs is None

    # an invalid geometry name should raise an error
    with pytest.raises(ValueError):
        read_parquet(filename, geometry_columns=["geometry"], geometry="notgeometry")

    # passed-in CRS should be returned
    pq_df = read_parquet(filename, geometry_columns=["geometry"], crs=orig.crs)
    assert_geodataframe_equal(orig, pq_df, check_like=True, check_crs=True)


def test_parquet_subset_columns(tmpdir):
    """Reading a subset of columns should correctly decode selected geometry
    columns.
    """

    test_dataset = "naturalearth_lowres"
    df = read_file(get_path(test_dataset))

    filename = os.path.join(str(tmpdir), "test.pq")
    df.to_parquet(filename)
    pq_df = read_parquet(filename, columns=["name", "geometry"])

    assert_geodataframe_equal(df[["name", "geometry"]], pq_df, check_like=True)


def test_parquet_promote_secondary_geometry(tmpdir):
    """Reading a subset of columns that does not include the primary geometry
    column should promote the first geometry column present.
    """

    test_dataset = "naturalearth_lowres"
    df = read_file(get_path(test_dataset))
    df["geom2"] = df.geometry.copy()

    filename = os.path.join(str(tmpdir), "test.pq")
    df.to_parquet(filename)
    pq_df = read_parquet(filename, columns=["name", "geom2"])

    assert_geodataframe_equal(
        df.set_geometry("geom2")[["name", "geom2"]], pq_df, check_like=True
    )


def test_parquet_columns_no_geometry(tmpdir):
    """Reading a parquet file that is missing all of the geometry columns
    should raise a ValueError"""

    test_dataset = "naturalearth_lowres"
    df = read_file(get_path(test_dataset))

    filename = os.path.join(str(tmpdir), "test.pq")
    df.to_parquet(filename)

    with pytest.raises(ValueError):
        read_parquet(filename, columns=["name"])


def test_parquet_missing_crs(tmpdir):
    """If CRS is `None`, it should be properly handled
    and remain `None` when read from parquet`.
    """

    test_dataset = "naturalearth_lowres"

    df = read_file(get_path(test_dataset))
    df = GeoDataFrame(df, crs=None)

    filename = os.path.join(str(tmpdir), "test.pq")
    df.to_parquet(filename)
    pq_df = read_parquet(filename)

    assert_geodataframe_equal(df, pq_df, check_crs=True)


def test_parquet_text_crs(tmpdir):
    """Text-based CRS should be properly handled and be identical to input
    CRS when read from file
    """

    test_dataset = "naturalearth_lowres"
    df = read_file(get_path(test_dataset))

    # Construct a GeoDataFrame with a PROJ4 version of crs
    df = GeoDataFrame(df, crs="+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs")

    filename = os.path.join(str(tmpdir), "test.pq")
    df.to_parquet(filename)
    pq_df = read_parquet(filename)

    print("orig crs", df.crs)
    print("output crs", pq_df.crs)

    assert_geodataframe_equal(df, pq_df, check_crs=True)


def test_parquet_overrides(tmpdir):
    """geometry_columns, geometry, and crs passed as parameters should override
    values in the parquet file metadata.
    """

    test_dataset = "naturalearth_lowres"
    df = read_file(get_path(test_dataset))

    # create an extra geometry column
    df["geom2"] = df.geometry.copy()

    filename = os.path.join(str(tmpdir), "test.pq")
    df.to_parquet(filename)

    # returned CRS should match parameter
    crs = "+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs"
    pq_df = read_parquet(filename, crs=crs)
    assert pq_df.crs == crs

    # specified geometry should be set in result
    pq_df = read_parquet(filename, geometry="geom2")
    assert pq_df._geometry_column_name == "geom2"

    # if specified geometry is not available, a ValueError should be raised
    with pytest.raises(ValueError):
        read_parquet(filename, geometry="geom3")

    # if specified geometry was written, but not read, a ValueError should be raised
    with pytest.raises(ValueError):
        read_parquet(filename, columns=["name", "geometry"], geometry="geom2")

    # if geometry_columns are specified but include missing columns,
    # a ValueError should be raised
    with pytest.raises(ValueError):
        read_parquet(filename, geometry_columns=["geom3"])

    # if geometry_columns are specified but include columns missing
    # from a subset that are read, a ValueError should be raised
    with pytest.raises(ValueError):
        read_parquet(filename, columns=["name", "geom2"], geometry_columns=["geometry"])

    # if fewer geometry_columns are specified than are actually present in the file,
    # the remaining geometry columns should still be encoded as WKB
    pq_df = read_parquet(filename, geometry_columns=["geometry"])
    assert np.array_equal(to_wkb(df.geom2.values), pq_df.geom2.values)
