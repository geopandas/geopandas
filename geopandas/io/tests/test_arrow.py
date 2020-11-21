from __future__ import absolute_import

from distutils.version import LooseVersion
import os

import pytest
from pandas import DataFrame, read_parquet as pd_read_parquet
from pandas.testing import assert_frame_equal
import numpy as np

import geopandas
from geopandas import GeoDataFrame, read_file, read_parquet, read_feather
from geopandas.array import to_wkb
from geopandas.datasets import get_path
from geopandas.io.arrow import (
    _create_metadata,
    _decode_metadata,
    _encode_metadata,
    _validate_dataframe,
    _validate_metadata,
    METADATA_VERSION,
)
from geopandas.testing import assert_geodataframe_equal, assert_geoseries_equal


# Skip all tests in this module if pyarrow is not available
pyarrow = pytest.importorskip("pyarrow")

# TEMPORARY: hide warning from to_parquet
pytestmark = pytest.mark.filterwarnings("ignore:.*initial implementation of Parquet.*")


@pytest.fixture(
    params=[
        "parquet",
        pytest.param(
            "feather",
            marks=pytest.mark.skipif(
                pyarrow.__version__ < LooseVersion("0.17.0"),
                reason="needs pyarrow >= 0.17",
            ),
        ),
    ]
)
def file_format(request):
    if request.param == "parquet":
        return read_parquet, GeoDataFrame.to_parquet
    elif request.param == "feather":
        return read_feather, GeoDataFrame.to_feather


def test_create_metadata():
    test_dataset = "naturalearth_lowres"
    df = read_file(get_path(test_dataset))
    metadata = _create_metadata(df)

    assert isinstance(metadata, dict)
    assert metadata["schema_version"] == METADATA_VERSION
    assert metadata["creator"]["library"] == "geopandas"
    assert metadata["creator"]["version"] == geopandas.__version__
    assert metadata["primary_column"] == "geometry"
    assert "geometry" in metadata["columns"]
    assert metadata["columns"]["geometry"]["crs"] == df.geometry.crs.to_wkt()
    assert metadata["columns"]["geometry"]["encoding"] == "WKB"

    assert np.array_equal(
        metadata["columns"]["geometry"]["bbox"], df.geometry.total_bounds
    )


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


def test_validate_metadata_valid():
    _validate_metadata(
        {
            "primary_column": "geometry",
            "columns": {"geometry": {"crs": None, "encoding": "WKB"}},
        }
    )

    _validate_metadata(
        {
            "primary_column": "geometry",
            "columns": {"geometry": {"crs": "WKT goes here", "encoding": "WKB"}},
        }
    )


@pytest.mark.parametrize(
    "metadata,error",
    [
        ({}, "Missing or malformed geo metadata in Parquet/Feather file"),
        (
            {"primary_column": "foo"},
            "'geo' metadata in Parquet/Feather file is missing required key:",
        ),
        (
            {"primary_column": "foo", "columns": None},
            "'geo' metadata in Parquet/Feather file is missing required key",
        ),
        (
            {"primary_column": "foo", "columns": []},
            "'columns' in 'geo' metadata must be a dict",
        ),
        (
            {"primary_column": "foo", "columns": {"foo": {}}},
            (
                "'geo' metadata in Parquet/Feather file is missing required key 'crs' "
                "for column 'foo'"
            ),
        ),
        (
            {"primary_column": "foo", "columns": {"foo": {"crs": None}}},
            "'geo' metadata in Parquet/Feather file is missing required key",
        ),
        (
            {"primary_column": "foo", "columns": {"foo": {"encoding": None}}},
            "'geo' metadata in Parquet/Feather file is missing required key",
        ),
        (
            {
                "primary_column": "foo",
                "columns": {"foo": {"crs": None, "encoding": None}},
            },
            "Only WKB geometry encoding is supported",
        ),
        (
            {
                "primary_column": "foo",
                "columns": {"foo": {"crs": None, "encoding": "BKW"}},
            },
            "Only WKB geometry encoding is supported",
        ),
    ],
)
def test_validate_metadata_invalid(metadata, error):
    with pytest.raises(ValueError, match=error):
        _validate_metadata(metadata)


# TEMPORARY: used to determine if pyarrow fails for roundtripping pandas data
# without geometries
def test_pandas_parquet_roundtrip1(tmpdir):
    df = DataFrame({"a": [1, 2, 3], "b": ["a", "b", "c"]})

    filename = os.path.join(str(tmpdir), "test.pq")
    df.to_parquet(filename)

    pq_df = pd_read_parquet(filename)

    assert_frame_equal(df, pq_df)


@pytest.mark.parametrize(
    "test_dataset", ["naturalearth_lowres", "naturalearth_cities", "nybb"]
)
def test_pandas_parquet_roundtrip2(test_dataset, tmpdir):
    test_dataset = "naturalearth_lowres"
    df = DataFrame(read_file(get_path(test_dataset)).drop(columns=["geometry"]))

    filename = os.path.join(str(tmpdir), "test.pq")
    df.to_parquet(filename)

    pq_df = pd_read_parquet(filename)

    assert_frame_equal(df, pq_df)


@pytest.mark.parametrize(
    "test_dataset", ["naturalearth_lowres", "naturalearth_cities", "nybb"]
)
def test_roundtrip(tmpdir, file_format, test_dataset):
    """Writing to parquet should not raise errors, and should not alter original
    GeoDataFrame
    """
    reader, writer = file_format

    df = read_file(get_path(test_dataset))
    orig = df.copy()

    filename = os.path.join(str(tmpdir), "test.pq")

    # TEMP: Initial implementation should raise a UserWarning
    with pytest.warns(UserWarning, match="initial implementation"):
        writer(df, filename)

    assert os.path.exists(filename)

    # make sure that the original data frame is unaltered
    assert_geodataframe_equal(df, orig)

    # make sure that we can roundtrip the data frame
    pq_df = reader(filename)

    assert isinstance(pq_df, GeoDataFrame)
    assert_geodataframe_equal(df, pq_df)


def test_index(tmpdir, file_format):
    """Setting index=`True` should preserve index in output, and
    setting index=`False` should drop index from output.
    """
    reader, writer = file_format

    test_dataset = "naturalearth_lowres"
    df = read_file(get_path(test_dataset)).set_index("iso_a3")

    filename = os.path.join(str(tmpdir), "test_with_index.pq")
    writer(df, filename, index=True)
    pq_df = reader(filename)
    assert_geodataframe_equal(df, pq_df)

    filename = os.path.join(str(tmpdir), "drop_index.pq")
    writer(df, filename, index=False)
    pq_df = reader(filename)
    assert_geodataframe_equal(df.reset_index(drop=True), pq_df)


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
    assert_geodataframe_equal(df, pq_df)


@pytest.mark.skipif(
    pyarrow.__version__ < LooseVersion("0.17.0"),
    reason="Feather only supported for pyarrow >= 0.17",
)
@pytest.mark.parametrize("compression", ["uncompressed", "lz4", "zstd"])
def test_feather_compression(compression, tmpdir):
    """Using compression options should not raise errors, and should
    return identical GeoDataFrame.
    """

    test_dataset = "naturalearth_lowres"
    df = read_file(get_path(test_dataset))

    filename = os.path.join(str(tmpdir), "test.feather")
    df.to_feather(filename, compression=compression)
    pq_df = read_feather(filename)

    assert isinstance(pq_df, GeoDataFrame)
    assert_geodataframe_equal(df, pq_df)


def test_parquet_multiple_geom_cols(tmpdir, file_format):
    """If multiple geometry columns are present when written to parquet,
    they should all be returned as such when read from parquet.
    """
    reader, writer = file_format

    test_dataset = "naturalearth_lowres"
    df = read_file(get_path(test_dataset))
    df["geom2"] = df.geometry.copy()

    filename = os.path.join(str(tmpdir), "test.pq")
    writer(df, filename)

    assert os.path.exists(filename)

    pq_df = reader(filename)

    assert isinstance(pq_df, GeoDataFrame)
    assert_geodataframe_equal(df, pq_df)

    assert_geoseries_equal(df.geom2, pq_df.geom2, check_geom_type=True)


def test_parquet_missing_metadata(tmpdir):
    """Missing geo metadata, such as from a parquet file created
    from a pandas DataFrame, will raise a ValueError.
    """

    test_dataset = "naturalearth_lowres"
    df = read_file(get_path(test_dataset))

    # convert to DataFrame
    df = DataFrame(df)

    # convert the geometry column so we can extract later
    df["geometry"] = to_wkb(df["geometry"].values)

    filename = os.path.join(str(tmpdir), "test.pq")

    # use pandas to_parquet (no geo metadata)
    df.to_parquet(filename)

    # missing metadata will raise ValueError
    with pytest.raises(
        ValueError, match="Missing geo metadata in Parquet/Feather file."
    ):
        read_parquet(filename)


@pytest.mark.parametrize(
    "geo_meta,error",
    [
        ({"geo": b""}, "Missing or malformed geo metadata in Parquet/Feather file"),
        (
            {"geo": _encode_metadata({})},
            "Missing or malformed geo metadata in Parquet/Feather file",
        ),
        (
            {"geo": _encode_metadata({"foo": "bar"})},
            "'geo' metadata in Parquet/Feather file is missing required key",
        ),
    ],
)
def test_parquet_invalid_metadata(tmpdir, geo_meta, error):
    """Has geo metadata with missing required fields will raise a ValueError.

    This requires writing the parquet file directly below, so that we can
    control the metadata that is written for this test.
    """

    from pyarrow import parquet, Table

    test_dataset = "naturalearth_lowres"
    df = read_file(get_path(test_dataset))

    # convert to DataFrame and encode geometry to WKB
    df = DataFrame(df)
    df["geometry"] = to_wkb(df["geometry"].values)

    table = Table.from_pandas(df)
    metadata = table.schema.metadata
    metadata.update(geo_meta)
    table = table.replace_schema_metadata(metadata)

    filename = os.path.join(str(tmpdir), "test.pq")
    parquet.write_table(table, filename)

    with pytest.raises(ValueError, match=error):
        read_parquet(filename)


def test_subset_columns(tmpdir, file_format):
    """Reading a subset of columns should correctly decode selected geometry
    columns.
    """
    reader, writer = file_format

    test_dataset = "naturalearth_lowres"
    df = read_file(get_path(test_dataset))

    filename = os.path.join(str(tmpdir), "test.pq")
    writer(df, filename)
    pq_df = reader(filename, columns=["name", "geometry"])

    assert_geodataframe_equal(df[["name", "geometry"]], pq_df)

    with pytest.raises(
        ValueError, match="No geometry columns are included in the columns read"
    ):
        reader(filename, columns=["name"])


def test_promote_secondary_geometry(tmpdir, file_format):
    """Reading a subset of columns that does not include the primary geometry
    column should promote the first geometry column present.
    """
    reader, writer = file_format

    test_dataset = "naturalearth_lowres"
    df = read_file(get_path(test_dataset))
    df["geom2"] = df.geometry.copy()

    filename = os.path.join(str(tmpdir), "test.pq")
    writer(df, filename)
    pq_df = reader(filename, columns=["name", "geom2"])

    assert_geodataframe_equal(df.set_geometry("geom2")[["name", "geom2"]], pq_df)

    df["geom3"] = df.geometry.copy()

    writer(df, filename)
    with pytest.warns(
        UserWarning,
        match="Multiple non-primary geometry columns read from Parquet/Feather file.",
    ):
        pq_df = reader(filename, columns=["name", "geom2", "geom3"])

    assert_geodataframe_equal(
        df.set_geometry("geom2")[["name", "geom2", "geom3"]], pq_df
    )


def test_columns_no_geometry(tmpdir, file_format):
    """Reading a parquet file that is missing all of the geometry columns
    should raise a ValueError"""
    reader, writer = file_format

    test_dataset = "naturalearth_lowres"
    df = read_file(get_path(test_dataset))

    filename = os.path.join(str(tmpdir), "test.pq")
    writer(df, filename)

    with pytest.raises(ValueError):
        reader(filename, columns=["name"])


def test_missing_crs(tmpdir, file_format):
    """If CRS is `None`, it should be properly handled
    and remain `None` when read from parquet`.
    """
    reader, writer = file_format

    test_dataset = "naturalearth_lowres"

    df = read_file(get_path(test_dataset))
    df.crs = None

    filename = os.path.join(str(tmpdir), "test.pq")
    writer(df, filename)
    pq_df = reader(filename)

    assert pq_df.crs is None

    assert_geodataframe_equal(df, pq_df, check_crs=True)


@pytest.mark.skipif(
    pyarrow.__version__ >= LooseVersion("0.17.0"),
    reason="Feather only supported for pyarrow >= 0.17",
)
def test_feather_arrow_version(tmpdir):
    df = read_file(get_path("naturalearth_lowres"))
    filename = os.path.join(str(tmpdir), "test.feather")

    with pytest.raises(
        ImportError, match="pyarrow >= 0.17 required for Feather support"
    ):
        df.to_feather(filename)
