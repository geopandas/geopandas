from __future__ import absolute_import
import os
import pytest
from pandas import DataFrame

from geopandas import GeoDataFrame, read_file, read_parquet
from geopandas.array import to_wkb
from geopandas.datasets import get_path
from geopandas.io.parquet import (
    _encode_metadata,
    _decode_metadata,
    _validate_dataframe,
)
from geopandas.testing import assert_geodataframe_equal, assert_geoseries_equal


try:
    import pyarrow  # noqa

    HAS_PYARROW = True

except ImportError:
    HAS_PYARROW = False


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


@pytest.mark.skipif(not HAS_PYARROW, reason="requires pyarrow")
@pytest.mark.parametrize(
    "test_dataset", ["naturalearth_lowres", "naturalearth_cities", "nybb"]
)
def test_parquet_roundtrip(test_dataset, tmpdir):
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
    assert_geodataframe_equal(df, pq_df)


@pytest.mark.skipif(not HAS_PYARROW, reason="requires pyarrow")
def test_parquet_index(tmpdir):
    """Setting index=`True` should preserve index in output, and
    setting index=`False` should drop index from output.
    """

    test_dataset = "naturalearth_lowres"
    df = read_file(get_path(test_dataset)).set_index("iso_a3")

    filename = os.path.join(str(tmpdir), "test_with_index.pq")
    df.to_parquet(filename, index=True)
    pq_df = read_parquet(filename)
    assert_geodataframe_equal(df, pq_df)

    filename = os.path.join(str(tmpdir), "drop_index.pq")
    df.to_parquet(filename, index=False)
    pq_df = read_parquet(filename)
    assert_geodataframe_equal(df.reset_index(drop=True), pq_df)


@pytest.mark.skipif(not HAS_PYARROW, reason="requires pyarrow")
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


@pytest.mark.skipif(not HAS_PYARROW, reason="requires pyarrow")
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
    assert_geodataframe_equal(df, pq_df)

    assert_geoseries_equal(df.geom2, pq_df.geom2, check_geom_type=True)


@pytest.mark.skipif(not HAS_PYARROW, reason="requires pyarrow")
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
        ValueError, match="Missing or malformed geo metadata in parquet file"
    ):
        read_parquet(filename)


@pytest.mark.skipif(not HAS_PYARROW, reason="requires pyarrow")
@pytest.mark.parametrize(
    "geo_meta,error",
    [
        ({"geo": b""}, "Missing or malformed geo metadata in parquet file"),
        (
            {"geo": _encode_metadata({})},
            "Missing or malformed geo metadata in parquet file",
        ),
        (
            {"geo": _encode_metadata({"foo": "bar"})},
            "'geo' metadata in parquet file is missing required key",
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


@pytest.mark.skipif(not HAS_PYARROW, reason="requires pyarrow")
def test_parquet_subset_columns(tmpdir):
    """Reading a subset of columns should correctly decode selected geometry
    columns.
    """

    test_dataset = "naturalearth_lowres"
    df = read_file(get_path(test_dataset))

    filename = os.path.join(str(tmpdir), "test.pq")
    df.to_parquet(filename)
    pq_df = read_parquet(filename, columns=["name", "geometry"])

    assert_geodataframe_equal(df[["name", "geometry"]], pq_df)

    with pytest.raises(
        ValueError, match="No geometry columns are included in the columns read"
    ):
        read_parquet(filename, columns=[])


@pytest.mark.skipif(not HAS_PYARROW, reason="requires pyarrow")
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

    assert_geodataframe_equal(df.set_geometry("geom2")[["name", "geom2"]], pq_df)


@pytest.mark.skipif(not HAS_PYARROW, reason="requires pyarrow")
def test_parquet_columns_no_geometry(tmpdir):
    """Reading a parquet file that is missing all of the geometry columns
    should raise a ValueError"""

    test_dataset = "naturalearth_lowres"
    df = read_file(get_path(test_dataset))

    filename = os.path.join(str(tmpdir), "test.pq")
    df.to_parquet(filename)

    with pytest.raises(ValueError):
        read_parquet(filename, columns=["name"])


@pytest.mark.skipif(not HAS_PYARROW, reason="requires pyarrow")
def test_parquet_missing_crs(tmpdir):
    """If CRS is `None`, it should be properly handled
    and remain `None` when read from parquet`.
    """

    test_dataset = "naturalearth_lowres"

    df = read_file(get_path(test_dataset))
    df.crs = None

    filename = os.path.join(str(tmpdir), "test.pq")
    df.to_parquet(filename)
    pq_df = read_parquet(filename)

    assert pq_df.crs is None

    assert_geodataframe_equal(df, pq_df, check_crs=True)
