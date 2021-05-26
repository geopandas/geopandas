from distutils.version import LooseVersion
import json
import warnings

from pandas import DataFrame

from geopandas._compat import import_optional_dependency
from geopandas.array import from_wkb
from geopandas import GeoDataFrame
import geopandas


METADATA_VERSION = "0.1.0"
# reference: https://github.com/geopandas/geo-arrow-spec

# Metadata structure:
# {
#     "geo": {
#         "columns": {
#             "<name>": {
#                 "crs": "<WKT or None: REQUIRED>",
#                 "encoding": "WKB"
#             }
#         },
#         "creator": {
#             "library": "geopandas",
#             "version": "<geopandas.__version__>"
#         }
#         "primary_column": "<str: REQUIRED>",
#         "schema_version": "<METADATA_VERSION>"
#     }
# }


def _create_metadata(df):
    """Create and encode geo metadata dict.

    Parameters
    ----------
    df : GeoDataFrame

    Returns
    -------
    dict
    """

    # Construct metadata for each geometry
    column_metadata = {}
    for col in df.columns[df.dtypes == "geometry"]:
        series = df[col]
        column_metadata[col] = {
            "crs": series.crs.to_wkt() if series.crs else None,
            "encoding": "WKB",
            "bbox": series.total_bounds.tolist(),
        }

    return {
        "primary_column": df._geometry_column_name,
        "columns": column_metadata,
        "schema_version": METADATA_VERSION,
        "creator": {"library": "geopandas", "version": geopandas.__version__},
    }


def _encode_metadata(metadata):
    """Encode metadata dict to UTF-8 JSON string

    Parameters
    ----------
    metadata : dict

    Returns
    -------
    UTF-8 encoded JSON string
    """
    return json.dumps(metadata).encode("utf-8")


def _decode_metadata(metadata_str):
    """Decode a UTF-8 encoded JSON string to dict

    Parameters
    ----------
    metadata_str : string (UTF-8 encoded)

    Returns
    -------
    dict
    """
    if metadata_str is None:
        return None

    return json.loads(metadata_str.decode("utf-8"))


def _validate_dataframe(df):
    """Validate that the GeoDataFrame conforms to requirements for writing
    to Parquet format.

    Raises `ValueError` if the GeoDataFrame is not valid.

    copied from `pandas.io.parquet`

    Parameters
    ----------
    df : GeoDataFrame
    """

    if not isinstance(df, DataFrame):
        raise ValueError("Writing to Parquet/Feather only supports IO with DataFrames")

    # must have value column names (strings only)
    if df.columns.inferred_type not in {"string", "unicode", "empty"}:
        raise ValueError("Writing to Parquet/Feather requires string column names")

    # index level names must be strings
    valid_names = all(
        isinstance(name, str) for name in df.index.names if name is not None
    )
    if not valid_names:
        raise ValueError("Index level names must be strings")


def _validate_metadata(metadata):
    """Validate geo metadata.
    Must not be empty, and must contain the structure specified above.

    Raises ValueError if metadata is not valid.

    Parameters
    ----------
    metadata : dict
    """

    if not metadata:
        raise ValueError("Missing or malformed geo metadata in Parquet/Feather file")

    required_keys = ("primary_column", "columns")
    for key in required_keys:
        if metadata.get(key, None) is None:
            raise ValueError(
                "'geo' metadata in Parquet/Feather file is missing required key: "
                "'{key}'".format(key=key)
            )

    if not isinstance(metadata["columns"], dict):
        raise ValueError("'columns' in 'geo' metadata must be a dict")

    # Validate that geometry columns have required metadata and values
    required_col_keys = ("crs", "encoding")
    for col, column_metadata in metadata["columns"].items():
        for key in required_col_keys:
            if key not in column_metadata:
                raise ValueError(
                    "'geo' metadata in Parquet/Feather file is missing required key "
                    "'{key}' for column '{col}'".format(key=key, col=col)
                )

        if column_metadata["encoding"] != "WKB":
            raise ValueError("Only WKB geometry encoding is supported")


def _geopandas_to_arrow(df, index=None):
    """
    Helper function with main, shared logic for to_parquet/to_feather.
    """
    from pyarrow import Table

    warnings.warn(
        "this is an initial implementation of Parquet/Feather file support and "
        "associated metadata.  This is tracking version 0.1.0 of the metadata "
        "specification at "
        "https://github.com/geopandas/geo-arrow-spec\n\n"
        "This metadata specification does not yet make stability promises.  "
        "We do not yet recommend using this in a production setting unless you "
        "are able to rewrite your Parquet/Feather files.\n\n"
        "To further ignore this warning, you can do: \n"
        "import warnings; warnings.filterwarnings('ignore', "
        "message='.*initial implementation of Parquet.*')",
        UserWarning,
        stacklevel=4,
    )

    _validate_dataframe(df)

    # create geo metadata before altering incoming data frame
    geo_metadata = _create_metadata(df)

    df = df.to_wkb()

    table = Table.from_pandas(df, preserve_index=index)

    # Store geopandas specific file-level metadata
    # This must be done AFTER creating the table or it is not persisted
    metadata = table.schema.metadata
    metadata.update({b"geo": _encode_metadata(geo_metadata)})
    return table.replace_schema_metadata(metadata)


def _to_parquet(df, path, index=None, compression="snappy", **kwargs):
    """
    Write a GeoDataFrame to the Parquet format.

    Any geometry columns present are serialized to WKB format in the file.

    Requires 'pyarrow'.

    WARNING: this is an initial implementation of Parquet file support and
    associated metadata.  This is tracking version 0.1.0 of the metadata
    specification at:
    https://github.com/geopandas/geo-arrow-spec

    This metadata specification does not yet make stability promises.  As such,
    we do not yet recommend using this in a production setting unless you are
    able to rewrite your Parquet files.


    .. versionadded:: 0.8

    Parameters
    ----------
    path : str, path object
    index : bool, default None
        If ``True``, always include the dataframe's index(es) as columns
        in the file output.
        If ``False``, the index(es) will not be written to the file.
        If ``None``, the index(ex) will be included as columns in the file
        output except `RangeIndex` which is stored as metadata only.
    compression : {'snappy', 'gzip', 'brotli', None}, default 'snappy'
        Name of the compression to use. Use ``None`` for no compression.
    kwargs
        Additional keyword arguments passed to pyarrow.parquet.write_table().
    """
    parquet = import_optional_dependency(
        "pyarrow.parquet", extra="pyarrow is required for Parquet support."
    )

    table = _geopandas_to_arrow(df, index=index)
    parquet.write_table(table, path, compression=compression, **kwargs)


def _to_feather(df, path, index=None, compression=None, **kwargs):
    """
    Write a GeoDataFrame to the Feather format.

    Any geometry columns present are serialized to WKB format in the file.

    Requires 'pyarrow' >= 0.17.

    WARNING: this is an initial implementation of Feather file support and
    associated metadata.  This is tracking version 0.1.0 of the metadata
    specification at:
    https://github.com/geopandas/geo-arrow-spec

    This metadata specification does not yet make stability promises.  As such,
    we do not yet recommend using this in a production setting unless you are
    able to rewrite your Feather files.

    .. versionadded:: 0.8

    Parameters
    ----------
    path : str, path object
    index : bool, default None
        If ``True``, always include the dataframe's index(es) as columns
        in the file output.
        If ``False``, the index(es) will not be written to the file.
        If ``None``, the index(ex) will be included as columns in the file
        output except `RangeIndex` which is stored as metadata only.
    compression : {'zstd', 'lz4', 'uncompressed'}, optional
        Name of the compression to use. Use ``"uncompressed"`` for no
        compression. By default uses LZ4 if available, otherwise uncompressed.
    kwargs
        Additional keyword arguments passed to pyarrow.feather.write_feather().
    """
    feather = import_optional_dependency(
        "pyarrow.feather", extra="pyarrow is required for Feather support."
    )
    # TODO move this into `import_optional_dependency`
    import pyarrow

    if pyarrow.__version__ < LooseVersion("0.17.0"):
        raise ImportError("pyarrow >= 0.17 required for Feather support")

    table = _geopandas_to_arrow(df, index=index)
    feather.write_feather(table, path, compression=compression, **kwargs)


def _arrow_to_geopandas(table):
    """
    Helper function with main, shared logic for read_parquet/read_feather.
    """
    df = table.to_pandas()

    metadata = table.schema.metadata
    if b"geo" not in metadata:
        raise ValueError(
            """Missing geo metadata in Parquet/Feather file.
            Use pandas.read_parquet/read_feather() instead."""
        )

    try:
        metadata = _decode_metadata(metadata.get(b"geo", b""))

    except (TypeError, json.decoder.JSONDecodeError):
        raise ValueError("Missing or malformed geo metadata in Parquet/Feather file")

    _validate_metadata(metadata)

    # Find all geometry columns that were read from the file.  May
    # be a subset if 'columns' parameter is used.
    geometry_columns = df.columns.intersection(metadata["columns"])

    if not len(geometry_columns):
        raise ValueError(
            """No geometry columns are included in the columns read from
            the Parquet/Feather file.  To read this file without geometry columns,
            use pandas.read_parquet/read_feather() instead."""
        )

    geometry = metadata["primary_column"]

    # Missing geometry likely indicates a subset of columns was read;
    # promote the first available geometry to the primary geometry.
    if len(geometry_columns) and geometry not in geometry_columns:
        geometry = geometry_columns[0]

        # if there are multiple non-primary geometry columns, raise a warning
        if len(geometry_columns) > 1:
            warnings.warn(
                "Multiple non-primary geometry columns read from Parquet/Feather "
                "file. The first column read was promoted to the primary geometry."
            )

    # Convert the WKB columns that are present back to geometry.
    for col in geometry_columns:
        df[col] = from_wkb(df[col].values, crs=metadata["columns"][col]["crs"])

    return GeoDataFrame(df, geometry=geometry)


def _read_parquet(path, columns=None, **kwargs):
    """
    Load a Parquet object from the file path, returning a GeoDataFrame.

    You can read a subset of columns in the file using the ``columns`` parameter.
    However, the structure of the returned GeoDataFrame will depend on which
    columns you read:

    * if no geometry columns are read, this will raise a ``ValueError`` - you
      should use the pandas `read_parquet` method instead.
    * if the primary geometry column saved to this file is not included in
      columns, the first available geometry column will be set as the geometry
      column of the returned GeoDataFrame.

    Requires 'pyarrow'.

    .. versionadded:: 0.8

    Parameters
    ----------
    path : str, path object
    columns : list-like of strings, default=None
        If not None, only these columns will be read from the file.  If
        the primary geometry column is not included, the first secondary
        geometry read from the file will be set as the geometry column
        of the returned GeoDataFrame.  If no geometry columns are present,
        a ``ValueError`` will be raised.
    **kwargs
        Any additional kwargs passed to pyarrow.parquet.read_table().

    Returns
    -------
    GeoDataFrame

    Examples
    --------
    >>> df = geopandas.read_parquet("data.parquet")  # doctest: +SKIP

    Specifying columns to read:

    >>> df = geopandas.read_parquet(
    ...     "data.parquet",
    ...     columns=["geometry", "pop_est"]
    ... )  # doctest: +SKIP
    """

    parquet = import_optional_dependency(
        "pyarrow.parquet", extra="pyarrow is required for Parquet support."
    )

    kwargs["use_pandas_metadata"] = True
    table = parquet.read_table(path, columns=columns, **kwargs)

    return _arrow_to_geopandas(table)


def _read_feather(path, columns=None, **kwargs):
    """
    Load a Feather object from the file path, returning a GeoDataFrame.

    You can read a subset of columns in the file using the ``columns`` parameter.
    However, the structure of the returned GeoDataFrame will depend on which
    columns you read:

    * if no geometry columns are read, this will raise a ``ValueError`` - you
      should use the pandas `read_feather` method instead.
    * if the primary geometry column saved to this file is not included in
      columns, the first available geometry column will be set as the geometry
      column of the returned GeoDataFrame.

    Requires 'pyarrow' >= 0.17.

    .. versionadded:: 0.8

    Parameters
    ----------
    path : str, path object
    columns : list-like of strings, default=None
        If not None, only these columns will be read from the file.  If
        the primary geometry column is not included, the first secondary
        geometry read from the file will be set as the geometry column
        of the returned GeoDataFrame.  If no geometry columns are present,
        a ``ValueError`` will be raised.
    **kwargs
        Any additional kwargs passed to pyarrow.feather.read_table().

    Returns
    -------
    GeoDataFrame

    Examples
    --------
    >>> df = geopandas.read_feather("data.feather")  # doctest: +SKIP

    Specifying columns to read:

    >>> df = geopandas.read_feather(
    ...     "data.feather",
    ...     columns=["geometry", "pop_est"]
    ... )  # doctest: +SKIP
    """

    feather = import_optional_dependency(
        "pyarrow.feather", extra="pyarrow is required for Feather support."
    )
    # TODO move this into `import_optional_dependency`
    import pyarrow

    if pyarrow.__version__ < LooseVersion("0.17.0"):
        raise ImportError("pyarrow >= 0.17 required for Feather support")

    table = feather.read_table(path, columns=columns, **kwargs)
    return _arrow_to_geopandas(table)
