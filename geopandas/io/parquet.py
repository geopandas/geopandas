import json


from pandas.io.common import get_filepath_or_buffer
from pandas import DataFrame

from geopandas._compat import import_optional_dependency
from geopandas.array import from_wkb, to_wkb, GeometryArray
from geopandas import GeoDataFrame
import geopandas


METADATA_VERSION = "0.1.0"
# reference: https://github.com/geopandas/geo-arrow-spec

# Metadata structure:
# {
#     "crs": {
#         "primary_column": "<str: REQUIRED>",
#         "columns": {
#             "<name>": {
#                 "crs": "<WKT or None: REQUIRED>",
#                 "encoding": "WKB"
#             }
#         }
#     }
# }


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
    to parquet format.

    Raises `ValueError` if the GeoDataFrame is not valid.

    copied from `pandas.io.parquet`

    Parameters
    ----------
    df : GeoDataFrame
    """

    if not isinstance(df, DataFrame):
        raise ValueError("to_parquet only supports IO with DataFrames")

    # must have value column names (strings only)
    if df.columns.inferred_type not in {"string", "unicode", "empty"}:
        raise ValueError("parquet must have string column names")

    # index level names must be strings
    valid_names = all(
        isinstance(name, str) for name in df.index.names if name is not None
    )
    if not valid_names:
        raise ValueError("Index level names must be strings")


def to_parquet(df, path, compression="snappy", index=None, **kwargs):
    """
    Write a GeoDataFrame to the parquet format.

    Any geometry columns present are serialized to WKB format in the file.

    Requires 'pyarrow'.

    Parameters
    ----------
    path : str
        File path or Root Directory path. Will be used as Root Directory path
        while writing a partitioned dataset.
    compression : {'snappy', 'gzip', 'brotli', None}, default 'snappy'
        Name of the compression to use. Use ``None`` for no compression.
    index : bool, default None
        If ``True``, always include the dataframe's index(es) as columns
        in the file output.
        If ``False``, the index(es) will not be written to the file.
        If ``None``, the index(ex) will be included as columns in the file
        output except `RangeIndex` which is stored as metadata only.
    kwargs
        Additional keyword arguments passed to parquet.write_table().
    """

    import_optional_dependency(
        "pyarrow.parquet", extra="pyarrow is required for parquet support."
    )
    from pyarrow import parquet, Table

    _validate_dataframe(df)

    geometry_column = df._geometry_column_name
    geometry_columns = df.columns[df.dtypes == "geometry"]

    # Construct metadata for each geometry
    column_metadata = {}
    for col in geometry_columns:
        series = df[col]
        column_metadata[col] = {
            "crs": series.crs.to_wkt() if series.crs else None,
            "encoding": "WKB",
            "bounds": series.total_bounds.tolist(),
        }

    # Convert to a DataFrame so we can convert geometries to WKB while
    # retaining original column names.
    df = DataFrame(df.copy())

    # Encode all geometry columns to WKB
    for col in geometry_columns:
        df[col] = to_wkb(df[col].values)

    table = Table.from_pandas(df, preserve_index=index)

    # Store geopandas specific file-level metadata
    # This must be done AFTER creating the table or it is not persisted
    metadata = table.schema.metadata
    metadata.update(
        {
            "geo": _encode_metadata(
                {
                    "primary_column": geometry_column,
                    "columns": column_metadata,
                    "schema_version": METADATA_VERSION,
                    "creator": {
                        "library": "geopandas",
                        "version": geopandas.__version__,
                    },
                }
            )
        }
    )

    table = table.replace_schema_metadata(metadata)
    parquet.write_table(table, path, compression=compression, **kwargs)


def read_parquet(path, columns=None, **kwargs):
    """
    Load a parquet object from the file path, returning a GeoDataFrame.

    You can read a subset of columns in the file using the ``columns`` parameter.
    However, the structure of the returned GeoDataFrame will depend on which
    columns you read:
    * if no geometry columns are read, this will raise a ``ValueError`` - you
      should use the pandas `read_parquet` method instead.
    * if the primary geometry column saved to this file is not included in
      columns, the first available geometry column will be set as the geometry
      column of the returned GeoDataFrame.

    Requires 'pyarrow'.

    Parameters
    ----------
    path : str, path object or file-like object
        Any valid string path is acceptable. The string could be a URL. Valid
        URL schemes include http, ftp, s3, and file. For file URLs, a host is
        expected. A local file could be:
        ``file://localhost/path/to/table.parquet``.

        If you want to pass in a path object, geopandas accepts any
        ``os.PathLike``.

        By file-like object, we refer to objects with a ``read()`` method,
        such as a file handler (e.g. via builtin ``open`` function)
        or ``StringIO``.
    columns : list-like of strings, default=None
        If not None, only these columns will be read from the file.  If
        the primary geometry column is not included, the first secondary
        geometry read from the file will be set as the geometry column
        of the returned GeoDataFrame.  If no geometry columns are present,
        a ``ValueError`` will be raised.
    **kwargs
        Any additional kwargs passed to parquet.read_table().

    Returns
    -------
    GeoDataFrame
    """

    import_optional_dependency(
        "pyarrow", extra="pyarrow is required for parquet support."
    )
    from pyarrow import parquet

    path, _, _, should_close = get_filepath_or_buffer(path)

    kwargs["use_pandas_metadata"] = True
    table = parquet.read_table(path, columns=columns, **kwargs)

    df = table.to_pandas()

    if should_close:
        try:
            path.close()
        except:  # noqa: flake8
            pass

    metadata = None

    try:
        metadata = table.schema.metadata
        if metadata is not None:
            if b"geo" not in metadata:
                raise ValueError("Missing or malformed geo metadata in parquet file")

            metadata = _decode_metadata(metadata.get(b"geo", b""))

    except (TypeError, json.decoder.JSONDecodeError):
        raise ValueError("Missing or malformed geo metadata in parquet file")

    if not metadata:
        raise ValueError("Missing or malformed geo metadata in parquet file")

    # Validate that required keys are present
    required_keys = ("primary_column", "columns")
    for key in required_keys:
        if key not in metadata:
            raise ValueError(
                f"""'geo' metadata in parquet file is missing required key:
                '{key}'"""
            )

    # Find all geometry columns that were read from the file.  May
    # be a subset if 'columns' parameter is used.
    geometry_columns = df.columns.intersection(metadata["columns"])

    if not len(geometry_columns):
        raise ValueError(
            """No geometry columns are included in the columns read from
            the parquet file.  To read this file without geometry columns,
            use pandas.read_parquet() instead."""
        )

    column_metadata = metadata["columns"]

    # Validate that geometry columns have required metadata and values
    required_col_keys = ("crs", "encoding")
    for col in geometry_columns:
        for key in required_col_keys:
            if key not in column_metadata[col]:
                raise ValueError(
                    f"""'geo' metadata in parquet file is missing required key
                    {key} for column '{col}'"""
                )

        if column_metadata[col]["encoding"] != "WKB":
            raise ValueError("Only WKB geometry encoding is supported")

    geometry = metadata["primary_column"]

    # Missing geometry likely indicates a subset of columns was read;
    # promote the first available geometry to the primary geometry.
    if len(geometry_columns) and geometry not in geometry_columns:
        geometry = geometry_columns[0]

    # Convert the WKB columns that are present back to geometry.
    for col in geometry_columns:
        df[col] = GeometryArray(
            from_wkb(df[col].values), crs=column_metadata[col]["crs"]
        )

    return GeoDataFrame(df, geometry=geometry)
