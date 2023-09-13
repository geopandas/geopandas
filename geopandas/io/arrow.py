from packaging.version import Version
import json
import warnings

import numpy as np
from pandas import DataFrame, Series

import geopandas._compat as compat
from geopandas._compat import import_optional_dependency
from geopandas.array import from_wkb
from geopandas import GeoDataFrame
import geopandas
from .file import _expand_user

METADATA_VERSION = "1.0.0-beta.1"
SUPPORTED_VERSIONS = ["0.1.0", "0.4.0", "1.0.0-beta.1"]
# reference: https://github.com/opengeospatial/geoparquet

# Metadata structure:
# {
#     "geo": {
#         "columns": {
#             "<name>": {
#                 "encoding": "WKB"
#                 "geometry_types": <list of str: REQUIRED>
#                 "crs": "<PROJJSON or None: OPTIONAL>",
#                 "orientation": "<'counterclockwise' or None: OPTIONAL>"
#                 "edges": "planar"
#                 "bbox": <list of [xmin, ymin, xmax, ymax]: OPTIONAL>
#                 "epoch": <float: OPTIONAL>
#             }
#         },
#         "primary_column": "<str: REQUIRED>",
#         "version": "<METADATA_VERSION>",
#
#         # Additional GeoPandas specific metadata (not in metadata spec)
#         "creator": {
#             "library": "geopandas",
#             "version": "<geopandas.__version__>"
#         }
#     }
# }


def _is_fsspec_url(url):
    return (
        isinstance(url, str)
        and "://" in url
        and not url.startswith(("http://", "https://"))
    )


def _remove_id_from_member_of_ensembles(json_dict):
    """
    Older PROJ versions will not recognize IDs of datum ensemble members that
    were added in more recent PROJ database versions.

    Cf https://github.com/opengeospatial/geoparquet/discussions/110
    and https://github.com/OSGeo/PROJ/pull/3221

    Mimicking the patch to GDAL from https://github.com/OSGeo/gdal/pull/5872
    """
    for key, value in json_dict.items():
        if isinstance(value, dict):
            _remove_id_from_member_of_ensembles(value)
        elif key == "members" and isinstance(value, list):
            for member in value:
                member.pop("id", None)


def _create_metadata(df, schema_version=None):
    """Create and encode geo metadata dict.

    Parameters
    ----------
    df : GeoDataFrame
    schema_version : {'0.1.0', '0.4.0', '1.0.0-beta.1', None}
        GeoParquet specification version; if not provided will default to
        latest supported version.

    Returns
    -------
    dict
    """

    schema_version = schema_version or METADATA_VERSION

    if schema_version not in SUPPORTED_VERSIONS:
        raise ValueError(
            f"schema_version must be one of: {', '.join(SUPPORTED_VERSIONS)}"
        )

    # Construct metadata for each geometry
    column_metadata = {}
    for col in df.columns[df.dtypes == "geometry"]:
        series = df[col]
        geometry_types = sorted(Series(series.geom_type.unique()).dropna())
        if schema_version[0] == "0":
            geometry_types_name = "geometry_type"
            if len(geometry_types) == 1:
                geometry_types = geometry_types[0]
        else:
            geometry_types_name = "geometry_types"

        crs = None
        if series.crs:
            if schema_version == "0.1.0":
                crs = series.crs.to_wkt()
            else:  # version >= 0.4.0
                crs = series.crs.to_json_dict()
                _remove_id_from_member_of_ensembles(crs)

        column_metadata[col] = {
            "encoding": "WKB",
            "crs": crs,
            geometry_types_name: geometry_types,
        }

        bbox = series.total_bounds.tolist()
        if np.isfinite(bbox).all():
            # don't add bbox with NaNs for empty / all-NA geometry column
            column_metadata[col]["bbox"] = bbox

    return {
        "primary_column": df._geometry_column_name,
        "columns": column_metadata,
        "version": schema_version or METADATA_VERSION,
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

    # version was schema_version in 0.1.0
    version = metadata.get("version", metadata.get("schema_version"))
    if not version:
        raise ValueError(
            "'geo' metadata in Parquet/Feather file is missing required key: "
            "'version'"
        )

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
    # leaving out "geometry_type" for compatibility with 0.1
    required_col_keys = ("encoding",)
    for col, column_metadata in metadata["columns"].items():
        for key in required_col_keys:
            if key not in column_metadata:
                raise ValueError(
                    "'geo' metadata in Parquet/Feather file is missing required key "
                    "'{key}' for column '{col}'".format(key=key, col=col)
                )

        if column_metadata["encoding"] != "WKB":
            raise ValueError("Only WKB geometry encoding is supported")

        if column_metadata.get("edges", "planar") == "spherical":
            warnings.warn(
                f"The geo metadata indicate that column '{col}' has spherical edges, "
                "but because GeoPandas currently does not support spherical "
                "geometry, it ignores this metadata and will interpret the edges of "
                "the geometries as planar.",
                UserWarning,
                stacklevel=4,
            )


def _geopandas_to_arrow(df, index=None, schema_version=None):
    """
    Helper function with main, shared logic for to_parquet/to_feather.
    """
    from pyarrow import Table

    _validate_dataframe(df)

    # create geo metadata before altering incoming data frame
    geo_metadata = _create_metadata(df, schema_version=schema_version)

    kwargs = {}
    if compat.USE_SHAPELY_20:
        kwargs = {"flavor": "iso"}
    else:
        for col in df.columns[df.dtypes == "geometry"]:
            series = df[col]
            if series.has_z.any():
                warnings.warn(
                    "The GeoDataFrame contains 3D geometries, and when using "
                    "shapely < 2.0, such geometries will be written not exactly "
                    "following to the GeoParquet spec (not using ISO WKB). For "
                    "most use cases this should not be a problem (GeoPandas can "
                    "read such files fine).",
                    stacklevel=2,
                )
                break
    df = df.to_wkb(**kwargs)

    table = Table.from_pandas(df, preserve_index=index)

    # Store geopandas specific file-level metadata
    # This must be done AFTER creating the table or it is not persisted
    metadata = table.schema.metadata
    metadata.update({b"geo": _encode_metadata(geo_metadata)})

    return table.replace_schema_metadata(metadata)


def _to_parquet(
    df, path, index=None, compression="snappy", schema_version=None, **kwargs
):
    """
    Write a GeoDataFrame to the Parquet format.

    Any geometry columns present are serialized to WKB format in the file.

    Requires 'pyarrow'.

    This is tracking version 1.0.0-beta.1 of the GeoParquet specification at:
    https://github.com/opengeospatial/geoparquet. Writing older versions is
    supported using the `schema_version` keyword.

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
    schema_version : {'0.1.0', '0.4.0', '1.0.0-beta.1', None}
        GeoParquet specification version; if not provided will default to
        latest supported version.
    **kwargs
        Additional keyword arguments passed to pyarrow.parquet.write_table().
    """
    parquet = import_optional_dependency(
        "pyarrow.parquet", extra="pyarrow is required for Parquet support."
    )

    if kwargs and "version" in kwargs and kwargs["version"] is not None:
        if schema_version is None and kwargs["version"] in SUPPORTED_VERSIONS:
            warnings.warn(
                "the `version` parameter has been replaced with `schema_version`. "
                "`version` will instead be passed directly to the underlying "
                "parquet writer unless `version` is 0.1.0 or 0.4.0.",
                FutureWarning,
                stacklevel=2,
            )
            schema_version = kwargs.pop("version")

    path = _expand_user(path)
    table = _geopandas_to_arrow(df, index=index, schema_version=schema_version)
    parquet.write_table(table, path, compression=compression, **kwargs)


def _to_feather(df, path, index=None, compression=None, schema_version=None, **kwargs):
    """
    Write a GeoDataFrame to the Feather format.

    Any geometry columns present are serialized to WKB format in the file.

    Requires 'pyarrow' >= 0.17.

    This is tracking version 1.0.0-beta.1 of the GeoParquet specification for
    the metadata at: https://github.com/opengeospatial/geoparquet. Writing
    older versions is supported using the `schema_version` keyword.

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
    schema_version : {'0.1.0', '0.4.0', '1.0.0-beta.1', None}
        GeoParquet specification version for the metadata; if not provided
        will default to latest supported version.
    kwargs
        Additional keyword arguments passed to pyarrow.feather.write_feather().
    """
    feather = import_optional_dependency(
        "pyarrow.feather", extra="pyarrow is required for Feather support."
    )
    # TODO move this into `import_optional_dependency`
    import pyarrow

    if Version(pyarrow.__version__) < Version("0.17.0"):
        raise ImportError("pyarrow >= 0.17 required for Feather support")

    if kwargs and "version" in kwargs and kwargs["version"] is not None:
        if schema_version is None and kwargs["version"] in SUPPORTED_VERSIONS:
            warnings.warn(
                "the `version` parameter has been replaced with `schema_version`. "
                "`version` will instead be passed directly to the underlying "
                "feather writer unless `version` is 0.1.0 or 0.4.0.",
                FutureWarning,
                stacklevel=2,
            )
            schema_version = kwargs.pop("version")

    path = _expand_user(path)
    table = _geopandas_to_arrow(df, index=index, schema_version=schema_version)
    feather.write_feather(table, path, compression=compression, **kwargs)


def _arrow_to_geopandas(table, metadata=None):
    """
    Helper function with main, shared logic for read_parquet/read_feather.
    """
    df = table.to_pandas()

    metadata = metadata or table.schema.metadata

    if metadata is None or b"geo" not in metadata:
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
                "file. The first column read was promoted to the primary geometry.",
                stacklevel=3,
            )

    # Convert the WKB columns that are present back to geometry.
    for col in geometry_columns:
        col_metadata = metadata["columns"][col]
        if "crs" in col_metadata:
            crs = col_metadata["crs"]
            if isinstance(crs, dict):
                _remove_id_from_member_of_ensembles(crs)
        else:
            # per the GeoParquet spec, missing CRS is to be interpreted as
            # OGC:CRS84
            crs = "OGC:CRS84"

        df[col] = from_wkb(df[col].values, crs=crs)

    return GeoDataFrame(df, geometry=geometry)


def _get_filesystem_path(path, filesystem=None, storage_options=None):
    """
    Get the filesystem and path for a given filesystem and path.

    If the filesystem is not None then it's just returned as is.
    """
    import pyarrow

    if (
        isinstance(path, str)
        and storage_options is None
        and filesystem is None
        and Version(pyarrow.__version__) >= Version("5.0.0")
    ):
        # Use the native pyarrow filesystem if possible.
        try:
            from pyarrow.fs import FileSystem

            filesystem, path = FileSystem.from_uri(path)
        except Exception:
            # fallback to use get_handle / fsspec for filesystems
            # that pyarrow doesn't support
            pass

    if _is_fsspec_url(path) and filesystem is None:
        fsspec = import_optional_dependency(
            "fsspec", extra="fsspec is requred for 'storage_options'."
        )
        filesystem, path = fsspec.core.url_to_fs(path, **(storage_options or {}))

    if filesystem is None and storage_options:
        raise ValueError(
            "Cannot provide 'storage_options' with non-fsspec path '{}'".format(path)
        )

    return filesystem, path


def _ensure_arrow_fs(filesystem):
    """
    Simplified version of pyarrow.fs._ensure_filesystem. This is only needed
    below because `pyarrow.parquet.read_metadata` does not yet accept a
    filesystem keyword (https://issues.apache.org/jira/browse/ARROW-16719)
    """
    from pyarrow import fs

    if isinstance(filesystem, fs.FileSystem):
        return filesystem

    # handle fsspec-compatible filesystems
    try:
        import fsspec
    except ImportError:
        pass
    else:
        if isinstance(filesystem, fsspec.AbstractFileSystem):
            return fs.PyFileSystem(fs.FSSpecHandler(filesystem))

    return filesystem


def _read_parquet(path, columns=None, storage_options=None, **kwargs):
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

    Supports versions 0.1.0, 0.4.0 and 1.0.0-beta.1 of the GeoParquet
    specification at: https://github.com/opengeospatial/geoparquet

    If 'crs' key is not present in the GeoParquet metadata associated with the
    Parquet object, it will default to "OGC:CRS84" according to the specification.

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
    storage_options : dict, optional
        Extra options that make sense for a particular storage connection, e.g. host,
        port, username, password, etc. For HTTP(S) URLs the key-value pairs are
        forwarded to urllib as header options. For other URLs (e.g. starting with
        "s3://", and "gcs://") the key-value pairs are forwarded to fsspec. Please
        see fsspec and urllib for more details.

        When no storage options are provided and a filesystem is implemented by
        both ``pyarrow.fs`` and ``fsspec`` (e.g. "s3://") then the ``pyarrow.fs``
        filesystem is preferred. Provide the instantiated fsspec filesystem using
        the ``filesystem`` keyword if you wish to use its implementation.
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
    # TODO(https://github.com/pandas-dev/pandas/pull/41194): see if pandas
    # adds filesystem as a keyword and match that.
    filesystem = kwargs.pop("filesystem", None)
    filesystem, path = _get_filesystem_path(
        path, filesystem=filesystem, storage_options=storage_options
    )

    path = _expand_user(path)
    kwargs["use_pandas_metadata"] = True
    table = parquet.read_table(path, columns=columns, filesystem=filesystem, **kwargs)

    # read metadata separately to get the raw Parquet FileMetaData metadata
    # (pyarrow doesn't properly exposes those in schema.metadata for files
    # created by GDAL - https://issues.apache.org/jira/browse/ARROW-16688)
    metadata = None
    if table.schema.metadata is None or b"geo" not in table.schema.metadata:
        try:
            # read_metadata does not accept a filesystem keyword, so need to
            # handle this manually (https://issues.apache.org/jira/browse/ARROW-16719)
            if filesystem is not None:
                pa_filesystem = _ensure_arrow_fs(filesystem)
                with pa_filesystem.open_input_file(path) as source:
                    metadata = parquet.read_metadata(source).metadata
            else:
                metadata = parquet.read_metadata(path).metadata
        except Exception:
            pass

    return _arrow_to_geopandas(table, metadata)


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

    Supports versions 0.1.0, 0.4.0 and 1.0.0-beta.1 of the GeoParquet
    specification at: https://github.com/opengeospatial/geoparquet

    If 'crs' key is not present in the Feather metadata associated with the
    Parquet object, it will default to "OGC:CRS84" according to the specification.

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

    if Version(pyarrow.__version__) < Version("0.17.0"):
        raise ImportError("pyarrow >= 0.17 required for Feather support")

    path = _expand_user(path)
    table = feather.read_table(path, columns=columns, **kwargs)
    return _arrow_to_geopandas(table)
