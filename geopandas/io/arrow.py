import json
import warnings
from packaging.version import Version

import numpy as np
from pandas import DataFrame, Series

import shapely

import geopandas
from geopandas import GeoDataFrame
from geopandas._compat import import_optional_dependency
from geopandas.array import from_shapely, from_wkb

from .file import _expand_user

METADATA_VERSION = "1.0.0"
SUPPORTED_VERSIONS = ["0.1.0", "0.4.0", "1.0.0-beta.1", "1.0.0", "1.1.0"]
GEOARROW_ENCODINGS = [
    "point",
    "linestring",
    "polygon",
    "multipoint",
    "multilinestring",
    "multipolygon",
]
SUPPORTED_ENCODINGS = ["WKB"] + GEOARROW_ENCODINGS

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


# type ids 0 to 7
_geometry_type_names = [
    "Point",
    "LineString",
    "LineString",
    "Polygon",
    "MultiPoint",
    "MultiLineString",
    "MultiPolygon",
    "GeometryCollection",
]
_geometry_type_names += [geom_type + " Z" for geom_type in _geometry_type_names]


def _get_geometry_types(series):
    """
    Get unique geometry types from a GeoSeries.
    """
    arr_geometry_types = shapely.get_type_id(series.array._data)
    # ensure to include "... Z" for 3D geometries
    has_z = shapely.has_z(series.array._data)
    arr_geometry_types[has_z] += 8

    geometry_types = Series(arr_geometry_types).unique().tolist()
    # drop missing values (shapely.get_type_id returns -1 for those)
    if -1 in geometry_types:
        geometry_types.remove(-1)

    return sorted([_geometry_type_names[idx] for idx in geometry_types])


def _create_metadata(
    df, schema_version=None, geometry_encoding=None, write_covering_bbox=False
):
    """Create and encode geo metadata dict.

    Parameters
    ----------
    df : GeoDataFrame
    schema_version : {'0.1.0', '0.4.0', '1.0.0-beta.1', '1.0.0', None}
        GeoParquet specification version; if not provided will default to
        latest supported version.
    write_covering_bbox : bool, default False
        Writes the bounding box column for each row entry with column
        name 'bbox'. Writing a bbox column can be computationally
        expensive, hence is default setting is False.

    Returns
    -------
    dict
    """
    if schema_version is None:
        if geometry_encoding and any(
            encoding != "WKB" for encoding in geometry_encoding.values()
        ):
            schema_version = "1.1.0"
        else:
            schema_version = METADATA_VERSION

    if schema_version not in SUPPORTED_VERSIONS:
        raise ValueError(
            f"schema_version must be one of: {', '.join(SUPPORTED_VERSIONS)}"
        )

    # Construct metadata for each geometry
    column_metadata = {}
    for col in df.columns[df.dtypes == "geometry"]:
        series = df[col]

        geometry_types = _get_geometry_types(series)
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
            "encoding": geometry_encoding[col],
            "crs": crs,
            geometry_types_name: geometry_types,
        }

        bbox = series.total_bounds.tolist()
        if np.isfinite(bbox).all():
            # don't add bbox with NaNs for empty / all-NA geometry column
            column_metadata[col]["bbox"] = bbox

        if write_covering_bbox:
            column_metadata[col]["covering"] = {
                "bbox": {
                    "xmin": ["bbox", "xmin"],
                    "ymin": ["bbox", "ymin"],
                    "xmax": ["bbox", "xmax"],
                    "ymax": ["bbox", "ymax"],
                },
            }

    return {
        "primary_column": df._geometry_column_name,
        "columns": column_metadata,
        "version": schema_version,
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


def _validate_geo_metadata(metadata):
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

        if column_metadata["encoding"] not in SUPPORTED_ENCODINGS:
            raise ValueError(
                "Only WKB geometry encoding or one of the native encodings "
                f"({GEOARROW_ENCODINGS!r}) are supported, "
                f"got: {column_metadata['encoding']}"
            )

        if column_metadata.get("edges", "planar") == "spherical":
            warnings.warn(
                f"The geo metadata indicate that column '{col}' has spherical edges, "
                "but because GeoPandas currently does not support spherical "
                "geometry, it ignores this metadata and will interpret the edges of "
                "the geometries as planar.",
                UserWarning,
                stacklevel=4,
            )

        if "covering" in column_metadata:
            covering = column_metadata["covering"]
            if "bbox" in covering:
                bbox = covering["bbox"]
                for var in ["xmin", "ymin", "xmax", "ymax"]:
                    if var not in bbox.keys():
                        raise ValueError("Metadata for bbox column is malformed.")


def _geopandas_to_arrow(
    df,
    index=None,
    geometry_encoding="WKB",
    schema_version=None,
    write_covering_bbox=None,
):
    """
    Helper function with main, shared logic for to_parquet/to_feather.
    """
    from pyarrow import StructArray

    from geopandas.io._geoarrow import geopandas_to_arrow

    _validate_dataframe(df)

    if schema_version is not None:
        if geometry_encoding != "WKB" and schema_version != "1.1.0":
            raise ValueError(
                "'geoarrow' encoding is only supported with schema version >= 1.1.0"
            )

    table, geometry_encoding_dict = geopandas_to_arrow(
        df, geometry_encoding=geometry_encoding, index=index, interleaved=False
    )
    geo_metadata = _create_metadata(
        df,
        schema_version=schema_version,
        geometry_encoding=geometry_encoding_dict,
        write_covering_bbox=write_covering_bbox,
    )

    if write_covering_bbox:
        if "bbox" in df.columns:
            raise ValueError(
                "An existing column 'bbox' already exists in the dataframe. "
                "Please rename to write covering bbox."
            )
        bounds = df.bounds
        bbox_array = StructArray.from_arrays(
            [bounds["minx"], bounds["miny"], bounds["maxx"], bounds["maxy"]],
            names=["xmin", "ymin", "xmax", "ymax"],
        )
        table = table.append_column("bbox", bbox_array)

    # Store geopandas specific file-level metadata
    # This must be done AFTER creating the table or it is not persisted
    metadata = table.schema.metadata
    metadata.update({b"geo": _encode_metadata(geo_metadata)})

    return table.replace_schema_metadata(metadata)


def _to_parquet(
    df,
    path,
    index=None,
    compression="snappy",
    geometry_encoding="WKB",
    schema_version=None,
    write_covering_bbox=False,
    **kwargs,
):
    """
    Write a GeoDataFrame to the Parquet format.

    Any geometry columns present are serialized to WKB format in the file.

    Requires 'pyarrow'.

    This is tracking version 1.0.0 of the GeoParquet specification at:
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
    geometry_encoding : {'WKB', 'geoarrow'}, default 'WKB'
        The encoding to use for the geometry columns. Defaults to "WKB"
        for maximum interoperability. Specify "geoarrow" to use one of the
        native GeoArrow-based single-geometry type encodings.
    schema_version : {'0.1.0', '0.4.0', '1.0.0', None}
        GeoParquet specification version; if not provided will default to
        latest supported version.
    write_covering_bbox : bool, default False
        Writes the bounding box column for each row entry with column
        name 'bbox'. Writing a bbox column can be computationally
        expensive, hence is default setting is False.
    **kwargs
        Additional keyword arguments passed to pyarrow.parquet.write_table().
    """
    parquet = import_optional_dependency(
        "pyarrow.parquet", extra="pyarrow is required for Parquet support."
    )

    path = _expand_user(path)
    table = _geopandas_to_arrow(
        df,
        index=index,
        geometry_encoding=geometry_encoding,
        schema_version=schema_version,
        write_covering_bbox=write_covering_bbox,
    )
    parquet.write_table(table, path, compression=compression, **kwargs)


def _to_feather(df, path, index=None, compression=None, schema_version=None, **kwargs):
    """
    Write a GeoDataFrame to the Feather format.

    Any geometry columns present are serialized to WKB format in the file.

    Requires 'pyarrow' >= 0.17.

    This is tracking version 1.0.0 of the GeoParquet specification for
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
    schema_version : {'0.1.0', '0.4.0', '1.0.0', None}
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

    path = _expand_user(path)
    table = _geopandas_to_arrow(df, index=index, schema_version=schema_version)
    feather.write_feather(table, path, compression=compression, **kwargs)


def _arrow_to_geopandas(table, geo_metadata=None):
    """
    Helper function with main, shared logic for read_parquet/read_feather.
    """
    if geo_metadata is None:
        # Note: this path of not passing metadata is also used by dask-geopandas
        geo_metadata = _validate_and_decode_metadata(table.schema.metadata)

    # Find all geometry columns that were read from the file.  May
    # be a subset if 'columns' parameter is used.
    geometry_columns = [
        col for col in geo_metadata["columns"] if col in table.column_names
    ]
    result_column_names = list(table.slice(0, 0).to_pandas().columns)
    geometry_columns.sort(key=result_column_names.index)

    if not len(geometry_columns):
        raise ValueError(
            """No geometry columns are included in the columns read from
            the Parquet/Feather file.  To read this file without geometry columns,
            use pandas.read_parquet/read_feather() instead."""
        )

    geometry = geo_metadata["primary_column"]

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

    table_attr = table.drop(geometry_columns)
    df = table_attr.to_pandas()

    # Convert the WKB columns that are present back to geometry.
    for col in geometry_columns:
        col_metadata = geo_metadata["columns"][col]
        if "crs" in col_metadata:
            crs = col_metadata["crs"]
            if isinstance(crs, dict):
                _remove_id_from_member_of_ensembles(crs)
        else:
            # per the GeoParquet spec, missing CRS is to be interpreted as
            # OGC:CRS84
            crs = "OGC:CRS84"

        if col_metadata["encoding"] == "WKB":
            geom_arr = from_wkb(np.array(table[col]), crs=crs)
        else:
            from geopandas.io._geoarrow import construct_shapely_array

            geom_arr = from_shapely(
                construct_shapely_array(
                    table[col].combine_chunks(), "geoarrow." + col_metadata["encoding"]
                ),
                crs=crs,
            )

        df.insert(result_column_names.index(col), col, geom_arr)

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


def _validate_and_decode_metadata(metadata):
    if metadata is None or b"geo" not in metadata:
        raise ValueError(
            """Missing geo metadata in Parquet/Feather file.
            Use pandas.read_parquet/read_feather() instead."""
        )

    # check for malformed metadata
    try:
        decoded_geo_metadata = _decode_metadata(metadata.get(b"geo", b""))
    except (TypeError, json.decoder.JSONDecodeError):
        raise ValueError("Missing or malformed geo metadata in Parquet/Feather file")

    _validate_geo_metadata(decoded_geo_metadata)
    return decoded_geo_metadata


def _read_parquet_schema_and_metadata(path, filesystem):
    """
    Opening the Parquet file/dataset a first time to get the schema and metadata.

    TODO: we should look into how we can reuse opened dataset for reading the
    actual data, to avoid discovering the dataset twice (problem right now is
    that the ParquetDataset interface doesn't allow passing the filters on read)

    """
    import pyarrow
    from pyarrow import parquet

    kwargs = {}
    if Version(pyarrow.__version__) < Version("15.0.0"):
        kwargs = dict(use_legacy_dataset=False)

    try:
        schema = parquet.ParquetDataset(path, filesystem=filesystem, **kwargs).schema
    except Exception:
        schema = parquet.read_schema(path, filesystem=filesystem)

    metadata = schema.metadata

    # read metadata separately to get the raw Parquet FileMetaData metadata
    # (pyarrow doesn't properly exposes those in schema.metadata for files
    # created by GDAL - https://issues.apache.org/jira/browse/ARROW-16688)
    if metadata is None or b"geo" not in metadata:
        try:
            metadata = parquet.read_metadata(path, filesystem=filesystem).metadata
        except Exception:
            pass

    return schema, metadata


def _read_parquet(path, columns=None, storage_options=None, bbox=None, **kwargs):
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

    Supports versions 0.1.0, 0.4.0 and 1.0.0 of the GeoParquet
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
    bbox : tuple, optional
        Bounding box to be used to filter selection from geoparquet data. This
        is only usable if the data was saved with the bbox covering metadata.
        Input is of the tuple format (xmin, ymin, xmax, ymax).

    **kwargs
        Any additional kwargs passed to :func:`pyarrow.parquet.read_table`.

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
    import geopandas.io._pyarrow_hotfix  # noqa: F401

    # TODO(https://github.com/pandas-dev/pandas/pull/41194): see if pandas
    # adds filesystem as a keyword and match that.
    filesystem = kwargs.pop("filesystem", None)
    filesystem, path = _get_filesystem_path(
        path, filesystem=filesystem, storage_options=storage_options
    )
    path = _expand_user(path)
    schema, metadata = _read_parquet_schema_and_metadata(path, filesystem)

    geo_metadata = _validate_and_decode_metadata(metadata)

    bbox_filter = (
        _get_parquet_bbox_filter(geo_metadata, bbox) if bbox is not None else None
    )

    if_bbox_column_exists = _check_if_covering_in_geo_metadata(geo_metadata)

    # by default, bbox column is not read in, so must specify which
    # columns are read in if it exists.
    if not columns and if_bbox_column_exists:
        columns = _get_non_bbox_columns(schema, geo_metadata)

    # if both bbox and filters kwargs are used, must splice together.
    if "filters" in kwargs:
        filters_kwarg = kwargs.pop("filters")
        filters = _splice_bbox_and_filters(filters_kwarg, bbox_filter)
    else:
        filters = bbox_filter

    kwargs["use_pandas_metadata"] = True

    table = parquet.read_table(
        path, columns=columns, filesystem=filesystem, filters=filters, **kwargs
    )

    return _arrow_to_geopandas(table, geo_metadata)


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

    Supports versions 0.1.0, 0.4.0 and 1.0.0 of the GeoParquet
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

    import geopandas.io._pyarrow_hotfix  # noqa: F401

    if Version(pyarrow.__version__) < Version("0.17.0"):
        raise ImportError("pyarrow >= 0.17 required for Feather support")

    path = _expand_user(path)

    table = feather.read_table(path, columns=columns, **kwargs)
    return _arrow_to_geopandas(table)


def _get_parquet_bbox_filter(geo_metadata, bbox):
    primary_column = geo_metadata["primary_column"]

    if _check_if_covering_in_geo_metadata(geo_metadata):
        bbox_column_name = _get_bbox_encoding_column_name(geo_metadata)
        return _convert_bbox_to_parquet_filter(bbox, bbox_column_name)

    elif geo_metadata["columns"][primary_column]["encoding"] == "point":
        import pyarrow.compute as pc

        return (
            (pc.field((primary_column, "x")) >= bbox[0])
            & (pc.field((primary_column, "x")) <= bbox[2])
            & (pc.field((primary_column, "y")) >= bbox[1])
            & (pc.field((primary_column, "y")) <= bbox[3])
        )

    else:
        raise ValueError(
            "Specifying 'bbox' not supported for this Parquet file (it should either "
            "have a bbox covering column or use 'point' encoding)."
        )


def _convert_bbox_to_parquet_filter(bbox, bbox_column_name):
    import pyarrow.compute as pc

    return ~(
        (pc.field((bbox_column_name, "xmin")) > bbox[2])
        | (pc.field((bbox_column_name, "ymin")) > bbox[3])
        | (pc.field((bbox_column_name, "xmax")) < bbox[0])
        | (pc.field((bbox_column_name, "ymax")) < bbox[1])
    )


def _check_if_covering_in_geo_metadata(geo_metadata):
    primary_column = geo_metadata["primary_column"]
    return "covering" in geo_metadata["columns"][primary_column].keys()


def _get_bbox_encoding_column_name(geo_metadata):
    primary_column = geo_metadata["primary_column"]
    return geo_metadata["columns"][primary_column]["covering"]["bbox"]["xmin"][0]


def _get_non_bbox_columns(schema, geo_metadata):

    bbox_column_name = _get_bbox_encoding_column_name(geo_metadata)
    columns = schema.names
    if bbox_column_name in columns:
        columns.remove(bbox_column_name)
    return columns


def _splice_bbox_and_filters(kwarg_filters, bbox_filter):
    parquet = import_optional_dependency(
        "pyarrow.parquet", extra="pyarrow is required for Parquet support."
    )
    if bbox_filter is None:
        return kwarg_filters

    filters_expression = parquet.filters_to_expression(kwarg_filters)
    return bbox_filter & filters_expression
