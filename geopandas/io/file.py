from __future__ import annotations

import os
import urllib.request
import warnings
from http import HTTPStatus
from io import IOBase
from packaging.version import Version
from pathlib import Path

# Adapted from pandas.io.common
from urllib.parse import urlparse as parse_url
from urllib.parse import uses_netloc, uses_params, uses_relative
from urllib.request import Request

import numpy as np
import pandas as pd
from pandas.api.types import (
    is_datetime64_any_dtype,
    is_integer_dtype,
    is_object_dtype,
    is_string_dtype,
)

import shapely
from shapely.geometry import mapping
from shapely.geometry.base import BaseGeometry

from geopandas import GeoDataFrame, GeoSeries
from geopandas._compat import HAS_PYPROJ
from geopandas.io.util import vsi_path

_VALID_URLS = set(uses_relative + uses_netloc + uses_params)
_VALID_URLS.discard("")
# file:// URIs are supported by fiona/pyogrio -> don't already open + read the file here
_VALID_URLS.discard("file")

fiona = None
fiona_env = None
fiona_import_error = None
FIONA_GE_19 = False


def _import_fiona():
    global fiona
    global fiona_env
    global fiona_import_error
    global FIONA_GE_19

    if fiona is None:
        try:
            import fiona

            # only try to import fiona.Env if the main fiona import succeeded
            # (otherwise you can get confusing "AttributeError: module 'fiona'
            # has no attribute '_loading'" / partially initialized module errors)
            try:
                from fiona import Env as fiona_env
            except ImportError:
                try:
                    from fiona import drivers as fiona_env
                except ImportError:
                    fiona_env = None

            FIONA_GE_19 = Version(Version(fiona.__version__).base_version) >= Version(
                "1.9.0"
            )

        except ImportError as err:
            fiona = False
            fiona_import_error = str(err)


pyogrio = None
pyogrio_import_error = None


def _import_pyogrio():
    global pyogrio
    global pyogrio_import_error

    if pyogrio is None:
        try:
            import pyogrio

        except ImportError as err:
            pyogrio = False
            pyogrio_import_error = str(err)


def _check_fiona(func):
    if not fiona:
        raise ImportError(
            f"the {func} requires the 'fiona' package, but it is not installed or does "
            f"not import correctly.\nImporting fiona resulted in: {fiona_import_error}"
        )


def _check_pyogrio(func):
    if not pyogrio:
        raise ImportError(
            f"the {func} requires the 'pyogrio' package, but it is not installed "
            "or does not import correctly."
            "\nImporting pyogrio resulted in: {pyogrio_import_error}"
        )


def _check_metadata_supported(metadata: str | None, engine: str, driver: str) -> None:
    if metadata is None:
        return
    if driver != "GPKG":
        raise NotImplementedError(
            "The 'metadata' keyword is only supported for the GPKG driver."
        )

    if engine == "fiona" and not FIONA_GE_19:
        raise NotImplementedError(
            "The 'metadata' keyword is only supported for Fiona >= 1.9."
        )


def _check_engine(engine, func):
    # if not specified through keyword or option, then default to "pyogrio" if
    # installed, otherwise try fiona
    if engine is None:
        import geopandas

        engine = geopandas.options.io_engine

    if engine is None:
        _import_pyogrio()
        if pyogrio:
            engine = "pyogrio"
        else:
            _import_fiona()
            if fiona:
                engine = "fiona"

    if engine == "pyogrio":
        _import_pyogrio()
        _check_pyogrio(func)
    elif engine == "fiona":
        _import_fiona()
        _check_fiona(func)
    elif engine is None:
        raise ImportError(
            f"The {func} requires the 'pyogrio' or 'fiona' package, "
            "but neither is installed or imports correctly."
            f"\nImporting pyogrio resulted in: {pyogrio_import_error}"
            f"\nImporting fiona resulted in: {fiona_import_error}"
        )

    return engine


_EXTENSION_TO_DRIVER = {
    ".bna": "BNA",
    ".dxf": "DXF",
    ".csv": "CSV",
    ".shp": "ESRI Shapefile",
    ".dbf": "ESRI Shapefile",
    ".json": "GeoJSON",
    ".geojson": "GeoJSON",
    ".geojsonl": "GeoJSONSeq",
    ".geojsons": "GeoJSONSeq",
    ".gpkg": "GPKG",
    ".gml": "GML",
    ".xml": "GML",
    ".gpx": "GPX",
    ".gtm": "GPSTrackMaker",
    ".gtz": "GPSTrackMaker",
    ".tab": "MapInfo File",
    ".mif": "MapInfo File",
    ".mid": "MapInfo File",
    ".dgn": "DGN",
    ".fgb": "FlatGeobuf",
}


def _expand_user(path):
    """Expand paths that use ~."""
    if isinstance(path, str):
        path = os.path.expanduser(path)
    elif isinstance(path, Path):
        path = path.expanduser()
    return path


def _is_url(url):
    """Check to see if *url* has a valid protocol."""
    try:
        return parse_url(url).scheme in _VALID_URLS
    except Exception:
        return False


def _read_file(
    filename, bbox=None, mask=None, columns=None, rows=None, engine=None, **kwargs
):
    """Return a GeoDataFrame from a file or URL.

    Parameters
    ----------
    filename : str, path object or file-like object
        Either the absolute or relative path to the file or URL to
        be opened, or any object with a read() method (such as an open file
        or StringIO)
    bbox : tuple | GeoDataFrame or GeoSeries | shapely Geometry, default None
        Filter features by given bounding box, GeoSeries, GeoDataFrame or a shapely
        geometry. With engine="fiona", CRS mis-matches are resolved if given a GeoSeries
        or GeoDataFrame. With engine="pyogrio", bbox must be in the same CRS as the
        dataset. Tuple is (minx, miny, maxx, maxy) to match the bounds property of
        shapely geometry objects. Cannot be used with mask.
    mask : dict | GeoDataFrame or GeoSeries | shapely Geometry, default None
        Filter for features that intersect with the given dict-like geojson
        geometry, GeoSeries, GeoDataFrame or shapely geometry.
        CRS mis-matches are resolved if given a GeoSeries or GeoDataFrame.
        Cannot be used with bbox. If multiple geometries are passed, this will
        first union all geometries, which may be computationally expensive.
    columns : list, optional
        List of column names to import from the data source. Column names
        must exactly match the names in the data source. To avoid reading
        any columns (besides the geometry column), pass an empty list-like.
        By default reads all columns.
    rows : int or slice, default None
        Load in specific rows by passing an integer (first `n` rows) or a
        slice() object.
    engine : str,  "pyogrio" or "fiona"
        The underlying library that is used to read the file. Currently, the
        supported options are "pyogrio" and "fiona". Defaults to "pyogrio" if
        installed, otherwise tries "fiona". Engine can also be set globally
        with the ``geopandas.options.io_engine`` option.
    **kwargs :
        Keyword args to be passed to the engine, and can be used to write
        to multi-layer data, store data within archives (zip files), etc.
        In case of the "pyogrio" engine, the keyword arguments are passed to
        `pyogrio.read_dataframe`. In case of the "fiona" engine, the keyword
        arguments are passed to fiona.open`. For more information on possible
        keywords, type: ``import pyogrio; help(pyogrio.read_dataframe)``.


    Examples
    --------
    >>> df = geopandas.read_file("nybb.shp")  # doctest: +SKIP

    Specifying layer of GPKG:

    >>> df = geopandas.read_file("file.gpkg", layer='cities')  # doctest: +SKIP

    Reading only first 10 rows:

    >>> df = geopandas.read_file("nybb.shp", rows=10)  # doctest: +SKIP

    Reading only geometries intersecting ``mask``:

    >>> df = geopandas.read_file("nybb.shp", mask=polygon)  # doctest: +SKIP

    Reading only geometries intersecting ``bbox``:

    >>> df = geopandas.read_file("nybb.shp", bbox=(0, 0, 10, 20))  # doctest: +SKIP

    Returns
    -------
    :obj:`geopandas.GeoDataFrame` or :obj:`pandas.DataFrame` :
        If `ignore_geometry=True` a :obj:`pandas.DataFrame` will be returned.

    Notes
    -----
    The format drivers will attempt to detect the encoding of your data, but
    may fail. In this case, the proper encoding can be specified explicitly
    by using the encoding keyword parameter, e.g. ``encoding='utf-8'``.

    For faster data reading with the default pyogrio engine when
    pyarrow is installed, pass ``use_arrow=True`` as an argument. See the User
    Guide page :doc:`../../user_guide/io` for details.


    When specifying a URL, geopandas will check if the server supports reading
    partial data and in that case pass the URL as is to the underlying engine,
    which will then use the network file system handler of GDAL to read from
    the URL. Otherwise geopandas will download the data from the URL and pass
    all data in-memory to the underlying engine.
    If you need more control over how the URL is read, you can specify the
    GDAL virtual filesystem manually (e.g. ``/vsicurl/https://...``). See the
    GDAL documentation on filesystems for more details
    (https://gdal.org/user/virtual_file_systems.html#vsicurl-http-https-ftp-files-random-access).

    """
    engine = _check_engine(engine, "'read_file' function")

    filename = _expand_user(filename)

    from_bytes = False
    if _is_url(filename):
        # if it is a url that supports random access -> pass through to
        # pyogrio/fiona as is (to support downloading only part of the file)
        # otherwise still download manually because pyogrio/fiona don't support
        # all types of urls (https://github.com/geopandas/geopandas/issues/2908)
        try:
            with urllib.request.urlopen(
                Request(filename, headers={"Range": "bytes=0-1"})
            ) as response:
                if (
                    response.headers.get("Accept-Ranges") == "none"
                    or response.status != HTTPStatus.PARTIAL_CONTENT
                ):
                    from_bytes = True
        except ConnectionError:
            from_bytes = True

        if from_bytes:
            with urllib.request.urlopen(filename) as response:
                filename = response.read()

    if engine == "pyogrio":
        return _read_file_pyogrio(
            filename, bbox=bbox, mask=mask, columns=columns, rows=rows, **kwargs
        )

    elif engine == "fiona":
        if pd.api.types.is_file_like(filename):
            data = filename.read()
            path_or_bytes = data.encode("utf-8") if isinstance(data, str) else data
            from_bytes = True
        else:
            path_or_bytes = filename

        return _read_file_fiona(
            path_or_bytes,
            from_bytes,
            bbox=bbox,
            mask=mask,
            columns=columns,
            rows=rows,
            **kwargs,
        )

    else:
        raise ValueError(f"unknown engine '{engine}'")


def _read_file_fiona(
    path_or_bytes,
    from_bytes,
    bbox=None,
    mask=None,
    columns=None,
    rows=None,
    where=None,
    **kwargs,
):
    if where is not None and not FIONA_GE_19:
        raise NotImplementedError("where requires fiona 1.9+")

    if columns is not None:
        if "include_fields" in kwargs:
            raise ValueError(
                "Cannot specify both 'include_fields' and 'columns' keywords"
            )
        if not FIONA_GE_19:
            raise NotImplementedError("'columns' keyword requires fiona 1.9+")
        kwargs["include_fields"] = columns
    elif "include_fields" in kwargs:
        # alias to columns, as this variable is used below to specify column order
        # in the dataframe creation
        columns = kwargs["include_fields"]

    if not from_bytes:
        # Opening a file via URL or file-like-object above automatically detects a
        # zipped file. In order to match that behavior, attempt to add a zip scheme
        # if missing.
        path_or_bytes = vsi_path(str(path_or_bytes))

    if from_bytes:
        reader = fiona.BytesCollection
    else:
        reader = fiona.open

    with fiona_env():
        with reader(path_or_bytes, **kwargs) as features:
            crs = features.crs_wkt  # returns "" if empty
            crs = crs or None
            # attempt to get EPSG code
            try:
                # fiona 1.9+
                epsg = features.crs.to_epsg(confidence_threshold=100)
                if epsg is not None:
                    crs = epsg
            except AttributeError:
                # fiona <= 1.8
                try:
                    crs = features.crs["init"]
                except (TypeError, KeyError):
                    pass

            # handle loading the bounding box
            if bbox is not None:
                if isinstance(bbox, GeoDataFrame | GeoSeries):
                    bbox = tuple(bbox.to_crs(crs).total_bounds)
                elif isinstance(bbox, BaseGeometry):
                    bbox = bbox.bounds
                assert len(bbox) == 4
            # handle loading the mask
            elif isinstance(mask, GeoDataFrame | GeoSeries):
                if crs is not None and mask.crs is not None:
                    mask = mask.to_crs(crs)
                else:
                    _warn_missing_crs_of_dataframe_and_mask(crs, mask)
                mask = mapping(mask.union_all())
            elif isinstance(mask, BaseGeometry):
                mask = mapping(mask)

            filters = {}
            if bbox is not None:
                filters["bbox"] = bbox
            if mask is not None:
                filters["mask"] = mask
            if where is not None:
                filters["where"] = where

            # setup the data loading filter
            if rows is not None:
                if isinstance(rows, int):
                    rows = slice(rows)
                elif not isinstance(rows, slice):
                    raise TypeError("'rows' must be an integer or a slice.")
                f_filt = features.filter(rows.start, rows.stop, rows.step, **filters)
            elif filters:
                f_filt = features.filter(**filters)
            else:
                f_filt = features
            # get list of columns
            columns = columns or list(features.schema["properties"])
            datetime_fields = [
                k for (k, v) in features.schema["properties"].items() if v == "datetime"
            ]
            if (
                kwargs.get("ignore_geometry", False)
                or features.schema["geometry"] == "None"
            ):
                df = pd.DataFrame(
                    [record["properties"] for record in f_filt], columns=columns
                )
            else:
                df = GeoDataFrame.from_features(
                    f_filt, crs=crs, columns=columns + ["geometry"]
                )
            for k in datetime_fields:
                as_dt = None
                # plain try catch for when pandas will raise in the future
                # TODO we can tighten the exception type in future when it does
                try:
                    with warnings.catch_warnings():
                        # pandas 2.x does not yet enforce this behaviour but raises a
                        # warning  -> we want to to suppress this warning for our users,
                        # and do this by turning it into an error so we take the
                        # `except` code path to try again with utc=True
                        warnings.filterwarnings(
                            "error",
                            "In a future version of pandas, parsing datetimes with "
                            "mixed time zones will raise an error",
                            FutureWarning,
                        )
                        as_dt = pd.to_datetime(df[k])
                except Exception:
                    pass
                if as_dt is None or as_dt.dtype == "object":
                    # if to_datetime failed, try again for mixed timezone offsets
                    # This can still fail if there are invalid datetimes
                    try:
                        as_dt = pd.to_datetime(df[k], utc=True)
                    except Exception:
                        pass
                # if to_datetime succeeded, round datetimes as
                # fiona only supports up to ms precision (any microseconds are
                # floating point rounding error)
                if as_dt is not None and not (as_dt.dtype == "object"):
                    df[k] = as_dt.dt.as_unit("ms")
            return df


def _read_file_pyogrio(path_or_bytes, bbox=None, mask=None, rows=None, **kwargs):
    import pyogrio

    if rows is not None:
        if isinstance(rows, int):
            kwargs["max_features"] = rows
        elif isinstance(rows, slice):
            if rows.start is not None:
                if rows.start < 0:
                    raise ValueError(
                        "Negative slice start not supported with the 'pyogrio' engine."
                    )
                kwargs["skip_features"] = rows.start
            if rows.stop is not None:
                kwargs["max_features"] = rows.stop - (rows.start or 0)
            if rows.step is not None:
                raise ValueError("slice with step is not supported")
        else:
            raise TypeError("'rows' must be an integer or a slice.")

    if bbox is not None and mask is not None:
        # match error message from Fiona
        raise ValueError("mask and bbox can not be set together")

    if bbox is not None:
        if isinstance(bbox, GeoDataFrame | GeoSeries):
            crs = pyogrio.read_info(path_or_bytes, layer=kwargs.get("layer")).get("crs")
            if isinstance(path_or_bytes, IOBase):
                path_or_bytes.seek(0)

            bbox = tuple(bbox.to_crs(crs).total_bounds)
        elif isinstance(bbox, BaseGeometry):
            bbox = bbox.bounds
        if len(bbox) != 4:
            raise ValueError("'bbox' should be a length-4 tuple.")

    if mask is not None:
        # NOTE: mask cannot be used at same time as bbox keyword
        if isinstance(mask, GeoDataFrame | GeoSeries):
            crs = pyogrio.read_info(path_or_bytes, layer=kwargs.get("layer")).get("crs")
            if crs is not None and mask.crs is not None:
                mask = mask.to_crs(crs)
            else:
                _warn_missing_crs_of_dataframe_and_mask(crs, mask)

            if isinstance(path_or_bytes, IOBase):
                path_or_bytes.seek(0)

            mask = shapely.unary_union(mask.geometry.values)
        elif isinstance(mask, BaseGeometry):
            mask = shapely.unary_union(mask)
        elif isinstance(mask, dict) or hasattr(mask, "__geo_interface__"):
            # convert GeoJSON to shapely geometry
            mask = shapely.geometry.shape(mask)

        kwargs["mask"] = mask

    if kwargs.pop("ignore_geometry", False):
        kwargs["read_geometry"] = False

    # translate `ignore_fields`/`include_fields` keyword for back compat with fiona
    if "ignore_fields" in kwargs and "include_fields" in kwargs:
        raise ValueError("Cannot specify both 'ignore_fields' and 'include_fields'")
    elif "ignore_fields" in kwargs:
        if kwargs.get("columns", None) is not None:
            raise ValueError(
                "Cannot specify both 'columns' and 'ignore_fields' keywords"
            )
        warnings.warn(
            "The 'include_fields' and 'ignore_fields' keywords are deprecated, and "
            "will be removed in a future release. You can use the 'columns' keyword "
            "instead to select which columns to read.",
            DeprecationWarning,
            stacklevel=3,
        )
        ignore_fields = kwargs.pop("ignore_fields")
        fields = pyogrio.read_info(path_or_bytes, layer=kwargs.get("layer"))["fields"]
        include_fields = [col for col in fields if col not in ignore_fields]
        kwargs["columns"] = include_fields
    elif "include_fields" in kwargs:
        # translate `include_fields` keyword for back compat with fiona engine
        if kwargs.get("columns", None) is not None:
            raise ValueError(
                "Cannot specify both 'columns' and 'include_fields' keywords"
            )
        warnings.warn(
            "The 'include_fields' and 'ignore_fields' keywords are deprecated, and "
            "will be removed in a future release. You can use the 'columns' keyword "
            "instead to select which columns to read.",
            DeprecationWarning,
            stacklevel=3,
        )
        kwargs["columns"] = kwargs.pop("include_fields")

    return pyogrio.read_dataframe(path_or_bytes, bbox=bbox, **kwargs)


def _warn_missing_crs_of_dataframe_and_mask(source_dataset_crs, mask):
    """
    Warn if one, or both, of the source dataset or mask does not
    have a crs.
    """
    if (source_dataset_crs is None) and (mask.crs is None):
        msg = "There is no CRS defined in the source dataset nor mask. "
    elif (source_dataset_crs is None) and (mask.crs is not None):
        msg = "There is no CRS defined in the source dataset. "
    else:  # crs not None and mask.crs is None
        msg = "There is no CRS defined in the mask. "
    msg += (
        "This may lead to a misalignment of the mask and the "
        "source dataset, leading to incorrect masking. Ensure "
        "both inputs share the same CRS."
    )
    warnings.warn(msg, UserWarning, stacklevel=3)


def _detect_driver(path):
    """Attempt to auto-detect driver based on the extension."""
    try:
        # in case the path is a file handle
        path = path.name
    except AttributeError:
        pass
    try:
        return _EXTENSION_TO_DRIVER[Path(path).suffix.lower()]
    except KeyError:
        # Assume it is a shapefile folder for now. In the future,
        # will likely raise an exception when the expected
        # folder writing behavior is more clearly defined.
        return "ESRI Shapefile"


def _to_file(
    df,
    filename,
    driver=None,
    schema=None,
    index=None,
    mode="w",
    crs=None,
    engine=None,
    metadata=None,
    **kwargs,
):
    """Write this GeoDataFrame to an OGR data source.

    A dictionary of supported OGR providers is available via:

    >>> import pyogrio
    >>> pyogrio.list_drivers()  # doctest: +SKIP

    Parameters
    ----------
    df : GeoDataFrame to be written
    filename : string
        File path or file handle to write to. The path may specify a
        GDAL VSI scheme.
    driver : string, default None
        The OGR format driver used to write the vector file.
        If not specified, it attempts to infer it from the file extension.
        If no extension is specified, it saves ESRI Shapefile to a folder.
    schema : dict, default None
        If specified, the schema dictionary is passed to Fiona to
        better control how the file is written. If None, GeoPandas
        will determine the schema based on each column's dtype.
        Not supported for the "pyogrio" engine.
    index : bool, default None
        If True, write index into one or more columns (for MultiIndex).
        Default None writes the index into one or more columns only if
        the index is named, is a MultiIndex, or has a non-integer data
        type. If False, no index is written.

        .. versionadded:: 0.7
            Previously the index was not written.
    mode : string, default 'w'
        The write mode, 'w' to overwrite the existing file and 'a' to append;
        when using the pyogrio engine, you can also pass ``append=True``.
        Not all drivers support appending. For the fiona engine, the drivers
        that support appending are listed in fiona.supported_drivers or
        https://github.com/Toblerity/Fiona/blob/master/fiona/drvsupport.py.
        For the pyogrio engine, you should be able to use any driver that
        is available in your installation of GDAL that supports append
        capability; see the specific driver entry at
        https://gdal.org/drivers/vector/index.html for more information.
    crs : pyproj.CRS, default None
        If specified, the CRS is passed to Fiona to
        better control how the file is written. If None, GeoPandas
        will determine the crs based on crs df attribute.
        The value can be anything accepted
        by :meth:`pyproj.CRS.from_user_input() <pyproj.crs.CRS.from_user_input>`,
        such as an authority string (eg "EPSG:4326") or a WKT string.
    engine : str,  "pyogrio" or "fiona"
        The underlying library that is used to read the file. Currently, the
        supported options are "pyogrio" and "fiona". Defaults to "pyogrio" if
        installed, otherwise tries "fiona". Engine can also be set globally
        with the ``geopandas.options.io_engine`` option.
    metadata : dict[str, str], default None
        Optional metadata to be stored in the file. Keys and values must be
        strings. Only supported for the "GPKG" driver
        (requires Fiona >= 1.9 or pyogrio >= 0.6).
    **kwargs :
        Keyword args to be passed to the engine, and can be used to write
        to multi-layer data, store data within archives (zip files), etc.
        In case of the "fiona" engine, the keyword arguments are passed to
        fiona.open`. For more information on possible keywords, type:
        ``import fiona; help(fiona.open)``. In case of the "pyogrio" engine,
        the keyword arguments are passed to `pyogrio.write_dataframe`.

    Notes
    -----
    The format drivers will attempt to detect the encoding of your data, but
    may fail. In this case, the proper encoding can be specified explicitly
    by using the encoding keyword parameter, e.g. ``encoding='utf-8'``.
    """
    engine = _check_engine(engine, "'to_file' method")

    filename = _expand_user(filename)

    if index is None:
        # Determine if index attribute(s) should be saved to file
        # (only if they are named or are non-integer)
        index = list(df.index.names) != [None] or not is_integer_dtype(df.index.dtype)
    if index:
        df = df.reset_index(drop=False)

    if driver is None:
        driver = _detect_driver(filename)

    if driver == "ESRI Shapefile" and any(len(c) > 10 for c in df.columns.tolist()):
        warnings.warn(
            "Column names longer than 10 characters will be truncated when saved to "
            "ESRI Shapefile.",
            stacklevel=3,
        )

    if (df.dtypes == "geometry").sum() > 1:
        raise ValueError(
            "GeoDataFrame contains multiple geometry columns but GeoDataFrame.to_file "
            "supports only a single geometry column. Use a GeoDataFrame.to_parquet or "
            "GeoDataFrame.to_feather, drop additional geometry columns or convert them "
            "to a supported format like a well-known text (WKT) using "
            "`GeoSeries.to_wkt()`.",
        )
    _check_metadata_supported(metadata, engine, driver)

    if mode not in ("w", "a"):
        raise ValueError(f"'mode' should be one of 'w' or 'a', got '{mode}' instead")

    if engine == "pyogrio":
        _to_file_pyogrio(df, filename, driver, schema, crs, mode, metadata, **kwargs)
    elif engine == "fiona":
        _to_file_fiona(df, filename, driver, schema, crs, mode, metadata, **kwargs)
    else:
        raise ValueError(f"unknown engine '{engine}'")


def _to_file_fiona(df, filename, driver, schema, crs, mode, metadata, **kwargs):
    if not HAS_PYPROJ and crs:
        raise ImportError(
            "The 'pyproj' package is required to write a file with a CRS, but it is not"
            " installed or does not import correctly."
        )

    if schema is None:
        schema = infer_schema(df)

    if crs:
        from pyproj import CRS

        crs = CRS.from_user_input(crs)
    else:
        crs = df.crs

    with fiona_env():
        crs_wkt = None
        try:
            gdal_version = Version(
                fiona.env.get_gdal_release_name().strip("e")
            )  # GH3147
        except (AttributeError, ValueError):
            gdal_version = Version("2.0.0")  # just assume it is not the latest
        if gdal_version >= Version("3.0.0") and crs:
            crs_wkt = crs.to_wkt()
        elif crs:
            crs_wkt = crs.to_wkt("WKT1_GDAL")
        with fiona.open(
            filename, mode=mode, driver=driver, crs_wkt=crs_wkt, schema=schema, **kwargs
        ) as colxn:
            if metadata is not None:
                colxn.update_tags(metadata)
            colxn.writerecords(df.iterfeatures())


def _to_file_pyogrio(df, filename, driver, schema, crs, mode, metadata, **kwargs):
    import pyogrio

    if schema is not None:
        raise ValueError(
            "The 'schema' argument is not supported with the 'pyogrio' engine."
        )

    if mode == "a":
        kwargs["append"] = True

    if crs is not None:
        raise ValueError("Passing 'crs' is not supported with the 'pyogrio' engine.")

    # for the fiona engine, this check is done in gdf.iterfeatures()
    if not df.columns.is_unique:
        raise ValueError("GeoDataFrame cannot contain duplicated column names.")

    pyogrio.write_dataframe(df, filename, driver=driver, metadata=metadata, **kwargs)


def infer_schema(df):
    from collections import OrderedDict

    # TODO: test pandas string type and boolean type once released
    types = {
        "Int32": "int32",
        "int32": "int32",
        "Int64": "int",
        "string": "str",
        "boolean": "bool",
    }

    def convert_type(column, in_type):
        if is_object_dtype(in_type) or is_string_dtype(in_type):
            return "str"
        if is_datetime64_any_dtype(in_type):
            # numpy datetime type regardless of frequency
            return "datetime"
        if str(in_type) in types:
            out_type = types[str(in_type)]
        else:
            out_type = type(np.zeros(1, in_type).item()).__name__
        if out_type == "long":
            out_type = "int"
        return out_type

    properties = OrderedDict(
        [
            (col, convert_type(col, _type))
            for col, _type in zip(df.columns, df.dtypes)
            if col != df._geometry_column_name
        ]
    )

    if df.empty:
        warnings.warn(
            "You are attempting to write an empty DataFrame to file. "
            "For some drivers, this operation may fail.",
            UserWarning,
            stacklevel=3,
        )

    # Since https://github.com/Toblerity/Fiona/issues/446 resolution,
    # Fiona allows a list of geometry types
    geom_types = _geometry_types(df)

    schema = {"geometry": geom_types, "properties": properties}

    return schema


def _geometry_types(df):
    """Determine the geometry types in the GeoDataFrame for the schema."""
    geom_types_2D = df[~df.geometry.has_z].geometry.geom_type.unique()
    geom_types_2D = list(geom_types_2D[pd.notna(geom_types_2D)])
    geom_types_3D = df[df.geometry.has_z].geometry.geom_type.unique()
    geom_types_3D = list(geom_types_3D[pd.notna(geom_types_3D)])
    geom_types_3D = ["3D " + gtype for gtype in geom_types_3D]
    geom_types = geom_types_3D + geom_types_2D

    if len(geom_types) == 0:
        # Default geometry type supported by Fiona
        # (Since https://github.com/Toblerity/Fiona/issues/446 resolution)
        return "Unknown"

    if len(geom_types) == 1:
        geom_types = geom_types[0]

    return geom_types


def _list_layers(filename) -> pd.DataFrame:
    """List layers available in a file.

    Provides an overview of layers available in a file or URL together with their
    geometry types. When supported by the data source, this includes both spatial and
    non-spatial layers. Non-spatial layers are indicated by the ``"geometry_type"``
    column being ``None``. GeoPandas will not read such layers but they can be read into
    a pd.DataFrame using :func:`pyogrio.read_dataframe`.

    Parameters
    ----------
    filename : str, path object or file-like object
        Either the absolute or relative path to the file or URL to
        be opened, or any object with a read() method (such as an open file
        or StringIO)

    Returns
    -------
    pandas.DataFrame
        A DataFrame with columns "name" and "geometry_type" and one row per layer.
    """
    _import_pyogrio()
    _check_pyogrio("list_layers")

    import pyogrio

    return pd.DataFrame(
        pyogrio.list_layers(filename), columns=["name", "geometry_type"]
    )
