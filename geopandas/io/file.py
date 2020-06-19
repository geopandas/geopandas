from distutils.version import LooseVersion

import io
import numpy as np
import pandas as pd

import fiona
import pyproj
from shapely.geometry import mapping
from shapely.geometry.base import BaseGeometry

try:
    from fiona import Env as fiona_env
except ImportError:
    from fiona import drivers as fiona_env

from geopandas import GeoDataFrame, GeoSeries


# Adapted from pandas.io.common
from urllib.request import urlopen as _urlopen
from urllib.parse import urlparse as parse_url
from urllib.parse import uses_netloc, uses_params, uses_relative


_FIONA18 = LooseVersion(fiona.__version__) >= LooseVersion("1.8")
_VALID_URLS = set(uses_relative + uses_netloc + uses_params)
_VALID_URLS.discard("")


def _is_url(url):
    """Check to see if *url* has a valid protocol."""
    try:
        return parse_url(url).scheme in _VALID_URLS
    except Exception:
        return False


def _read_file(filename, bbox=None, mask=None, rows=None, **kwargs):
    """
    Returns a GeoDataFrame from a file or URL.

    .. versionadded:: 0.7.0 mask, rows

    Parameters
    ----------
    filename : str, path object or file-like object
        Either the absolute or relative path to the file or URL to
        be opened, or any object with a read() method (such as an open file
        or StringIO)
    bbox : tuple | GeoDataFrame or GeoSeries | shapely Geometry, default None
        Filter features by given bounding box, GeoSeries, GeoDataFrame or a
        shapely geometry. CRS mis-matches are resolved if given a GeoSeries
        or GeoDataFrame. Cannot be used with mask.
    mask : dict | GeoDataFrame or GeoSeries | shapely Geometry, default None
        Filter for features that intersect with the given dict-like geojson
        geometry, GeoSeries, GeoDataFrame or shapely geometry.
        CRS mis-matches are resolved if given a GeoSeries or GeoDataFrame.
        Cannot be used with bbox.
    rows : int or slice, default None
        Load in specific rows by passing an integer (first `n` rows) or a
        slice() object.
    **kwargs :
        Keyword args to be passed to the `open` or `BytesCollection` method
        in the fiona library when opening the file. For more information on
        possible keywords, type:
        ``import fiona; help(fiona.open)``

    Examples
    --------
    >>> df = geopandas.read_file("nybb.shp")

    Returns
    -------
    :obj:`geopandas.GeoDataFrame` or :obj:`pandas.DataFrame` :
        If `ignore_geometry=True` a :obj:`pandas.DataFrame` will be returned.

    Notes
    -----
    The format drivers will attempt to detect the encoding of your data, but
    may fail. In this case, the proper encoding can be specified explicitly
    by using the encoding keyword parameter, e.g. ``encoding='utf-8'``.
    """
    if _is_url(filename):
        req = _urlopen(filename)
        path_or_bytes = req.read()
        reader = fiona.BytesCollection
    elif isinstance(filename, io.TextIOBase):
        path_or_bytes = filename.read()
        reader = fiona.open
    else:
        path_or_bytes = filename
        reader = fiona.open

    with fiona_env():
        with reader(path_or_bytes, **kwargs) as features:

            # In a future Fiona release the crs attribute of features will
            # no longer be a dict, but will behave like a dict. So this should
            # be forwards compatible
            crs = (
                features.crs["init"]
                if features.crs and "init" in features.crs
                else features.crs_wkt
            )

            # handle loading the bounding box
            if bbox is not None:
                if isinstance(bbox, (GeoDataFrame, GeoSeries)):
                    bbox = tuple(bbox.to_crs(crs).total_bounds)
                elif isinstance(bbox, BaseGeometry):
                    bbox = bbox.bounds
                assert len(bbox) == 4
            # handle loading the mask
            elif isinstance(mask, (GeoDataFrame, GeoSeries)):
                mask = mapping(mask.to_crs(crs).unary_union)
            elif isinstance(mask, BaseGeometry):
                mask = mapping(mask)
            # setup the data loading filter
            if rows is not None:
                if isinstance(rows, int):
                    rows = slice(rows)
                elif not isinstance(rows, slice):
                    raise TypeError("'rows' must be an integer or a slice.")
                f_filt = features.filter(
                    rows.start, rows.stop, rows.step, bbox=bbox, mask=mask
                )
            elif any((bbox, mask)):
                f_filt = features.filter(bbox=bbox, mask=mask)
            else:
                f_filt = features
            # get list of columns
            columns = list(features.schema["properties"])
            if kwargs.get("ignore_geometry", False):
                return pd.DataFrame(
                    [record["properties"] for record in f_filt], columns=columns
                )

            return GeoDataFrame.from_features(
                f_filt, crs=crs, columns=columns + ["geometry"]
            )


def read_file(*args, **kwargs):
    import warnings

    warnings.warn(
        "geopandas.io.file.read_file() is intended for internal "
        "use only, and will be deprecated. Use geopandas.read_file() instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    return _read_file(*args, **kwargs)


def to_file(*args, **kwargs):
    import warnings

    warnings.warn(
        "geopandas.io.file.to_file() is intended for internal "
        "use only, and will be deprecated. Use GeoDataFrame.to_file() "
        "or GeoSeries.to_file() instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    return _to_file(*args, **kwargs)


def _to_file(
    df,
    filename,
    driver="ESRI Shapefile",
    schema=None,
    index=None,
    mode="w",
    crs=None,
    **kwargs
):
    """
    Write this GeoDataFrame to an OGR data source

    A dictionary of supported OGR providers is available via:
    >>> import fiona
    >>> fiona.supported_drivers

    Parameters
    ----------
    df : GeoDataFrame to be written
    filename : string
        File path or file handle to write to.
    driver : string, default 'ESRI Shapefile'
        The OGR format driver used to write the vector file.
    schema : dict, default None
        If specified, the schema dictionary is passed to Fiona to
        better control how the file is written. If None, GeoPandas
        will determine the schema based on each column's dtype
    index : bool, default None
        If True, write index into one or more columns (for MultiIndex).
        Default None writes the index into one or more columns only if
        the index is named, is a MultiIndex, or has a non-integer data
        type. If False, no index is written.

        .. versionadded:: 0.7
            Previously the index was not written.
    mode : string, default 'w'
        The write mode, 'w' to overwrite the existing file and 'a' to append.
        Not all drivers support appending. The drivers that support appending
        are listed in fiona.supported_drivers or
        https://github.com/Toblerity/Fiona/blob/master/fiona/drvsupport.py
    crs : pyproj.CRS, default None
        If specified, the CRS is passed to Fiona to
        better control how the file is written. If None, GeoPandas
        will determine the crs based on crs df attribute.
        The value can be anything accepted
        by :meth:`pyproj.CRS.from_user_input() <pyproj.crs.CRS.from_user_input>`,
        such as an authority string (eg "EPSG:4326") or a WKT string.

    The *kwargs* are passed to fiona.open and can be used to write
    to multi-layer data, store data within archives (zip files), etc.
    The path may specify a fiona VSI scheme.

    Notes
    -----
    The format drivers will attempt to detect the encoding of your data, but
    may fail. In this case, the proper encoding can be specified explicitly
    by using the encoding keyword parameter, e.g. ``encoding='utf-8'``.
    """
    if index is None:
        # Determine if index attribute(s) should be saved to file
        index = list(df.index.names) != [None] or type(df.index) not in (
            pd.RangeIndex,
            pd.Int64Index,
        )
    if index:
        df = df.reset_index(drop=False)
    if schema is None:
        schema = infer_schema(df)
    if crs:
        crs = pyproj.CRS.from_user_input(crs)
    else:
        crs = df.crs
    with fiona_env():
        crs_wkt = None
        try:
            gdal_version = fiona.env.get_gdal_release_name()
        except AttributeError:
            gdal_version = "2.0.0"  # just assume it is not the latest
        if LooseVersion(gdal_version) >= LooseVersion("3.0.0") and crs:
            crs_wkt = crs.to_wkt()
        elif crs:
            crs_wkt = crs.to_wkt("WKT1_GDAL")
        with fiona.open(
            filename, mode=mode, driver=driver, crs_wkt=crs_wkt, schema=schema, **kwargs
        ) as colxn:
            colxn.writerecords(df.iterfeatures())


def infer_schema(df):
    from collections import OrderedDict

    # TODO: test pandas string type and boolean type once released
    types = {"Int64": "int", "string": "str", "boolean": "bool"}

    def convert_type(column, in_type):
        if in_type == object:
            return "str"
        if in_type.name.startswith("datetime64"):
            # numpy datetime type regardless of frequency
            return "datetime"
        if str(in_type) in types:
            out_type = types[str(in_type)]
        else:
            out_type = type(np.zeros(1, in_type).item()).__name__
        if out_type == "long":
            out_type = "int"
        if not _FIONA18 and out_type == "bool":
            raise ValueError(
                'column "{}" is boolean type, '.format(column)
                + "which is unsupported in file writing with fiona "
                "< 1.8. Consider casting the column to int type."
            )
        return out_type

    properties = OrderedDict(
        [
            (col, convert_type(col, _type))
            for col, _type in zip(df.columns, df.dtypes)
            if col != df._geometry_column_name
        ]
    )

    if df.empty:
        raise ValueError("Cannot write empty DataFrame to file.")

    # Since https://github.com/Toblerity/Fiona/issues/446 resolution,
    # Fiona allows a list of geometry types
    geom_types = _geometry_types(df)

    schema = {"geometry": geom_types, "properties": properties}

    return schema


def _geometry_types(df):
    """
    Determine the geometry types in the GeoDataFrame for the schema.
    """
    if _FIONA18:
        # Starting from Fiona 1.8, schema submitted to fiona to write a gdf
        # can have mixed geometries:
        # - 3D and 2D shapes can coexist in inferred schema
        # - Shape and MultiShape types can (and must) coexist in inferred
        #   schema
        geom_types_2D = df[~df.geometry.has_z].geometry.geom_type.unique()
        geom_types_2D = [gtype for gtype in geom_types_2D if gtype is not None]
        geom_types_3D = df[df.geometry.has_z].geometry.geom_type.unique()
        geom_types_3D = ["3D " + gtype for gtype in geom_types_3D if gtype is not None]
        geom_types = geom_types_3D + geom_types_2D

    else:
        # Before Fiona 1.8, schema submitted to write a gdf should have
        # one single geometry type whenever possible:
        # - 3D and 2D shapes cannot coexist in inferred schema
        # - Shape and MultiShape can not coexist in inferred schema
        geom_types = _geometry_types_back_compat(df)

    if len(geom_types) == 0:
        # Default geometry type supported by Fiona
        # (Since https://github.com/Toblerity/Fiona/issues/446 resolution)
        return "Unknown"

    if len(geom_types) == 1:
        geom_types = geom_types[0]

    return geom_types


def _geometry_types_back_compat(df):
    """
    for backward compatibility with Fiona<1.8 only
    """
    unique_geom_types = df.geometry.geom_type.unique()
    unique_geom_types = [gtype for gtype in unique_geom_types if gtype is not None]

    # merge single and Multi types (eg Polygon and MultiPolygon)
    unique_geom_types = [
        gtype
        for gtype in unique_geom_types
        if not gtype.startswith("Multi") or gtype[5:] not in unique_geom_types
    ]

    if df.geometry.has_z.any():
        # declare all geometries as 3D geometries
        unique_geom_types = ["3D " + type for type in unique_geom_types]
    # by default, all geometries are 2D geometries

    return unique_geom_types
