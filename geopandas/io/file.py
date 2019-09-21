from distutils.version import LooseVersion

import numpy as np
import six

import fiona

from geopandas import GeoDataFrame, GeoSeries

try:
    from fiona import Env as fiona_env
except ImportError:
    from fiona import drivers as fiona_env


_FIONA18 = LooseVersion(fiona.__version__) >= LooseVersion("1.8")


# Adapted from pandas.io.common
if six.PY3:
    from urllib.request import urlopen as _urlopen
    from urllib.parse import urlparse as parse_url
    from urllib.parse import uses_relative, uses_netloc, uses_params
else:
    from urllib2 import urlopen as _urlopen
    from urlparse import urlparse as parse_url
    from urlparse import uses_relative, uses_netloc, uses_params

_VALID_URLS = set(uses_relative + uses_netloc + uses_params)
_VALID_URLS.discard("")


def _is_url(url):
    """Check to see if *url* has a valid protocol."""
    try:
        return parse_url(url).scheme in _VALID_URLS
    except Exception:
        return False


def read_file(filename, bbox=None, **kwargs):
    """
    Returns a GeoDataFrame from a file or URL.

    Parameters
    ----------
    filename: str
        Either the absolute or relative path to the file or URL to
        be opened.
    bbox : tuple | GeoDataFrame or GeoSeries, default None
        Filter features by given bounding box, GeoSeries, or GeoDataFrame.
        CRS mis-matches are resolved if given a GeoSeries or GeoDataFrame.
    **kwargs:
        Keyword args to be passed to the `open` or `BytesCollection` method
        in the fiona library when opening the file. For more information on
        possible keywords, type:
        ``import fiona; help(fiona.open)``

    Examples
    --------
    >>> df = geopandas.read_file("nybb.shp")

    Returns
    -------
    geodataframe : GeoDataFrame
    """
    if _is_url(filename):
        req = _urlopen(filename)
        path_or_bytes = req.read()
        reader = fiona.BytesCollection
    else:
        path_or_bytes = filename
        reader = fiona.open

    with fiona_env():
        with reader(path_or_bytes, **kwargs) as features:

            # In a future Fiona release the crs attribute of features will
            # no longer be a dict. The following code will be both forward
            # and backward compatible.
            if hasattr(features.crs, "to_dict"):
                crs = features.crs.to_dict()
            else:
                crs = features.crs

            if bbox is not None:
                if isinstance(bbox, GeoDataFrame) or isinstance(bbox, GeoSeries):
                    bbox = tuple(bbox.to_crs(crs).total_bounds)
                assert len(bbox) == 4
                f_filt = features.filter(bbox=bbox)
            else:
                f_filt = features

            columns = list(features.meta["schema"]["properties"]) + ["geometry"]
            gdf = GeoDataFrame.from_features(f_filt, crs=crs, columns=columns)

    return gdf


def to_file(df, filename, driver="ESRI Shapefile", schema=None, **kwargs):
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

    The *kwargs* are passed to fiona.open and can be used to write
    to multi-layer data, store data within archives (zip files), etc.
    The path may specify a fiona VSI scheme.
    """
    if schema is None:
        schema = infer_schema(df)
    with fiona_env():
        with fiona.open(
            filename, "w", driver=driver, crs=df.crs, schema=schema, **kwargs
        ) as colxn:
            colxn.writerecords(df.iterfeatures())


def infer_schema(df):
    try:
        from collections import OrderedDict
    except ImportError:
        from ordereddict import OrderedDict

    def convert_type(column, in_type):
        if in_type == object:
            return "str"
        if in_type.name.startswith("datetime64"):
            # numpy datetime type regardless of frequency
            return "datetime"
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
