import os

import fiona
import numpy as np
import six

try:
    from fiona import Env as fiona_env
except ImportError:
    from fiona import drivers as fiona_env

from geopandas import GeoDataFrame, GeoSeries

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
_VALID_URLS.discard('')


def _is_url(url):
    """Check to see if *url* has a valid protocol."""
    try:
        return parse_url(url).scheme in _VALID_URLS
    except:
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


def to_file(df, filename, driver="ESRI Shapefile", schema=None,
            **kwargs):
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
    """
    if schema is None:
        schema = infer_schema(df)
    filename = os.path.abspath(os.path.expanduser(filename))
    with fiona_env():
        with fiona.open(filename, 'w', driver=driver, crs=df.crs,
                        schema=schema, **kwargs) as colxn:
            colxn.writerecords(df.iterfeatures())


def infer_schema(df):
    try:
        from collections import OrderedDict
    except ImportError:
        from ordereddict import OrderedDict

    def convert_type(column, in_type):
        if in_type == object:
            return 'str'
        out_type = type(np.zeros(1, in_type).item()).__name__
        if out_type == 'long':
            out_type = 'int'
        if out_type == 'bool':
            raise ValueError('column "{}" is boolean type, '.format(column) +
                             'which is unsupported in file writing. '
                             'Consider casting the column to int type.')
        return out_type

    properties = OrderedDict([
        (col, convert_type(col, _type)) for col, _type in
        zip(df.columns, df.dtypes) if col != df._geometry_column_name
    ])

    if df.empty:
        raise ValueError("Cannot write empty DataFrame to file.")

    geom_type = _common_geom_type(df)
    
    if not geom_type:
        raise ValueError("Geometry column cannot contain mutiple "
                         "geometry types when writing to file.")

    schema = {'geometry': geom_type, 'properties': properties}

    return schema


def _common_geom_type(df):
    # Need to check geom_types before we write to file...
    # Some (most?) providers expect a single geometry type:
    # Point, LineString, or Polygon
    geom_types = df.geometry.geom_type.unique()

    from os.path import commonprefix
    # use reversed geom types and commonprefix to find the common suffix,
    # then reverse the result to get back to a geom type
    geom_type = commonprefix([g[::-1] for g in geom_types if g])[::-1]
    if not geom_type:
        return None

    if df.geometry.has_z.any():
        geom_type = "3D " + geom_type

    return geom_type
