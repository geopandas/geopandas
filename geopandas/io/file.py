import os

import fiona
import numpy as np
import six

from geopandas import GeoDataFrame

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


def read_file(filename, **kwargs):
    """
    Returns a GeoDataFrame from a file or URL.

    Parameters
    ----------
    filename: str
        Either the absolute or relative path to the file or URL to
        be opened.
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
    bbox = kwargs.pop('bbox', None)
    if _is_url(filename):
        req = _urlopen(filename)
        path_or_bytes = req.read()
        reader = fiona.BytesCollection
    else:
        path_or_bytes = filename
        reader = fiona.open
    with reader(path_or_bytes, **kwargs) as f:
        crs = f.crs
        if bbox is not None:
            assert len(bbox) == 4
            f_filt = f.filter(bbox=bbox)
        else:
            f_filt = f
        gdf = GeoDataFrame.from_features(f_filt, crs=crs)
        # re-order with column order from metadata, with geometry last
        columns = list(f.meta["schema"]["properties"]) + ["geometry"]
        gdf = gdf[columns]

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
    with fiona.drivers():
        with fiona.open(filename, 'w', driver=driver, crs=df.crs,
                        schema=schema, **kwargs) as colxn:
            colxn.writerecords(df.iterfeatures())


def infer_schema(df):
    try:
        from collections import OrderedDict
    except ImportError:
        from ordereddict import OrderedDict

    def convert_type(in_type):
        if in_type == object:
            return 'str'
        out_type = type(np.asscalar(np.zeros(1, in_type))).__name__
        if out_type == 'long':
            out_type = 'int'
        return out_type

    properties = OrderedDict([
        (col, convert_type(_type)) for col, _type in
        zip(df.columns, df.dtypes) if col != df._geometry_column_name
    ])

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

    from os.path import commonprefix   # To find longest common prefix
    geom_type = commonprefix([g[::-1] for g in geom_types if g])[::-1]  # Reverse
    if not geom_type:
        geom_type = None

    return geom_type
