import os

import fiona
import numpy as np
from shapely.geometry import mapping

from six import iteritems
from geopandas import GeoDataFrame


def read_file(filename, **kwargs):
    """
    Returns a GeoDataFrame from a file.

    *filename* is either the absolute or relative path to the file to be
    opened and *kwargs* are keyword args to be passed to the `open` method
    in the fiona library when opening the file. For more information on 
    possible keywords, type: ``import fiona; help(fiona.open)``
    """
    bbox = kwargs.pop('bbox', None)
    with fiona.open(filename, **kwargs) as f:
        crs = f.crs
        if bbox is not None:
            assert len(bbox)==4
            f_filt = f.filter(bbox=bbox)
        else:
            f_filt = f
        gdf = GeoDataFrame.from_features(f, crs=crs)

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
    with fiona.open(filename, 'w', driver=driver, crs=df.crs,
                    schema=schema, **kwargs) as c:
        for feature in df.iterfeatures():
            c.write(feature)


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
