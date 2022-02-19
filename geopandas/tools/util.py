import pandas as pd

from shapely.geometry import MultiLineString, MultiPoint, MultiPolygon
from shapely.geometry.base import BaseGeometry

_multi_type_map = {
    "Point": MultiPoint,
    "LineString": MultiLineString,
    "Polygon": MultiPolygon,
}


def collect(x, multi=False):
    """
    Collect single part geometries into their Multi* counterpart

    Parameters
    ----------
    x : an iterable or Series of Shapely geometries, a GeoSeries, or
        a single Shapely geometry
    multi : boolean, default False
        if True, force returned geometries to be Multi* even if they
        only have one component.

    """
    if isinstance(x, BaseGeometry):
        x = [x]
    elif isinstance(x, pd.Series):
        x = list(x)

    # We cannot create GeometryCollection here so all types
    # must be the same. If there is more than one element,
    # they cannot be Multi*, i.e., can't pass in combination of
    # Point and MultiPoint... or even just MultiPoint
    t = x[0].type
    if not all(g.type == t for g in x):
        raise ValueError("Geometry type must be homogeneous")
    if len(x) > 1 and t.startswith("Multi"):
        raise ValueError("Cannot collect {0}. Must have single geometries".format(t))

    if len(x) == 1 and (t.startswith("Multi") or not multi):
        # If there's only one single part geom and we're not forcing to
        # multi, then just return it
        return x[0]
    return _multi_type_map[t](x)
