import numpy as np

from shapely.geometry.base import BaseGeometry


def _binary_geo(op, left, right):
    # type: (str, GeometryArray, [GeometryArray/BaseGeometry]) -> GeometryArray
    """ Apply geometry-valued operation

    Supports:

    -   difference
    -   symmetric_difference
    -   intersection
    -   union

    Parameters
    ----------
    right: GeometryArray or single shapely BaseGeoemtry
    op: string
    """
    if isinstance(right, BaseGeometry):
        # intersection can return empty GeometryCollections, and if the
        # result are only those, numpy will coerce it to empty 2D array
        data = np.empty(len(left), dtype=object)
        data[:] = [getattr(s, op)(right) for s in left.data]
        return GeometryArray(data)
    elif isinstance(right, GeometryArray):
        if len(left) != len(right):
            msg = (
                "Lengths of inputs to not match. "
                "Left: {0}, Right: {1}".format(len(left), len(right)))
            raise ValueError(msg)
        data = np.empty(len(left), dtype=object)
        data[:] = [getattr(this_elem, op)(other_elem)
                   for this_elem, other_elem in zip(left.data, right.data)]
        return GeometryArray(data)
    else:
        raise TypeError(
            "Type not known: {0} vs {1}".format(type(left), type(right)))


def _binary_op(op, left, right, *args, **kwargs):
    # type: (str, GeometryArray, GeometryArray/BaseGeometry, args/kwargs)
    #        -> array
    """Binary operation on GeometryArray that returns a ndarray"""
    if op in ['distance', 'project']:
        null_value = np.nan
    elif op == 'relate':
        null_value = None
    else:
        null_value = False
    if op in ['distance', 'project']:
        dtype = float
    elif op == 'relate':
        dtype = object
    else:
        dtype = bool

    if isinstance(right, BaseGeometry):
        data = [
            getattr(s, op)(right, *args, **kwargs) if s else null_value
            for s in left.data]
        return np.array(data, dtype=dtype)
    elif isinstance(right, GeometryArray):
        if len(left) != len(right):
            msg = (
                "Lengths of inputs to not match. "
                "Left: {0}, Right: {1}".format(len(left), len(right)))
            raise ValueError(msg)
        data = [
            getattr(this_elem, op)(other_elem, *args, **kwargs)
            if not this_elem.is_empty | other_elem.is_empty else null_value
            for this_elem, other_elem in zip(left.data, right.data)]
        return np.array(data, dtype=dtype)
    else:
        raise TypeError(
            "Type not known: {0} vs {1}".format(type(left), type(right)))


def _unary_geo(op, left):
    # type: (str, GeometryArray) -> GeometryArray
    """Unary operation that returns new geometries"""
    # ensure 1D output, see note above
    data = np.empty(len(left), dtype=object)
    data[:] = [getattr(geom, op) for geom in left.data]
    return GeometryArray(data)


def _unary_op(op, left, null_value=False):
    # type: (str, GeometryArray, Any) -> Series
    """Unary operation that returns a Series"""
    data = [getattr(geom, op, null_value) for geom in left.data]
    return np.array(data, dtype=np.dtype(type(null_value)))


class GeometryArray:
    """
    Class wrapping a numpy array of Shapely objects and
    holding the array-based implementations.
    """

    def __init__(self, data):
        if isinstance(data, self.__class__):
            data = data.data
        elif not isinstance(data, np.ndarray):
            raise ValueError(
                "'data' should be array of geometry objects. Use from_shapely,"
                " from_wkb, from_wkt functions to construct a GeometryArray.")
        self.data = data

    def __len__(self):
        return len(self.data)

    # -------------------------------------------------------------------------
    # Geometry related methods
    # -------------------------------------------------------------------------

    def covers(self, other):
        return _binary_op('covers', self, other)

    def contains(self, other):
        return _binary_op('contains', self, other)

    def crosses(self, other):
        return _binary_op('crosses', self, other)

    def disjoint(self, other):
        return _binary_op('disjoint', self, other)

    def equals(self, other):
        return _binary_op('equals', self, other)

    def intersects(self, other):
        return _binary_op('intersects', self, other)

    def overlaps(self, other):
        return _binary_op('overlaps', self, other)

    def touches(self, other):
        return _binary_op('touches', self, other)

    def within(self, other):
        return _binary_op('within', self, other)

    def equals_exact(self, other, tolerance):
        return _binary_op('equals_exact', self, other, tolerance=tolerance)

    def almost_equals(self, other, decimal):
        return _binary_op('almost_equals', self, other, decimal=decimal)

    def is_valid(self):
        return _unary_op('is_valid', self, null_value=False)

    def is_empty(self):
        return _unary_op('is_empty', self, null_value=False)

    def is_simple(self):
        return _unary_op('is_simple', self, null_value=False)

    def is_ring(self):
        return _unary_op('is_ring', self, null_value=False)

    def has_z(self):
        return _unary_op('has_z', self, null_value=False)

    def is_closed(self):
        return _unary_op('is_closed', self, null_value=False)

    def boundary(self):
        return _unary_geo('boundary', self)

    def centroid(self):
        return _unary_geo('centroid', self)

    def convex_hull(self):
        return _unary_geo('convex_hull', self)

    def envelope(self):
        return _unary_geo('envelope', self)

    def exterior(self):
        return _unary_geo('exterior', self)

    def representative_point(self):
        return _unary_geo(self, 'representative_point')

    def distance(self, other):
        return _binary_op('distance', self, other)

    def project(self, other, normalized=False):
        return _binary_op('project', self, other, normalized=normalized)

    def relate(self, other):
        return _binary_op('relate', self, other)

    def area(self):
        return _unary_op('area', self, null_value=np.nan)

    def length(self):
        return _unary_op('length', self, null_value=np.nan)

    def difference(self, other):
        return _binary_geo('difference', self, other)

    def symmetric_difference(self, other):
        return _binary_geo('symmetric_difference', self, other)

    def union(self, other):
        return _binary_geo('union', self, other)

    def intersection(self, other):
        return _binary_geo('intersection', self, other)

    # def buffer(self, distance, resolution=16, cap_style=CAP_STYLE.round,
    #            join_style=JOIN_STYLE.round, mitre_limit=5.0):
    #     """ Buffer operation on array of GEOSGeometry objects """
    #     return GeometryArray(
    #         vectorized.buffer(self.data, distance, resolution, cap_style,
    #                           join_style, mitre_limit))

    def geom_type(self):
        return _unary_op('geom_type', self, null_value=None)

    # def unary_union(self):
    #     """ Unary union.

    #     Returns a single shapely geometry
    #     """
    #     return vectorized.unary_union(self.data)

    # def coords(self):
    #     return vectorized.coords(self.data)

    def x(self):
        """Return the x location of point geometries in a GeoSeries"""
        if (self.geom_type() == "Point").all():
            return _unary_op('x', self, null_value=np.nan)
        else:
            message = "x attribute access only provided for Point geometries"
            raise ValueError(message)

    def y(self):
        """Return the y location of point geometries in a GeoSeries"""
        if (self.geom_type() == "Point").all():
            return _unary_op('y', self, null_value=np.nan)
        else:
            message = "y attribute access only provided for Point geometries"
            raise ValueError(message)
