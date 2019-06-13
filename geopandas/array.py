import warnings

import numpy as np

from shapely.geometry.base import BaseGeometry
import shapely.geometry
import shapely.ops
import shapely.affinity


# -----------------------------------------------------------------------------
# Constructors / converters to other formats
# -----------------------------------------------------------------------------


def from_shapely(data):
    """
    Convert a list or array of shapely objects to a GeometryArray.

    Validates the elements.
    """
    n = len(data)

    out = []

    for idx in range(n):
        geom = data[idx]
        if isinstance(geom, BaseGeometry):
            out.append(geom)
        elif hasattr(geom, '__geo_interface__'):
            geom = shapely.geometry.asShape(geom)
            out.append(geom)
        elif geom is None or (isinstance(geom, float) and np.isnan(geom)):
            out.append(None)

    out = np.array(out, dtype=object)
    return GeometryArray(out)


def to_shapely(geoms):
    """
    Convert GeometryArray to numpy object array of shapely objects.
    """
    if not isinstance(geoms, GeometryArray):
        raise ValueError("'geoms' must be a GeometryArray")
    return geoms.data


def from_wkb(data):
    """
    Convert a list or array of WKB objects to a GeometryArray.
    """
    import shapely.wkb

    n = len(data)

    out = []

    for idx in range(n):
        geom = data[idx]
        if geom is not None and len(geom):
            geom = shapely.wkb.loads(geom)
        else:
            geom = None
        out.append(geom)

    out = np.array(out, dtype=object)
    return GeometryArray(out)


def to_wkb(geoms):
    """
    Convert GeometryArray to a numpy object array of WKB objects.
    """
    if not isinstance(geoms, GeometryArray):
        raise ValueError("'geoms' must be a GeometryArray")
    out = [geom.wkb if geom is not None else None for geom in geoms]
    return np.array(out, dtype=object)


def from_wkt(data):
    """
    Convert a list or array of WKT objects to a GeometryArray.
    """
    import shapely.wkt

    n = len(data)

    out = []

    for idx in range(n):
        geom = data[idx]
        if geom is not None and len(geom):
            if isinstance(geom, bytes):
                geom = geom.decode('utf-8')
            geom = shapely.wkt.loads(geom)
        else:
            geom = None
        out.append(geom)

    out = np.array(out, dtype=object)
    return GeometryArray(out)


def to_wkt(geoms):
    """
    Convert GeometryArray to a numpy object array of WKT objects.
    """
    if not isinstance(geoms, GeometryArray):
        raise ValueError("'geoms' must be a GeometryArray")
    out = [geom.wkt if geom is not None else None for geom in geoms]
    return np.array(out, dtype=object)


def _points_from_xy(x, y, z=None):
    """
    Generate list of shapely Point geometries from x, y(, z) coordinates.

    Parameters
    ----------
    x, y, z : array

    Examples
    --------
    >>> geometry = geopandas.points_from_xy(x=[1, 0], y=[0, 1])
    >>> geometry = geopandas.points_from_xy(df['x'], df['y'], df['z'])
    >>> gdf = geopandas.GeoDataFrame(
            df, geometry=geopandas.points_from_xy(df['x'], df['y']))

    Returns
    -------
    list : list
    """
    if not len(x) == len(y):
        raise ValueError("x and y arrays must be equal length.")
    if z is not None:
        if not len(z) == len(x):
            raise ValueError("z array must be same length as x and y.")
        geom = [shapely.geometry.Point(i, j, k) for i, j, k in zip(x, y, z)]
    else:
        geom = [shapely.geometry.Point(i, j) for i, j in zip(x, y)]
    return geom


def points_from_xy(x, y, z=None):
    """Convert arrays of x and y values to a GeometryArray of points."""
    x = np.asarray(x, dtype='float64')
    y = np.asarray(y, dtype='float64')
    if z is not None:
        z = np.asarray(z, dtype='float64')
    out = _points_from_xy(x, y, z)
    out = np.array(out, dtype=object)
    return GeometryArray(out)


# -----------------------------------------------------------------------------
# Helper methods for the vectorized operations
# -----------------------------------------------------------------------------


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
    op: string
    right: GeometryArray or single shapely BaseGeoemtry
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


def _unary_geo(op, left, *args, **kwargs):
    # type: (str, GeometryArray) -> GeometryArray
    """Unary operation that returns new geometries"""
    # ensure 1D output, see note above
    data = np.empty(len(left), dtype=object)
    data[:] = [getattr(geom, op) for geom in left.data]
    return GeometryArray(data)


def _unary_op(op, left, null_value=False):
    # type: (str, GeometryArray, Any) -> array
    """Unary operation that returns a Series"""
    data = [getattr(geom, op, null_value) for geom in left.data]
    return np.array(data, dtype=np.dtype(type(null_value)))


def _affinity_method(op, left, *args, **kwargs):
    # type: (str, GeometryArray, ...) -> GeometryArray
    data = [getattr(shapely.affinity, op)(s, *args, **kwargs)
            for s in left.data]
    return GeometryArray(np.array(data, dtype=object))


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
        elif not data.ndim == 1:
            raise ValueError(
                "'data' should be a 1-dimensional array of geometry objects.")
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        assert isinstance(i, int)
        return self.data[i]

    # -------------------------------------------------------------------------
    # Geometry related methods
    # -------------------------------------------------------------------------

    @property
    def is_valid(self):
        return _unary_op('is_valid', self, null_value=False)

    @property
    def is_empty(self):
        return _unary_op('is_empty', self, null_value=False)

    @property
    def is_simple(self):
        return _unary_op('is_simple', self, null_value=False)

    @property
    def is_ring(self):
        # operates on the exterior, so can't use _unary_op()
        return np.array(
            [geom.exterior.is_ring for geom in self.data], dtype=bool)

    @property
    def is_closed(self):
        return _unary_op('is_closed', self, null_value=False)

    @property
    def has_z(self):
        return _unary_op('has_z', self, null_value=False)

    @property
    def geom_type(self):
        return _unary_op('geom_type', self, null_value=None)

    @property
    def area(self):
        return _unary_op('area', self, null_value=np.nan)

    @property
    def length(self):
        return _unary_op('length', self, null_value=np.nan)

    #
    # Unary operations that return new geometries
    #

    @property
    def boundary(self):
        return _unary_geo('boundary', self)

    @property
    def centroid(self):
        return _unary_geo('centroid', self)

    @property
    def convex_hull(self):
        return _unary_geo('convex_hull', self)

    @property
    def envelope(self):
        return _unary_geo('envelope', self)

    @property
    def exterior(self):
        return _unary_geo('exterior', self)

    @property
    def interiors(self):
        has_non_poly = False
        inner_rings = []
        for geom in self.data:
            interior_ring_seq = getattr(geom, 'interiors', None)
            # polygon case
            if interior_ring_seq is not None:
                inner_rings.append(list(interior_ring_seq))
            # non-polygon case
            else:
                has_non_poly = True
                inner_rings.append(None)
        if has_non_poly:
            warnings.warn(
                "Only Polygon objects have interior rings. For other "
                "geometry types, None is returned.")

        return np.array(inner_rings, dtype=object)

    def representative_point(self):
        # method and not a property -> can't use _unary_geo
        data = np.empty(len(self), dtype=object)
        data[:] = [geom.representative_point() for geom in self.data]
        return GeometryArray(data)

    #
    # Binary predicates
    #

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

    #
    # Binary operations that return new geometries
    #

    def difference(self, other):
        return _binary_geo('difference', self, other)

    def intersection(self, other):
        return _binary_geo('intersection', self, other)

    def symmetric_difference(self, other):
        return _binary_geo('symmetric_difference', self, other)

    def union(self, other):
        return _binary_geo('union', self, other)

    #
    # Other operations
    #

    def distance(self, other):
        return _binary_op('distance', self, other)

    def buffer(self, distance, resolution=16, **kwargs):
        if isinstance(distance, np.ndarray):
            if len(distance) != len(self):
                raise ValueError("Length of distance sequence does not match "
                                 "length of the GeoSeries")
            data = [
                geom.buffer(dist, resolution, **kwargs)
                for geom, dist in zip(self.data, distance)]
            return GeometryArray(np.array(data, dtype=object))

        data = [geom.buffer(distance, resolution, **kwargs)
                for geom in self.data]
        return GeometryArray(np.array(data, dtype=object))

    def interpolate(self, distance, normalized=False):
        if isinstance(distance, np.ndarray):
            if len(distance) != len(self):
                raise ValueError("Length of distance sequence does not match "
                                 "length of the GeoSeries")
            data = [
                geom.interpolate(dist, normalized=normalized)
                for geom, dist in zip(self.data, distance)]
            return GeometryArray(np.array(data, dtype=object))

        data = [geom.interpolate(distance, normalized=normalized)
                for geom in self.data]
        return GeometryArray(np.array(data, dtype=object))

    def simplify(self, *args, **kwargs):
        # method and not a property -> can't use _unary_geo
        data = np.empty(len(self), dtype=object)
        data[:] = [geom.simplify(*args, **kwargs) for geom in self.data]
        return GeometryArray(data)

    def project(self, other, normalized=False):
        return _binary_op('project', self, other, normalized=normalized)

    def relate(self, other):
        return _binary_op('relate', self, other)

    #
    # Reduction operations that return a Shapely geometry
    #

    def unary_union(self):
        return shapely.ops.unary_union(self.data)

    #
    # Affinity operations
    #

    def affine_transform(self, matrix):
        return _affinity_method('affine_transform', self, matrix)

    def translate(self, xoff=0.0, yoff=0.0, zoff=0.0):
        return _affinity_method('translate', self, xoff, yoff, zoff)

    def rotate(self, angle, origin='center', use_radians=False):
        return _affinity_method('rotate', self, angle, origin=origin,
                                use_radians=use_radians)

    def scale(self, xfact=1.0, yfact=1.0, zfact=1.0, origin='center'):
        return _affinity_method(
            'scale', self, xfact, yfact, zfact, origin=origin)

    def skew(self, xs=0.0, ys=0.0, origin='center', use_radians=False):
        return _affinity_method('skew', self, xs, ys, origin=origin,
                                use_radians=use_radians)

    #
    # Coordinate related properties
    #

    @property
    def x(self):
        """Return the x location of point geometries in a GeoSeries"""
        if (self.geom_type == "Point").all():
            return _unary_op('x', self, null_value=np.nan)
        else:
            message = "x attribute access only provided for Point geometries"
            raise ValueError(message)

    @property
    def y(self):
        """Return the y location of point geometries in a GeoSeries"""
        if (self.geom_type == "Point").all():
            return _unary_op('y', self, null_value=np.nan)
        else:
            message = "y attribute access only provided for Point geometries"
            raise ValueError(message)

    @property
    def bounds(self):
        # TODO fix for empty / missing geometries
        bounds = np.array([geom.bounds for geom in self.data])
        return bounds

    @property
    def total_bounds(self):
        b = self.bounds
        return np.array((b[:, 0].min(),  # minx
                         b[:, 1].min(),  # miny
                         b[:, 2].max(),  # maxx
                         b[:, 3].max()))  # maxy
