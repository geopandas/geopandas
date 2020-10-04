"""
Compatibility shim for the vectorized geometry operations.

Uses PyGEOS if available/set, otherwise loops through Shapely geometries.

"""
import warnings

import numpy as np

import shapely.geometry
import shapely.geos
import shapely.wkb
import shapely.wkt

from shapely.geometry.base import BaseGeometry

from . import _compat as compat

try:
    import pygeos
except ImportError:
    geos = None


_names = {
    "MISSING": None,
    "NAG": None,
    "POINT": "Point",
    "LINESTRING": "LineString",
    "LINEARRING": "LinearRing",
    "POLYGON": "Polygon",
    "MULTIPOINT": "MultiPoint",
    "MULTILINESTRING": "MultiLineString",
    "MULTIPOLYGON": "MultiPolygon",
    "GEOMETRYCOLLECTION": "GeometryCollection",
}

if compat.USE_PYGEOS:
    type_mapping = {p.value: _names[p.name] for p in pygeos.GeometryType}
    geometry_type_ids = list(type_mapping.keys())
    geometry_type_values = np.array(list(type_mapping.values()), dtype=object)
else:
    type_mapping, geometry_type_ids, geometry_type_values = None, None, None


def _isna(value):
    """
    Check if scalar value is NA-like (None or np.nan).

    Custom version that only works for scalars (returning True or False),
    as `pd.isna` also works for array-like input returning a boolean array.
    """
    if value is None:
        return True
    elif isinstance(value, float) and np.isnan(value):
        return True
    else:
        return False


def _pygeos_to_shapely(geom):
    if geom is None:
        return None

    if compat.PYGEOS_SHAPELY_COMPAT:
        geom = shapely.geos.lgeos.GEOSGeom_clone(geom._ptr)
        return shapely.geometry.base.geom_factory(geom)

    # fallback going through WKB
    if pygeos.is_empty(geom) and pygeos.get_type_id(geom) == 0:
        # empty point does not roundtrip through WKB
        return shapely.wkt.loads("POINT EMPTY")
    else:
        return shapely.wkb.loads(pygeos.to_wkb(geom))


def _shapely_to_pygeos(geom):
    if geom is None:
        return None

    if compat.PYGEOS_SHAPELY_COMPAT:
        return pygeos.from_shapely(geom)

    # fallback going through WKB
    if geom.is_empty and geom.geom_type == "Point":
        # empty point does not roundtrip through WKB
        return pygeos.from_wkt("POINT EMPTY")
    else:
        return pygeos.from_wkb(geom.wkb)


def from_shapely(data):
    """
    Convert a list or array of shapely objects to an object-dtype numpy
    array of validated geometry elements.

    """
    # First try a fast path for pygeos if possible, but do this in a try-except
    # block because pygeos.from_shapely only handles Shapely objects, while
    # the rest of this function is more forgiving (also __geo_interface__).
    if compat.USE_PYGEOS and compat.PYGEOS_SHAPELY_COMPAT:
        if not isinstance(data, np.ndarray):
            arr = np.empty(len(data), dtype=object)
            arr[:] = data
        else:
            arr = data
        try:
            return pygeos.from_shapely(arr)
        except TypeError:
            pass

    out = []

    for geom in data:
        if compat.USE_PYGEOS and isinstance(geom, pygeos.Geometry):
            out.append(geom)
        elif isinstance(geom, BaseGeometry):
            if compat.USE_PYGEOS:
                out.append(_shapely_to_pygeos(geom))
            else:
                out.append(geom)
        elif hasattr(geom, "__geo_interface__"):
            geom = shapely.geometry.asShape(geom)
            # asShape returns GeometryProxy -> trigger actual materialization
            # with one of its methods
            geom.wkb
            if compat.USE_PYGEOS:
                out.append(_shapely_to_pygeos(geom))
            else:
                out.append(geom)
        elif _isna(geom):
            out.append(None)
        else:
            raise TypeError("Input must be valid geometry objects: {0}".format(geom))

    if compat.USE_PYGEOS:
        return np.array(out, dtype=object)
    else:
        # numpy can expand geometry collections into 2D arrays, use this
        # two-step construction to avoid this
        aout = np.empty(len(data), dtype=object)
        aout[:] = out
        return aout


def to_shapely(data):
    if compat.USE_PYGEOS:
        out = np.empty(len(data), dtype=object)
        out[:] = [_pygeos_to_shapely(geom) for geom in data]
        return out
    else:
        return data


def from_wkb(data):
    """
    Convert a list or array of WKB objects to a np.ndarray[geoms].
    """
    if compat.USE_PYGEOS:
        return pygeos.from_wkb(data)

    import shapely.wkb

    out = []

    for geom in data:
        if geom is not None and len(geom):
            geom = shapely.wkb.loads(geom)
        else:
            geom = None
        out.append(geom)

    aout = np.empty(len(data), dtype=object)
    aout[:] = out
    return aout


def to_wkb(data, hex=False):
    if compat.USE_PYGEOS:
        return pygeos.to_wkb(data, hex=hex)
    else:
        if hex:
            out = [geom.wkb_hex if geom is not None else None for geom in data]
        else:
            out = [geom.wkb if geom is not None else None for geom in data]
        return np.array(out, dtype=object)


def from_wkt(data):
    """
    Convert a list or array of WKT objects to a np.ndarray[geoms].
    """
    if compat.USE_PYGEOS:
        return pygeos.from_wkt(data)

    import shapely.wkt

    out = []

    for geom in data:
        if geom is not None and len(geom):
            if isinstance(geom, bytes):
                geom = geom.decode("utf-8")
            geom = shapely.wkt.loads(geom)
        else:
            geom = None
        out.append(geom)

    aout = np.empty(len(data), dtype=object)
    aout[:] = out
    return aout


def to_wkt(data, **kwargs):
    if compat.USE_PYGEOS:
        return pygeos.to_wkt(data, **kwargs)
    else:
        out = [geom.wkt if geom is not None else None for geom in data]
        return np.array(out, dtype=object)


def _points_from_xy(x, y, z=None):
    # helper method for shapely-based function
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

    x = np.asarray(x, dtype="float64")
    y = np.asarray(y, dtype="float64")
    if z is not None:
        z = np.asarray(z, dtype="float64")

    if compat.USE_PYGEOS:
        return pygeos.points(x, y, z)
    else:
        out = _points_from_xy(x, y, z)
        aout = np.empty(len(x), dtype=object)
        aout[:] = out
        return aout


# -----------------------------------------------------------------------------
# Helper methods for the vectorized operations
# -----------------------------------------------------------------------------


def _binary_method(op, left, right, **kwargs):
    # type: (str, np.array[geoms], [np.array[geoms]/BaseGeometry]) -> array-like
    if isinstance(right, BaseGeometry):
        right = from_shapely([right])[0]
    return getattr(pygeos, op)(left, right, **kwargs)


def _binary_geo(op, left, right):
    # type: (str, np.array[geoms], [np.array[geoms]/BaseGeometry]) -> np.array[geoms]
    """ Apply geometry-valued operation

    Supports:

    -   difference
    -   symmetric_difference
    -   intersection
    -   union

    Parameters
    ----------
    op: string
    right: np.array[geoms] or single shapely BaseGeoemtry
    """
    if isinstance(right, BaseGeometry):
        # intersection can return empty GeometryCollections, and if the
        # result are only those, numpy will coerce it to empty 2D array
        data = np.empty(len(left), dtype=object)
        data[:] = [
            getattr(s, op)(right) if s is not None and right is not None else None
            for s in left
        ]
        return data
    elif isinstance(right, np.ndarray):
        if len(left) != len(right):
            msg = "Lengths of inputs do not match. Left: {0}, Right: {1}".format(
                len(left), len(right)
            )
            raise ValueError(msg)
        data = np.empty(len(left), dtype=object)
        data[:] = [
            getattr(this_elem, op)(other_elem)
            if this_elem is not None and other_elem is not None
            else None
            for this_elem, other_elem in zip(left, right)
        ]
        return data
    else:
        raise TypeError("Type not known: {0} vs {1}".format(type(left), type(right)))


def _binary_predicate(op, left, right, *args, **kwargs):
    # type: (str, np.array[geoms], np.array[geoms]/BaseGeometry, args/kwargs)
    #        -> array[bool]
    """Binary operation on np.array[geoms] that returns a boolean ndarray

    Supports:

    -  contains
    -  disjoint
    -  intersects
    -  touches
    -  crosses
    -  within
    -  overlaps
    -  covers
    -  covered_by
    -  equals

    Parameters
    ----------
    op: string
    right: np.array[geoms] or single shapely BaseGeoemtry
    """
    # empty geometries are handled by shapely (all give False except disjoint)
    if isinstance(right, BaseGeometry):
        data = [
            getattr(s, op)(right, *args, **kwargs) if s is not None else False
            for s in left
        ]
        return np.array(data, dtype=bool)
    elif isinstance(right, np.ndarray):
        data = [
            getattr(this_elem, op)(other_elem, *args, **kwargs)
            if not (this_elem is None or other_elem is None)
            else False
            for this_elem, other_elem in zip(left, right)
        ]
        return np.array(data, dtype=bool)
    else:
        raise TypeError("Type not known: {0} vs {1}".format(type(left), type(right)))


def _binary_op_float(op, left, right, *args, **kwargs):
    # type: (str, np.array[geoms], np.array[geoms]/BaseGeometry, args/kwargs)
    #        -> array
    """Binary operation on np.array[geoms] that returns a ndarray"""
    # used for distance -> check for empty as we want to return np.nan instead 0.0
    # as shapely does currently (https://github.com/Toblerity/Shapely/issues/498)
    if isinstance(right, BaseGeometry):
        data = [
            getattr(s, op)(right, *args, **kwargs)
            if not (s is None or s.is_empty or right.is_empty)
            else np.nan
            for s in left
        ]
        return np.array(data, dtype=float)
    elif isinstance(right, np.ndarray):
        if len(left) != len(right):
            msg = "Lengths of inputs do not match. Left: {0}, Right: {1}".format(
                len(left), len(right)
            )
            raise ValueError(msg)
        data = [
            getattr(this_elem, op)(other_elem, *args, **kwargs)
            if not (this_elem is None or this_elem.is_empty)
            | (other_elem is None or other_elem.is_empty)
            else np.nan
            for this_elem, other_elem in zip(left, right)
        ]
        return np.array(data, dtype=float)
    else:
        raise TypeError("Type not known: {0} vs {1}".format(type(left), type(right)))


def _binary_op(op, left, right, *args, **kwargs):
    # type: (str, np.array[geoms], np.array[geoms]/BaseGeometry, args/kwargs)
    #        -> array
    """Binary operation on np.array[geoms] that returns a ndarray"""
    # pass empty to shapely (relate handles this correctly, project only
    # for linestrings and points)
    if op == "project":
        null_value = np.nan
        dtype = float
    elif op == "relate":
        null_value = None
        dtype = object
    else:
        raise AssertionError("wrong op")

    if isinstance(right, BaseGeometry):
        data = [
            getattr(s, op)(right, *args, **kwargs) if s is not None else null_value
            for s in left
        ]
        return np.array(data, dtype=dtype)
    elif isinstance(right, np.ndarray):
        if len(left) != len(right):
            msg = "Lengths of inputs do not match. Left: {0}, Right: {1}".format(
                len(left), len(right)
            )
            raise ValueError(msg)
        data = [
            getattr(this_elem, op)(other_elem, *args, **kwargs)
            if not (this_elem is None or other_elem is None)
            else null_value
            for this_elem, other_elem in zip(left, right)
        ]
        return np.array(data, dtype=dtype)
    else:
        raise TypeError("Type not known: {0} vs {1}".format(type(left), type(right)))


def _affinity_method(op, left, *args, **kwargs):
    # type: (str, np.array[geoms], ...) -> np.array[geoms]

    # not all shapely.affinity methods can handle empty geometries:
    # affine_transform itself works (as well as translate), but rotate, scale
    # and skew fail (they try to unpack the bounds).
    # Here: consistently returning empty geom for input empty geom
    left = to_shapely(left)
    out = []
    for geom in left:
        if geom is None or geom.is_empty:
            res = geom
        else:
            res = getattr(shapely.affinity, op)(geom, *args, **kwargs)
        out.append(res)
    data = np.empty(len(left), dtype=object)
    data[:] = out
    return from_shapely(data)


# -----------------------------------------------------------------------------
# Vectorized operations
# -----------------------------------------------------------------------------


#
# Unary operations that return non-geometry (bool or float)
#


def _unary_op(op, left, null_value=False):
    # type: (str, np.array[geoms], Any) -> np.array
    """Unary operation that returns a Series"""
    data = [getattr(geom, op, null_value) for geom in left]
    return np.array(data, dtype=np.dtype(type(null_value)))


def is_valid(data):
    if compat.USE_PYGEOS:
        return pygeos.is_valid(data)
    else:
        return _unary_op("is_valid", data, null_value=False)


def is_empty(data):
    if compat.USE_PYGEOS:
        return pygeos.is_empty(data)
    else:
        return _unary_op("is_empty", data, null_value=False)


def is_simple(data):
    if compat.USE_PYGEOS:
        return pygeos.is_simple(data)
    else:
        return _unary_op("is_simple", data, null_value=False)


def is_ring(data):
    if compat.USE_PYGEOS:
        return pygeos.is_ring(pygeos.get_exterior_ring(data))
    else:
        # operates on the exterior, so can't use _unary_op()
        # XXX needed to change this because there is now a geometry collection
        # in the shapely ones that was something else before?
        return np.array(
            [
                geom.exterior.is_ring
                if geom is not None
                and hasattr(geom, "exterior")
                and geom.exterior is not None
                else False
                for geom in data
            ],
            dtype=bool,
        )


def is_closed(data):
    if compat.USE_PYGEOS:
        return pygeos.is_closed(data)
    else:
        return _unary_op("is_closed", data, null_value=False)


def has_z(data):
    if compat.USE_PYGEOS:
        return pygeos.has_z(data)
    else:
        return _unary_op("has_z", data, null_value=False)


def geom_type(data):
    if compat.USE_PYGEOS:
        res = pygeos.get_type_id(data)
        return geometry_type_values[np.searchsorted(geometry_type_ids, res)]
    else:
        return _unary_op("geom_type", data, null_value=None)


def area(data):
    if compat.USE_PYGEOS:
        return pygeos.area(data)
    else:
        return _unary_op("area", data, null_value=np.nan)


def length(data):
    if compat.USE_PYGEOS:
        return pygeos.length(data)
    else:
        return _unary_op("length", data, null_value=np.nan)


#
# Unary operations that return new geometries
#


def _unary_geo(op, left, *args, **kwargs):
    # type: (str, np.array[geoms]) -> np.array[geoms]
    """Unary operation that returns new geometries"""
    # ensure 1D output, see note above
    data = np.empty(len(left), dtype=object)
    data[:] = [getattr(geom, op, None) for geom in left]
    return data


def boundary(data):
    if compat.USE_PYGEOS:
        return pygeos.boundary(data)
    else:
        return _unary_geo("boundary", data)


def centroid(data):
    if compat.USE_PYGEOS:
        return pygeos.centroid(data)
    else:
        return _unary_geo("centroid", data)


def convex_hull(data):
    if compat.USE_PYGEOS:
        return pygeos.convex_hull(data)
    else:
        return _unary_geo("convex_hull", data)


def envelope(data):
    if compat.USE_PYGEOS:
        return pygeos.envelope(data)
    else:
        return _unary_geo("envelope", data)


def exterior(data):
    if compat.USE_PYGEOS:
        return pygeos.get_exterior_ring(data)
    else:
        return _unary_geo("exterior", data)


def interiors(data):
    data = to_shapely(data)
    has_non_poly = False
    inner_rings = []
    for geom in data:
        interior_ring_seq = getattr(geom, "interiors", None)
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
            "geometry types, None is returned."
        )
    data = np.empty(len(data), dtype=object)
    data[:] = inner_rings
    return data


def representative_point(data):
    if compat.USE_PYGEOS:
        return pygeos.point_on_surface(data)
    else:
        # method and not a property -> can't use _unary_geo
        out = np.empty(len(data), dtype=object)
        out[:] = [
            geom.representative_point() if geom is not None else None for geom in data
        ]
        return out


#
# Binary predicates
#


def covers(data, other):
    if compat.USE_PYGEOS:
        return _binary_method("covers", data, other)
    else:
        return _binary_predicate("covers", data, other)


def covered_by(data, other):
    if compat.USE_PYGEOS:
        return _binary_method("covered_by", data, other)
    else:
        raise NotImplementedError(
            "covered_by is only implemented for pygeos, not shapely"
        )


def contains(data, other):
    if compat.USE_PYGEOS:
        return _binary_method("contains", data, other)
    else:
        return _binary_predicate("contains", data, other)


def crosses(data, other):
    if compat.USE_PYGEOS:
        return _binary_method("crosses", data, other)
    else:
        return _binary_predicate("crosses", data, other)


def disjoint(data, other):
    if compat.USE_PYGEOS:
        return _binary_method("disjoint", data, other)
    else:
        return _binary_predicate("disjoint", data, other)


def equals(data, other):
    if compat.USE_PYGEOS:
        return _binary_method("equals", data, other)
    else:
        return _binary_predicate("equals", data, other)


def intersects(data, other):
    if compat.USE_PYGEOS:
        return _binary_method("intersects", data, other)
    else:
        return _binary_predicate("intersects", data, other)


def overlaps(data, other):
    if compat.USE_PYGEOS:
        return _binary_method("overlaps", data, other)
    else:
        return _binary_predicate("overlaps", data, other)


def touches(data, other):
    if compat.USE_PYGEOS:
        return _binary_method("touches", data, other)
    else:
        return _binary_predicate("touches", data, other)


def within(data, other):
    if compat.USE_PYGEOS:
        return _binary_method("within", data, other)
    else:
        return _binary_predicate("within", data, other)


def equals_exact(data, other, tolerance):
    if compat.USE_PYGEOS:
        return _binary_method("equals_exact", data, other, tolerance=tolerance)
    else:
        return _binary_predicate("equals_exact", data, other, tolerance=tolerance)


def almost_equals(self, other, decimal):
    if compat.USE_PYGEOS:
        return self.equals_exact(other, 0.5 * 10 ** (-decimal))
    else:
        return _binary_predicate("almost_equals", self, other, decimal=decimal)


#
# Binary operations that return new geometries
#


def difference(data, other):
    if compat.USE_PYGEOS:
        return _binary_method("difference", data, other)
    else:
        return _binary_geo("difference", data, other)


def intersection(data, other):
    if compat.USE_PYGEOS:
        return _binary_method("intersection", data, other)
    else:
        return _binary_geo("intersection", data, other)


def symmetric_difference(data, other):
    if compat.USE_PYGEOS:
        return _binary_method("symmetric_difference", data, other)
    else:
        return _binary_geo("symmetric_difference", data, other)


def union(data, other):
    if compat.USE_PYGEOS:
        return _binary_method("union", data, other)
    else:
        return _binary_geo("union", data, other)


#
# Other operations
#


def distance(data, other):
    if compat.USE_PYGEOS:
        return _binary_method("distance", data, other)
    else:
        return _binary_op_float("distance", data, other)


def buffer(data, distance, resolution=16, **kwargs):
    if compat.USE_PYGEOS:
        return pygeos.buffer(data, distance, quadsegs=resolution, **kwargs)
    else:
        out = np.empty(len(data), dtype=object)
        if isinstance(distance, np.ndarray):
            if len(distance) != len(data):
                raise ValueError(
                    "Length of distance sequence does not match "
                    "length of the GeoSeries"
                )

            out[:] = [
                geom.buffer(dist, resolution, **kwargs) if geom is not None else None
                for geom, dist in zip(data, distance)
            ]
            return out

        out[:] = [
            geom.buffer(distance, resolution, **kwargs) if geom is not None else None
            for geom in data
        ]
        return out


def interpolate(data, distance, normalized=False):
    if compat.USE_PYGEOS:
        return pygeos.line_interpolate_point(data, distance, normalize=normalized)
    else:
        out = np.empty(len(data), dtype=object)
        if isinstance(distance, np.ndarray):
            if len(distance) != len(data):
                raise ValueError(
                    "Length of distance sequence does not match "
                    "length of the GeoSeries"
                )
            out[:] = [
                geom.interpolate(dist, normalized=normalized)
                for geom, dist in zip(data, distance)
            ]
            return out

        out[:] = [geom.interpolate(distance, normalized=normalized) for geom in data]
        return out


def simplify(data, tolerance, preserve_topology=True):
    if compat.USE_PYGEOS:
        # preserve_topology has different default as pygeos!
        return pygeos.simplify(data, tolerance, preserve_topology=preserve_topology)
    else:
        # method and not a property -> can't use _unary_geo
        out = np.empty(len(data), dtype=object)
        out[:] = [
            geom.simplify(tolerance, preserve_topology=preserve_topology)
            for geom in data
        ]
        return out


def project(data, other, normalized=False):
    if compat.USE_PYGEOS:
        return pygeos.line_locate_point(data, other, normalize=normalized)
    else:
        return _binary_op("project", data, other, normalized=normalized)


def relate(data, other):
    data = to_shapely(data)
    if isinstance(other, np.ndarray):
        other = to_shapely(other)
    return _binary_op("relate", data, other)


def unary_union(data):
    if compat.USE_PYGEOS:
        return _pygeos_to_shapely(pygeos.union_all(data))
    else:
        return shapely.ops.unary_union(data)


#
# Coordinate related properties
#


def get_x(data):
    if compat.USE_PYGEOS:
        return pygeos.get_x(data)
    else:
        return _unary_op("x", data, null_value=np.nan)


def get_y(data):
    if compat.USE_PYGEOS:
        return pygeos.get_y(data)
    else:
        return _unary_op("y", data, null_value=np.nan)


def bounds(data):
    if compat.USE_PYGEOS:
        return pygeos.bounds(data)
    # ensure that for empty arrays, the result has the correct shape
    if len(data) == 0:
        return np.empty((0, 4), dtype="float64")
    # need to explicitly check for empty (in addition to missing) geometries,
    # as those return an empty tuple, not resulting in a 2D array
    bounds = np.array(
        [
            geom.bounds
            if not (geom is None or geom.is_empty)
            else (np.nan, np.nan, np.nan, np.nan)
            for geom in data
        ]
    )
    return bounds


#
# Coordinate transformation
#


def transform(data, func):
    if compat.USE_PYGEOS:
        coords = pygeos.get_coordinates(data)
        new_coords = func(coords[:, 0], coords[:, 1])
        result = pygeos.set_coordinates(data.copy(), np.array(new_coords).T)
        return result
    else:
        from shapely.ops import transform

        n = len(data)
        result = np.empty(n, dtype=object)
        for i in range(n):
            geom = data[i]
            result[i] = transform(func, geom)

        return result
