"""
Compatibility shim for the vectorized geometry operations.
"""
import warnings

import numpy as np
import pandas as pd
import shapely
import shapely.geometry
import shapely.geos
import shapely.ops
import shapely.validation
import shapely.wkb
import shapely.wkt
from shapely.geometry.base import BaseGeometry


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


type_mapping = {p.value: _names[p.name] for p in shapely.GeometryType}
geometry_type_ids = list(type_mapping.keys())
geometry_type_values = np.array(list(type_mapping.values()), dtype=object)


def isna(value):
    """
    Check if scalar value is NA-like (None, np.nan or pd.NA).

    Custom version that only works for scalars (returning True or False),
    as `pd.isna` also works for array-like input returning a boolean array.
    """
    if value is None:
        return True
    elif isinstance(value, float) and np.isnan(value):
        return True
    elif value is pd.NA:
        return True
    else:
        return False


def from_shapely(data):
    """
    Convert a list or array of shapely objects to an object-dtype numpy
    array of validated geometry elements.

    """
    # First try a fast path for pygeos if possible, but do this in a try-except
    # block because pygeos.from_shapely only handles Shapely objects, while
    # the rest of this function is more forgiving (also __geo_interface__).

    out = []

    for geom in data:
        if isinstance(geom, BaseGeometry):
            out.append(geom)
        elif hasattr(geom, "__geo_interface__"):
            geom = shapely.geometry.shape(geom)
            out.append(geom)
        elif isna(geom):
            out.append(None)
        else:
            raise TypeError("Input must be valid geometry objects: {0}".format(geom))

    return np.array(out, dtype=object)


def from_wkb(data):
    """
    Convert a list or array of WKB objects to a np.ndarray[geoms].
    """
    return shapely.from_wkb(data)


def to_wkb(data, hex=False, **kwargs):
    return shapely.to_wkb(data, hex=hex, **kwargs)


def from_wkt(data):
    """
    Convert a list or array of WKT objects to a np.ndarray[geoms].
    """
    return shapely.from_wkt(data)


def to_wkt(data, **kwargs):
    return shapely.to_wkt(data, **kwargs)


def points_from_xy(x, y, z=None):
    x = np.asarray(x, dtype="float64")
    y = np.asarray(y, dtype="float64")
    if z is not None:
        z = np.asarray(z, dtype="float64")

    return shapely.points(x, y, z)


def _affinity_method(op, left, *args, **kwargs):
    # type: (str, np.array[geoms], ...) -> np.array[geoms]

    # not all shapely.affinity methods can handle empty geometries:
    # affine_transform itself works (as well as translate), but rotate, scale
    # and skew fail (they try to unpack the bounds).
    # Here: consistently returning empty geom for input empty geom
    out = []
    for geom in left:
        if geom is None or geom.is_empty:
            res = geom
        else:
            res = getattr(shapely.affinity, op)(geom, *args, **kwargs)
        out.append(res)
    data = np.empty(len(left), dtype=object)
    data[:] = out
    return data


# -----------------------------------------------------------------------------
# Vectorized operations
# -----------------------------------------------------------------------------


#
# Unary operations that return non-geometry (bool or float)
#


def is_valid(data):
    return shapely.is_valid(data)


def is_empty(data):
    return shapely.is_empty(data)


def is_simple(data):
    return shapely.is_simple(data)


def is_ring(data):
    return shapely.is_ring(data)


def is_closed(data):
    return shapely.is_closed(data)


def has_z(data):
    return shapely.has_z(data)


def geom_type(data):
    res = shapely.get_type_id(data)
    return geometry_type_values[np.searchsorted(geometry_type_ids, res)]


def area(data):
    return shapely.area(data)


def length(data):
    return shapely.length(data)


#
# Unary operations that return new geometries
#


def boundary(data):
    return shapely.boundary(data)


def centroid(data):
    return shapely.centroid(data)


def concave_hull(data, **kwargs):
    return shapely.concave_hull(data, **kwargs)


def convex_hull(data):
    return shapely.convex_hull(data)


def delaunay_triangles(data, tolerance, only_edges):
    return shapely.delaunay_triangles(data, tolerance, only_edges)


def envelope(data):
    return shapely.envelope(data)


def minimum_rotated_rectangle(data):
    return shapely.oriented_envelope(data)


def exterior(data):
    return shapely.get_exterior_ring(data)


def extract_unique_points(data):
    return shapely.extract_unique_points(data)


def offset_curve(data, distance, quad_segs=8, join_style="round", mitre_limit=5.0):
    return shapely.offset_curve(
        data,
        distance=distance,
        quad_segs=quad_segs,
        join_style=join_style,
        mitre_limit=mitre_limit,
    )


def interiors(data):
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
            "geometry types, None is returned.",
            stacklevel=2,
        )
    data = np.empty(len(data), dtype=object)
    data[:] = inner_rings
    return data


def remove_repeated_points(data, tolerance=0.0):
    return shapely.remove_repeated_points(data, tolerance=tolerance)


def representative_point(data):
    return shapely.point_on_surface(data)


def minimum_bounding_circle(data):
    return shapely.minimum_bounding_circle(data)


def minimum_bounding_radius(data):
    return shapely.minimum_bounding_radius(data)


def segmentize(data, max_segment_length):
    return shapely.segmentize(data, max_segment_length)


#
# Binary predicates
#


def covers(data, other):
    return shapely.covers(data, other)


def covered_by(data, other):
    return shapely.covered_by(data, other)


def contains(data, other):
    return shapely.contains(data, other)


def crosses(data, other):
    return shapely.crosses(data, other)


def disjoint(data, other):
    return shapely.disjoint(data, other)


def equals(data, other):
    return shapely.equals(data, other)


def intersects(data, other):
    return shapely.intersects(data, other)


def overlaps(data, other):
    return shapely.overlaps(data, other)


def touches(data, other):
    return shapely.touches(data, other)


def within(data, other):
    return shapely.within(data, other)


def equals_exact(data, other, tolerance):
    return shapely.equals_exact(data, other, tolerance=tolerance)


def almost_equals(self, other, decimal):
    return self.equals_exact(other, 0.5 * 10 ** (-decimal))


#
# Binary operations that return new geometries
#


def clip_by_rect(data, xmin, ymin, xmax, ymax):
    return shapely.clip_by_rect(data, xmin, ymin, xmax, ymax)


def difference(data, other):
    return shapely.difference(data, other)


def intersection(data, other):
    return shapely.intersection(data, other)


def symmetric_difference(data, other):
    return shapely.symmetric_difference(data, other)


def union(data, other):
    return shapely.union(data, other)


def shortest_line(data, other):
    return shapely.shortest_line(data, other)


#
# Other operations
#


def distance(data, other):
    return shapely.distance(data, other)


def hausdorff_distance(data, other, densify=None, **kwargs):
    return shapely.hausdorff_distance(data, other, densify=densify, **kwargs)


def frechet_distance(data, other, densify=None, **kwargs):
    return shapely.frechet_distance(data, other, densify=densify, **kwargs)


def buffer(data, distance, resolution=16, **kwargs):
    return shapely.buffer(data, distance, quad_segs=resolution, **kwargs)


def interpolate(data, distance, normalized=False):
    return shapely.line_interpolate_point(data, distance, normalized=normalized)


def simplify(data, tolerance, preserve_topology=True):
    return shapely.simplify(data, tolerance, preserve_topology=preserve_topology)


def normalize(data):
    return shapely.normalize(data)


def make_valid(data):
    return shapely.make_valid(data)


def reverse(data):
    return shapely.reverse(data)


def project(data, other, normalized=False):
    return shapely.line_locate_point(data, other, normalized=normalized)


def relate(data, other):
    return shapely.relate(data, other)


def unary_union(data):
    warning_msg = (
        "`unary_union` returned None due to all-None GeoSeries. In future, "
        "`unary_union` will return 'GEOMETRYCOLLECTION EMPTY' instead."
    )
    data = shapely.union_all(data)
    if data is None or data.is_empty:  # shapely 2.0a1 and 2.0
        warnings.warn(
            warning_msg,
            FutureWarning,
            stacklevel=4,
        )
        return None
    else:
        return data


#
# Coordinate related properties
#


def get_x(data):
    return shapely.get_x(data)


def get_y(data):
    return shapely.get_y(data)


def get_z(data):
    return shapely.get_z(data)


def bounds(data):
    return shapely.bounds(data)


#
# Coordinate transformation
#


def transform(data, func):
    has_z = shapely.has_z(data)

    result = np.empty_like(data)

    coords = shapely.get_coordinates(data[~has_z], include_z=False)
    new_coords_z = func(coords[:, 0], coords[:, 1])
    result[~has_z] = shapely.set_coordinates(
        data[~has_z].copy(), np.array(new_coords_z).T
    )

    coords_z = shapely.get_coordinates(data[has_z], include_z=True)
    new_coords_z = func(coords_z[:, 0], coords_z[:, 1], coords_z[:, 2])
    result[has_z] = shapely.set_coordinates(
        data[has_z].copy(), np.array(new_coords_z).T
    )

    return result
