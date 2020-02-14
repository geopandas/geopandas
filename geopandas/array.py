import numbers
import operator
import warnings

import numpy as np
import pandas as pd
from pandas.api.extensions import ExtensionArray, ExtensionDtype

import shapely
import shapely.affinity
import shapely.geometry
from shapely.geometry.base import BaseGeometry
import shapely.ops
import shapely.wkt

from collections.abc import Iterable

from ._compat import PANDAS_GE_024, PANDAS_GE_10


class GeometryDtype(ExtensionDtype):
    type = BaseGeometry
    name = "geometry"
    na_value = np.nan

    @classmethod
    def construct_from_string(cls, string):
        if not isinstance(string, str):
            raise TypeError(
                "'construct_from_string' expects a string, got {}".format(type(string))
            )
        elif string == cls.name:
            return cls()
        else:
            raise TypeError(
                "Cannot construct a '{}' from '{}'".format(cls.__name__, string)
            )

    @classmethod
    def construct_array_type(cls):
        return GeometryArray


if PANDAS_GE_024:
    from pandas.api.extensions import register_extension_dtype

    register_extension_dtype(GeometryDtype)


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
        elif hasattr(geom, "__geo_interface__"):
            geom = shapely.geometry.asShape(geom)
            out.append(geom)
        elif _isna(geom):
            out.append(None)
        else:
            raise TypeError("Input must be valid geometry objects: {0}".format(geom))

    aout = np.empty(n, dtype=object)
    aout[:] = out
    return GeometryArray(aout)


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

    aout = np.empty(n, dtype=object)
    aout[:] = out
    return GeometryArray(aout)


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
                geom = geom.decode("utf-8")
            geom = shapely.wkt.loads(geom)
        else:
            geom = None
        out.append(geom)

    aout = np.empty(n, dtype=object)
    aout[:] = out
    return GeometryArray(aout)


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
    x = np.asarray(x, dtype="float64")
    y = np.asarray(y, dtype="float64")
    if z is not None:
        z = np.asarray(z, dtype="float64")
    out = _points_from_xy(x, y, z)
    aout = np.empty(len(x), dtype=object)
    aout[:] = out
    return GeometryArray(aout)


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
        data[:] = [
            getattr(s, op)(right) if s is not None and right is not None else None
            for s in left.data
        ]
        return GeometryArray(data)
    elif isinstance(right, GeometryArray):
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
            for this_elem, other_elem in zip(left.data, right.data)
        ]
        return GeometryArray(data)
    else:
        raise TypeError("Type not known: {0} vs {1}".format(type(left), type(right)))


def _binary_predicate(op, left, right, *args, **kwargs):
    # type: (str, GeometryArray, GeometryArray/BaseGeometry, args/kwargs)
    #        -> array[bool]
    """Binary operation on GeometryArray that returns a boolean ndarray

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
    right: GeometryArray or single shapely BaseGeoemtry
    """
    # empty geometries are handled by shapely (all give False except disjoint)
    if isinstance(right, BaseGeometry):
        data = [
            getattr(s, op)(right, *args, **kwargs) if s is not None else False
            for s in left.data
        ]
        return np.array(data, dtype=bool)
    elif isinstance(right, GeometryArray):
        if len(left) != len(right):
            msg = "Lengths of inputs do not match. Left: {0}, Right: {1}".format(
                len(left), len(right)
            )
            raise ValueError(msg)
        data = [
            getattr(this_elem, op)(other_elem, *args, **kwargs)
            if not (this_elem is None or other_elem is None)
            else False
            for this_elem, other_elem in zip(left.data, right.data)
        ]
        return np.array(data, dtype=bool)
    else:
        raise TypeError("Type not known: {0} vs {1}".format(type(left), type(right)))


def _binary_op_float(op, left, right, *args, **kwargs):
    # type: (str, GeometryArray, GeometryArray/BaseGeometry, args/kwargs)
    #        -> array
    """Binary operation on GeometryArray that returns a ndarray"""
    # used for distance -> check for empty as we want to return np.nan instead 0.0
    # as shapely does currently (https://github.com/Toblerity/Shapely/issues/498)
    if isinstance(right, BaseGeometry):
        data = [
            getattr(s, op)(right, *args, **kwargs)
            if not (s is None or s.is_empty or right.is_empty)
            else np.nan
            for s in left.data
        ]
        return np.array(data, dtype=float)
    elif isinstance(right, GeometryArray):
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
            for this_elem, other_elem in zip(left.data, right.data)
        ]
        return np.array(data, dtype=float)
    else:
        raise TypeError("Type not known: {0} vs {1}".format(type(left), type(right)))


def _binary_op(op, left, right, *args, **kwargs):
    # type: (str, GeometryArray, GeometryArray/BaseGeometry, args/kwargs)
    #        -> array
    """Binary operation on GeometryArray that returns a ndarray"""
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
            for s in left.data
        ]
        return np.array(data, dtype=dtype)
    elif isinstance(right, GeometryArray):
        if len(left) != len(right):
            msg = "Lengths of inputs do not match. Left: {0}, Right: {1}".format(
                len(left), len(right)
            )
            raise ValueError(msg)
        data = [
            getattr(this_elem, op)(other_elem, *args, **kwargs)
            if not (this_elem is None or other_elem is None)
            else null_value
            for this_elem, other_elem in zip(left.data, right.data)
        ]
        return np.array(data, dtype=dtype)
    else:
        raise TypeError("Type not known: {0} vs {1}".format(type(left), type(right)))


def _unary_geo(op, left, *args, **kwargs):
    # type: (str, GeometryArray) -> GeometryArray
    """Unary operation that returns new geometries"""
    # ensure 1D output, see note above
    data = np.empty(len(left), dtype=object)
    data[:] = [getattr(geom, op, None) for geom in left.data]
    return GeometryArray(data)


def _unary_op(op, left, null_value=False):
    # type: (str, GeometryArray, Any) -> np.array
    """Unary operation that returns a Series"""
    data = [getattr(geom, op, null_value) for geom in left.data]
    return np.array(data, dtype=np.dtype(type(null_value)))


def _affinity_method(op, left, *args, **kwargs):
    # type: (str, GeometryArray, ...) -> GeometryArray

    # not all shapely.affinity methods can handle empty geometries:
    # affine_transform itself works (as well as translate), but rotate, scale
    # and skew fail (they try to unpack the bounds).
    # Here: consistently returning empty geom for input empty geom
    out = []
    for geom in left.data:
        if geom is None or geom.is_empty:
            res = geom
        else:
            res = getattr(shapely.affinity, op)(geom, *args, **kwargs)
        out.append(res)
    data = np.empty(len(left), dtype=object)
    data[:] = out
    return GeometryArray(data)


class GeometryArray(ExtensionArray):
    """
    Class wrapping a numpy array of Shapely objects and
    holding the array-based implementations.
    """

    _dtype = GeometryDtype()

    def __init__(self, data):
        if isinstance(data, self.__class__):
            data = data.data
        elif not isinstance(data, np.ndarray):
            raise TypeError(
                "'data' should be array of geometry objects. Use from_shapely, "
                "from_wkb, from_wkt functions to construct a GeometryArray."
            )
        elif not data.ndim == 1:
            raise ValueError(
                "'data' should be a 1-dimensional array of geometry objects."
            )
        self.data = data

    @property
    def dtype(self):
        return self._dtype

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, idx):
        if isinstance(idx, numbers.Integral):
            return self.data[idx]
        # array-like, slice
        if PANDAS_GE_10:
            # for pandas >= 1.0, validate and convert IntegerArray/BooleanArray
            # to numpy array, pass-through non-array-like indexers
            idx = pd.api.indexers.check_array_indexer(self, idx)
        if isinstance(idx, (Iterable, slice)):
            return GeometryArray(self.data[idx])
        else:
            raise TypeError("Index type not supported", idx)

    def __setitem__(self, key, value):
        if PANDAS_GE_10:
            # for pandas >= 1.0, validate and convert IntegerArray/BooleanArray
            # keys to numpy array, pass-through non-array-like indexers
            key = pd.api.indexers.check_array_indexer(self, key)
        if isinstance(value, pd.Series):
            value = value.values
        if isinstance(value, (list, np.ndarray)):
            value = from_shapely(value)
        if isinstance(value, GeometryArray):
            if isinstance(key, numbers.Integral):
                raise ValueError("cannot set a single element with an array")
            self.data[key] = value.data
        elif isinstance(value, BaseGeometry) or _isna(value):
            if _isna(value):
                # internally only use None as missing value indicator
                # but accept others
                value = None
            if isinstance(key, (slice, list, np.ndarray)):
                value_array = np.empty(1, dtype=object)
                value_array[:] = [value]
                self.data[key] = value_array
            else:
                self.data[key] = value
        else:
            raise TypeError(
                "Value should be either a BaseGeometry or None, got %s" % str(value)
            )

    # -------------------------------------------------------------------------
    # Geometry related methods
    # -------------------------------------------------------------------------

    @property
    def is_valid(self):
        return _unary_op("is_valid", self, null_value=False)

    @property
    def is_empty(self):
        return _unary_op("is_empty", self, null_value=False)

    @property
    def is_simple(self):
        return _unary_op("is_simple", self, null_value=False)

    @property
    def is_ring(self):
        # operates on the exterior, so can't use _unary_op()
        return np.array(
            [
                geom.exterior.is_ring
                if geom is not None and geom.exterior is not None
                else False
                for geom in self.data
            ],
            dtype=bool,
        )

    @property
    def is_closed(self):
        return _unary_op("is_closed", self, null_value=False)

    @property
    def has_z(self):
        return _unary_op("has_z", self, null_value=False)

    @property
    def geom_type(self):
        return _unary_op("geom_type", self, null_value=None)

    @property
    def area(self):
        return _unary_op("area", self, null_value=np.nan)

    @property
    def length(self):
        return _unary_op("length", self, null_value=np.nan)

    #
    # Unary operations that return new geometries
    #

    @property
    def boundary(self):
        return _unary_geo("boundary", self)

    @property
    def centroid(self):
        return _unary_geo("centroid", self)

    @property
    def convex_hull(self):
        return _unary_geo("convex_hull", self)

    @property
    def envelope(self):
        return _unary_geo("envelope", self)

    @property
    def exterior(self):
        return _unary_geo("exterior", self)

    @property
    def interiors(self):
        has_non_poly = False
        inner_rings = []
        for geom in self.data:
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
        data = np.empty(len(self), dtype=object)
        data[:] = inner_rings
        return data

    def representative_point(self):
        # method and not a property -> can't use _unary_geo
        data = np.empty(len(self), dtype=object)
        data[:] = [
            geom.representative_point() if geom is not None else None
            for geom in self.data
        ]
        return GeometryArray(data)

    #
    # Binary predicates
    #

    def covers(self, other):
        return _binary_predicate("covers", self, other)

    def contains(self, other):
        return _binary_predicate("contains", self, other)

    def crosses(self, other):
        return _binary_predicate("crosses", self, other)

    def disjoint(self, other):
        return _binary_predicate("disjoint", self, other)

    def equals(self, other):
        return _binary_predicate("equals", self, other)

    def intersects(self, other):
        return _binary_predicate("intersects", self, other)

    def overlaps(self, other):
        return _binary_predicate("overlaps", self, other)

    def touches(self, other):
        return _binary_predicate("touches", self, other)

    def within(self, other):
        return _binary_predicate("within", self, other)

    def equals_exact(self, other, tolerance):
        return _binary_predicate("equals_exact", self, other, tolerance=tolerance)

    def almost_equals(self, other, decimal):
        return _binary_predicate("almost_equals", self, other, decimal=decimal)

    #
    # Binary operations that return new geometries
    #

    def difference(self, other):
        return _binary_geo("difference", self, other)

    def intersection(self, other):
        return _binary_geo("intersection", self, other)

    def symmetric_difference(self, other):
        return _binary_geo("symmetric_difference", self, other)

    def union(self, other):
        return _binary_geo("union", self, other)

    #
    # Other operations
    #

    def distance(self, other):
        return _binary_op_float("distance", self, other)

    def buffer(self, distance, resolution=16, **kwargs):
        data = data = np.empty(len(self), dtype=object)
        if isinstance(distance, np.ndarray):
            if len(distance) != len(self):
                raise ValueError(
                    "Length of distance sequence does not match "
                    "length of the GeoSeries"
                )

            data[:] = [
                geom.buffer(dist, resolution, **kwargs) if geom is not None else None
                for geom, dist in zip(self.data, distance)
            ]
            return GeometryArray(data)

        data[:] = [
            geom.buffer(distance, resolution, **kwargs) if geom is not None else None
            for geom in self.data
        ]
        return GeometryArray(data)

    def interpolate(self, distance, normalized=False):
        data = data = np.empty(len(self), dtype=object)
        if isinstance(distance, np.ndarray):
            if len(distance) != len(self):
                raise ValueError(
                    "Length of distance sequence does not match "
                    "length of the GeoSeries"
                )
            data[:] = [
                geom.interpolate(dist, normalized=normalized)
                for geom, dist in zip(self.data, distance)
            ]
            return GeometryArray(data)

        data[:] = [
            geom.interpolate(distance, normalized=normalized) for geom in self.data
        ]
        return GeometryArray(data)

    def simplify(self, *args, **kwargs):
        # method and not a property -> can't use _unary_geo
        data = np.empty(len(self), dtype=object)
        data[:] = [geom.simplify(*args, **kwargs) for geom in self.data]
        return GeometryArray(data)

    def project(self, other, normalized=False):
        return _binary_op("project", self, other, normalized=normalized)

    def relate(self, other):
        return _binary_op("relate", self, other)

    #
    # Reduction operations that return a Shapely geometry
    #

    def unary_union(self):
        return shapely.ops.unary_union(self.data)

    #
    # Affinity operations
    #

    def affine_transform(self, matrix):
        return _affinity_method("affine_transform", self, matrix)

    def translate(self, xoff=0.0, yoff=0.0, zoff=0.0):
        return _affinity_method("translate", self, xoff, yoff, zoff)

    def rotate(self, angle, origin="center", use_radians=False):
        return _affinity_method(
            "rotate", self, angle, origin=origin, use_radians=use_radians
        )

    def scale(self, xfact=1.0, yfact=1.0, zfact=1.0, origin="center"):
        return _affinity_method("scale", self, xfact, yfact, zfact, origin=origin)

    def skew(self, xs=0.0, ys=0.0, origin="center", use_radians=False):
        return _affinity_method(
            "skew", self, xs, ys, origin=origin, use_radians=use_radians
        )

    #
    # Coordinate related properties
    #

    @property
    def x(self):
        """Return the x location of point geometries in a GeoSeries"""
        if (self.geom_type[~self.isna()] == "Point").all():
            return _unary_op("x", self, null_value=np.nan)
        else:
            message = "x attribute access only provided for Point geometries"
            raise ValueError(message)

    @property
    def y(self):
        """Return the y location of point geometries in a GeoSeries"""
        if (self.geom_type[~self.isna()] == "Point").all():
            return _unary_op("y", self, null_value=np.nan)
        else:
            message = "y attribute access only provided for Point geometries"
            raise ValueError(message)

    @property
    def bounds(self):
        # ensure that for empty arrays, the result has the correct shape
        if len(self) == 0:
            return np.empty((0, 4), dtype="float64")
        # need to explicitly check for empty (in addition to missing) geometries,
        # as those return an empty tuple, not resulting in a 2D array
        bounds = np.array(
            [
                geom.bounds
                if not (geom is None or geom.is_empty)
                else (np.nan, np.nan, np.nan, np.nan)
                for geom in self.data
            ]
        )
        return bounds

    @property
    def total_bounds(self):
        if len(self) == 0:
            # numpy 'min' cannot handle empty arrays
            # TODO with numpy >= 1.15, the 'initial' argument can be used
            return np.array([np.nan, np.nan, np.nan, np.nan])
        b = self.bounds
        return np.array(
            (
                b[:, 0].min(),  # minx
                b[:, 1].min(),  # miny
                b[:, 2].max(),  # maxx
                b[:, 3].max(),
            )
        )  # maxy

    # -------------------------------------------------------------------------
    # general array like compat
    # -------------------------------------------------------------------------

    @property
    def size(self):
        return self.data.size

    @property
    def shape(self):
        return (self.size,)

    @property
    def ndim(self):
        return len(self.shape)

    def copy(self, *args, **kwargs):
        # still taking args/kwargs for compat with pandas 0.24
        return GeometryArray(self.data.copy())

    def take(self, indices, allow_fill=False, fill_value=None):
        from pandas.api.extensions import take

        if allow_fill:
            if fill_value is None or pd.isna(fill_value):
                fill_value = 0

        result = take(self.data, indices, allow_fill=allow_fill, fill_value=fill_value)
        if fill_value == 0:
            result[result == 0] = None
        return GeometryArray(result)

    def _fill(self, idx, value):
        """ Fill index locations with value

        Value should be a BaseGeometry
        """
        if not (isinstance(value, BaseGeometry) or value is None):
            raise TypeError(
                "Value should be either a BaseGeometry or None, got %s" % str(value)
            )
        # self.data[idx] = value
        self.data[idx] = np.array([value], dtype=object)
        return self

    def fillna(self, value=None, method=None, limit=None):
        """ Fill NA/NaN values using the specified method.

        Parameters
        ----------
        value : scalar, array-like
            If a scalar value is passed it is used to fill all missing values.
            Alternatively, an array-like 'value' can be given. It's expected
            that the array-like have the same length as 'self'.
        method : {'backfill', 'bfill', 'pad', 'ffill', None}, default None
            Method to use for filling holes in reindexed Series
            pad / ffill: propagate last valid observation forward to next valid
            backfill / bfill: use NEXT valid observation to fill gap
        limit : int, default None
            If method is specified, this is the maximum number of consecutive
            NaN values to forward/backward fill. In other words, if there is
            a gap with more than this number of consecutive NaNs, it will only
            be partially filled. If method is not specified, this is the
            maximum number of entries along the entire axis where NaNs will be
            filled.

        Returns
        -------
        filled : ExtensionArray with NA/NaN filled
        """
        if method is not None:
            raise NotImplementedError("fillna with a method is not yet supported")

        mask = self.isna()
        new_values = self.copy()

        if mask.any():
            # fill with value
            if _isna(value):
                value = None
            elif not isinstance(value, BaseGeometry):
                raise NotImplementedError(
                    "fillna currently only supports filling with a scalar geometry"
                )
            new_values = new_values._fill(mask, value)

        return new_values

    def astype(self, dtype, copy=True):
        """
        Cast to a NumPy array with 'dtype'.

        Parameters
        ----------
        dtype : str or dtype
            Typecode or data-type to which the array is cast.
        copy : bool, default True
            Whether to copy the data, even if not necessary. If False,
            a copy is made only if the old dtype does not match the
            new dtype.

        Returns
        -------
        array : ndarray
            NumPy ndarray with 'dtype' for its dtype.
        """
        if isinstance(dtype, GeometryDtype):
            if copy:
                return self.copy()
            else:
                return self
        elif pd.api.types.is_string_dtype(dtype) and not pd.api.types.is_object_dtype(
            dtype
        ):
            return to_wkt(self).astype(dtype, copy=False)
        else:
            return np.array(self, dtype=dtype, copy=copy)

    def isna(self):
        """
        Boolean NumPy array indicating if each value is missing
        """
        return np.array([g is None for g in self.data], dtype="bool")

    def unique(self):
        """Compute the ExtensionArray of unique values.

        Returns
        -------
        uniques : ExtensionArray
        """
        from pandas import factorize

        _, uniques = factorize(self)
        return uniques

    @property
    def nbytes(self):
        return self.data.nbytes

    # -------------------------------------------------------------------------
    # ExtensionArray specific
    # -------------------------------------------------------------------------

    @classmethod
    def _from_sequence(cls, scalars, dtype=None, copy=False):
        """
        Construct a new ExtensionArray from a sequence of scalars.

        Parameters
        ----------
        scalars : Sequence
            Each element will be an instance of the scalar type for this
            array, ``cls.dtype.type``.
        dtype : dtype, optional
            Construct for this particular dtype. This should be a Dtype
            compatible with the ExtensionArray.
        copy : boolean, default False
            If True, copy the underlying data.

        Returns
        -------
        ExtensionArray
        """
        return from_shapely(scalars)

    def _values_for_factorize(self):
        # type: () -> Tuple[np.ndarray, Any]
        """Return an array and missing value suitable for factorization.

        Returns
        -------
        values : ndarray
            An array suitable for factoraization. This should maintain order
            and be a supported dtype (Float64, Int64, UInt64, String, Object).
            By default, the extension array is cast to object dtype.
        na_value : object
            The value in `values` to consider missing. This will be treated
            as NA in the factorization routines, so it will be coded as
            `na_sentinal` and not included in `uniques`. By default,
            ``np.nan`` is used.
        """
        vals = to_wkb(self)
        return vals, None

    @classmethod
    def _from_factorized(cls, values, original):
        """
        Reconstruct an ExtensionArray after factorization.

        Parameters
        ----------
        values : ndarray
            An integer ndarray with the factorized values.
        original : ExtensionArray
            The original ExtensionArray that factorize was called on.

        See Also
        --------
        pandas.factorize
        ExtensionArray.factorize
        """
        return from_wkb(values)

    def _values_for_argsort(self):
        # type: () -> np.ndarray
        """Return values for sorting.

        Returns
        -------
        ndarray
            The transformed values should maintain the ordering between values
            within the array.

        See Also
        --------
        ExtensionArray.argsort
        """
        # Note: this is used in `ExtensionArray.argsort`.
        raise TypeError("geometries are not orderable")

    def _formatter(self, boxed=False):
        """Formatting function for scalar values.

        This is used in the default '__repr__'. The returned formatting
        function receives instances of your scalar type.

        Parameters
        ----------
        boxed: bool, default False
            An indicated for whether or not your array is being printed
            within a Series, DataFrame, or Index (True), or just by
            itself (False). This may be useful if you want scalar values
            to appear differently within a Series versus on its own (e.g.
            quoted or not).

        Returns
        -------
        Callable[[Any], str]
            A callable that gets instances of the scalar type and
            returns a string. By default, :func:`repr` is used
            when ``boxed=False`` and :func:`str` is used when
            ``boxed=True``.
        """
        if boxed:
            import geopandas

            precision = geopandas.options.display_precision
            if precision is None:
                # dummy heuristic based on 10 first geometries that should
                # work in most cases
                xmin, ymin, xmax, ymax = self[~self.isna()][:10].total_bounds
                if (
                    (-180 <= xmin <= 180)
                    and (-180 <= xmax <= 180)
                    and (-90 <= ymin <= 90)
                    and (-90 <= ymax <= 90)
                ):
                    # geographic coordinates
                    precision = 5
                else:
                    # typically projected coordinates
                    # (in case of unit meter: mm precision)
                    precision = 3
            return lambda geom: shapely.wkt.dumps(geom, rounding_precision=precision)
        return repr

    @classmethod
    def _concat_same_type(cls, to_concat):
        """
        Concatenate multiple array

        Parameters
        ----------
        to_concat : sequence of this type

        Returns
        -------
        ExtensionArray
        """
        data = np.concatenate([ga.data for ga in to_concat])
        return GeometryArray(data)

    def _reduce(self, name, skipna=True, **kwargs):
        # including the base class version here (that raises by default)
        # because this was not yet defined in pandas 0.23
        if name == "any" or name == "all":
            return getattr(self.data, name)()
        raise TypeError(
            "cannot perform {name} with type {dtype}".format(
                name=name, dtype=self.dtype
            )
        )

    def __array__(self, dtype=None):
        """
        The numpy array interface.

        Returns
        -------
        values : numpy array
        """
        return self.data

    def _binop(self, other, op):
        def convert_values(param):
            if isinstance(param, ExtensionArray) or pd.api.types.is_list_like(param):
                ovalues = param
            else:  # Assume its an object
                ovalues = [param] * len(self)
            return ovalues

        if isinstance(other, (pd.Series, pd.Index)):
            # rely on pandas to unbox and dispatch to us
            return NotImplemented

        lvalues = self
        rvalues = convert_values(other)

        if len(lvalues) != len(rvalues):
            raise ValueError("Lengths must match to compare")

        # If the operator is not defined for the underlying objects,
        # a TypeError should be raised
        res = [op(a, b) for (a, b) in zip(lvalues, rvalues)]

        res = np.asarray(res)
        return res

    def __eq__(self, other):
        return self._binop(other, operator.eq)

    def __ne__(self, other):
        return self._binop(other, operator.ne)
