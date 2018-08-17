import collections
import numbers
import operator

import numpy as np
import pandas as pd
from pandas.core import ops

import shapely
from shapely.geometry.base import BaseGeometry
from shapely.geometry.base import (
    GEOMETRY_TYPES as GEOMETRY_NAMES, CAP_STYLE, JOIN_STYLE)
from shapely.geometry import asShape

from six import PY3

try:
    from pandas.api.extensions import ExtensionArray, ExtensionDtype

    class GeometryDtype(ExtensionDtype):
        type = BaseGeometry
        name = 'geometry'
        na_value = None

        @classmethod
        def construct_from_string(cls, string):
            if string == cls.name:
                return cls()
            else:
                raise TypeError("Cannot construct a '{}' from "
                                "'{}'".format(cls, string))
        
        @classmethod
        def construct_array_type(cls):
            return GeometryArray

    # TODO expose registry in pandas.api.types
    from pandas.core.dtypes.dtypes import registry
    registry.register(GeometryDtype)

    _HAS_EXTENSION_ARRAY = True
except ImportError:
    ExtensionArray = object
    GeometryDtype = lambda: np.dtype('object')
    _HAS_EXTENSION_ARRAY = False


GEOMETRY_TYPES = [getattr(shapely.geometry, name) for name in GEOMETRY_NAMES]

opposite_predicates = {'contains': 'within',
                       'intersects': 'intersects',
                       'touches': 'touches',
                       'covers': 'covered_by',
                       'crosses': 'crosses',
                       'overlaps': 'overlaps'}

for k, v in list(opposite_predicates.items()):
    opposite_predicates[v] = k


def to_shapely(geoms):
    """ Convert array of pointers to an array of shapely objects """
    return vectorized.to_shapely(geoms)


def from_shapely(L):
    """
    Convert a list or array of shapely objects to a GeometryArray.

    Validates the elements.
    """
    n = len(L)

    out = []

    for idx in range(n):
        g = L[idx]
        if isinstance(g, BaseGeometry):
            out.append(g)
        elif hasattr(g, '__geo_interface__'):
            g = asShape(g)
            out.append(g)
        elif g is None or (isinstance(g, float) and np.isnan(g)):
            out.append(None)

    out = np.array(out, dtype=object)
    return GeometryArray(out)


def from_wkb(L):
    """ Convert a list or array of WKB objects to a GeometryArray """
    import shapely.wkb

    n = len(L)

    out = []

    for idx in range(n):
        g = L[idx]
        if g is not None:
            g = shapely.wkb.loads(g)
        out.append(g)

    out = np.array(out, dtype=object)
    return GeometryArray(out)


def from_wkt(L):
    """ Convert a list or array of WKT objects to a GeometryArray """
    import shapely.wkt

    n = len(L)

    out = []

    for idx in range(n):
        g = L[idx]
        if g is not None:
            g = shapely.wkt.loads(g)
        out.append(g)

    out = np.array(out, dtype=object)
    return GeometryArray(out)


def points_from_xy(x, y):
    """ Convert numpy arrays of x and y values to a GeometryArray of points """
    x = np.asarray(x, dtype='float64')
    y = np.asarray(y, dtype='float64')

    n = len(x)
    out = []

    for xi, yi in range(n):
        out.append(shapely.goemetry.Point(xi, yi))

    out = np.array(out, dtype=object)
    return GeometryArray(out)


class GeometryArray(ExtensionArray):
    dtype = GeometryDtype()

    def __init__(self, data):
        if isinstance(data, self.__class__):
            data = data.data
        elif not isinstance(data, np.ndarray):
            raise TypeError(
                "'data' should be array of geometry objects. Use from_shapely, "
                "from_wkb, from_wkt functions to construct a GeometryArray.")
        self.data = data

    def __getitem__(self, idx):
        if isinstance(idx, numbers.Integral):
            return self.data[idx]
        elif isinstance(idx, (collections.Iterable, slice)):
            return GeometryArray(self.data[idx])
        else:
            raise TypeError("Index type not supported", idx)

    def __setitem__(self, key, value):
        if isinstance(value, list):
            value = np.array(value, dtype=object)
        if isinstance(value, np.ndarray) or isinstance(value, GeometryArray):
            # TODO validate array
            self.data[key] = value
        elif isinstance(value, BaseGeometry) or value is None:
            # self.data[idx] = value
            if isinstance(key, np.ndarray):
                self.data[key] = np.array([value], dtype=object)
            else:
                self.data[key] = value
        else:
            raise TypeError("Value should be either a BaseGeometry or None, "
                            "got %s" % str(value))

    def __len__(self):
        return len(self.data)

    @property
    def size(self):
        return len(self.data)

    @property
    def ndim(self):
        return 1

    def copy(self, *args, **kwargs):
        return GeometryArray(self.data.copy())

    def take(self, idx, allow_fill=False, fill_value=None):

        if _HAS_EXTENSION_ARRAY:
            from pandas.api.extensions import take

            if allow_fill:
                if fill_value is None: # or pd.isna(fill_value):
                    fill_value = 0

            result = take(self.data, idx, allow_fill=allow_fill,
                          fill_value=fill_value)
            if fill_value == 0:
                result[result == 0] = None
            return GeometryArray(result)
        else:
            if allow_fill:
                # take on empty array
                if not len(self):
                    # only valid if result is an all-missing array
                    if (np.asarray(idx) == -1).all():
                        return GeometryArray(
                            np.array([None]*len(idx), dtype=object))
                    else:
                        raise IndexError(
                            "cannot do a non-empty take from an empty array.")

                result = self[idx]
                result.data[idx == -1] = None
                return result
            else:
                return self[idx]

    def _fill(self, idx, value):
        """ Fill index locations with value

        Value should be a BaseGeometry

        Returns a copy
        """
        if not (isinstance(value, BaseGeometry) or value is None):
            raise TypeError("Value should be either a BaseGeometry or None, "
                            "got %s" % str(value))
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
            raise NotImplementedError(
                "fillna with a method is not yet supported")
        elif not isinstance(value, BaseGeometry):
            raise NotImplementedError(
                "fillna currently only supports filling with a scalar "
                "geometry")

        mask = self.isna()
        new_values = self.copy()

        if mask.any():
            # fill with value
            new_values = new_values._fill(mask, value)

        return new_values

    # def __getstate__(self):
    #     return vectorized.serialize(self.data)

    # def __setstate__(self, state):
    #     geoms = vectorized.deserialize(*state)
    #     self.data = geoms
    #     self.base = None

    # -------------------------------------------------------------------------
    # Geometry related methods
    # -------------------------------------------------------------------------

    def _binary_geo(self, other, op):
        """ Apply geometry-valued operation

        Supports:

        -   difference
        -   symmetric_difference
        -   intersection
        -   union

        Parameters
        ----------
        other: GeometryArray or single shapely BaseGeoemtry
        op: string
        """
        if isinstance(other, BaseGeometry):
            return GeometryArray(vectorized.binary_geo(op, self.data, other))
        elif isinstance(other, GeometryArray):
            if len(self) != len(other):
                msg = ("Lengths of inputs to not match.  Left: %d, Right: %d" %
                       (len(self), len(other)))
                raise ValueError(msg)
            return GeometryArray(
                vectorized.vector_binary_geo(op, self.data, other.data))
        else:
            raise NotImplementedError("type not known %s" % type(other))

    def _binop_predicate(self, other, op, extra=None):
        """ Apply boolean-valued operation

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
        other: GeometryArray or single shapely BaseGeoemtry
        op: string
        """
        if isinstance(other, BaseGeometry):
            if extra is not None:
                return vectorized.binary_predicate_with_arg(
                    op, self.data, other, extra)
            elif op in opposite_predicates:
                op2 = opposite_predicates[op]
                return vectorized.prepared_binary_predicate(
                    op2, self.data, other)
            else:
                return vectorized.binary_predicate(op, self.data, other)
        elif isinstance(other, GeometryArray):
            if len(self) != len(other):
                msg = ("Shapes of inputs to not match.  Left: %d, Right: %d" %
                       (len(self), len(other)))
                raise ValueError(msg)
            if extra is not None:
                return vectorized.vector_binary_predicate_with_arg(
                    op, self.data, other.data, extra)
            else:
                return vectorized.vector_binary_predicate(
                    op, self.data, other.data)
        else:
            raise NotImplementedError("type not known %s" % type(other))

    def covers(self, other):
        return self._binop_predicate(other, 'covers')

    def contains(self, other):
        return self._binop_predicate(other, 'contains')

    def crosses(self, other):
        return self._binop_predicate(other, 'crosses')

    def disjoint(self, other):
        return self._binop_predicate(other, 'disjoint')

    def equals(self, other):
        return self._binop_predicate(other, 'equals')

    def intersects(self, other):
        return self._binop_predicate(other, 'intersects')

    def overlaps(self, other):
        return self._binop_predicate(other, 'overlaps')

    def touches(self, other):
        return self._binop_predicate(other, 'touches')

    def within(self, other):
        return self._binop_predicate(other, 'within')

    def equals_exact(self, other, tolerance):
        return self._binop_predicate(other, 'equals_exact', tolerance)

    def is_valid(self):
        return vectorized.unary_predicate('is_valid', self.data)

    def is_empty(self):
        return vectorized.unary_predicate('is_empty', self.data)

    def is_simple(self):
        return vectorized.unary_predicate('is_simple', self.data)

    def is_ring(self):
        return vectorized.unary_predicate('is_ring', self.data)

    def has_z(self):
        return vectorized.unary_predicate('has_z', self.data)

    def is_closed(self):
        return vectorized.unary_predicate('is_closed', self.data)

    def _geo_unary_op(self, op):
        return GeometryArray(vectorized.geo_unary_op(op, self.data))

    def boundary(self):
        return self._geo_unary_op('boundary')

    def centroid(self):
        return self._geo_unary_op('centroid')

    def convex_hull(self):
        return self._geo_unary_op('convex_hull')

    def envelope(self):
        return self._geo_unary_op('envelope')

    def exterior(self):
        out = self._geo_unary_op('exterior')
        out.base = self  # exterior shares data with self
        return out

    def representative_point(self):
        return self._geo_unary_op('representative_point')

    def distance(self, other):
        if isinstance(other, GeometryArray):
            return vectorized.binary_vector_float(
                'distance', self.data, other.data)
        else:
            return vectorized.binary_float('distance', self.data, other)

    def project(self, other, normalized=False):
        op = 'project' if not normalized else 'project-normalized'
        if isinstance(other, GeometryArray):
            return vectorized.binary_vector_float_return(
                op, self.data, other.data)
        else:
            return vectorized.binary_float_return(op, self.data, other)

    def area(self):
        return vectorized.unary_vector_float('area', self.data)

    def length(self):
        return vectorized.unary_vector_float('length', self.data)

    def difference(self, other):
        return self._binary_geo(other, 'difference')

    def symmetric_difference(self, other):
        return self._binary_geo(other, 'symmetric_difference')

    def union(self, other):
        return self._binary_geo(other, 'union')

    def intersection(self, other):
        return self._binary_geo(other, 'intersection')

    def buffer(self, distance, resolution=16, cap_style=CAP_STYLE.round,
               join_style=JOIN_STYLE.round, mitre_limit=5.0):
        """ Buffer operation on array of GEOSGeometry objects """
        return GeometryArray(
            vectorized.buffer(self.data, distance, resolution, cap_style,
                              join_style, mitre_limit))

    def geom_type(self):
        """
        Types of the underlying Geometries

        Returns
        -------
        Pandas categorical with types for each geometry
        """
        x = vectorized.geom_type(self.data)

        import pandas as pd
        return pd.Categorical.from_codes(x, GEOMETRY_NAMES)

    def unary_union(self):
        """ Unary union.

        Returns a single shapely geometry
        """
        return vectorized.unary_union(self.data)

    def coords(self):
        return vectorized.coords(self.data)

    # -------------------------------------------------------------------------
    # for Series/ndarray like compat
    # -------------------------------------------------------------------------

    @property
    def shape(self):
        """ Shape of the ...

        For internal compatibility with numpy arrays.

        Returns
        -------
        shape : tuple
        """
        return tuple([len(self)])

    def to_dense(self):
        """Return my 'dense' representation

        For internal compatibility with numpy arrays.

        Returns
        -------
        dense : array
        """
        return self.data

    def isna(self):
        """
        Boolean NumPy array indicating if each value is missing
        """
        return np.array([g is None for g in self.data], dtype='bool')

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

    # ExtensionArray specific

    @classmethod
    def _from_sequence(cls, scalars, dtype=None, copy=False):
        """Construct a new ExtensionArray from a sequence of scalars.

        Parameters
        ----------
        scalars : Sequence
            Each element will be an instance of the scalar type for this
            array, ``cls.dtype.type``.

        Returns
        -------
        ExtensionArray
        """
        return from_shapely(scalars)

    @classmethod
    def _from_factorized(cls, values, original):
        """Reconstruct an ExtensionArray after factorization.

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
        # type: () -> ndarray
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

    def _values_for_factorize(self):
        # type: () -> Tuple[ndarray, Any]
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
        vals = np.array([getattr(x, 'wkb', None) for x in self], dtype=object)
        return vals, None

    @classmethod
    def _concat_same_type(cls, to_concat):
        """
        Concatenate list of single blocks of the same type.
        """
        L = list(to_concat)
        data = np.concatenate([ga.data for ga in L])
        return GeometryArray(data)

    def __array__(self, dtype=None):
        """
        The numpy array interface.

        Returns
        -------
        values : numpy array
        """
        return self.data
