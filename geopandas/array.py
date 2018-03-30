import collections
import numbers

import shapely
from shapely.geometry.base import BaseGeometry
from shapely.geometry.base import (
    GEOMETRY_TYPES as GEOMETRY_NAMES, CAP_STYLE, JOIN_STYLE)

import numpy as np

from . import vectorized


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
    """ Convert a list or array of shapely objects to a GeometryArray """
    out = vectorized.from_shapely(L)
    return GeometryArray(out)


def from_wkb(L):
    """ Convert a list or array of WKB objects to a GeometryArray """
    out = vectorized.from_wkb(L)
    return GeometryArray(out)


def from_wkt(L):
    """ Convert a list or array of WKT objects to a GeometryArray """
    out = vectorized.from_wkt(L)
    return GeometryArray(out)


def points_from_xy(x, y):
    """ Convert numpy arrays of x and y values to a GeometryArray of points """
    out = vectorized.points_from_xy(x, y)
    return GeometryArray(out)


class GeometryArray(object):
    dtype = np.dtype('O')

    def __init__(self, data, base=False):
        self.data = data
        self.base = base

    def __getitem__(self, idx):
        if isinstance(idx, numbers.Integral):
            return vectorized.get_element(self.data, idx)
        elif isinstance(idx, (collections.Iterable, slice)):
            return GeometryArray(self.data[idx], base=self)
        else:
            raise TypeError("Index type not supported", idx)

    def take(self, idx):
        result = self[idx]
        result.data[idx == -1] = 0
        return result

    def fill(self, idx, value):
        """ Fill index locations with value

        Value should be a BaseGeometry

        Returns a copy
        """
        base = [self]
        if isinstance(value, BaseGeometry):
            base.append(value)
            value = value.__geom__
        elif value is None:
            value = 0
        else:
            raise TypeError("Value should be either a BaseGeometry or None, "
                            "got %s" % str(value))
        new = GeometryArray(self.data.copy(), base=base)
        new.data[idx] = value
        return new

    def fillna(self, value=None):
        return self.fill(self.data == 0, value)

    def __len__(self):
        return len(self.data)

    @property
    def size(self):
        return len(self.data)

    def __del__(self):
        if self.base is False:
            try:
                vectorized.vec_free(self.data)
            except (TypeError, AttributeError):
                # the vectorized module can already be removed, therefore
                # ignoring such an error to not output this as a warning
                pass

    def copy(self):
        return self  # assume immutable for now

    @property
    def ndim(self):
        return 1

    def __getstate__(self):
        return vectorized.serialize(self.data)

    def __setstate__(self, state):
        geoms = vectorized.deserialize(*state)
        self.data = geoms
        self.base = None

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

    # for Series/ndarray like compat

    @property
    def shape(self):
        """ Shape of the ...

        For internal compatibility with numpy arrays.

        Returns
        -------
        shape : tuple
        """

        return tuple([len(self)])

    def ravel(self, order='C'):
        """ Return a flattened (numpy) array.

        For internal compatibility with numpy arrays.

        Returns
        -------
        raveled : numpy array
        """
        return np.array(self)

    def view(self):
        """Return a view of myself.

        For internal compatibility with numpy arrays.

        Returns
        -------
        view : Categorical
           Returns `self`!
        """
        return self

    def to_dense(self):
        """Return my 'dense' representation

        For internal compatibility with numpy arrays.

        Returns
        -------
        dense : array
        """
        return to_shapely(self.data)

    def get_values(self):
        """ Return the values.

        For internal compatibility with pandas formatting.

        Returns
        -------
        values : numpy array
            A numpy array of the same dtype as categorical.categories.dtype or
            Index if datetime / periods
        """
        # if we are a datetime and period index, return Index to keep metadata
        return to_shapely(self.data)

    def tolist(self):
        """
        Return the array as a list of geometries
        """
        return self.to_dense().tolist()

    def __array__(self, dtype=None):
        """
        The numpy array interface.

        Returns
        -------
        values : numpy array
            A numpy array of either the specified dtype or,
            if dtype==None (default), the same dtype as
            categorical.categories.dtype
        """
        return to_shapely(self.data)

    def unary_union(self):
        """ Unary union.

        Returns a single shapely geometry
        """
        return vectorized.unary_union(self.data)

    def coords(self):
        return vectorized.coords(self.data)
