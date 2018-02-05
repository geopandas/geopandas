# distutils: sources = geopandas/algos.c

import collections
import numbers

import shapely
from shapely.geometry import MultiPoint, MultiLineString, MultiPolygon
from shapely.geometry.base import BaseGeometry, geom_factory

import cython

cimport numpy as np
import numpy as np

include "geopandas/_geos.pxi"

from geopandas import vectorized

from shapely.geometry.base import (GEOMETRY_TYPES as GEOMETRY_NAMES, CAP_STYLE,
        JOIN_STYLE)

GEOMETRY_TYPES = [getattr(shapely.geometry, name) for name in GEOMETRY_NAMES]

opposite_predicates = {'contains': 'within',
                       'intersects': 'intersects',
                       'touches': 'touches',
                       'covers': 'covered_by',
                       'crosses': 'crosses',
                       'overlaps': 'overlaps'}

for k, v in list(opposite_predicates.items()):
    opposite_predicates[v] = k


cdef get_element(np.ndarray[np.uintp_t, ndim=1, cast=True] geoms, int idx):
    """
    Get a single shape from a GeometryArray as a Shapely object

    This allocates a new GEOSGeometry object
    """
    cdef GEOSGeometry *geom
    with get_geos_handle() as handle:
        geom = <GEOSGeometry *> geoms[idx]

        if geom is NULL:
            return None
        else:
            geom = GEOSGeom_clone_r(handle, geom)  # create a copy rather than deal with gc

    return geom_factory(<np.uintp_t> geom)


cpdef to_shapely(np.ndarray[np.uintp_t, ndim=1, cast=True] geoms):
    """ Convert array of pointers to an array of shapely objects """
    cdef GEOSGeometry *geom
    cdef unsigned int n = geoms.size
    cdef np.ndarray[object, ndim=1] out = np.empty(n, dtype=object)
    with get_geos_handle() as handle:
        for i in range(n):
            geom = <GEOSGeometry *> geoms[i]

    with get_geos_handle() as handle:
        for i in range(n):
            geom = <GEOSGeometry *> geoms[i]

            if not geom:
                out[i] = None
            else:
                geom = GEOSGeom_clone_r(handle, geom)  # create a copy rather than deal with gc
                out[i] = geom_factory(<np.uintp_t> geom)

    return out


cpdef from_shapely(object L):
    """ Convert a list or array of shapely objects to a GeometryArray """
    cdef Py_ssize_t idx
    cdef GEOSGeometry *geom
    cdef uintptr_t geos_geom
    cdef unsigned int n

    n = len(L)
    cdef np.ndarray[np.uintp_t, ndim=1] out = np.empty(n, dtype=np.uintp)

    with get_geos_handle() as handle:

        for idx in xrange(n):
            g = L[idx]
            if g is not None and not (isinstance(g, float) and np.isnan(g)):
                try:
                    geos_geom = <np.uintp_t> g.__geom__
                except AttributeError:
                    msg = ("Inputs to from_shapely must be shapely geometries. "
                           "Got %s" % str(g))
                    raise TypeError(msg)
                geom = GEOSGeom_clone_r(handle, <GEOSGeometry *> geos_geom)  # create a copy rather than deal with gc
                out[idx] = <np.uintp_t> geom
            else:
                out[idx] = 0

    return GeometryArray(out)


cpdef from_wkb(object L):
    """ Convert a list or array of WKB objects to a GeometryArray """
    cdef Py_ssize_t idx
    cdef GEOSGeometry *geom
    cdef unsigned int n
    cdef unsigned char* c_string
    cdef bytes py_wkb
    cdef size_t size

    n = len(L)
    cdef np.ndarray[np.uintp_t, ndim=1] out = np.empty(n, dtype=np.uintp)

    with get_geos_handle() as handle:
        reader = GEOSWKBReader_create_r(handle)

        for idx in xrange(n):
            py_wkb = L[idx]
            if py_wkb is not None:
                size = len(py_wkb)
                if size:
                    c_string = <unsigned char*> py_wkb
                    geom = GEOSWKBReader_read_r(handle, reader, c_string, size)
                    out[idx] = <np.uintp_t> geom
                else:
                    out[idx] = 0
            else:
                out[idx] = 0

        GEOSWKBReader_destroy_r(handle, reader)

    return GeometryArray(out)


cpdef from_wkt(object L):
    """ Convert a list or array of WKT objects to a GeometryArray """
    cdef Py_ssize_t idx
    cdef GEOSGeometry *geom
    cdef unsigned int n
    cdef char* c_string
    #cdef bytes py_wkt

    n = len(L)
    cdef np.ndarray[np.uintp_t, ndim=1] out = np.empty(n, dtype=np.uintp)

    with get_geos_handle() as handle:
        reader = GEOSWKTReader_create_r(handle)

        for idx in xrange(n):
            py_wkt = L[idx]
            if isinstance(py_wkt, unicode):
                py_wkt = (<unicode>py_wkt).encode('utf8')
            if py_wkt:
                c_string = <char*> py_wkt
                geom = GEOSWKTReader_read_r(handle, reader, c_string)
                out[idx] = <np.uintp_t> geom
            else:
                out[idx] = 0

        GEOSWKTReader_destroy_r(handle, reader)

    return GeometryArray(out)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef points_from_xy(np.ndarray[double, ndim=1, cast=True] x,
                     np.ndarray[double, ndim=1, cast=True] y):
    """ Convert numpy arrays of x and y values to a GeometryArray of points """
    cdef Py_ssize_t idx
    cdef GEOSCoordSequence *sequence
    cdef GEOSGeometry *geom
    cdef uintptr_t geos_geom
    cdef unsigned int n = x.size
    cdef np.ndarray[np.uintp_t, ndim=1] out = np.empty(n, dtype=np.uintp)
    with get_geos_handle() as handle:

        with nogil:
            for idx in xrange(n):
                sequence = GEOSCoordSeq_create_r(handle, 1, 2)
                GEOSCoordSeq_setX_r(handle, sequence, 0, x[idx])
                GEOSCoordSeq_setY_r(handle, sequence, 0, y[idx])
                geom = GEOSGeom_createPoint_r(handle, sequence)
                geos_geom = <np.uintp_t> geom
                out[idx] = <np.uintp_t> geom

    return GeometryArray(out)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef vec_free(np.ndarray[np.uintp_t, ndim=1, cast=True] geoms):
    """ Free an array of GEOSGeometry pointers """
    cdef Py_ssize_t idx
    cdef GEOSGeometry *geom
    cdef uintptr_t geos_geom
    cdef unsigned int n = geoms.size

    with get_geos_handle() as handle:
        with nogil:
            for idx in xrange(n):
                geos_geom = geoms[idx]
                geom = <GEOSGeometry *> geos_geom
                if geom is not NULL:
                    GEOSGeom_destroy_r(handle, geom)


class GeometryArray(object):
    dtype = np.dtype('O')

    def __init__(self, data, base=False):
        self.data = data
        self.base = base

    def __getitem__(self, idx):
        if isinstance(idx, numbers.Integral):
            return get_element(self.data, idx)
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
            vec_free(self.data)

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

    def binary_geo(self, other, op):
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
            return GeometryArray(vectorized.vector_binary_geo(op, self.data, other.data))
        else:
            raise NotImplementedError("type not known %s" % type(other))

    def binop_predicate(self, other, op, extra=None):
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
                return vectorized.binary_predicate_with_arg(op, self.data, other, extra)
            elif op in opposite_predicates:
                op2 = opposite_predicates[op]
                return vectorized.prepared_binary_predicate(op2, self.data, other)
            else:
                return vectorized.binary_predicate(op, self.data, other)
        elif isinstance(other, GeometryArray):
            if len(self) != len(other):
                msg = ("Shapes of inputs to not match.  Left: %d, Right: %d" %
                        (len(self), len(other)))
                raise ValueError(msg)
            if extra is not None:
                return vectorized.vector_binary_predicate_with_arg(op, self.data, other.data, extra)
            else:
                return vectorized.vector_binary_predicate(op, self.data, other.data)
        else:
            raise NotImplementedError("type not known %s" % type(other))

    def covers(self, other):
        return self.binop_predicate(other, 'covers')

    def contains(self, other):
        return self.binop_predicate(other, 'contains')

    def crosses(self, other):
        return self.binop_predicate(other, 'crosses')

    def disjoint(self, other):
        return self.binop_predicate(other, 'disjoint')

    def equals(self, other):
        return self.binop_predicate(other, 'equals')

    def intersects(self, other):
        return self.binop_predicate(other, 'intersects')

    def overlaps(self, other):
        return self.binop_predicate(other, 'overlaps')

    def touches(self, other):
        return self.binop_predicate(other, 'touches')

    def within(self, other):
        return self.binop_predicate(other, 'within')

    def equals_exact(self, other, tolerance):
        return self.binop_predicate(other, 'equals_exact', tolerance)

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
            return vectorized.binary_vector_float('distance', self.data, other.data)
        else:
            return vectorized.binary_float('distance', self.data, other)

    def project(self, other, normalized=False):
        op = 'project' if not normalized else 'project-normalized'
        if isinstance(other, GeometryArray):
            return vectorized.binary_vector_float_return(op, self.data, other.data)
        else:
            return vectorized.binary_float_return(op, self.data, other)

    def area(self):
        return vectorized.unary_vector_float('area', self.data)

    def length(self):
        return vectorized.unary_vector_float('length', self.data)

    def difference(self, other):
        return self.binary_geo(other, 'difference')

    def symmetric_difference(self, other):
        return self.binary_geo(other, 'symmetric_difference')

    def union(self, other):
        return self.binary_geo(other, 'union')

    def intersection(self, other):
        return self.binary_geo(other, 'intersection')

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


cdef class get_geos_handle:
    cdef GEOSContextHandle_t handle

    cdef GEOSContextHandle_t __enter__(self):
        self.handle = GEOS_init_r()
        return self.handle

    def __exit__(self, type, value, traceback):
        GEOS_finish_r(self.handle)
