import collections
import numbers

import shapely
from shapely.geometry import MultiPoint, MultiLineString, MultiPolygon
from shapely.geometry.base import BaseGeometry, geom_factory
from shapely.ops import cascaded_union, unary_union
import shapely.affinity as affinity

import cython
cimport cpython.array

cimport numpy as np
import numpy as np
import pandas as pd
from pandas import Series, DataFrame, MultiIndex

import geopandas as gpd

from .base import GeoPandasBase

include "_geos.pxi"

from shapely.geometry.base import GEOMETRY_TYPES as GEOMETRY_NAMES

GEOMETRY_TYPES = [getattr(shapely.geometry, name) for name in GEOMETRY_NAMES]


cdef get_element(np.ndarray[np.uintp_t, ndim=1, cast=True] geoms, int idx):
    cdef GEOSGeometry *geom
    cdef GEOSContextHandle_t handle
    geom = <GEOSGeometry *> geoms[idx]

    handle = get_geos_context_handle()
    geom = GEOSGeom_clone_r(handle, geom)  # create a copy rather than deal with gc
    typ = GEOMETRY_TYPES[GEOSGeomTypeId_r(handle, geom)]

    return geom_factory(<np.uintp_t> geom)


cpdef points_from_xy(np.ndarray[double, ndim=1, cast=True] x,
                     np.ndarray[double, ndim=1, cast=True] y):
    cdef Py_ssize_t idx
    cdef GEOSContextHandle_t handle
    cdef GEOSCoordSequence *sequence
    cdef GEOSGeometry *geom
    cdef uintptr_t geos_geom
    cdef unsigned int n = x.size

    cdef np.ndarray[np.uintp_t, ndim=1] out = np.empty(n, dtype=np.uintp)

    handle = get_geos_context_handle()

    for idx in xrange(n):
        sequence = GEOSCoordSeq_create_r(handle, 1, 2)
        GEOSCoordSeq_setX_r(handle, sequence, 0, x[idx])
        GEOSCoordSeq_setY_r(handle, sequence, 0, y[idx])
        geom = GEOSGeom_createPoint_r(handle, sequence)
        geos_geom = <np.uintp_t> geom
        out[idx] = <np.uintp_t> geom

    return VectorizedGeometry(out)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef prepared_binary_op(str op,
                        np.ndarray[np.uintp_t, ndim=1, cast=True] geoms,
                        object other):
    cdef Py_ssize_t idx
    cdef GEOSContextHandle_t handle
    cdef GEOSGeometry *geom
    cdef GEOSGeometry *other_geom
    cdef GEOSPreparedGeometry *prepared_geom
    cdef unsigned int n = geoms.size

    cdef np.ndarray[np.uint8_t, ndim=1, cast=True] out = np.empty(n, dtype=np.bool_)

    handle = get_geos_context_handle()
    other_geom = <GEOSGeometry *> other.__geom__

    # Prepare the geometry if it hasn't already been prepared.
    # TODO: why can't we do the following instead?
    #   prepared_geom = GEOSPrepare_r(handle, other_geom)
    if not isinstance(other, shapely.prepared.PreparedGeometry):
        other = shapely.prepared.prep(other)

    geos_handle = get_geos_context_handle()
    prepared_geom = geos_from_prepared(other)

    if op == 'contains':
        func = GEOSPreparedContains_r
    elif op == 'intersects':
        func = GEOSPreparedIntersects_r
    elif op == 'touches':
        func = GEOSPreparedTouches_r
    elif op == 'crosses':
        func = GEOSPreparedCrosses_r
    elif op == 'within':
        func = GEOSPreparedWithin_r
    elif op == 'contains_properly':
        func = GEOSPreparedContainsProperly_r
    elif op == 'overlaps':
        func = GEOSPreparedOverlaps_r
    elif op == 'covers':
        func = GEOSPreparedCovers_r
    elif op == 'covered_by':
        func = GEOSPreparedCoveredBy_r

    with nogil:
        for idx in xrange(n):
            geom = <GEOSGeometry *> geoms[idx]
            out[idx] = func(handle, prepared_geom, geom)

    return out


"""
@cython.boundscheck(False)
@cython.wraparound(False)
cdef binary_op(str op,
               np.ndarray[np.uintp_t, ndim=1, cast=True] geoms,
               object other):
    cdef Py_ssize_t idx
    cdef GEOSContextHandle_t handle
    cdef GEOSGeometry *geom
    cdef GEOSGeometry *other_geom
    cdef unsigned int n = geoms.size

    cdef np.ndarray[np.uint8_t, ndim=1, cast=True] out = np.empty(n, dtype=np.bool_)

    handle = get_geos_context_handle()
    other_geom = <GEOSGeometry *> other.__geom__


    if op == 'contains':
        func = GEOSContains_r
    elif op == 'equals':
        func = GEOSEquals_r
    elif op == 'intersects':
        func = GEOSIntersects_r
    elif op == 'touches':
        func = GEOSTouches_r
    elif op == 'crosses':
        func = GEOSCrosses_r
    elif op == 'within':
        func = GEOSWithin_r
    elif op == 'contains_properly':
        func = GEOSContainsProperly_r
    elif op == 'overlaps':
        func = GEOSOverlaps_r
    elif op == 'covers':
        func = GEOSCovers_r
    elif op == 'covered_by':
        func = GEOSCoveredBy_r

    with nogil:
        for idx in xrange(n):
            geom = <GEOSGeometry *> geoms[idx]
            out[idx] = func(handle, geom, other_geom)

    return out
"""


@cython.boundscheck(False)
@cython.wraparound(False)
cdef geo_unary_op(str op, np.ndarray[np.uintp_t, ndim=1, cast=True] geoms):
    cdef Py_ssize_t idx
    cdef GEOSContextHandle_t handle
    cdef GEOSGeometry *geom
    cdef uintptr_t geos_geom
    cdef GEOSGeometry *other_geom
    cdef unsigned int n = geoms.size

    cdef np.ndarray[np.uintp_t, ndim=1, cast=True] out = np.empty(n, dtype=np.uintp)

    handle = get_geos_context_handle()

    if op == 'boundary':
        func = GEOSBoundary_r
    elif op == 'centroid':
        func = GEOSGetCentroid_r
    elif op == 'convex_hull':
        func = GEOSConvexHull_r
    # elif op == 'exterior':
    #     func = GEOSGetExteriorRing_r  # segfaults on cleanup?
    elif op == 'envelope':
        func = GEOSEnvelope_r
    else:
        raise NotImplementedError("Op %s not known" % op)

    for idx in xrange(n):
        geos_geom = geoms[idx]
        geom = <GEOSGeometry *> geos_geom
        out[idx] = <np.uintp_t> func(handle, geom)

    return VectorizedGeometry(out)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef get_coordinate_point(np.ndarray[np.uintp_t, ndim=1, cast=True] geoms,
                          int coordinate):
    cdef Py_ssize_t idx
    cdef GEOSContextHandle_t handle
    cdef GEOSGeometry *geom
    cdef GEOSCoordSequence *sequence
    cdef unsigned int n = geoms.size
    cdef double value

    cdef np.ndarray[double, ndim=1, cast=True] out = np.empty(n, dtype=np.float)

    handle = get_geos_context_handle()

    if coordinate == 0:
        func = GEOSCoordSeq_getX_r
    elif coordinate == 1:
        func = GEOSCoordSeq_getY_r
    elif coordinate == 2:
        func = GEOSCoordSeq_getZ_r
    else:
        raise NotImplementedError("Coordinate must be between 0-x, 1-y, 2-z")

    with nogil:
        for idx in xrange(n):
            geom = <GEOSGeometry *> geoms[idx]
            sequence = GEOSGeom_getCoordSeq_r(handle, geom)
            func(handle, sequence, 0, &value)
            out[idx] = value

    return out


cpdef from_shapely(object L):
    cdef Py_ssize_t idx
    cdef GEOSContextHandle_t handle
    cdef GEOSGeometry *geom
    cdef uintptr_t geos_geom
    cdef unsigned int n

    n = len(L)

    cdef np.ndarray[np.uintp_t, ndim=1] out = np.empty(n, dtype=np.uintp)

    handle = get_geos_context_handle()

    for idx in xrange(n):
        g = L[idx]
        geos_geom = <np.uintp_t> g.__geom__
        geom = GEOSGeom_clone_r(handle, <GEOSGeometry *> geos_geom)  # create a copy rather than deal with gc
        out[idx] = <np.uintp_t> geom

    return VectorizedGeometry(out)


cdef free(np.ndarray[np.uintp_t, ndim=1, cast=True] geoms):
    cdef Py_ssize_t idx
    cdef GEOSContextHandle_t handle
    cdef GEOSGeometry *geom
    cdef uintptr_t geos_geom

    handle = get_geos_context_handle()

    for idx in xrange(geoms.size):
        geos_geom = geoms[idx]
        geom = <GEOSGeometry *> geos_geom
        GEOSGeom_destroy_r(handle, geom)


class VectorizedGeometry(object):
    def __init__(self, data, parent=None):
        self.data = data
        self.parent = parent

    def __getitem__(self, idx):
        if isinstance(idx, numbers.Integral):
            return get_element(self.data, idx)
        elif isinstance(idx, collections.Iterable):
            return VectorizedGeometry(self.data[idx], parent=self)

    def __len__(self):
        return len(self.data)

    def __del__(self):
        if self.parent is None:
            free(self.data)

    @property
    def x(self):
        return get_coordinate_point(self.data, 0)

    @property
    def y(self):
        return get_coordinate_point(self.data, 1)

    def contains(self, other):
        return prepared_binary_op('within', self.data, other)

    def covers(self, other):
        return prepared_binary_op('covered_by', self.data, other)

    def covered_by(self, other):
        return prepared_binary_op('covers', self.data, other)

    def rcontains(self, other):
        return prepared_binary_op('contains', self.data, other)

    def rcovers(self, other):
        return prepared_binary_op('covers', self.data, other)

    def rcovered_by(self, other):
        return prepared_binary_op('covered_by', self.data, other)

    def centroid(self):
        return geo_unary_op('centroid', self.data)
