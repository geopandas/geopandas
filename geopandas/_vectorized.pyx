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
        print(idx, x[idx], y[idx], <np.uintp_t> sequence, geos_geom)
        out[idx] = <np.uintp_t> geom

    return VectorizedGeometry(out)


cdef prepared_binary_op(str op,
                        np.ndarray[np.uintp_t, ndim=1, cast=True] geoms,
                        object other):
    cdef Py_ssize_t idx
    cdef GEOSContextHandle_t handle
    cdef GEOSGeometry *geom
    cdef GEOSGeometry *other_geom
    cdef GEOSPreparedGeometry *prepared_geom
    cdef unsigned int n = geoms.size

    cdef np.ndarray[np.uintp_t, ndim=1, cast=True] out = np.empty(n, dtype=np.uintp)

    handle = get_geos_context_handle()
    other_geom = <GEOSGeometry *> other.__geom__
    prepared_geom = GEOSPrepare_r(handle, other_geom)

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
            out[idx] = <np.uintp_t> func(handle, prepared_geom, geom)

    return VectorizedGeometry(out)


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


class VectorizedGeometry(object):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, idx):
        return get_element(self.data, idx)

    @property
    def x(self):
        return get_coordinate_point(self.data, 0)

    @property
    def y(self):
        return get_coordinate_point(self.data, 1)
