from warnings import warn

import shapely
from shapely.geometry import MultiPoint, MultiLineString, MultiPolygon
from shapely.geometry.base import BaseGeometry
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


def _series_op(this, other, op, **kwargs):
    if kwargs or not isinstance(other, BaseGeometry):
        return _cy_series_op_slow(this, other, op, kwargs)
    try:
        if op in ['equals']:
            x = _cy_series_op_fast_unprepared(this.values, other, op)
        else:
            x = _cy_series_op_fast(this.values, other, op)
        return Series(x, index=this.index)
    except NotImplementedError:
        return _cy_series_op_slow(this, other, op, kwargs)


cdef _cy_series_op_slow(object this, object other, str op, kwargs):
    """Geometric operation that returns a pandas Series"""
    null_val = False if op != 'distance' else np.nan

    if isinstance(other, GeoPandasBase):
        this = this.geometry
        this, other = this.align(other.geometry)
        return Series([getattr(this_elem, op)(other_elem, **kwargs)
                    if not this_elem.is_empty | other_elem.is_empty else null_val
                    for this_elem, other_elem in zip(this, other)],
                    index=this.index)
    else:
        return Series([getattr(s, op)(other, **kwargs) if s else null_val
                      for s in this.geometry], index=this.index)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef _cy_series_op_fast(np.ndarray[object, ndim=1] array, object geometry, str op):

    cdef Py_ssize_t idx
    cdef unsigned int n = array.size
    cdef np.ndarray[np.uint8_t, ndim=1, cast=True] result = np.empty(n, dtype=np.uint8)
    cdef np.ndarray[np.uintp_t, ndim=1, cast=True] geometries = np.empty(n, dtype=np.uintp)

    cdef GEOSContextHandle_t geos_handle
    cdef GEOSPreparedGeometry *geom1
    cdef GEOSGeometry *geom2
    cdef uintptr_t geos_geom

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

    else:
        raise NotImplementedError("Op %s not known" % op)

    # Prepare the geometry if it hasn't already been prepared.
    if not isinstance(geometry, shapely.prepared.PreparedGeometry):
        geometry = shapely.prepared.prep(geometry)

    geos_handle = get_geos_context_handle()
    geom1 = geos_from_prepared(geometry)

    for idx in xrange(n):
        g = array[idx]
        if g is None:
            geometries[idx] = 0
        else:
            geometries[idx] = g.__geom__

    with nogil:
        for idx in xrange(n):
            geos_geom = geometries[idx]
            if geos_geom == 0:
                result[idx] = <np.uint8_t> 0
            else:
                geom2 = <GEOSGeometry *> geos_geom
                result[idx] = <np.uint8_t> func(geos_handle, geom1, geom2)

    return result.view(dtype=np.bool)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef _cy_series_op_fast_unprepared(np.ndarray[object, ndim=1] array, object geometry, str op):

    cdef Py_ssize_t idx
    cdef unsigned int n = array.size
    cdef np.ndarray[np.uint8_t, ndim=1, cast=True] result = np.empty(n, dtype=np.uint8)

    cdef GEOSContextHandle_t geos_handle
    cdef GEOSGeometry *geom1
    cdef GEOSGeometry *geom2
    cdef uintptr_t geos_geom
    cdef uintptr_t geos_geom_1

    if op == 'equals':
        func = GEOSEquals_r
    else:
        raise NotImplementedError("Op %s not known" % op)

    geos_handle = get_geos_context_handle()
    geos_geom_1 = geometry.__geom__
    geom1 = <GEOSGeometry *> geos_geom_1

    for idx in xrange(n):
        g = array[idx]
        if g is None:
            result[idx] = 0
        else:
            geos_geom = g.__geom__
            geom2 = <GEOSGeometry *>geos_geom
            result[idx] = <np.uint8_t> func(geos_handle, geom1, geom2)

    return result.view(dtype=np.bool)


def _geo_unary_op(this, op):
    try:
        x = _cy_geo_unary_op(this.geometry.values, op)
    except NotImplementedError:
        x = _py_geo_unary_op(this.geometry.values, op)
    return gpd.GeoSeries(x, index=this.index, crs=this.crs)


cdef _py_geo_unary_op(np.ndarray[object, ndim=1] this, str op):
    """Unary operation that returns a GeoSeries"""
    return [getattr(geom, op) for geom in this]


@cython.boundscheck(False)
@cython.wraparound(False)
cdef _cy_geo_unary_op(np.ndarray[object, ndim=1] array, str op):

    cdef Py_ssize_t idx
    cdef unsigned int n = array.size
    cdef np.ndarray[np.uintp_t, ndim=1, cast=True] geometries = np.empty(n, dtype=np.uintp)
    cdef np.ndarray[object, ndim=1] out = np.empty(n, dtype=object)

    cdef GEOSContextHandle_t geos_handle
    cdef GEOSGeometry *geom
    cdef uintptr_t geos_geom

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

    geos_handle = get_geos_context_handle()

    for idx in xrange(n):
        g = array[idx]
        if g is None:
            geometries[idx] = 0
        else:
            geometries[idx] = <np.uintp_t> g.__geom__

    with nogil:
        for idx in xrange(n):
            geos_geom = geometries[idx]
            if geos_geom == 0:
                geometries[idx] = <np.uintp_t> 0
            else:
                geom = <GEOSGeometry *> geos_geom
                result = func(geos_handle, geom)
                geometries[idx] = <np.uintp_t> result

    for idx in xrange(n):
        geom = <GEOSGeometry *> geometries[idx]
        if geom == NULL:
            out[idx] = None
        else:
            typ = GEOMETRY_TYPES[GEOSGeomTypeId_r(geos_handle, geom)]
            obj = BaseGeometry()
            obj.__class = typ
            obj.__geom__ = <np.uintp_t> geom
            obj.__p__ = None
            if GEOSHasZ_r(geos_handle, geom):
                obj._ndim = 3
            else:
                obj._ndim = 2
            obj._is_empty = False
            out[idx] = obj
    return out
