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


def _series_op(this, other, op, **kwargs):
    if kwargs or not isinstance(other, BaseGeometry):
        return _cy_series_op_slow(this, other, op, kwargs)
    try:
        if op == 'equals':
            func = _cy_series_op_fast_unprepared
        else:
            func = _cy_series_op_fast
        return Series(func(this.values, other, op), index=this.index)
    except NotImplementedError:
        return _cy_series_op_slow(this, other, op, kwargs)


cdef _cy_series_op_slow(this, other, op, kwargs):
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
cdef _cy_series_op_fast(array, geometry, op):

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
            if geos_geom != 0:
                geom2 = <GEOSGeometry *> geos_geom
                result[idx] = <np.uint8_t> func(geos_handle, geom1, geom2)

    return result.view(dtype=np.bool)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef _cy_series_op_fast_unprepared(array, geometry, op):

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
