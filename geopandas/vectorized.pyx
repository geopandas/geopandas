# distutils: sources = geopandas/algos.c

import collections
import numbers
from libc.stdlib cimport free as cfree


import shapely
import shapely.prepared
from shapely.geometry import MultiPoint, MultiLineString, MultiPolygon
from shapely.geometry.base import BaseGeometry, geom_factory
from shapely.ops import cascaded_union
import shapely.affinity as affinity

import cython
cimport cpython.array
from libc.stdlib cimport malloc, free


cimport numpy as np
import numpy as np

include "geopandas/_geos.pxi"

from shapely.geometry.base import (GEOMETRY_TYPES as GEOMETRY_NAMES, CAP_STYLE,
        JOIN_STYLE)

cdef extern from "algos.h":
    ctypedef char (*GEOSPredicate)(GEOSContextHandle_t handler,
                                   const GEOSGeometry *left,
                                   const GEOSGeometry *right) nogil
    ctypedef char (*GEOSPreparedPredicate)(GEOSContextHandle_t handler,
                                           const GEOSPreparedGeometry *left,
                                           const GEOSGeometry *right) nogil
    ctypedef struct size_vector:
        size_t n
        size_t m
        size_t *a
    size_vector sjoin(GEOSContextHandle_t handle,
                      GEOSPreparedPredicate predicate,
                      GEOSGeometry *left, size_t nleft,
                      GEOSGeometry *right, size_t nright) nogil


cdef int GEOS_POINT = 0
cdef int GEOS_LINESTRING = 1
cdef int GEOS_LINEARRING = 2
cdef int GEOS_POLYGON = 3
cdef int GEOS_MULTIPOINT = 4
cdef int GEOS_MULTILINESTRING = 5
cdef int GEOS_MULTIPOLYGON = 6
cdef int GEOS_GEOMETRYCOLLECTION = 7


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
    cdef GEOSContextHandle_t handle
    geom = <GEOSGeometry *> geoms[idx]

    handle = get_geos_context_handle()

    if not geom:
        geom = GEOSGeom_createEmptyPolygon_r(handle)
    else:
        geom = GEOSGeom_clone_r(handle, geom)  # create a copy rather than deal with gc

    return geom_factory(<np.uintp_t> geom)


cpdef to_shapely(np.ndarray[np.uintp_t, ndim=1, cast=True] geoms):
    """ Convert array of pointers to an array of shapely objects """
    cdef GEOSGeometry *geom
    cdef GEOSContextHandle_t handle
    cdef unsigned int n = geoms.size
    cdef np.ndarray[object, ndim=1] out = np.empty(n, dtype=object)

    handle = get_geos_context_handle()

    for i in range(n):
        geom = <GEOSGeometry *> geoms[i]

        if not geom:
            geom = GEOSGeom_createEmptyPolygon_r(handle)
        else:
            geom = GEOSGeom_clone_r(handle, geom)  # create a copy rather than deal with gc

        out[i] = geom_factory(<np.uintp_t> geom)

    return out


cpdef from_shapely(object L):
    """ Convert a list or array of shapely objects to a GeometryArray """
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
        if g is not None and not (isinstance(g, float) and np.isnan(g)):
            try:
                geos_geom = <np.uintp_t> g.__geom__
            except AttributeError:
                msg = ("Inputs to from_shapely must be shapely eometries. "
                       "Got %s" % str(g))
                raise TypeError(msg)
            geom = GEOSGeom_clone_r(handle, <GEOSGeometry *> geos_geom)  # create a copy rather than deal with gc
            out[idx] = <np.uintp_t> geom
        else:
            out[idx] = 0

    return GeometryArray(out)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef points_from_xy(np.ndarray[double, ndim=1, cast=True] x,
                     np.ndarray[double, ndim=1, cast=True] y):
    """ Convert numpy arrays of x and y values to a GeometryArray of points """
    cdef Py_ssize_t idx
    cdef GEOSContextHandle_t handle
    cdef GEOSCoordSequence *sequence
    cdef GEOSGeometry *geom
    cdef uintptr_t geos_geom
    cdef unsigned int n = x.size

    cdef np.ndarray[np.uintp_t, ndim=1] out = np.empty(n, dtype=np.uintp)

    handle = get_geos_context_handle()

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
cpdef prepared_binary_predicate(str op,
                               np.ndarray[np.uintp_t, ndim=1, cast=True] geoms,
                               object other):
    """
    Apply predicate to a GeometryArray and an individual shapely object

    This uses a prepared geometry for other

    Parameters
    ----------
    op: str
        string like 'contains', or 'intersects'
    geoms: numpy.ndarray
        Array of pointers to GEOSGeometry objects
    other: shapely BaseGeometry

    Returns
    -------
    out: boolean array
    """
    cdef Py_ssize_t idx
    cdef GEOSContextHandle_t handle
    cdef GEOSGeometry *geom
    cdef GEOSGeometry *other_geom
    cdef GEOSPreparedGeometry *prepared_geom
    cdef GEOSPreparedPredicate predicate
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

    predicate = get_prepared_predicate(op)

    with nogil:
        for idx in xrange(n):
            geom = <GEOSGeometry *> geoms[idx]
            if geom != NULL:
                out[idx] = predicate(handle, prepared_geom, geom)
            else:
                out[idx] = 0

    return out

cdef GEOSPreparedPredicate get_prepared_predicate(str op) except NULL:
    if op == 'contains':
        func = GEOSPreparedContains_r
    elif op == 'disjoint':
        func = GEOSPreparedDisjoint_r
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
        raise NotImplementedError(op)

    return func


cdef GEOSPredicate get_predicate(str op) except NULL:
    cdef GEOSPredicate func

    if op == 'contains':
        func = GEOSContains_r
    elif op == 'disjoint':
        func = GEOSDisjoint_r
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
    elif op == 'overlaps':
        func = GEOSOverlaps_r
    elif op == 'covers':
        func = GEOSCovers_r
    elif op == 'covered_by':
        func = GEOSCoveredBy_r
    else:
        raise NotImplementedError(op)

    return func


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef vector_binary_predicate(str op,
                             np.ndarray[np.uintp_t, ndim=1, cast=True] left,
                             np.ndarray[np.uintp_t, ndim=1, cast=True] right):
    """
    Apply predicate to two arrays of GEOSGeometry pointers

    Parameters
    ----------
    op: str
        string like 'contains', or 'intersects'
    left : numpy.ndarray
        Array of pointers to GEOSGeometry objects
    right: numpy.ndarray
        Array of pointers to GEOSGeometry objects

    Returns
    -------
    out: boolean array
    """
    cdef Py_ssize_t idx
    cdef GEOSContextHandle_t handle
    cdef GEOSGeometry *left_geom
    cdef GEOSGeometry *right_geom
    cdef GEOSPredicate func
    cdef unsigned int n = left.size

    cdef np.ndarray[np.uint8_t, ndim=1, cast=True] out = np.empty(n, dtype=np.bool_)

    handle = get_geos_context_handle()

    func = get_predicate(op)
    if not func:
        raise NotImplementedError(op)

    with nogil:
        for idx in xrange(n):
            left_geom = <GEOSGeometry *> left[idx]
            right_geom = <GEOSGeometry *> right[idx]
            if left_geom != NULL and right_geom != NULL:
                out[idx] = func(handle, left_geom, right_geom)
            else:
                out[idx] = False

    return out


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef vector_binary_predicate_with_arg(
        str op,
        np.ndarray[np.uintp_t, ndim=1, cast=True] left,
        np.ndarray[np.uintp_t, ndim=1, cast=True] right,
        double arg):
    """
    This is a copy of vector_binary_predicate but supporting a double arg

    This is currently only used for equals_exact
    """
    cdef Py_ssize_t idx
    cdef GEOSContextHandle_t handle
    cdef GEOSGeometry *left_geom
    cdef GEOSGeometry *right_geom
    cdef unsigned int n = left.size

    cdef np.ndarray[np.uint8_t, ndim=1, cast=True] out = np.empty(n, dtype=np.bool_)

    handle = get_geos_context_handle()

    if op == 'equals_exact':
        func = GEOSEqualsExact_r
    else:
        raise NotImplementedError(op)

    with nogil:
        for idx in xrange(n):
            left_geom = <GEOSGeometry *> left[idx]
            right_geom = <GEOSGeometry *> right[idx]
            if left_geom != NULL and right_geom != NULL:
                out[idx] = func(handle, left_geom, right_geom, arg)
            else:
                out[idx] = False

    return out


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef unary_predicate(str op, np.ndarray[np.uintp_t, ndim=1, cast=True] geoms):
    """
    Apply predicate to a single array of GEOSGeometry pointers

    Parameters
    ----------
    op: str
        string like 'contains', or 'intersects'
    goems: numpy.ndarray
        Array of pointers to GEOSGeometry objects

    Returns
    -------
    out: boolean array
    """
    cdef Py_ssize_t idx
    cdef GEOSContextHandle_t handle
    cdef GEOSGeometry *geom
    cdef unsigned int n = geoms.size

    cdef np.ndarray[np.uint8_t, ndim=1, cast=True] out = np.empty(n, dtype=np.bool_)

    handle = get_geos_context_handle()

    if op == 'is_empty':
        func = GEOSisEmpty_r
    elif op == 'is_valid':
        func = GEOSisValid_r
    elif op == 'is_simple':
        func = GEOSisSimple_r
    elif op == 'is_ring':
        func = GEOSisRing_r
    elif op == 'has_z':
        func = GEOSHasZ_r
    elif op == 'is_closed':
        func = GEOSisClosed_r
    else:
        raise NotImplementedError(op)

    with nogil:
        for idx in xrange(n):
            geom = <GEOSGeometry *> geoms[idx]
            if geom is not NULL:
                out[idx] = func(handle, geom)
            else:
                out[idx] = False

    return out

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef binary_predicate(str op,
                      np.ndarray[np.uintp_t, ndim=1, cast=True] geoms,
                      object other):
    """
    Apply predicate to an array of GEOSGeometry pointers and a shapely object

    Parameters
    ----------
    op: str
        string like 'contains', or 'intersects'
    goems: numpy.ndarray
        Array of pointers to GEOSGeometry objects
    other: shapely BaseGeometry

    Returns
    -------
    out: boolean array
    """
    cdef Py_ssize_t idx
    cdef GEOSContextHandle_t handle
    cdef GEOSGeometry *geom
    cdef GEOSGeometry *other_geom
    cdef uintptr_t other_pointer
    cdef unsigned int n = geoms.size
    cdef GEOSPredicate predicate

    cdef np.ndarray[np.uint8_t, ndim=1, cast=True] out = np.empty(n, dtype=np.bool_)

    handle = get_geos_context_handle()
    other_pointer = <np.uintp_t> other.__geom__
    other_geom = <GEOSGeometry *> other_pointer

    predicate = get_predicate(op)

    with nogil:
        for idx in xrange(n):
            geom = <GEOSGeometry *> geoms[idx]
            if geom is not NULL:
                out[idx] = predicate(handle, geom, other_geom)
            else:
                out[idx] = False

    return out

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef binary_predicate_with_arg(str op,
                                np.ndarray[np.uintp_t, ndim=1, cast=True] geoms,
                                object other,
                                double arg):
    """
    This is a copy of binary_predicate but supporting a double arg

    This is currently only used for equals_exact
    """
    cdef Py_ssize_t idx
    cdef GEOSContextHandle_t handle
    cdef GEOSGeometry *geom
    cdef GEOSGeometry *other_geom
    cdef uintptr_t other_pointer
    cdef unsigned int n = geoms.size

    cdef np.ndarray[np.uint8_t, ndim=1, cast=True] out = np.empty(n, dtype=np.bool_)

    handle = get_geos_context_handle()
    other_pointer = <np.uintp_t> other.__geom__
    other_geom = <GEOSGeometry *> other_pointer

    if op == 'equals_exact':
        func = GEOSEqualsExact_r
    else:
        raise NotImplementedError(op)

    with nogil:
        for idx in xrange(n):
            geom = <GEOSGeometry *> geoms[idx]
            if geom is not NULL:
                out[idx] = func(handle, geom, other_geom, arg)
            else:
                out[idx] = False

    return out


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef unary_vector_float(str op, np.ndarray[np.uintp_t, ndim=1, cast=True] geoms):
    """
    Evaluate float-valued function on array of GEOSGeometry pointers

    Parameters
    ----------
    op: str
        string like 'area', or 'length'
    goems: numpy.ndarray
        Array of pointers to GEOSGeometry objects

    Returns
    -------
    out: float array
    """
    cdef Py_ssize_t idx
    cdef GEOSContextHandle_t handle
    cdef GEOSGeometry *geom
    cdef unsigned int n = geoms.size
    cdef double nan = np.nan
    cdef double * location

    cdef np.ndarray[double, ndim=1, cast=True] out = np.empty(n, dtype=np.float64)
    location = <double *> out.data  # need to pass a pointer to function

    handle = get_geos_context_handle()

    if op == 'area':
        func = GEOSArea_r
    elif op == 'length':
        func = GEOSLength_r
    else:
        raise NotImplementedError(op)

    with nogil:
        for idx in xrange(n):
            geom = <GEOSGeometry *> geoms[idx]
            if geom != NULL:
                func(handle, geom, <double*> location + idx)
            else:
                out[idx] = nan

    return out


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef binary_vector_float(str op,
                          np.ndarray[np.uintp_t, ndim=1, cast=True] left,
                          np.ndarray[np.uintp_t, ndim=1, cast=True] right):
    """
    Evaluate float-valued function on array of GEOSGeometry pointers

    Parameters
    ----------
    op: str
        string like 'distance'
    goems: numpy.ndarray
        Array of pointers to GEOSGeometry objects

    Returns
    -------
    out: float array
    """
    cdef Py_ssize_t idx
    cdef GEOSContextHandle_t handle
    cdef GEOSGeometry *left_geom
    cdef GEOSGeometry *right_geom
    cdef unsigned int n = left.size
    cdef double nan = np.nan
    cdef double * location

    cdef np.ndarray[double, ndim=1, cast=True] out = np.empty(n, dtype=np.float64)
    location = <double *> out.data  # need to pass a pointer to function

    handle = get_geos_context_handle()

    if op == 'distance':
        func = GEOSDistance_r
    else:
        raise NotImplementedError(op)

    with nogil:
        for idx in xrange(n):
            left_geom = <GEOSGeometry *> left[idx]
            right_geom = <GEOSGeometry *> right[idx]
            if left_geom != NULL and right_geom != NULL:
                func(handle, left_geom, right_geom, <double*> location + idx)
            else:
                out[idx] = nan

    return out


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef binary_vector_float_return(str op,
                          np.ndarray[np.uintp_t, ndim=1, cast=True] left,
                          np.ndarray[np.uintp_t, ndim=1, cast=True] right):
    """
    Evaluate float-valued function on array of GEOSGeometry pointers

    Parameters
    ----------
    op: str
        string like 'project'
    goems: numpy.ndarray
        Array of pointers to GEOSGeometry objects

    Returns
    -------
    out: float array
    """
    cdef Py_ssize_t idx
    cdef GEOSContextHandle_t handle
    cdef GEOSGeometry *left_geom
    cdef GEOSGeometry *right_geom
    cdef unsigned int n = left.size
    cdef double nan = np.nan

    cdef np.ndarray[double, ndim=1, cast=True] out = np.empty(n, dtype=np.float64)

    handle = get_geos_context_handle()

    if op == 'project':
        func = GEOSProject_r
    elif op == 'project-normalized':
        func = GEOSProjectNormalized_r
    else:
        raise NotImplementedError(op)

    with nogil:
        for idx in xrange(n):
            left_geom = <GEOSGeometry *> left[idx]
            right_geom = <GEOSGeometry *> right[idx]
            if left_geom != NULL and right_geom != NULL:
                out[idx] = <double>func(handle, left_geom, right_geom)
            else:
                out[idx] = nan

    return out


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef binary_float(str op,
                   np.ndarray[np.uintp_t, ndim=1, cast=True] left,
                   object right):
    """
    Evaluate float-valued function on array of GEOSGeometry pointers

    Parameters
    ----------
    op: str
        string like 'area', or 'length'
    goems: numpy.ndarray
        Array of pointers to GEOSGeometry objects

    Returns
    -------
    out: float array
    """
    cdef Py_ssize_t idx
    cdef GEOSContextHandle_t handle
    cdef uintptr_t right_ptr
    cdef GEOSGeometry *left_geom
    cdef GEOSGeometry *right_geom
    cdef unsigned int n = left.size
    cdef double nan = np.nan
    cdef double * location

    cdef np.ndarray[double, ndim=1, cast=True] out = np.empty(n, dtype=np.float64)
    location = <double *> out.data  # need to pass a pointer to function

    right_ptr = <np.uintp_t> right.__geom__
    right_geom = <GEOSGeometry *> right_ptr

    handle = get_geos_context_handle()

    if op == 'distance':
        func = GEOSDistance_r
    else:
        raise NotImplementedError(op)

    with nogil:
        for idx in xrange(n):
            left_geom = <GEOSGeometry *> left[idx]
            if left_geom != NULL and right_geom != NULL:
                func(handle, left_geom, right_geom, <double*> location + idx)
            else:
                out[idx] = nan

    return out


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef binary_float_return(str op,
                   np.ndarray[np.uintp_t, ndim=1, cast=True] left,
                   object right):
    """
    Evaluate float-valued function on array of GEOSGeometry pointers

    Parameters
    ----------
    op: str
        string like 'project'
    goems: numpy.ndarray
        Array of pointers to GEOSGeometry objects

    Returns
    -------
    out: float array
    """
    cdef Py_ssize_t idx
    cdef GEOSContextHandle_t handle
    cdef uintptr_t right_ptr
    cdef GEOSGeometry *left_geom
    cdef GEOSGeometry *right_geom
    cdef unsigned int n = left.size
    cdef double nan = np.nan

    cdef np.ndarray[double, ndim=1, cast=True] out = np.empty(n, dtype=np.float64)

    right_ptr = <np.uintp_t> right.__geom__
    right_geom = <GEOSGeometry *> right_ptr

    handle = get_geos_context_handle()

    if op == 'project':
        func = GEOSProject_r
    elif op == 'project-normalized':
        func = GEOSProjectNormalized_r
    else:
        raise NotImplementedError(op)

    with nogil:
        for idx in xrange(n):
            left_geom = <GEOSGeometry *> left[idx]
            if left_geom != NULL and right_geom != NULL:
                out[idx] = <double>func(handle, left_geom, right_geom)
            else:
                out[idx] = nan

    return out




@cython.boundscheck(False)
@cython.wraparound(False)
cpdef geo_unary_op(str op, np.ndarray[np.uintp_t, ndim=1, cast=True] geoms):
    """
    Evaluate Geometry-valued function on array of GEOSGeometry pointers

    Parameters
    ----------
    op: str
        string like 'centroid', or 'convex_hull'
    goems: numpy.ndarray
        Array of pointers to GEOSGeometry objects

    Returns
    -------
    out: array of pointers to GEOSGeometry objects
    """
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
    elif op == 'exterior':
        func = GEOSGetExteriorRing_r  # segfaults on cleanup?
    elif op == 'envelope':
        func = GEOSEnvelope_r
    elif op == 'representative_point':
        func = GEOSPointOnSurface_r
    else:
        raise NotImplementedError("Op %s not known" % op)

    with nogil:
        for idx in xrange(n):
            geos_geom = geoms[idx]
            geom = <GEOSGeometry *> geos_geom
            if geom is not NULL:
                out[idx] = <np.uintp_t> func(handle, geom)
            else:
                out[idx] = 0

    return GeometryArray(out)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef vector_binary_geo(str op,
                       np.ndarray[np.uintp_t, ndim=1, cast=True] left,
                       np.ndarray[np.uintp_t, ndim=1, cast=True] right):
    """
    Evaluate Geometry-valued function on two arrays of GEOSGeometry pointers

    Parameters
    ----------
    op: str
        string like 'intersection', or 'union'
    left: numpy.ndarray
        Array of pointers to GEOSGeometry objects
    right: numpy.ndarray
        Array of pointers to GEOSGeometry objects

    Returns
    -------
    out: array of pointers to GEOSGeometry objects
    """
    cdef Py_ssize_t idx
    cdef GEOSContextHandle_t handle
    cdef GEOSGeometry *left_geom
    cdef GEOSGeometry *right_geom
    cdef unsigned int n = left.size

    cdef np.ndarray[np.uintp_t, ndim=1, cast=True] out = np.empty(n, dtype=np.uintp)

    handle = get_geos_context_handle()

    if op == 'difference':
        func = GEOSDifference_r
    elif op == 'symmetric_difference':
        func = GEOSSymDifference_r
    elif op == 'union':
        func = GEOSUnion_r
    elif op == 'intersection':
        func = GEOSIntersection_r
    else:
        raise NotImplementedError("Op %s not known" % op)

    with nogil:
        for idx in xrange(n):
            left_geom = <GEOSGeometry *> left[idx]
            right_geom = <GEOSGeometry *> right[idx]
            if left_geom and right_geom:
                out[idx] = <np.uintp_t> func(handle, left_geom, right_geom)
            else:
                out[idx] = 0

    return GeometryArray(out)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef binary_geo(str op,
                np.ndarray[np.uintp_t, ndim=1, cast=True] left,
                object right):
    """
    Evaluate Geometry-valued function on arrays GEOSGeometry pointers and a
    shapely object

    Parameters
    ----------
    op: str
        string like 'intersection', or 'union'
    left: numpy.ndarray
        Array of pointers to GEOSGeometry objects
    right: shapely BaseGeometry

    Returns
    -------
    out: array of pointers to GEOSGeometry objects
    """
    cdef Py_ssize_t idx
    cdef GEOSContextHandle_t handle
    cdef GEOSGeometry *left_geom
    cdef GEOSGeometry *right_geom
    cdef uintptr_t right_ptr
    cdef unsigned int n = left.size

    cdef np.ndarray[np.uintp_t, ndim=1, cast=True] out = np.empty(n, dtype=np.uintp)

    right_ptr = <np.uintp_t> right.__geom__
    right_geom = <GEOSGeometry *> right_ptr

    handle = get_geos_context_handle()

    if op == 'difference':
        func = GEOSDifference_r
    elif op == 'symmetric_difference':
        func = GEOSSymDifference_r
    elif op == 'union':
        func = GEOSUnion_r
    elif op == 'intersection':
        func = GEOSIntersection_r
    else:
        raise NotImplementedError("Op %s not known" % op)

    with nogil:
        for idx in xrange(n):
            left_geom = <GEOSGeometry *> left[idx]
            if left_geom and right_geom:
                out[idx] = <np.uintp_t> func(handle, left_geom, right_geom)
            else:
                out[idx] = 0

    return GeometryArray(out)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef buffer(np.ndarray[np.uintp_t, ndim=1, cast=True] geoms, double distance,
            int resolution, int cap_style, int join_style, double mitre_limit):
    """ Buffer operation on array of GEOSGeometry objects """
    cdef Py_ssize_t idx
    cdef GEOSContextHandle_t handle
    cdef GEOSGeometry *geom
    cdef uintptr_t geos_geom
    cdef GEOSGeometry *other_geom
    cdef unsigned int n = geoms.size

    cdef np.ndarray[np.uintp_t, ndim=1, cast=True] out = np.empty(n, dtype=np.uintp)
    handle = get_geos_context_handle()

    with nogil:
        for idx in xrange(n):
            geos_geom = geoms[idx]
            geom = <GEOSGeometry *> geos_geom
            if geom is not NULL:
                out[idx] = <np.uintp_t> GEOSBufferWithStyle_r(handle, geom,
                        distance, resolution, cap_style, join_style, mitre_limit)
            else:
                out[idx] = 0

    return GeometryArray(out)



@cython.boundscheck(False)
@cython.wraparound(False)
cdef get_coordinate_point(np.ndarray[np.uintp_t, ndim=1, cast=True] geoms,
                          int coordinate):
    """ Get x, y, or z value for an array of points """
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
            if geom is not NULL:
                sequence = GEOSGeom_getCoordSeq_r(handle, geom)
                func(handle, sequence, 0, &value)
                out[idx] = value
            else:
                out[idx] = 0

    return out


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef serialize(np.ndarray[np.uintp_t, ndim=1, cast=True] geoms):
    """ Serialize array of pointers

    Returns
    -------
    out: bytearray of data
    sizes: array of sizes of wkb strings

    See Also
    --------
    deserialize
    """
    cdef Py_ssize_t idx
    cdef GEOSContextHandle_t handle
    cdef GEOSGeometry *geom
    cdef uintptr_t geos_geom
    cdef unsigned int n = geoms.size
    cdef size_t size
    cdef GEOSWKBWriter *writer
    cdef unsigned char* c_string
    cdef bytes py_string

    cdef np.ndarray[np.uintp_t, ndim=1] sizes = np.empty(n, dtype=np.uintp)
    cdef bytearray out = bytearray()

    handle = get_geos_context_handle()

    writer = GEOSWKBWriter_create_r(handle)

    for idx in xrange(n):
        geos_geom = geoms[idx]
        geom = <GEOSGeometry *> geos_geom
        if geom is not NULL:
            c_string = GEOSWKBWriter_write_r(handle, writer, geom, &size)
            py_string = c_string[:size]
            out.extend(py_string)
            free(c_string)
            sizes[idx] = size
        else:
            sizes[idx] = 0

    GEOSWKBWriter_destroy_r(handle, writer)
    return out, sizes


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef deserialize(const unsigned char* data, np.ndarray[np.uintp_t, ndim=1, cast=True] sizes):
    """ Serialize array of pointers

    Parameters
    -------
    out: bytearray of data
    sizes: array of sizes of wkb strings

    See Also
    --------
    serialize
    """
    cdef Py_ssize_t idx
    cdef GEOSContextHandle_t handle
    cdef GEOSGeometry *geom
    cdef unsigned int n = sizes.size
    cdef size_t size
    cdef GEOSWKBReader *reader

    cdef np.ndarray[np.uintp_t, ndim=1] out = np.empty(n, dtype=np.uintp)

    handle = get_geos_context_handle()

    reader = GEOSWKBReader_create_r(handle)

    with nogil:
        for idx in xrange(n):
            size = sizes[idx]
            if size:
                geom = GEOSWKBReader_read_r(handle, reader, data, size)
                out[idx] = <np.uintp_t> geom
            else:
                out[idx] = 0
            data += size  # march the data pointer forward

    GEOSWKBReader_destroy_r(handle, reader)
    return out


@cython.boundscheck(False)
@cython.wraparound(False)
cdef vec_free(np.ndarray[np.uintp_t, ndim=1, cast=True] geoms):
    """ Free an array of GEOSGeometry pointers """
    cdef Py_ssize_t idx
    cdef GEOSContextHandle_t handle
    cdef GEOSGeometry *geom
    cdef uintptr_t geos_geom
    cdef unsigned int n = geoms.size

    handle = get_geos_context_handle()

    with nogil:
        for idx in xrange(n):
            geos_geom = geoms[idx]
            geom = <GEOSGeometry *> geos_geom
            if geom is not NULL:
                GEOSGeom_destroy_r(handle, geom)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef geom_type(np.ndarray[np.uintp_t, ndim=1, cast=True] geoms):
    """ Free an array of GEOSGeometry pointers """
    cdef Py_ssize_t idx
    cdef GEOSContextHandle_t handle
    cdef GEOSGeometry *geom
    cdef unsigned int n = geoms.size

    cdef np.ndarray[np.int8_t, ndim=1] out = np.empty(n, dtype=np.int8)

    handle = get_geos_context_handle()

    with nogil:
        for idx in xrange(n):
            geom = <GEOSGeometry *> geoms[idx]
            if geom is NULL:
                out[idx] = -1
            else:
                out[idx] = GEOSGeomTypeId_r(handle, geom)

    return out


cdef unary_union(np.ndarray[np.uintp_t, ndim=1, cast=True] geoms):
    cdef GEOSContextHandle_t handle
    cdef GEOSGeometry *collection
    cdef GEOSGeometry *out
    cdef size_t n

    handle = get_geos_context_handle()
    geoms = geoms[geoms != 0]
    n = geoms.size

    with nogil:
        collection = GEOSGeom_createCollection_r(handle, GEOS_MULTIPOLYGON,
                                                 <GEOSGeometry **> geoms.data,
                                                 n)
        out = GEOSUnaryUnion_r(handle, collection)

    return geom_factory(<np.uintp_t> out)


class GeometryArray(object):
    dtype = np.dtype('O')

    def __init__(self, data, parent=False):
        self.data = data
        self.parent = parent

    def __getitem__(self, idx):
        if isinstance(idx, numbers.Integral):
            return get_element(self.data, idx)
        elif isinstance(idx, (collections.Iterable, slice)):
            return GeometryArray(self.data[idx], parent=self)
        else:
            raise TypeError("Index type not supported", idx)

    def take(self, idx):
        result = self[idx]
        result.data[idx == -1] = 0
        return result

    def __len__(self):
        return len(self.data)

    @property
    def size(self):
        return len(self.data)

    def __del__(self):
        if self.parent is False:
            vec_free(self.data)

    def copy(self):
        return self  # assume immutable for now

    @property
    def ndim(self):
        return 1

    def __getstate__(self):
        return serialize(self.data)

    def __setstate__(self, state):
        geoms = deserialize(*state)
        self.data = geoms
        self.parent = None

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
            return binary_geo(op, self.data, other)
        elif isinstance(other, GeometryArray):
            if len(self) != len(other):
                msg = ("Lengths of inputs to not match.  Left: %d, Right: %d" %
                        (len(self), len(other)))
                raise ValueError(msg)
            return vector_binary_geo(op, self.data, other.data)
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
                return binary_predicate_with_arg(op, self.data, other, extra)
            elif op in opposite_predicates:
                op2 = opposite_predicates[op]
                return prepared_binary_predicate(op2, self.data, other)
            else:
                return binary_predicate(op, self.data, other)
        elif isinstance(other, GeometryArray):
            if len(self) != len(other):
                msg = ("Shapes of inputs to not match.  Left: %d, Right: %d" %
                        (len(self), len(other)))
                raise ValueError(msg)
            if extra is not None:
                return vector_binary_predicate_with_arg(op, self.data, other.data, extra)
            else:
                return vector_binary_predicate(op, self.data, other.data)
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
        return unary_predicate('is_valid', self.data)

    def is_empty(self):
        return unary_predicate('is_empty', self.data)

    def is_simple(self):
        return unary_predicate('is_simple', self.data)

    def is_ring(self):
        return unary_predicate('is_ring', self.data)

    def has_z(self):
        return unary_predicate('has_z', self.data)

    def is_closed(self):
        return unary_predicate('is_closed', self.data)

    def boundary(self):
        return geo_unary_op('boundary', self.data)

    def centroid(self):
        return geo_unary_op('centroid', self.data)

    def convex_hull(self):
        return geo_unary_op('convex_hull', self.data)

    def envelope(self):
        return geo_unary_op('envelope', self.data)

    def exterior(self):
        out = geo_unary_op('exterior', self.data)
        out.parent = self  # exterior shares data with self
        return out

    def representative_point(self):
        return geo_unary_op('representative_point', self.data)

    def distance(self, other):
        if isinstance(other, GeometryArray):
            return binary_vector_float('distance', self.data, other.data)
        else:
            return binary_float('distance', self.data, other)

    def project(self, other, normalized=False):
        op = 'project' if not normalized else 'project-normalized'
        if isinstance(other, GeometryArray):
            return binary_vector_float_return(op, self.data, other.data)
        else:
            return binary_float_return(op, self.data, other)

    def area(self):
        return unary_vector_float('area', self.data)

    def length(self):
        return unary_vector_float('length', self.data)

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
        return buffer(self.data, distance, resolution, cap_style, join_style,
                      mitre_limit)

    def geom_type(self):
        """
        Types of the underlying Geometries

        Returns
        -------
        Pandas categorical with types for each geometry
        """
        x = geom_type(self.data)

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
        return unary_union(self.data)


cpdef cysjoin(np.ndarray[np.uintp_t, ndim=1, cast=True] left,
              np.ndarray[np.uintp_t, ndim=1, cast=True] right,
              str predicate_name):
    """ Spatial join

    Parameters
    ----------
    left: np.ndarray
        array of pointers to GEOSGeometry objects
    right : np.ndarray
        array of pointers to GEOSGeometry objects
    predicate_name: string
        contains, intersects, within, etc..

    Returns
    -------
    left_out: np.ndarray
        Array of indices to pass to take for the left side to match with right
    right_out: np.ndarray
        Array of indices to pass to take for the right side to match with left
    """
    cdef GEOSContextHandle_t handle
    cdef GEOSPreparedPredicate predicate
    cdef size_vector sv
    cdef Py_ssize_t idx
    cdef np.ndarray[np.uintp_t, ndim=1] left_out
    cdef np.ndarray[np.uintp_t, ndim=1] right_out
    cdef size_t left_size = left.size
    cdef size_t right_size = right.size

    handle = get_geos_context_handle()
    predicate = get_prepared_predicate(predicate_name)
    if not predicate:
        raise NotImplementedError(predicate_name)

    with nogil:
        sv = sjoin(handle, predicate,
                   <GEOSGeometry*> left.data, left_size,
                   <GEOSGeometry*> right.data, right_size)

    left_out = np.empty(sv.n // 2, dtype=np.uintp)
    right_out = np.empty(sv.n // 2, dtype=np.uintp)

    with nogil:
        for idx in range(0, sv.n // 2):
            left_out[idx] = sv.a[2 * idx]
            right_out[idx] = sv.a[2 * idx + 1]

    cfree(sv.a)
    return left_out, right_out
