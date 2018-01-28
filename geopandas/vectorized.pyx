# distutils: sources = geopandas/algos.c

import collections
import numbers
from libc.stdlib cimport free as cfree


import shapely
from shapely.geometry.base import BaseGeometry, geom_factory
import shapely.prepared

import cython
cimport cpython.array
from libc.stdlib cimport malloc, free


cimport numpy as np
import numpy as np

include "geopandas/_geos.pxi"


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


cpdef get_element(np.ndarray[np.uintp_t, ndim=1, cast=True] geoms, int idx):
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

    #TODO(cython) per difference by adding those?
    # if isinstance(geom, BaseGeometry):
    #     out.append(geom)
    # elif hasattr(geom, '__geo_interface__'):
    #     geom = shapely.geometry.asShape(geom)
    #     out.append(geom)
    # elif geom is None or (isinstance(geom, float) and np.isnan(geom)):
    #     out.append(None)
    # else:
    #     raise ValueError("Input is not a valid geometry: {0}".format(geom))

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

    return out


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

    return out


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

    return out


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

    return out


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
    cdef GEOSGeometry *geom
    cdef uintptr_t other_pointer
    cdef GEOSGeometry *other_geom
    cdef GEOSPreparedGeometry *prepared_geom
    cdef GEOSPreparedPredicate predicate
    cdef unsigned int n = geoms.size
    cdef np.ndarray[np.uint8_t, ndim=1, cast=True] out = np.empty(n, dtype=np.bool_)

    other_pointer = <np.uintp_t> other.__geom__
    other_geom = <GEOSGeometry *> other_pointer

    with get_geos_handle() as handle:
        prepared_geom = GEOSPrepare_r(handle, other_geom)

        predicate = get_prepared_predicate(op)

        with nogil:
            for idx in xrange(n):
                geom = <GEOSGeometry *> geoms[idx]
                if geom != NULL:
                    out[idx] = predicate(handle, prepared_geom, geom)
                else:
                    out[idx] = 0

        GEOSPreparedGeom_destroy_r(handle, prepared_geom)

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
    cdef GEOSGeometry *left_geom
    cdef GEOSGeometry *right_geom
    cdef GEOSPredicate func
    cdef unsigned int n = left.size
    cdef np.ndarray[np.uint8_t, ndim=1, cast=True] out = np.empty(n, dtype=np.bool_)

    func = get_predicate(op)
    if not func:
        raise NotImplementedError(op)

    with get_geos_handle() as handle:
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
    cdef GEOSGeometry *left_geom
    cdef GEOSGeometry *right_geom
    cdef unsigned int n = left.size
    cdef np.ndarray[np.uint8_t, ndim=1, cast=True] out = np.empty(n, dtype=np.bool_)

    if op == 'equals_exact':
        func = GEOSEqualsExact_r
    else:
        raise NotImplementedError(op)

    with get_geos_handle() as handle:
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
    cdef GEOSGeometry *geom
    cdef unsigned int n = geoms.size
    cdef np.ndarray[np.uint8_t, ndim=1, cast=True] out = np.empty(n, dtype=np.bool_)

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

    with get_geos_handle() as handle:
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
    cdef GEOSGeometry *geom
    cdef GEOSGeometry *other_geom
    cdef uintptr_t other_pointer
    cdef unsigned int n = geoms.size
    cdef GEOSPredicate predicate
    cdef np.ndarray[np.uint8_t, ndim=1, cast=True] out = np.empty(n, dtype=np.bool_)

    other_pointer = <np.uintp_t> other.__geom__
    other_geom = <GEOSGeometry *> other_pointer

    predicate = get_predicate(op)

    with get_geos_handle() as handle:
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
    cdef GEOSGeometry *geom
    cdef GEOSGeometry *other_geom
    cdef uintptr_t other_pointer
    cdef unsigned int n = geoms.size
    cdef np.ndarray[np.uint8_t, ndim=1, cast=True] out = np.empty(n, dtype=np.bool_)

    other_pointer = <np.uintp_t> other.__geom__
    other_geom = <GEOSGeometry *> other_pointer

    if op == 'equals_exact':
        func = GEOSEqualsExact_r
    else:
        raise NotImplementedError(op)

    with get_geos_handle() as handle:
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
    cdef GEOSGeometry *geom
    cdef unsigned int n = geoms.size
    cdef double nan = np.nan
    cdef double * location
    cdef np.ndarray[double, ndim=1, cast=True] out = np.empty(n, dtype=np.float64)

    location = <double *> out.data  # need to pass a pointer to function

    if op == 'area':
        func = GEOSArea_r
    elif op == 'length':
        func = GEOSLength_r
    else:
        raise NotImplementedError(op)

    with get_geos_handle() as handle:
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
    cdef GEOSGeometry *left_geom
    cdef GEOSGeometry *right_geom
    cdef unsigned int n = left.size
    cdef double nan = np.nan
    cdef double * location
    cdef np.ndarray[double, ndim=1, cast=True] out = np.empty(n, dtype=np.float64)

    location = <double *> out.data  # need to pass a pointer to function

    if op == 'distance':
        func = GEOSDistance_r
    else:
        raise NotImplementedError(op)

    with get_geos_handle() as handle:
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
    cdef GEOSGeometry *left_geom
    cdef GEOSGeometry *right_geom
    cdef unsigned int n = left.size
    cdef double nan = np.nan
    cdef np.ndarray[double, ndim=1, cast=True] out = np.empty(n, dtype=np.float64)

    if op == 'project':
        func = GEOSProject_r
    elif op == 'project-normalized':
        func = GEOSProjectNormalized_r
    else:
        raise NotImplementedError(op)

    with get_geos_handle() as handle:
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

    if op == 'distance':
        func = GEOSDistance_r
    else:
        raise NotImplementedError(op)

    with get_geos_handle() as handle:
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
    cdef uintptr_t right_ptr
    cdef GEOSGeometry *left_geom
    cdef GEOSGeometry *right_geom
    cdef unsigned int n = left.size
    cdef double nan = np.nan
    cdef np.ndarray[double, ndim=1, cast=True] out = np.empty(n, dtype=np.float64)

    right_ptr = <np.uintp_t> right.__geom__
    right_geom = <GEOSGeometry *> right_ptr

    if op == 'project':
        func = GEOSProject_r
    elif op == 'project-normalized':
        func = GEOSProjectNormalized_r
    else:
        raise NotImplementedError(op)

    with get_geos_handle() as handle:
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
    cdef GEOSGeometry *geom
    cdef uintptr_t geos_geom
    cdef GEOSGeometry *other_geom
    cdef unsigned int n = geoms.size
    cdef np.ndarray[np.uintp_t, ndim=1, cast=True] out = np.empty(n, dtype=np.uintp)

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

    with get_geos_handle() as handle:
        with nogil:
            for idx in xrange(n):
                geos_geom = geoms[idx]
                geom = <GEOSGeometry *> geos_geom
                if geom is not NULL:
                    out[idx] = <np.uintp_t> func(handle, geom)
                else:
                    out[idx] = 0

    return out


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
    cdef GEOSGeometry *left_geom
    cdef GEOSGeometry *right_geom
    cdef unsigned int n = left.size

    cdef np.ndarray[np.uintp_t, ndim=1, cast=True] out = np.empty(n, dtype=np.uintp)

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

    with get_geos_handle() as handle:
        with nogil:
            for idx in xrange(n):
                left_geom = <GEOSGeometry *> left[idx]
                right_geom = <GEOSGeometry *> right[idx]
                if left_geom and right_geom:
                    out[idx] = <np.uintp_t> func(handle, left_geom, right_geom)
                else:
                    out[idx] = 0

    return out


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
    cdef GEOSGeometry *left_geom
    cdef GEOSGeometry *right_geom
    cdef uintptr_t right_ptr
    cdef unsigned int n = left.size

    cdef np.ndarray[np.uintp_t, ndim=1, cast=True] out = np.empty(n, dtype=np.uintp)

    right_ptr = <np.uintp_t> right.__geom__
    right_geom = <GEOSGeometry *> right_ptr

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

    with get_geos_handle() as handle:
        with nogil:
            for idx in xrange(n):
                left_geom = <GEOSGeometry *> left[idx]
                if left_geom and right_geom:
                    out[idx] = <np.uintp_t> func(handle, left_geom, right_geom)
                else:
                    out[idx] = 0

    return out


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef buffer(np.ndarray[np.uintp_t, ndim=1, cast=True] geoms, double distance,
            int resolution, int cap_style, int join_style, double mitre_limit):
    """ Buffer operation on array of GEOSGeometry objects """
    cdef Py_ssize_t idx
    cdef GEOSGeometry *geom
    cdef uintptr_t geos_geom
    cdef GEOSGeometry *other_geom
    cdef unsigned int n = geoms.size

    cdef np.ndarray[np.uintp_t, ndim=1, cast=True] out = np.empty(n, dtype=np.uintp)

    with get_geos_handle() as handle:
        with nogil:
            for idx in xrange(n):
                geos_geom = geoms[idx]
                geom = <GEOSGeometry *> geos_geom
                if geom is not NULL:
                    out[idx] = <np.uintp_t> GEOSBufferWithStyle_r(handle, geom,
                            distance, resolution, cap_style, join_style, mitre_limit)
                else:
                    out[idx] = 0

    return out



@cython.boundscheck(False)
@cython.wraparound(False)
cdef get_coordinate_point(np.ndarray[np.uintp_t, ndim=1, cast=True] geoms,
                          int coordinate):
    """ Get x, y, or z value for an array of points """
    cdef Py_ssize_t idx
    cdef GEOSGeometry *geom
    cdef GEOSCoordSequence *sequence
    cdef unsigned int n = geoms.size
    cdef double value

    cdef np.ndarray[double, ndim=1, cast=True] out = np.empty(n, dtype=np.float)

    if coordinate == 0:
        func = GEOSCoordSeq_getX_r
    elif coordinate == 1:
        func = GEOSCoordSeq_getY_r
    elif coordinate == 2:
        func = GEOSCoordSeq_getZ_r
    else:
        raise NotImplementedError("Coordinate must be between 0-x, 1-y, 2-z")

    with get_geos_handle() as handle:
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
    cdef GEOSGeometry *geom
    cdef uintptr_t geos_geom
    cdef unsigned int n = geoms.size
    cdef size_t size
    cdef GEOSWKBWriter *writer
    cdef unsigned char* c_string
    cdef bytes py_string

    cdef np.ndarray[np.uintp_t, ndim=1] sizes = np.empty(n, dtype=np.uintp)
    cdef bytearray out = bytearray()

    with get_geos_handle() as handle:
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
    cdef GEOSGeometry *geom
    cdef unsigned int n = sizes.size
    cdef size_t size
    cdef GEOSWKBReader *reader

    cdef np.ndarray[np.uintp_t, ndim=1] out = np.empty(n, dtype=np.uintp)

    with get_geos_handle() as handle:
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


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef geom_type(np.ndarray[np.uintp_t, ndim=1, cast=True] geoms):
    """ Free an array of GEOSGeometry pointers """
    cdef Py_ssize_t idx
    cdef GEOSGeometry *geom
    cdef unsigned int n = geoms.size

    cdef np.ndarray[np.int8_t, ndim=1] out = np.empty(n, dtype=np.int8)

    with get_geos_handle() as handle:
        with nogil:
            for idx in xrange(n):
                geom = <GEOSGeometry *> geoms[idx]
                if geom is NULL:
                    out[idx] = -1
                else:
                    out[idx] = GEOSGeomTypeId_r(handle, geom)

    return out


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef unary_union(np.ndarray[np.uintp_t, ndim=1, cast=True] geoms):
    cdef GEOSGeometry *collection
    cdef GEOSGeometry *out
    cdef size_t n

    with get_geos_handle() as handle:
        geoms = geoms[geoms != 0]
        n = geoms.size

        with nogil:
            collection = GEOSGeom_createCollection_r(handle, GEOS_MULTIPOLYGON,
                                                     <GEOSGeometry **> geoms.data,
                                                     n)
            out = GEOSUnaryUnion_r(handle, collection)

    return geom_factory(<np.uintp_t> out)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef coords(np.ndarray[np.uintp_t, ndim=1, cast=True] geoms):
    """ Get coordinates of LineStrings or Points

    Parameters
    ----------
    geoms: np.ndarray
        Array of pointers to GEOSGeometry objects
        These must be either LineString objects, Point objects, or NULL

    Returns
    -------
    out: list
        List of tuples of coordinate points
    """
    cdef Py_ssize_t i, j
    cdef GEOSGeometry *geom
    cdef GEOSCoordSequence *sequence
    cdef unsigned int n = geoms.size
    cdef double x, y, z
    cdef char has_z

    with get_geos_handle() as handle:
        out = []

        for i in xrange(n):
            geom = <GEOSGeometry *> geoms[i]
            if geom is NULL:
                out.append(())
                continue

            sequence = GEOSGeom_getCoordSeq_r(handle, geom)
            if sequence is NULL:
                raise TypeError("Geometry must be LineString or Point")

            L = []

            has_z = GEOSHasZ_r(handle, geom)

            for j in xrange(GEOSGetNumCoordinates_r(handle, geom)):
                GEOSCoordSeq_getX_r(handle, sequence, j, &x)
                GEOSCoordSeq_getY_r(handle, sequence, j, &y)
                if has_z:
                    GEOSCoordSeq_getZ_r(handle, sequence, j, &z)
                    L.append(tuple((float(x), float(y), float(z))))
                else:
                    L.append(tuple((float(x), float(y))))
            out.append(tuple(L))

    return out


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
    cdef GEOSPreparedPredicate predicate
    cdef size_vector sv
    cdef Py_ssize_t idx
    cdef np.ndarray[np.uintp_t, ndim=1] left_out
    cdef np.ndarray[np.uintp_t, ndim=1] right_out
    cdef size_t left_size = left.size
    cdef size_t right_size = right.size

    with get_geos_handle() as handle:
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


cdef class get_geos_handle:
    cdef GEOSContextHandle_t handle

    cdef GEOSContextHandle_t __enter__(self):
        self.handle = GEOS_init_r()
        return self.handle

    def __exit__(self, type, value, traceback):
        GEOS_finish_r(self.handle)
