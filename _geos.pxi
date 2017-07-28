# The beginnings of a Cython definition of GEOS. In the future much of this
# could be auto-generated.

from libc.stdint cimport uintptr_t


cdef extern from "geos_c.h":
    ctypedef void *GEOSContextHandle_t
    ctypedef struct GEOSGeometry
    ctypedef struct GEOSCoordSequence
    ctypedef struct GEOSPreparedGeometry

    GEOSCoordSequence *GEOSCoordSeq_create_r(GEOSContextHandle_t, unsigned int, unsigned int) nogil
    GEOSCoordSequence *GEOSGeom_getCoordSeq_r(GEOSContextHandle_t, GEOSGeometry *) nogil

    int GEOSCoordSeq_getSize_r(GEOSContextHandle_t, GEOSCoordSequence *, unsigned int *) nogil
    int GEOSCoordSeq_setX_r(GEOSContextHandle_t, GEOSCoordSequence *, int, double) nogil
    int GEOSCoordSeq_setY_r(GEOSContextHandle_t, GEOSCoordSequence *, int, double) nogil
    int GEOSCoordSeq_setZ_r(GEOSContextHandle_t, GEOSCoordSequence *, int, double) nogil
    int GEOSCoordSeq_getX_r(GEOSContextHandle_t, GEOSCoordSequence *, int, double *) nogil
    int GEOSCoordSeq_getY_r(GEOSContextHandle_t, GEOSCoordSequence *, int, double *) nogil
    int GEOSCoordSeq_getZ_r(GEOSContextHandle_t, GEOSCoordSequence *, int, double *) nogil

    GEOSGeometry *GEOSGeom_createPoint_r(GEOSContextHandle_t, GEOSCoordSequence *) nogil
    GEOSGeometry *GEOSGeom_createLineString_r(GEOSContextHandle_t, GEOSCoordSequence *) nogil
    GEOSGeometry *GEOSGeom_createLinearRing_r(GEOSContextHandle_t, GEOSCoordSequence *) nogil
    GEOSGeometry *GEOSGeom_clone_r(GEOSContextHandle_t, GEOSGeometry *) nogil
    GEOSCoordSequence *GEOSCoordSeq_clone_r(GEOSContextHandle_t, GEOSCoordSequence *) nogil
    GEOSGeometry *GEOSGeom_createEmptyPolygon() 
    GEOSGeometry *GEOSGeom_createEmptyPolygon_r(GEOSContextHandle_t handle) nogil


    void GEOSGeom_destroy_r(GEOSContextHandle_t, GEOSGeometry *) nogil

    char GEOSContains_r(GEOSContextHandle_t, const GEOSGeometry*, const GEOSGeometry*) nogil
    char GEOSCoveredBy_r(GEOSContextHandle_t, const GEOSGeometry*, const GEOSGeometry*) nogil
    char GEOSCovers_r(GEOSContextHandle_t, const GEOSGeometry*, const GEOSGeometry*) nogil
    char GEOSCrosses_r(GEOSContextHandle_t, const GEOSGeometry*, const GEOSGeometry*) nogil
    char GEOSDisjoint_r(GEOSContextHandle_t, const GEOSGeometry*, const GEOSGeometry*) nogil
    char GEOSIntersects_r(GEOSContextHandle_t, const GEOSGeometry*, const GEOSGeometry*) nogil
    char GEOSOverlaps_r(GEOSContextHandle_t, const GEOSGeometry*, const GEOSGeometry*) nogil
    char GEOSTouches_r(GEOSContextHandle_t, const GEOSGeometry*, const GEOSGeometry*) nogil
    char GEOSWithin_r(GEOSContextHandle_t, const GEOSGeometry*, const GEOSGeometry*) nogil

    char GEOSPreparedContains_r(GEOSContextHandle_t, const GEOSPreparedGeometry*, const GEOSGeometry*) nogil
    char GEOSPreparedContainsProperly_r(GEOSContextHandle_t, const GEOSPreparedGeometry*, const GEOSGeometry*) nogil
    char GEOSPreparedCoveredBy_r(GEOSContextHandle_t, const GEOSPreparedGeometry*, const GEOSGeometry*) nogil
    char GEOSPreparedCovers_r(GEOSContextHandle_t, const GEOSPreparedGeometry*, const GEOSGeometry*) nogil
    char GEOSPreparedCrosses_r(GEOSContextHandle_t, const GEOSPreparedGeometry*, const GEOSGeometry*) nogil
    char GEOSPreparedDisjoint_r(GEOSContextHandle_t, const GEOSPreparedGeometry*, const GEOSGeometry*) nogil
    char GEOSPreparedIntersects_r(GEOSContextHandle_t, const GEOSPreparedGeometry*, const GEOSGeometry*) nogil
    char GEOSPreparedOverlaps_r(GEOSContextHandle_t, const GEOSPreparedGeometry*, const GEOSGeometry*) nogil
    char GEOSPreparedTouches_r(GEOSContextHandle_t, const GEOSPreparedGeometry*, const GEOSGeometry*) nogil
    char GEOSPreparedWithin_r(GEOSContextHandle_t, const GEOSPreparedGeometry*, const GEOSGeometry*) nogil

    GEOSPreparedGeometry *GEOSPrepare_r(GEOSContextHandle_t handle, const GEOSGeometry* g) nogil

    char GEOSHasZ(const GEOSGeometry*) nogil
    char GEOSHasZ_r(GEOSContextHandle_t, GEOSGeometry *) nogil
    char GEOSisRing_r(GEOSContextHandle_t, GEOSGeometry *) nogil
    char GEOSisClosed_r(GEOSContextHandle_t, GEOSGeometry *) nogil

    char GEOSEquals(const GEOSGeometry * , const GEOSGeometry *) nogil
    char GEOSEquals_r(GEOSContextHandle_t , const GEOSGeometry * , const GEOSGeometry *) nogil
    char GEOSEqualsExact(const GEOSGeometry * , const GEOSGeometry * , double) nogil
    char GEOSEqualsExact_r(GEOSContextHandle_t , const GEOSGeometry * , const GEOSGeometry * , double) nogil

    GEOSGeometry *GEOSDifference(const GEOSGeometry*, const GEOSGeometry*)
    GEOSGeometry *GEOSSymDifference(const GEOSGeometry*, const GEOSGeometry*)
    GEOSGeometry *GEOSBoundary(const GEOSGeometry*)
    GEOSGeometry *GEOSUnion(const GEOSGeometry*, const GEOSGeometry*)
    GEOSGeometry *GEOSUnaryUnion(const GEOSGeometry*)
    GEOSGeometry *GEOSGetCentroid(const GEOSGeometry*)

    GEOSGeometry *GEOSDifference_r(GEOSContextHandle_t, const GEOSGeometry*, const GEOSGeometry*) nogil
    GEOSGeometry *GEOSSymDifference_r(GEOSContextHandle_t, const GEOSGeometry*, const GEOSGeometry*) nogil
    GEOSGeometry *GEOSBoundary_r(GEOSContextHandle_t, const GEOSGeometry*) nogil
    GEOSGeometry *GEOSUnion_r(GEOSContextHandle_t, const GEOSGeometry*, const GEOSGeometry*) nogil
    GEOSGeometry *GEOSUnaryUnion_r(GEOSContextHandle_t, const GEOSGeometry*) nogil
    GEOSGeometry *GEOSGetCentroid_r(GEOSContextHandle_t, const GEOSGeometry*) nogil
    GEOSGeometry *GEOSPointOnSurface_r(GEOSContextHandle_t, const GEOSGeometry*) nogil


    GEOSGeometry *GEOSConvexHull_r(GEOSContextHandle_t, const GEOSGeometry*) nogil
    GEOSGeometry *GEOSConvexHull(GEOSContextHandle_t, const GEOSGeometry*)
    GEOSGeometry *GEOSEnvelope(const GEOSGeometry*)
    GEOSGeometry *GEOSEnvelope_r(GEOSContextHandle_t, const GEOSGeometry*) nogil
    GEOSGeometry *GEOSGetExteriorRing(const GEOSGeometry*)
    GEOSGeometry *GEOSGetExteriorRing_r(GEOSContextHandle_t, const GEOSGeometry*) nogil

    int GEOSGeomTypeId(const GEOSGeometry*) nogil
    int GEOSGeomTypeId_r(GEOSContextHandle_t, const GEOSGeometry*) nogil

    int GEOSArea_r(GEOSContextHandle_t, const GEOSGeometry*, double *) nogil
    int GEOSLength_r(GEOSContextHandle_t, const GEOSGeometry*, double *) nogil
    int GEOSDistance_r(GEOSContextHandle_t, const GEOSGeometry*, const GEOSGeometry*, double *) nogil
    int GEOSDistanceIndexed_r(GEOSContextHandle_t, const GEOSGeometry*, const GEOSGeometry*, double *) nogil
    int GEOSHausdorffDistance_r(GEOSContextHandle_t, const GEOSGeometry *, const GEOSGeometry *, double *) nogil
    int GEOSHausdorffDistanceDensify_r(GEOSContextHandle_t, const GEOSGeometry *, const GEOSGeometry *,double, double *) nogil
    int GEOSFrechetDistance_r(GEOSContextHandle_t, const GEOSGeometry *,const GEOSGeometry *, double *) nogil
    int GEOSFrechetDistanceDensify_r(GEOSContextHandle_t, const GEOSGeometry *, const GEOSGeometry *, double , double *) nogil
    int GEOSGeomGetLength_r(GEOSContextHandle_t, const GEOSGeometry *, double *) nogil


    GEOSGeometry *GEOSBufferWithStyle_r(GEOSContextHandle_t, const GEOSGeometry*, double, int, int, int, double) nogil






cdef GEOSContextHandle_t get_geos_context_handle():
    # Note: This requires that lgeos is defined, so needs to be imported as:
    from shapely.geos import lgeos
    cdef uintptr_t handle = lgeos.geos_handle
    return <GEOSContextHandle_t>handle


cdef GEOSPreparedGeometry *geos_from_prepared(shapely_geom) except *:
    """Get the Prepared GEOS geometry pointer from the given shapely geometry."""
    cdef uintptr_t geos_geom = shapely_geom.__geom__
    return <GEOSPreparedGeometry *>geos_geom
