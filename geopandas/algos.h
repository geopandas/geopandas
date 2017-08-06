#include <geos_c.h>
#include "kvec.h"

#ifndef GEOPANDAS_ALGOS_C
#define GEOPANDAS_ALGOS_C

typedef char (*GEOSPredicate)(GEOSContextHandle_t handle, const GEOSGeometry *left, const GEOSGeometry *right);
typedef char (*GEOSPreparedPredicate)(GEOSContextHandle_t handle, const GEOSPreparedGeometry *left, const GEOSGeometry *right);


enum overlay_strategy {INTERSECTION, UNION, DIFFERENCE, SYMMETRIC_DIFFERENCE, IDENTITY};

typedef struct
{
    size_t n, m;
    size_t *a;
} size_vector;

void sjoin_callback(void *item, void *vec);

size_vector sjoin(GEOSContextHandle_t handle,
                  GEOSPreparedPredicate predicate,
                  GEOSGeometry **left, size_t nleft,
                  GEOSGeometry **right, size_t nright);

size_vector overlay(GEOSContextHandle_t handle, int how,
                    GEOSGeometry **left, size_t n_left,
                    GEOSGeometry **right, size_t n_right);
#endif
