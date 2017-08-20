#include <geos_c.h>
#include "kvec.h"

#ifndef GEOPANDAS_ALGOS_C
#define GEOPANDAS_ALGOS_C

typedef char (*GEOSPredicate)(GEOSContextHandle_t handle, const GEOSGeometry *left, const GEOSGeometry *right);
typedef char (*GEOSPreparedPredicate)(GEOSContextHandle_t handle, const GEOSPreparedGeometry *left, const GEOSGeometry *right);

/* A resizable vector
 * This is the same as kvec_t(size_t)
 */
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

#endif
