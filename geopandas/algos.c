
#include <geos_c.h>
#include "kvec.h"

#ifndef GEOPANDAS_ALGOS_C
#define GEOPANDAS_ALGOS_C

typedef void (*GEOMPredicate)(GEOSContextHandle_t handle, GEOSGeometry *left, GEOSGeometry *right);

typedef struct 
{
    size_t n, m;
    size_t *a;
} size_vector;

void sjoin_callback(void *item, void *vec)
{
    kv_push(size_t, *((size_vector*) vec), (size_t) item);
}

size_vector sjoin(GEOSContextHandle_t handle, 
           GEOMPredicate predicate,
           GEOSGeometry **left, size_t nleft, 
           GEOSGeometry **right, size_t nright)
{
    size_t l, r;
    size_t index;
    size_vector out;
    kv_init(out);
    size_vector vec;
    kv_init(vec);

    GEOSSTRtree* tree = GEOSSTRtree_create_r(handle, nleft);

    for (l = 0; l < nleft; l++)
    {
        GEOSSTRtree_insert_r(handle, tree, left[l], (void*) l);
    }

    for (r = 0; r < nright; r++)
    {
        GEOSSTRtree_query_r(handle, tree, right[r], sjoin_callback, &vec);
        while (vec.n)
        {
            l = kv_pop(vec);
            kv_push(size_t, out, l);
            kv_push(size_t, out, r);
        }
    }

    GEOSSTRtree_destroy_r(handle, tree);
    kv_destroy(vec);
    return out;
}

#endif
