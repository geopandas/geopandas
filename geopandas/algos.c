
#include <geos_c.h>
#include "kvec.h"
#include "algos.h"
#include <stdio.h>
#include <time.h>
#include <stdbool.h>

typedef char (*GEOSPredicate)(GEOSContextHandle_t handle, const GEOSGeometry *left, const GEOSGeometry *right);
typedef char (*GEOSPreparedPredicate)(GEOSContextHandle_t handle, const GEOSPreparedGeometry *left, const GEOSGeometry *right);


typedef struct
{
    size_t n, m;
    GEOSGeometry **a;
} geom_vector;


/* Callback to give to strtree_query
 * Given the value returned from each intersecting geometry it inserts that
 * value (typically an index) into the given size_vector */
void strtree_query_callback(void *item, void *vec)
{
    kv_push(size_t, *((size_vector*) vec), (size_t) item);
}


/* Create STRTree spatial index from an array of Geometries */
GEOSSTRtree *create_index(GEOSContextHandle_t handle, GEOSGeometry **geoms, size_t n)
{
    GEOSSTRtree* tree = GEOSSTRtree_create_r(handle, n);

    for (int i = 0; i < n ; i++)
    {
        if (geoms[i] != NULL)
        {
            GEOSSTRtree_insert_r(handle, tree, geoms[i], (void*) i);
        }
    }
    return tree;
}


/* Spatial join of two arrays of geometries over the predicate
 * This creates an index for the right side
 * Finds all polygons in both sides that intersect with each other
 * Then filters by the spatial predicate like intersects, contains, etc.
 * This returns an array of indices in each side that match
 * Organized in a [left_0, right_0, left_1, right_1, ... ] order
 */
size_vector sjoin(GEOSContextHandle_t handle,
                  GEOSPreparedPredicate predicate,
                  GEOSGeometry **left, size_t nleft,
                  GEOSGeometry **right, size_t nright)
{
    // clock_t begin, begin_1, begin_2;
    // clock_t end, end_1, end_2;
    // double time_spent = 0, time_spent_1 = 0, time_spent_2 = 0;

    size_t l, r;                    // indices for left and right sides
    GEOSPreparedGeometry* prepared; // Temporary prepared geometry for right side
    size_vector out;                // output array of matching indices
    size_vector vec;                // Temporary array for matches for each geometry
    kv_init(out);
    kv_init(vec);

    // begin = clock();

    GEOSSTRtree* tree = create_index(handle, right, nright);

    // end = clock();
    // time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
    // printf("Create tree, %f\n", time_spent);
    // begin = end;

    for (l = 0; l < nleft; l++)
    {
        if (left[l] != NULL)
        {
            // begin_1 = clock();
            GEOSSTRtree_query_r(handle, tree, left[l], strtree_query_callback, &vec);
            // end_1 = clock();
            // time_spent_1 += (double)(end_1 - begin_1) / CLOCKS_PER_SEC;

            prepared = GEOSPrepare_r(handle, left[l]);

            // begin_2 = clock();
            while (vec.n)
            {
                r = kv_pop(vec);
                if (predicate(handle, prepared, right[r])){
                    kv_push(size_t, out, l);
                    kv_push(size_t, out, r);
                }
            }
            GEOSPreparedGeom_destroy_r(handle, prepared);
            // end_2 = clock();
            // time_spent_2 += (double)(end_2 - begin_2) / CLOCKS_PER_SEC;
        }
    }
    // end = clock();
    // time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
    // printf("Intersect, %f\n", time_spent);
    // printf("query, %f\n", time_spent_1);
    // printf("intersect, %f\n", time_spent_2);

    GEOSSTRtree_destroy_r(handle, tree);
    kv_destroy(vec);
    return out;
}
