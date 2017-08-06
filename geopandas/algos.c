
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

void strtree_query_callback(void *item, void *vec)
{
    kv_push(size_t, *((size_vector*) vec), (size_t) item);
}


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


size_vector sjoin(GEOSContextHandle_t handle,
                  GEOSPreparedPredicate predicate,
                  GEOSGeometry **left, size_t nleft,
                  GEOSGeometry **right, size_t nright)
{
    // clock_t begin, begin_1, begin_2;
    // clock_t end, end_1, end_2;
    // double time_spent = 0, time_spent_1 = 0, time_spent_2 = 0;

    size_t l, r;
    size_vector out;
    kv_init(out);
    size_vector vec;
    kv_init(vec);
    GEOSPreparedGeometry* prepared;

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


void process_polygon_rings(GEOSContextHandle_t handle, GEOSGeometry *geom, geom_vector *rings)
{
    int n_interior;
    GEOSGeometry *ring;

    if (!GEOSisValid_r(handle, geom))  // need to cleaup here?
        geom = GEOSBufferWithStyle_r(handle, geom, 0, 16, 1, 1, 5);

    // Exterior Ring
    ring = GEOSGetExteriorRing_r(handle, geom);
    kv_push(GEOSGeometry*, *rings, ring);

    // Interior Rings
    n_interior = GEOSGetNumInteriorRings_r(handle, geom);
    for (int i = 0; i < n_interior; i++)
    {
        ring = GEOSGetInteriorRingN_r(handle, geom, i);
        kv_push(GEOSGeometry*, *rings, ring);
    }
}


geom_vector extract_rings(GEOSContextHandle_t handle, GEOSGeometry **geoms, size_t n)
{
    GEOSGeometry *geom, *sub, *out;
    geom_vector rings;
    kv_init(rings);
    int n_geoms, i, j;
    enum GEOSGeomTypes type;

    for(i = 0; i < n; i++)
    {
        geom = geoms[i];
        type = GEOSGeomTypeId_r(handle, geom);
        if (type == 3)  // Polygon
        {
            process_polygon_rings(handle, geom, &rings);
        }
        else if (type == 6) // MultiPolygon
        {
            n_geoms = GEOSGetNumGeometries_r(handle, geom);
            for (j = 0; j < n_geoms; j++)
            {
                sub = GEOSGetGeometryN_r(handle, geom, j);
                process_polygon_rings(handle, sub, &rings);
            }
        }
        else
        {
            kv_destroy(rings);
            return rings;
        }
    }

    return rings;
}



size_vector overlay(GEOSContextHandle_t handle, int how,
                    GEOSGeometry **left_geoms, size_t n_left,
                    GEOSGeometry **right_geoms, size_t n_right)
{
    clock_t begin, begin_1, begin_2;
    clock_t end, end_1, end_2;
    double time_spent = 0, time_spent_1 = 0, time_spent_2 = 0;

    GEOSGeometry **all_rings, *polygons, *poly, *point, *left_geom, *right_geom, *collection, *uunion;
    geom_vector left_rings, right_rings;
    GEOSSTRtree *left_tree, *right_tree;
    int n_polys, left_ind = 0, right_ind = 0, i;
    bool hit_left, hit_right, hit;
    size_vector vec;
    kv_init(vec);

    size_vector out;
    kv_init(out);

    begin = clock();
    left_rings = extract_rings(handle, left_geoms, n_left);
    right_rings = extract_rings(handle, right_geoms, n_right);
    all_rings = malloc(sizeof(GEOSGeometry *) * (left_rings.n + right_rings.n));

    end = clock();
    time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
    printf("extract rings, %f\n", time_spent);
    begin = end;


    for (i = 0; i < left_rings.n; i++)
        all_rings[i] = left_rings.a[i];
    for (i = 0; i < right_rings.n; i++)
        all_rings[i + left_rings.n] = right_rings.a[i];

    begin = clock();

    collection = GEOSGeom_createCollection_r(handle, GEOS_MULTIPOLYGON, all_rings, left_rings.n + right_rings.n);
    uunion = GEOSUnaryUnion_r(handle, collection);
    end = clock();
    time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
    printf("unary union, %f\n", time_spent);
    begin = end;

    free(all_rings);
    n_polys = GEOSGetNumGeometries_r(handle, uunion);
    all_rings = malloc(sizeof(GEOSGeometry *) * n_polys);
    for (i = 0; i < n_polys; i++)
        all_rings[i] = GEOSGetGeometryN_r(handle, uunion, i);
    end = clock();
    time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
    printf("copy geometries, %f\n", time_spent);
    begin = end;

    polygons = GEOSPolygonize_r(handle, all_rings, n_polys);
    n_polys = GEOSGetNumGeometries_r(handle, polygons);

    end = clock();
    time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
    printf("copy geometries, %f\n", time_spent);
    begin = end;

    end = clock();
    time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
    printf("polygonize, %f\n", time_spent);
    begin = end;

    begin = clock();

    left_tree = create_index(handle, left_geoms, n_left);
    right_tree = create_index(handle, right_geoms, n_right);

    end = clock();
    time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
    printf("create index, %f\n", time_spent);
    begin = end;


    begin = clock();

    for (int i; i < n_polys; i++)
    {
        poly = GEOSGetGeometryN_r(handle, polygons, i);
        point = GEOSPointOnSurface_r(handle, poly);

        hit_left = false;
        hit_right = false;

        GEOSSTRtree_query_r(handle, left_tree, point, strtree_query_callback, &vec);
        while (vec.n)
        {
            left_ind = kv_pop(vec);
            left_geom = left_geoms[left_ind];
            if (GEOSIntersects_r(handle, left_geom, point))
            {
                hit_left = true;
                while (vec.n)
                    kv_pop(vec);
                break;
            }
        }

        GEOSSTRtree_query_r(handle, right_tree, point, strtree_query_callback, &vec);
        while (vec.n)
        {
            right_ind = kv_pop(vec);
            right_geom = right_geoms[right_ind];
            if (GEOSIntersects_r(handle, right_geom, point))
            {
                hit_right = true;
                while (vec.n)
                    kv_pop(vec);
                break;
            }
        }

        hit = false;
        if (how == INTERSECTION && (hit_left && hit_right))
            hit = true;
        else if (how == UNION && (hit_left || hit_right))
            hit = true;
        else if (how == IDENTITY && hit_left)
            hit = true;
        else if (how == SYMMETRIC_DIFFERENCE && !(hit_left && hit_right))
            hit = true;
        else if (how == DIFFERENCE && (hit_left && !hit_right))
            hit = true;

        if (hit)
        {
            kv_push(size_t, out, poly);
            kv_push(size_t, out, left_ind);
            kv_push(size_t, out, right_ind);
            GEOSGeom_destroy_r(handle, point);
        }
    }

    end = clock();
    time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
    printf("query, %f\n", time_spent);
    begin = end;

    // GEOSGeom_destroy_r(handle, polygons);  // destroys used data
    free(all_rings);
    kv_destroy(vec);
    kv_destroy(left_rings);
    kv_destroy(right_rings);
    return out;
}
