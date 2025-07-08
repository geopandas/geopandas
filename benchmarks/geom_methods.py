import random

import numpy as np
from geopandas import GeoSeries
from shapely.geometry import Point, Polygon, MultiPolygon


def with_attributes(**attrs):
    def decorator(func):
        for key, value in attrs.items():
            setattr(func, key, value)
        return func

    return decorator


class Bench:
    def setup(self, *args):
        self.points = GeoSeries([Point(i, i) for i in range(100000)])

        triangles = GeoSeries(
            [
                Polygon([(random.random(), random.random()) for _ in range(3)])
                for _ in range(1000)
            ]
        )
        triangles2 = triangles.copy().iloc[np.random.choice(1000, 1000)]
        triangles3 = GeoSeries(
            [
                Polygon([(random.random(), random.random()) for _ in range(3)])
                for _ in range(10000)
            ]
        )
        triangles4 = GeoSeries(
            [
                MultiPolygon(
                    [Polygon([(random.random(), random.random()) for _ in range(3)])]
                )
                for _ in range(10000)
            ]
        )
        triangle = Polygon([(random.random(), random.random()) for _ in range(3)])
        self.triangles, self.triangles2 = triangles, triangles2
        self.triangles_big = triangles3
        self.multi_triangles = triangles4
        self.triangle = triangle

    @with_attributes(
        param_names=["op"],
        params=[
            (
                "contains",
                "crosses",
                "disjoint",
                "intersects",
                "overlaps",
                "touches",
                "within",
                "geom_equals",
                "geom_equals_exact",
            )
        ],
    )
    def time_binary_predicate(self, op):
        getattr(self.triangles, op)(self.triangle)

    @with_attributes(
        param_names=["op"],
        params=[
            (
                "contains",
                "crosses",
                "disjoint",
                "intersects",
                "overlaps",
                "touches",
                "within",
                "geom_equals",
            )
        ],
    )  # 'geom_equals_exact')])
    def time_binary_predicate_vector(self, op):
        getattr(self.triangles, op)(self.triangles2)

    @with_attributes(param_names=["op"], params=[("distance")])
    def time_binary_float(self, op):
        getattr(self.triangles, op)(self.triangle)

    @with_attributes(param_names=["op"], params=[("distance")])
    def time_binary_float_vector(self, op):
        getattr(self.triangles, op)(self.triangles2)

    @with_attributes(
        param_names=["op"],
        params=[("difference", "symmetric_difference", "union", "intersection")],
    )
    def time_binary_geo(self, op):
        getattr(self.triangles, op)(self.triangle)

    @with_attributes(
        param_names=["op"],
        params=[("difference", "symmetric_difference", "union", "intersection")],
    )
    def time_binary_geo_vector(self, op):
        getattr(self.triangles, op)(self.triangles2)

    @with_attributes(
        param_names=["op"], params=[("is_valid", "is_empty", "is_simple", "is_ring")]
    )
    def time_unary_predicate(self, op):
        getattr(self.triangles, op)

    @with_attributes(param_names=["op"], params=[("area", "length")])
    def time_unary_float(self, op):
        getattr(self.triangles_big, op)

    @with_attributes(
        param_names=["op"],
        params=[
            ("boundary", "centroid", "convex_hull", "envelope", "exterior", "interiors")
        ],
    )
    def time_unary_geo(self, op):
        getattr(self.triangles, op)

    def time_unary_geo_representative_point(self, *args):
        getattr(self.triangles, "representative_point")()

    def time_geom_type(self, *args):
        self.triangles_big.geom_type

    def time_bounds(self, *args):
        self.triangles.bounds

    def time_union_all(self, *args):
        self.triangles.union_all()

    def time_buffer(self, *args):
        self.points.buffer(2)

    def time_explode(self, *args):
        self.multi_triangles.explode()


# TODO
# project, interpolate, affine_transform, translate, rotate, scale, skew
# cx indexer
