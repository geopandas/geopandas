import random

from geopandas import GeoSeries
from shapely.geometry import Point, LineString, Polygon


def with_attributes(**attrs):
    def decorator(func):
        for key, value in attrs.items():
            setattr(func, key, value)
        return func
    return decorator


class Bench:

    def setup(self, *args):
        self.s_points = GeoSeries([Point(i, i) for i in range(100000)])

        triangles = GeoSeries([Polygon([(random.random(), random.random())
                                        for _ in range(3)])
                               for _ in range(1000)])
        self.triangle = Polygon([(random.random(), random.random())
                                 for _ in range(3)])
        self.triangles = triangles

    @with_attributes(param_names=['op'],
                     params=[('contains', 'crosses', 'disjoint', 'intersects',
                              'overlaps', 'touches', 'within')])
    def time_binary_predicate(self, op):
        getattr(self.triangles, op)(self.triangle)

    @with_attributes(param_names=['op'],
                     params=[('difference', 'symmetric_difference', 'union',
                              'intersection')])
    def time_binary_geo(self, op):
        getattr(self.triangles, op)(self.triangle)

    @with_attributes(param_names=['op'],
                     params=[('area', 'length')])
    def time_unary_float(self, op):
        getattr(self.triangles, op)
