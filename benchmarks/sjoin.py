import random

from geopandas import GeoDataFrame, GeoSeries, sjoin
from shapely.geometry import Point, Polygon
import numpy as np


class Bench:

    param_names = ["op", "left_size", "right_size", "left_type", "right_type"]
    params = [
        ("intersects", "contains", "within"),
        (100, 1000),
        (100, 1000),
        ("points", "polygons"),
        ("points", "polygons"),
    ]

    def setup(self, *args):
        triangles = GeoSeries(
            [
                Polygon([(random.random(), random.random()) for _ in range(3)])
                for _ in range(1000)
            ]
        )

        points = GeoSeries(
            [
                Point(x, y)
                for x, y in zip(np.random.random(10000), np.random.random(10000))
            ]
        )

        poly_df = GeoDataFrame(
            {"val1": np.random.randn(len(triangles)), "geometry": triangles}
        )
        point_df = GeoDataFrame(
            {"val1": np.random.randn(len(points)), "geometry": points}
        )

        self.polygons, self.points = poly_df, point_df

    def time_sjoin(self, op):
        sjoin(self.polygons, self.points, op=op)

    def time_points_on_poly(self, op, left_size, right_size, left_type, right_type):
        if left_type == "polygons":
            left_df = self.polygons
        else:
            left_df = self.polygons
        if right_type == "polygons":
            right_df = self.polygons
        else:
            right_df = self.points
        left_df = left_df.sample(left_size, replace=True)
        right_df = right_df.sample(right_size, replace=True)
        sjoin(left_df, right_df, op=op)
