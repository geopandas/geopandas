import random

from geopandas import GeoDataFrame, GeoSeries, sjoin
from shapely.geometry import Point, LineString, Polygon
import numpy as np


class Bench:
    param_names = ["op"]
    params = [("intersects", "contains", "within")]

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

        df1 = GeoDataFrame(
            {"val1": np.random.randn(len(triangles)), "geometry": triangles}
        )
        df2 = GeoDataFrame({"val1": np.random.randn(len(points)), "geometry": points})

        self.df1, self.df2 = df1, df2

    def time_sjoin(self, predicate):
        sjoin(self.df1, self.df2, predicate=predicate)
