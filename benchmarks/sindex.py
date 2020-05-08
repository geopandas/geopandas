import random

from geopandas import GeoDataFrame, GeoSeries
from shapely.geometry import Point, Polygon
import numpy as np


class Bench:
    def setup(self, *args):
        triangles = GeoSeries(
            [
                Polygon([(random.random(), random.random()) for _ in range(3)])
                for _ in range(100)
            ]
        )

        points = GeoSeries(
            [
                Point(x, y)
                for x, y in zip(
                    np.random.random_sample(1000), np.random.random_sample(1000)
                )
            ]
        )

        df1 = GeoDataFrame(
            {"val1": np.random.randn(len(triangles)), "geometry": triangles}
        )
        df2 = GeoDataFrame({"val1": np.random.randn(len(points)), "geometry": points})

        # save dfs
        self.df, self.df2, self.data = df1, df2, df2.geometry.values.data
        # cache bounds so that bound creation is not counted in benchmarks
        self.bounds = [point.bounds for point in points]

    def time_sindex_index_creation(self, *args):
        """Time creation of spatial index.

        Note: pygeos will only create the index once; this benchmark
        is not intended to be used to compare rtree and pygeos.
        """
        self.df._invalidate_sindex()
        self.df._generate_sindex()

    def time_sindex_intersects(self, *args):
        for bounds in self.bounds:
            self.df.sindex.intersection(bounds)

    def time_sindex_intersects_objects(self, *args):
        for bounds in self.bounds:
            self.df.sindex.intersection(bounds, objects=True)
