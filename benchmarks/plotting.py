import random

from geopandas import GeoSeries
from shapely.geometry import Point, LineString, Polygon


class Bench:

    def setup(self):
        self.points = GeoSeries([Point(i, i) for i in range(1000)])

        lines = GeoSeries([LineString([(random.random(), random.random())
                                       for _ in range(5)])
                           for _ in range(100)])
        triangles = GeoSeries([Polygon([(random.random(), random.random())
                                        for _ in range(3)])
                               for _ in range(100)])
        self.lines, self.triangles = lines, triangles

    def time_plot_points(self):
        self.points.plot()

    def time_plot_linestrings(self):
        self.lines.plot()

    def time_plot_polygons(self):
        self.triangles.plot()
