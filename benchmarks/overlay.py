from geopandas import GeoDataFrame, GeoSeries, read_file, overlay
import numpy as np
from shapely.geometry import Point, Polygon
from geopandas.tests.util import _NATURALEARTH_LOWRES
from geopandas.tests.util import _NATURALEARTH_CITIES


class Countries:
    param_names = ["how"]
    params = [
        ("intersection", "union", "identity", "symmetric_difference", "difference")
    ]

    def setup(self, *args):
        world = read_file(_NATURALEARTH_LOWRES)
        capitals = read_file(_NATURALEARTH_CITIES)
        countries = world[["geometry", "name"]]
        countries = countries.to_crs("+init=epsg:3395")[countries.name != "Antarctica"]
        capitals = capitals.to_crs("+init=epsg:3395")
        capitals["geometry"] = capitals.buffer(500000)

        self.countries = countries
        self.capitals = capitals

    def time_overlay(self, how):
        overlay(self.countries, self.capitals, how=how)


class Small:
    param_names = ["how"]
    params = [
        ("intersection", "union", "identity", "symmetric_difference", "difference")
    ]

    def setup(self, *args):
        polys1 = GeoSeries(
            [
                Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]),
                Polygon([(2, 2), (4, 2), (4, 4), (2, 4)]),
            ]
        )
        polys2 = GeoSeries(
            [
                Polygon([(1, 1), (3, 1), (3, 3), (1, 3)]),
                Polygon([(3, 3), (5, 3), (5, 5), (3, 5)]),
            ]
        )

        df1 = GeoDataFrame({"geometry": polys1, "df1": [1, 2]})
        df2 = GeoDataFrame({"geometry": polys2, "df2": [1, 2]})

        self.df1, self.df2 = df1, df2

    def time_overlay(self, how):
        overlay(self.df1, self.df2, how=how)


class ManyPoints:
    param_names = ["how"]
    params = [
        ("intersection", "union", "identity", "symmetric_difference", "difference")
    ]

    def setup(self, *args):
        points = GeoDataFrame(geometry=[Point(i, i) for i in range(1000)])
        base = np.array([[0, 0], [0, 100], [100, 100], [100, 0]])
        polys = GeoDataFrame(geometry=[Polygon(base + i * 100) for i in range(10)])

        self.df1, self.df2 = points, polys

    def time_overlay(self, how):
        overlay(self.df1, self.df2, how=how)
