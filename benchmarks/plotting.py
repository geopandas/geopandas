import random

from geopandas import GeoDataFrame, GeoSeries
from shapely.geometry import Point, LineString, Polygon, MultiPolygon
import numpy as np


class Bench:
    param_names = ["geom_type"]
    params = [("Point", "LineString", "Polygon", "MultiPolygon", "mixed")]

    def setup(self, geom_type):
        if geom_type == "Point":
            geoms = GeoSeries([Point(i, i) for i in range(1000)])
        elif geom_type == "LineString":
            geoms = GeoSeries(
                [
                    LineString([(random.random(), random.random()) for _ in range(5)])
                    for _ in range(100)
                ]
            )
        elif geom_type == "Polygon":
            geoms = GeoSeries(
                [
                    Polygon([(random.random(), random.random()) for _ in range(3)])
                    for _ in range(100)
                ]
            )
        elif geom_type == "MultiPolygon":
            geoms = GeoSeries(
                [
                    MultiPolygon(
                        [
                            Polygon(
                                [(random.random(), random.random()) for _ in range(3)]
                            )
                            for _ in range(3)
                        ]
                    )
                    for _ in range(20)
                ]
            )
        elif geom_type == "mixed":
            g1 = GeoSeries([Point(i, i) for i in range(100)])
            g2 = GeoSeries(
                [
                    LineString([(random.random(), random.random()) for _ in range(5)])
                    for _ in range(100)
                ]
            )
            g3 = GeoSeries(
                [
                    Polygon([(random.random(), random.random()) for _ in range(3)])
                    for _ in range(100)
                ]
            )

            geoms = g1
            geoms.iloc[np.random.randint(0, 100, 50)] = g2.iloc[:50]
            geoms.iloc[np.random.randint(0, 100, 33)] = g3.iloc[:33]

            print(geoms.geom_type.value_counts())

        df = GeoDataFrame({"geometry": geoms, "values": np.random.randn(len(geoms))})

        self.geoms = geoms
        self.df = df

    def time_plot_series(self, *args):
        self.geoms.plot()

    def time_plot_values(self, *args):
        self.df.plot(column="values")
