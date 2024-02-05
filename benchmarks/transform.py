from geopandas import GeoDataFrame, GeoSeries, read_file
import numpy as np
import pandas as pd
from shapely.geometry import Point
from geopandas.tests.util import _NYBB


class CRS:
    def setup(self):
        nybb = read_file(_NYBB)
        self.long_nybb = GeoDataFrame(pd.concat(10 * [nybb]), crs=nybb.crs)

        num_points = 20000
        longitudes = np.random.rand(num_points) - 120
        latitudes = np.random.rand(num_points) + 38
        self.point_df = GeoSeries(
            [Point(x, y) for (x, y) in zip(longitudes, latitudes)]
        )
        self.point_df.crs = {"init": "epsg:4326"}

    def time_transform_wgs84(self):
        self.long_nybb.to_crs({"init": "epsg:4326"})

    def time_transform_many_points(self):
        self.point_df.to_crs({"init": "epsg:32610"})
