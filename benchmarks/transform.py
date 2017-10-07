from geopandas import GeoDataFrame, read_file, datasets
import pandas as pd


class Boroughs:

    def setup(self):
        nybb = read_file(datasets.get_path('nybb'))
        self.long_nybb = GeoDataFrame(pd.concat(10 * [nybb]),
                                      crs=nybb.crs)

    def time_transform_wgs84(self):
        self.long_nybb.to_crs({"init": "epsg:4326"})
