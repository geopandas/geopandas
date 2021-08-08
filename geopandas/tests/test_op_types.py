import os
import shutil
import tempfile

import pandas as pd

from shapely.geometry import Point

import geopandas
from geopandas import GeoDataFrame, read_file

from geopandas.tests.util import PACKAGE_DIR


class TestDataFrame:
    def setup_method(self):
        N = 10

        nybb_filename = geopandas.datasets.get_path("nybb")
        self.df = read_file(nybb_filename)
        self.tempdir = tempfile.mkdtemp()
        self.crs = "epsg:4326"
        self.df2 = GeoDataFrame(
            [
                {"geometry": Point(x, y), "value1": x + y, "value2": x * y}
                for x, y in zip(range(N), range(N))
            ],
            crs=self.crs,
        )
        self.df3 = read_file(
            os.path.join(PACKAGE_DIR, "geopandas", "tests", "data", "null_geom.geojson")
        )

    def teardown_method(self):
        shutil.rmtree(self.tempdir)

    def test_df_init(self):
        assert type(self.df2) is GeoDataFrame
        assert self.df2.crs == self.crs

    def test_drop(self):
        assert type(self.df.drop(columns="geometry")) is pd.DataFrame

    def test_getitem(self):
        assert type(self.df2[["value1", "value2"]]) is pd.DataFrame

    def test_loc(self):
        assert type(self.df2.loc[:, ["value1", "value2"]]) is pd.DataFrame

    def test_iloc(self):
        df = self.df2.iloc[:, 1:]
        print(df.columns)

        assert type(df) is pd.DataFrame

    def test_reindex(self):
        df = geopandas.read_file(geopandas.datasets.get_path("naturalearth_cities"))

        df_reindex = df.reindex(columns=["name"])
        print(df)
        print(df_reindex)
        assert type(df_reindex) is pd.DataFrame
