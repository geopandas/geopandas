from __future__ import absolute_import

from pandas import Series, DataFrame
from shapely.geometry import Point

from geopandas import GeoSeries, GeoDataFrame


class TestSeries:

    def setup_method(self):
        N = self.N = 10
        r = 0.5
        self.pts = GeoSeries([Point(x, y) for x, y in zip(range(N), range(N))])
        self.polys = self.pts.buffer(r)

    def test_slice(self):
        assert type(self.pts[:2]) is GeoSeries
        assert type(self.pts[::2]) is GeoSeries
        assert type(self.polys[:2]) is GeoSeries

    def test_head(self):
        assert type(self.pts.head()) is GeoSeries

    def test_tail(self):
        assert type(self.pts.tail()) is GeoSeries

    def test_sort_index(self):
        assert type(self.pts.sort_index()) is GeoSeries

    def test_loc(self):
        assert type(self.pts.loc[5:]) is GeoSeries

    def test_iloc(self):
        assert type(self.pts.iloc[5:]) is GeoSeries

    def test_fancy(self):
        idx = (self.pts.index.to_series() % 2).astype(bool)
        assert type(self.pts[idx]) is GeoSeries

    def test_take(self):
        assert type(self.pts.take(list(range(0, self.N, 2)))) is GeoSeries

    def test_select(self):
        assert type(self.pts.select(lambda x: x % 2 == 0)) is GeoSeries

    def test_groupby(self):
        for f, s in self.pts.groupby(lambda x: x % 2):
            assert type(s) is GeoSeries


class TestDataFrame:

    def setup_method(self):
        N = 10
        self.df = GeoDataFrame([
            {'geometry': Point(x, y), 'value1': x + y, 'value2': x*y}
            for x, y in zip(range(N), range(N))])

    def test_geometry(self):
        assert type(self.df.geometry) is GeoSeries
        # still GeoSeries if different name
        df2 = GeoDataFrame({"coords": [Point(x, y) for x, y in zip(range(5),
                                                                   range(5))],
                            "nums": range(5)}, geometry="coords")
        assert type(df2.geometry) is GeoSeries
        assert type(df2['coords']) is GeoSeries

    def test_nongeometry(self):
        assert type(self.df['value1']) is Series

    def test_geometry_multiple(self):
        assert type(self.df[['geometry', 'value1']]) is GeoDataFrame

    def test_nongeometry_multiple(self):
        assert type(self.df[['value1', 'value2']]) is DataFrame

    def test_slice(self):
        assert type(self.df[:2]) is GeoDataFrame
        assert type(self.df[::2]) is GeoDataFrame

    def test_fancy(self):
        idx = (self.df.index.to_series() % 2).astype(bool)
        assert type(self.df[idx]) is GeoDataFrame
