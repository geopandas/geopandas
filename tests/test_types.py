import unittest
from shapely.geometry import Point
from geopandas import GeoSeries


class TestSeries(unittest.TestCase):

    def setUp(self):
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

    def test_sort_order(self):
        assert type(self.pts.order()) is GeoSeries

    @unittest.skip('not yet implemented')
    def test_loc(self):
        assert type(self.pts.loc[5:]) is GeoSeries

    @unittest.skip('not yet implemented')
    def test_iloc(self):
        assert type(self.pts.iloc[5:]) is GeoSeries

    def test_fancy(self):
        idx = (self.pts.index % 2).astype(bool)
        assert type(self.pts[idx]) is GeoSeries

    def test_take(self):
        assert type(self.pts.take(range(0, self.N, 2))) is GeoSeries

    def test_select(self):
        assert type(self.pts.select(lambda x: x % 2 == 0)) is GeoSeries

    @unittest.skip('not yet implemented')
    def test_groupby(self):
        for f, s in self.pts.groupby(lambda x: x % 2):
            assert type(s) is GeoSeries
