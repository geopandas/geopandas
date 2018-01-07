import sys

from shapely.geometry import Polygon, Point

import geopandas
from geopandas import GeoSeries, GeoDataFrame, base, read_file

import pytest


@pytest.mark.skipif(sys.platform.startswith("win"), reason="fails on AppVeyor")
@pytest.mark.skipif(not base.HAS_SINDEX, reason='Rtree absent, skipping')
class TestSeriesSindex:

    def test_empty_index(self):
        assert GeoSeries().sindex is None

    def test_point(self):
        s = GeoSeries([Point(0, 0)])
        assert s.sindex.size == 1
        hits = s.sindex.intersection((-1, -1, 1, 1))
        assert len(list(hits)) == 1
        hits = s.sindex.intersection((-2, -2, -1, -1))
        assert len(list(hits)) == 0

    def test_empty_point(self):
        s = GeoSeries([Point()])
        assert s.sindex is None
        assert s._sindex_generated is True

    def test_empty_geo_series(self):
        assert GeoSeries().sindex is None

    def test_polygons(self):
        t1 = Polygon([(0, 0), (1, 0), (1, 1)])
        t2 = Polygon([(0, 0), (1, 1), (0, 1)])
        sq = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        s = GeoSeries([t1, t2, sq])
        assert s.sindex.size == 3

    def test_polygons_append(self):
        t1 = Polygon([(0, 0), (1, 0), (1, 1)])
        t2 = Polygon([(0, 0), (1, 1), (0, 1)])
        sq = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        s = GeoSeries([t1, t2, sq])
        t = GeoSeries([t1, t2, sq], [3, 4, 5])
        s = s.append(t)
        assert len(s) == 6
        assert s.sindex.size == 6

    def test_lazy_build(self):
        s = GeoSeries([Point(0, 0)])
        assert s._sindex is None
        assert s.sindex.size == 1
        assert s._sindex is not None


@pytest.mark.skipif(sys.platform.startswith("win"), reason="fails on AppVeyor")
@pytest.mark.skipif(not base.HAS_SINDEX, reason='Rtree absent, skipping')
class TestFrameSindex:
    def setup_method(self):
        data = {"A": range(5), "B": range(-5, 0),
                "location": [Point(x, y) for x, y in zip(range(5), range(5))]}
        self.df = GeoDataFrame(data, geometry='location')

    def test_sindex(self):
        self.df.crs = {'init': 'epsg:4326'}
        assert self.df.sindex.size == 5
        hits = list(self.df.sindex.intersection((2.5, 2.5, 4, 4),
                                                objects=True))
        assert len(hits) == 2
        assert hits[0].object == 3

    def test_lazy_build(self):
        assert self.df._sindex is None
        assert self.df.sindex.size == 5
        assert self.df._sindex is not None

    def test_sindex_rebuild_on_set_geometry(self):
        # First build the sindex
        assert self.df.sindex is not None
        self.df.set_geometry(
            [Point(x, y) for x, y in zip(range(5, 10), range(5, 10))],
            inplace=True)
        assert self.df._sindex_generated is False


# Skip to accommodate Shapely geometries being unhashable
@pytest.mark.skip
class TestJoinSindex:

    def setup_method(self):
        nybb_filename = geopandas.datasets.get_path('nybb')
        self.boros = read_file(nybb_filename)

    def test_merge_geo(self):
        # First check that we gets hits from the boros frame.
        tree = self.boros.sindex
        hits = tree.intersection((1012821.80, 229228.26), objects=True)
        res = [self.boros.loc[hit.object]['BoroName'] for hit in hits]
        assert res == ['Bronx', 'Queens']

        # Check that we only get the Bronx from this view.
        first = self.boros[self.boros['BoroCode'] < 3]
        tree = first.sindex
        hits = tree.intersection((1012821.80, 229228.26), objects=True)
        res = [first.loc[hit.object]['BoroName'] for hit in hits]
        assert res == ['Bronx']

        # Check that we only get Queens from this view.
        second = self.boros[self.boros['BoroCode'] >= 3]
        tree = second.sindex
        hits = tree.intersection((1012821.80, 229228.26), objects=True)
        res = [second.loc[hit.object]['BoroName'] for hit in hits],
        assert res == ['Queens']

        # Get both the Bronx and Queens again.
        merged = first.merge(second, how='outer')
        assert len(merged) == 5
        assert merged.sindex.size == 5
        tree = merged.sindex
        hits = tree.intersection((1012821.80, 229228.26), objects=True)
        res = [merged.loc[hit.object]['BoroName'] for hit in hits]
        assert res == ['Bronx', 'Queens']
