import sys

from shapely.geometry import Polygon, Point

from geopandas import GeoSeries, GeoDataFrame, base, read_file
from geopandas.tests.util import unittest


@unittest.skipIf(sys.platform.startswith("win"), "fails on AppVeyor")
@unittest.skipIf(not base.HAS_SINDEX, 'Rtree absent, skipping')
class TestSeriesSindex(unittest.TestCase):

    def test_empty_index(self):
        self.assert_(GeoSeries().sindex is None)

    def test_point(self):
        s = GeoSeries([Point(0, 0)])
        self.assertEqual(s.sindex.size, 1)
        hits = s.sindex.intersection((-1, -1, 1, 1))
        self.assertEqual(len(list(hits)), 1)
        hits = s.sindex.intersection((-2, -2, -1, -1))
        self.assertEqual(len(list(hits)), 0)

    def test_empty_point(self):
        s = GeoSeries([Point()])
        self.assertTrue(s.sindex is None)
        self.assertTrue(s._sindex_generated)

    def test_empty_geo_series(self):
        self.assert_(GeoSeries().sindex is None)

    def test_polygons(self):
        t1 = Polygon([(0, 0), (1, 0), (1, 1)])
        t2 = Polygon([(0, 0), (1, 1), (0, 1)])
        sq = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        s = GeoSeries([t1, t2, sq])
        self.assertEqual(s.sindex.size, 3)

    def test_polygons_append(self):
        t1 = Polygon([(0, 0), (1, 0), (1, 1)])
        t2 = Polygon([(0, 0), (1, 1), (0, 1)])
        sq = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        s = GeoSeries([t1, t2, sq])
        t = GeoSeries([t1, t2, sq], [3,4,5])
        s = s.append(t)
        self.assertEqual(len(s), 6)
        self.assertEqual(s.sindex.size, 6)

    def test_lazy_build(self):
        s = GeoSeries([Point(0, 0)])
        self.assert_(s._sindex is None)
        self.assertEqual(s.sindex.size, 1)
        self.assert_(s._sindex is not None)


@unittest.skipIf(sys.platform.startswith("win"), "fails on AppVeyor")
@unittest.skipIf(not base.HAS_SINDEX, 'Rtree absent, skipping')
class TestFrameSindex(unittest.TestCase):
    def setUp(self):
        data = {"A": range(5), "B": range(-5, 0),
                "location": [Point(x, y) for x, y in zip(range(5), range(5))]}
        self.df = GeoDataFrame(data, geometry='location')

    def test_sindex(self):
        self.df.crs = {'init': 'epsg:4326'}
        self.assertEqual(self.df.sindex.size, 5)
        hits = list(self.df.sindex.intersection((2.5, 2.5, 4, 4), objects=True))
        self.assertEqual(len(hits), 2)
        self.assertEqual(hits[0].object, 3)

    def test_lazy_build(self):
        self.assert_(self.df._sindex is None)
        self.assertEqual(self.df.sindex.size, 5)
        self.assert_(self.df._sindex is not None)

    def test_sindex_rebuild_on_set_geometry(self):
        # First build the sindex
        self.assert_(self.df.sindex is not None)
        self.df.set_geometry(
            [Point(x, y) for x, y in zip(range(5, 10), range(5, 10))],
            inplace=True)
        self.assertFalse(self.df._sindex_generated)


# Skip to accommodate Shapely geometries being unhashable
@unittest.skip
class TestJoinSindex(unittest.TestCase):

    def setUp(self):
        nybb_filename = geopandas.datasets.get_path('nybb')
        self.boros = read_file(nybb_filename)

    def test_merge_geo(self):
        # First check that we gets hits from the boros frame.
        tree = self.boros.sindex
        hits = tree.intersection((1012821.80, 229228.26), objects=True)
        self.assertEqual(
            [self.boros.ix[hit.object]['BoroName'] for hit in hits],
            ['Bronx', 'Queens'])

        # Check that we only get the Bronx from this view.
        first = self.boros[self.boros['BoroCode'] < 3]
        tree = first.sindex
        hits = tree.intersection((1012821.80, 229228.26), objects=True)
        self.assertEqual(
            [first.ix[hit.object]['BoroName'] for hit in hits],
            ['Bronx'])

        # Check that we only get Queens from this view.
        second = self.boros[self.boros['BoroCode'] >= 3]
        tree = second.sindex
        hits = tree.intersection((1012821.80, 229228.26), objects=True)
        self.assertEqual(
            [second.ix[hit.object]['BoroName'] for hit in hits],
            ['Queens'])

        # Get both the Bronx and Queens again.
        merged = first.merge(second, how='outer')
        self.assertEqual(len(merged), 5)
        self.assertEqual(merged.sindex.size, 5)
        tree = merged.sindex
        hits = tree.intersection((1012821.80, 229228.26), objects=True)
        self.assertEqual(
            [merged.ix[hit.object]['BoroName'] for hit in hits],
            ['Bronx', 'Queens'])
