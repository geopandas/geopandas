import sys

from shapely.geometry import Point, Polygon, box, GeometryCollection
from numpy.testing import assert_array_equal

import geopandas
from geopandas import _compat as compat
from geopandas import GeoDataFrame, GeoSeries, read_file, sindex

import pytest
import numpy as np


@pytest.mark.skipif(sindex.has_sindex(), reason="Spatial index present, skipping")
class TestNoSindex:
    def test_no_sindex(self):
        """Checks that a warning is given when no spatial index is present."""
        with pytest.warns():
            sindex.get_sindex_class()


@pytest.mark.skipif(sys.platform.startswith("win"), reason="fails on AppVeyor")
@pytest.mark.skipif(not sindex.has_sindex(), reason="Spatial index absent, skipping")
class TestSeriesSindex:
    def test_empty_geoseries(self):

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
@pytest.mark.skipif(not sindex.has_sindex(), reason="Spatial index absent, skipping")
class TestFrameSindex:
    def setup_method(self):
        data = {
            "A": range(5),
            "B": range(-5, 0),
            "location": [Point(x, y) for x, y in zip(range(5), range(5))],
        }
        self.df = GeoDataFrame(data, geometry="location")

    def test_sindex(self):
        self.df.crs = "epsg:4326"
        assert self.df.sindex.size == 5
        hits = list(self.df.sindex.intersection((2.5, 2.5, 4, 4), objects=True))
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
            [Point(x, y) for x, y in zip(range(5, 10), range(5, 10))], inplace=True
        )
        assert self.df._sindex_generated is False


# Skip to accommodate Shapely geometries being unhashable
@pytest.mark.skip
class TestJoinSindex:
    def setup_method(self):
        nybb_filename = geopandas.datasets.get_path("nybb")
        self.boros = read_file(nybb_filename)

    def test_merge_geo(self):
        # First check that we gets hits from the boros frame.
        tree = self.boros.sindex
        hits = tree.intersection((1012821.80, 229228.26), objects=True)
        res = [self.boros.loc[hit.object]["BoroName"] for hit in hits]
        assert res == ["Bronx", "Queens"]

        # Check that we only get the Bronx from this view.
        first = self.boros[self.boros["BoroCode"] < 3]
        tree = first.sindex
        hits = tree.intersection((1012821.80, 229228.26), objects=True)
        res = [first.loc[hit.object]["BoroName"] for hit in hits]
        assert res == ["Bronx"]

        # Check that we only get Queens from this view.
        second = self.boros[self.boros["BoroCode"] >= 3]
        tree = second.sindex
        hits = tree.intersection((1012821.80, 229228.26), objects=True)
        res = ([second.loc[hit.object]["BoroName"] for hit in hits],)
        assert res == ["Queens"]

        # Get both the Bronx and Queens again.
        merged = first.merge(second, how="outer")
        assert len(merged) == 5
        assert merged.sindex.size == 5
        tree = merged.sindex
        hits = tree.intersection((1012821.80, 229228.26), objects=True)
        res = [merged.loc[hit.object]["BoroName"] for hit in hits]
        assert res == ["Bronx", "Queens"]


@pytest.mark.skipif(not sindex.has_sindex(), reason="Spatial index absent, skipping")
class TestPygeosInterface:
    def setup_method(self):
        data = {
            "A": range(6),
            "B": range(-6, 0),
            "location": [Point(x, y) for x, y in zip(range(5), range(5))]
            + [box(10, 10, 20, 20)],  # include a box geometry
        }
        self.df = GeoDataFrame(data, geometry="location")
        self.expected_size = len(data["location"])

    @pytest.mark.parametrize(
        "predicate, test_geom, expected",
        (
            (None, (-1, -1, -0.5, -0.5), []),
            (None, (-0.5, -0.5, 0.5, 0.5), [0]),
            (None, (0, 0, 1, 1), [0, 1]),
            ("intersects", (-1, -1, -0.5, -0.5), []),
            ("intersects", (-0.5, -0.5, 0.5, 0.5), [0]),
            ("intersects", (0, 0, 1, 1), [0, 1]),
            ("within", (0, 0, 1, 1), []),
            ("within", (11, 11, 12, 12), [5]),
            ("contains", (0, 0, 1, 1), []),
            ("contains", (0.5, 0.5, 1.5, 1.5), [1]),
            ("contains", (-1, -1, 2, 2), [0, 1]),
        ),
    )
    def test_query(self, predicate, test_geom, expected):
        """Tests the `query` method with valid inputs and valid predicates."""
        test_geom = box(*test_geom)
        res = self.df.sindex.query(test_geom, predicate=predicate)
        assert_array_equal(res, expected)

    @pytest.mark.parametrize(
        "test_geom, expected",
        (
            ((-1, -1, -0.5, -0.5), []),
            ((-0.5, -0.5, 0.5, 0.5), [0]),
            ((0, 0, 1, 1), [0, 1]),
            ((0, 0), [0]),
        ),
    )
    def test_intersection_bounds_tuple(self, test_geom, expected):
        """Tests the `intersection` method with valid inputs."""
        res = list(self.df.sindex.intersection(test_geom))
        assert_array_equal(res, expected)

    @pytest.mark.parametrize(
        "test_geom", ((-1, -1, -0.5), -0.5, None, Point(0, 0),),
    )
    def test_intersection_invalid_bounds_tuple(self, test_geom):
        """Tests the `intersection` method with invalid inputs."""
        if compat.USE_PYGEOS:
            with pytest.raises(TypeError):
                # we raise a useful TypeError
                self.df.sindex.intersection(test_geom)
        else:
            with pytest.raises((TypeError, Exception)):
                # catch a general exception
                # rtree raises an RTreeError which we need to catch
                self.df.sindex.intersection(test_geom)

    @pytest.mark.parametrize(
        "predicate, test_geom, expected", (("test", (-1, -1, -0.5, -0.5), [[], []]),),
    )
    def test_query_invalid_predicate(self, predicate, test_geom, expected):
        """Tests the `query` method with invalid predicates.
        """
        test_geom = box(*test_geom)
        with pytest.raises(ValueError):
            self.df.sindex.query(test_geom, predicate=predicate)

    @pytest.mark.parametrize(
        "test_geom, expected_error", [("notavalidgeom", TypeError)],
    )
    def test_query_invalid_geometry(self, test_geom, expected_error):
        """Tests the `query` method with invalid geometry.
        """
        with pytest.raises(expected_error):
            self.df.sindex.query(test_geom)

    @pytest.mark.parametrize(
        "test_geom, expected_error, expected_value",
        [(GeometryCollection(), None, []), (None, None, [])],
    )
    def test_query_empty_geometry(self, test_geom, expected_error, expected_value):
        """Tests the `query` method with empty geometry.
        """
        if expected_error is not None:
            with pytest.raises(expected_error):
                self.df.sindex.query(test_geom)
        else:
            # no error expected
            res = self.df.sindex.query(test_geom)
            assert_array_equal(res, expected_value)

    @pytest.mark.parametrize(
        "test_geoms, expected_error, expected_value",
        [
            # single empty geometry
            ([GeometryCollection()], None, [[], []]),
            # None should be skipped
            ([GeometryCollection(), None], None, [[], []]),
            ([None], None, [[], []]),
            ([None, box(-0.5, -0.5, 0.5, 0.5), None], None, [[1], [0]]),
        ],
    )
    def test_query_bulk_empty_geometry(
        self, test_geoms, expected_error, expected_value
    ):
        """Tests the `query_bulk` method with an empty geometry.
        """
        # pass through GeoSeries to have GeoPandas
        # determine if it should use shapely or pygeos geometry objects
        # note: for this test, test_geoms (note plural) is a list already
        test_geoms = geopandas.GeoSeries(test_geoms, index=range(len(test_geoms)))
        if expected_error is not None:
            with pytest.raises(expected_error):
                self.df.sindex.query_bulk(test_geoms)
        else:
            # no error expected
            res = self.df.sindex.query_bulk(test_geoms)
            assert_array_equal(res, expected_value)

    @pytest.mark.parametrize(
        "test_array, expected_error, expected_value",
        ((np.array([], dtype=object), None, [[], []]),),
    )
    def test_query_bulk_empty_input_array(
        self, test_array, expected_error, expected_value
    ):
        """Tests the `query_bulk` method with an empty input array.
        """
        if expected_error is not None:
            with pytest.raises(expected_error):
                self.df.sindex.query_bulk(test_array)
        else:
            # no error expected
            res = self.df.sindex.query_bulk(test_array)
            assert_array_equal(res, expected_value)

    @pytest.mark.parametrize(
        "test_array, expected_error, expected_value",
        (("notanarray", TypeError, None),),
    )
    def test_query_bulk_invalid_input_geometry(
        self, test_array, expected_error, expected_value
    ):
        """Tests the `query_bulk` method with invalid input for the `geometry` parameter.
        """
        if expected_error is not None:
            with pytest.raises(expected_error):
                self.df.sindex.query_bulk(test_array)
        else:
            # no error expected
            res = self.df.sindex.query_bulk(test_array)
            assert_array_equal(res, expected_value)

    @pytest.mark.parametrize(
        "predicate, test_geom, expected", (("test", (-1, -1, -0.5, -0.5), [[], []]),),
    )
    def test_query_bulk_invalid_predicate(self, predicate, test_geom, expected):
        """Tests the `query_bulk` method with invalid predicates.
        """
        # pass through GeoSeries to have GeoPandas
        # determine if it should use shapely or pygeos geometry objects
        test_geom = geopandas.GeoSeries([box(*test_geom)], index=["0"])
        with pytest.raises(ValueError):
            self.df.sindex.query_bulk(test_geom.geometry, predicate=predicate)

    @pytest.mark.parametrize(
        "predicate, test_geom, expected",
        (
            (None, (-1, -1, -0.5, -0.5), [[], []]),
            (None, (-0.5, -0.5, 0.5, 0.5), [[0], [0]]),
            (None, (0, 0, 1, 1), [[0, 0], [0, 1]]),
            ("intersects", (-1, -1, -0.5, -0.5), [[], []]),
            ("intersects", (-0.5, -0.5, 0.5, 0.5), [[0], [0]]),
            ("intersects", (0, 0, 1, 1), [[0, 0], [0, 1]]),
            ("within", (0, 0, 1, 1), [[], []]),
            ("within", (11, 11, 12, 12), [[0], [5]]),
            ("contains", (0, 0, 1, 1), [[], []]),
            ("contains", (0.5, 0.5, 1.5, 1.5), [[0], [1]]),
            ("contains", (-1, -1, 2, 2), [[0, 0], [0, 1]]),
        ),
    )
    def test_query_bulk(self, predicate, test_geom, expected):
        """Tests the `query_bulk` method with valid
        inputs and valid predicates.
        """
        # pass through GeoSeries to have GeoPandas
        # determine if it should use shapely or pygeos geometry objects
        test_geom = geopandas.GeoSeries([box(*test_geom)], index=["0"])

        res = self.df.sindex.query_bulk(test_geom, predicate=predicate)
        assert_array_equal(res, expected)

    @pytest.mark.parametrize(
        "predicate, test_geom, expected", ((None, (-1, -1, -0.5, -0.5), [[], []]),),
    )
    def test_query_bulk_input_type(self, predicate, test_geom, expected):
        """Tests that query_bulk can accept a GeoSeries, GeometryArray or
        numpy array.
        """
        # pass through GeoSeries to have GeoPandas
        # determine if it should use shapely or pygeos geometry objects
        test_geom = geopandas.GeoSeries([box(*test_geom)], index=["0"])

        # test GeoSeries
        res = self.df.sindex.query_bulk(test_geom, predicate=predicate)
        assert_array_equal(res, expected)

        # test GeometryArray
        res = self.df.sindex.query_bulk(test_geom.geometry, predicate=predicate)
        assert_array_equal(res, expected)
        res = self.df.sindex.query_bulk(test_geom.geometry.values, predicate=predicate)
        assert_array_equal(res, expected)

        # test numpy array
        res = self.df.sindex.query_bulk(
            test_geom.geometry.values.data, predicate=predicate
        )
        assert_array_equal(res, expected)
        res = self.df.sindex.query_bulk(
            test_geom.geometry.values.data, predicate=predicate
        )
        assert_array_equal(res, expected)

    @pytest.mark.parametrize(
        "sort, expected",
        (
            (True, [[0, 0, 0], [0, 1, 2]]),
            # False could be anything, at least we'll know if it changes
            (False, [[0, 0, 0], [0, 1, 2]]),
        ),
    )
    def test_query_sorting(self, sort, expected):
        """Check that results from `query` don't depend on the
        order of geometries.
        """
        # these geometries come from a reported issue:
        # https://github.com/geopandas/geopandas/issues/1337
        # there is no theoretical reason they were chosen
        test_polys = GeoSeries([Polygon([(1, 1), (3, 1), (3, 3), (1, 3)])])
        tree_polys = GeoSeries(
            [
                Polygon([(1, 1), (3, 1), (3, 3), (1, 3)]),
                Polygon([(-1, 1), (1, 1), (1, 3), (-1, 3)]),
                Polygon([(3, 3), (5, 3), (5, 5), (3, 5)]),
            ]
        )
        expected = [0, 1, 2]

        # pass through GeoSeries to have GeoPandas
        # determine if it should use shapely or pygeos geometry objects
        tree_df = geopandas.GeoDataFrame(geometry=tree_polys)
        test_df = geopandas.GeoDataFrame(geometry=test_polys)

        test_geo = test_df.geometry.values.data[0]
        res = tree_df.sindex.query(test_geo, sort=sort)
        try:
            assert_array_equal(res, expected)
        except AssertionError as e:
            if not compat.USE_PYGEOS and sort is False:
                pytest.xfail(
                    "rtree results are known to be unordered, see "
                    "https://github.com/geopandas/geopandas/issues/1337"
                )
            raise e

    @pytest.mark.parametrize(
        "sort, expected",
        (
            (True, [[0, 0, 0], [0, 1, 2]]),
            # False could be anything, at least we'll know if it changes
            (False, [[0, 0, 0], [0, 1, 2]]),
        ),
    )
    def test_query_bulk_sorting(self, sort, expected):
        """Check that results from `query_bulk` don't depend
        on the order of geometries.
        """
        # these geometries come from a reported issue:
        # https://github.com/geopandas/geopandas/issues/1337
        # there is no theoretical reason they were chosen
        test_polys = GeoSeries([Polygon([(1, 1), (3, 1), (3, 3), (1, 3)])])
        tree_polys = GeoSeries(
            [
                Polygon([(1, 1), (3, 1), (3, 3), (1, 3)]),
                Polygon([(-1, 1), (1, 1), (1, 3), (-1, 3)]),
                Polygon([(3, 3), (5, 3), (5, 5), (3, 5)]),
            ]
        )

        # pass through GeoSeries to have GeoPandas
        # determine if it should use shapely or pygeos geometry objects
        tree_df = geopandas.GeoDataFrame(geometry=tree_polys)
        test_df = geopandas.GeoDataFrame(geometry=test_polys)

        res = tree_df.sindex.query_bulk(test_df.geometry, sort=sort)
        try:
            assert_array_equal(res, expected)
        except AssertionError as e:
            if not compat.USE_PYGEOS and sort is False:
                pytest.xfail(
                    "rtree results are known to be unordered, see "
                    "https://github.com/geopandas/geopandas/issues/1337"
                )
            raise e

    def test_size(self):
        """Tests the `size` property."""
        assert self.df.sindex.size == self.expected_size

    def test_is_empty(self):
        """Tests the `is_empty` property."""
        # create empty tree
        cls_ = sindex.get_sindex_class()
        empty = geopandas.GeoSeries([])
        tree = cls_(empty)
        assert tree.is_empty
        # create a non-empty tree
        non_empty = geopandas.GeoSeries([Point(0, 0)])
        tree = cls_(non_empty)
        assert not tree.is_empty
