import sys

from shapely.geometry import (
    Point,
    Polygon,
    MultiPolygon,
    box,
    GeometryCollection,
    LineString,
)
from numpy.testing import assert_array_equal

import geopandas
from geopandas import _compat as compat
from geopandas import GeoDataFrame, GeoSeries, read_file, sindex, datasets

import pytest
import numpy as np


class TestNoSindex:
    @pytest.mark.skipif(sindex.has_sindex(), reason="Spatial index present, skipping")
    def test_no_sindex_installed(self):
        """Checks that an error is raised when no spatial index is present."""
        with pytest.raises(ImportError):
            sindex.get_sindex_class()

    @pytest.mark.skipif(
        compat.HAS_RTREE or not compat.HAS_PYGEOS,
        reason="rtree cannot be disabled via flags",
    )
    def test_no_sindex_active(self):
        """Checks that an error is given when rtree is not installed
        and compat.USE_PYGEOS is False.
        """
        state = compat.USE_PYGEOS  # try to save state
        compat.set_use_pygeos(False)
        with pytest.raises(ImportError):
            sindex.get_sindex_class()
        compat.set_use_pygeos(state)  # try to restore state


@pytest.mark.skipif(sys.platform.startswith("win"), reason="fails on AppVeyor")
@pytest.mark.skipif(not sindex.has_sindex(), reason="Spatial index absent, skipping")
class TestSeriesSindex:
    def test_empty_geoseries(self):
        """Tests creating a spatial index from an empty GeoSeries."""
        with pytest.warns(FutureWarning, match="Generated spatial index is empty"):
            # TODO: add checking len(GeoSeries().sindex) == 0 once deprecated
            assert not GeoSeries(dtype=object).sindex

    def test_point(self):
        s = GeoSeries([Point(0, 0)])
        assert s.sindex.size == 1
        hits = s.sindex.intersection((-1, -1, 1, 1))
        assert len(list(hits)) == 1
        hits = s.sindex.intersection((-2, -2, -1, -1))
        assert len(list(hits)) == 0

    def test_empty_point(self):
        """Tests that a single empty Point results in an empty tree."""
        s = GeoSeries([Point()])

        with pytest.warns(FutureWarning, match="Generated spatial index is empty"):
            # TODO: add checking len(s) == 0 once deprecated
            assert not s.sindex

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
        with pytest.warns(FutureWarning, match="`objects` is deprecated"):
            # TODO: remove warning check once deprecated
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
        with pytest.warns(FutureWarning, match="`objects` is deprecated"):
            # TODO: remove warning check once deprecated
            hits = tree.intersection((1012821.80, 229228.26), objects=True)
        res = [self.boros.loc[hit.object]["BoroName"] for hit in hits]
        assert res == ["Bronx", "Queens"]

        # Check that we only get the Bronx from this view.
        first = self.boros[self.boros["BoroCode"] < 3]
        tree = first.sindex
        with pytest.warns(FutureWarning, match="`objects` is deprecated"):
            # TODO: remove warning check once deprecated
            hits = tree.intersection((1012821.80, 229228.26), objects=True)
        res = [first.loc[hit.object]["BoroName"] for hit in hits]
        assert res == ["Bronx"]

        # Check that we only get Queens from this view.
        second = self.boros[self.boros["BoroCode"] >= 3]
        tree = second.sindex
        with pytest.warns(FutureWarning, match="`objects` is deprecated"):
            # TODO: remove warning check once deprecated
            hits = tree.intersection((1012821.80, 229228.26), objects=True)
        res = ([second.loc[hit.object]["BoroName"] for hit in hits],)
        assert res == ["Queens"]

        # Get both the Bronx and Queens again.
        merged = first.merge(second, how="outer")
        assert len(merged) == 5
        assert merged.sindex.size == 5
        tree = merged.sindex
        with pytest.warns(FutureWarning, match="`objects` is deprecated"):
            # TODO: remove warning check once deprecated
            hits = tree.intersection((1012821.80, 229228.26), objects=True)
        res = [merged.loc[hit.object]["BoroName"] for hit in hits]
        assert res == ["Bronx", "Queens"]


@pytest.mark.skipif(not sindex.has_sindex(), reason="Spatial index absent, skipping")
class TestPygeosInterface:
    def setup_method(self):
        data = {
            "location": [Point(x, y) for x, y in zip(range(5), range(5))]
            + [box(10, 10, 20, 20)]  # include a box geometry
        }
        self.df = GeoDataFrame(data, geometry="location")
        self.expected_size = len(data["location"])

    # --------------------------- `intersection` tests -------------------------- #
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

    @pytest.mark.parametrize("test_geom", ((-1, -1, -0.5), -0.5, None, Point(0, 0)))
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

    # ------------------------------ `query` tests ------------------------------ #
    @pytest.mark.parametrize(
        "predicate, test_geom, expected",
        (
            (None, box(-1, -1, -0.5, -0.5), []),  # bbox does not intersect
            (None, box(-0.5, -0.5, 0.5, 0.5), [0]),  # bbox intersects
            (None, box(0, 0, 1, 1), [0, 1]),  # bbox intersects multiple
            (
                None,
                LineString([(0, 1), (1, 0)]),
                [0, 1],
            ),  # bbox intersects but not geometry
            ("intersects", box(-1, -1, -0.5, -0.5), []),  # bbox does not intersect
            (
                "intersects",
                box(-0.5, -0.5, 0.5, 0.5),
                [0],
            ),  # bbox and geometry intersect
            (
                "intersects",
                box(0, 0, 1, 1),
                [0, 1],
            ),  # bbox and geometry intersect multiple
            (
                "intersects",
                LineString([(0, 1), (1, 0)]),
                [],
            ),  # bbox intersects but not geometry
            ("within", box(0.25, 0.28, 0.75, 0.75), []),  # does not intersect
            ("within", box(0, 0, 10, 10), []),  # intersects but is not within
            ("within", box(11, 11, 12, 12), [5]),  # intersects and is within
            ("within", LineString([(0, 1), (1, 0)]), []),  # intersects but not within
            ("contains", box(0, 0, 1, 1), []),  # intersects but does not contain
            ("contains", box(0, 0, 1.001, 1.001), [1]),  # intersects and contains
            ("contains", box(0.5, 0.5, 1.5, 1.5), [1]),  # intersects and contains
            ("contains", box(-1, -1, 2, 2), [0, 1]),  # intersects and contains multiple
            (
                "contains",
                LineString([(0, 1), (1, 0)]),
                [],
            ),  # intersects but not contains
            ("touches", box(-1, -1, 0, 0), [0]),  # bbox intersects and touches
            (
                "touches",
                box(-0.5, -0.5, 1.5, 1.5),
                [],
            ),  # bbox intersects but geom does not touch
        ),
    )
    def test_query(self, predicate, test_geom, expected):
        """Tests the `query` method with valid inputs and valid predicates."""
        res = self.df.sindex.query(test_geom, predicate=predicate)
        assert_array_equal(res, expected)

    def test_query_invalid_geometry(self):
        """Tests the `query` method with invalid geometry.
        """
        with pytest.raises(TypeError):
            self.df.sindex.query("notavalidgeom")

    @pytest.mark.parametrize(
        "test_geom, expected_value",
        [
            (None, []),
            (GeometryCollection(), []),
            (Point(), []),
            (MultiPolygon(), []),
            (Polygon(), []),
        ],
    )
    def test_query_empty_geometry(self, test_geom, expected_value):
        """Tests the `query` method with empty geometry.
        """
        res = self.df.sindex.query(test_geom)
        assert_array_equal(res, expected_value)

    def test_query_invalid_predicate(self):
        """Tests the `query` method with invalid predicates.
        """
        test_geom = box(-1, -1, -0.5, -0.5)
        with pytest.raises(ValueError):
            self.df.sindex.query(test_geom, predicate="test")

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
                    "https://github.com/geopandas/geopandas/issues/1337\n"
                    "Expected:\n {}\n".format(expected)
                    + "Got:\n {}\n".format(res.tolist())
                )
            raise e

    # ------------------------- `query_bulk` tests -------------------------- #
    @pytest.mark.parametrize(
        "predicate, test_geom, expected",
        (
            (None, [(-1, -1, -0.5, -0.5)], [[], []]),
            (None, [(-0.5, -0.5, 0.5, 0.5)], [[0], [0]]),
            (None, [(0, 0, 1, 1)], [[0, 0], [0, 1]]),
            ("intersects", [(-1, -1, -0.5, -0.5)], [[], []]),
            ("intersects", [(-0.5, -0.5, 0.5, 0.5)], [[0], [0]]),
            ("intersects", [(0, 0, 1, 1)], [[0, 0], [0, 1]]),
            # only second geom intersects
            ("intersects", [(-1, -1, -0.5, -0.5), (-0.5, -0.5, 0.5, 0.5)], [[1], [0]]),
            # both geoms intersect
            (
                "intersects",
                [(-1, -1, 1, 1), (-0.5, -0.5, 0.5, 0.5)],
                [[0, 0, 1], [0, 1, 0]],
            ),
            ("within", [(0.25, 0.28, 0.75, 0.75)], [[], []]),  # does not intersect
            ("within", [(0, 0, 10, 10)], [[], []]),  # intersects but is not within
            ("within", [(11, 11, 12, 12)], [[0], [5]]),  # intersects and is within
            ("contains", [(0, 0, 1, 1)], [[], []]),  # intersects but does not contain
            (
                "contains",
                [(0, 0, 1.001, 1.001)],
                [[0], [1]],
            ),  # intersects 2 and contains 1
            (
                "contains",
                [(0.5, 0.5, 1.001, 1.001)],
                [[0], [1]],
            ),  # intersects 1 and contains 1
            ("contains", [(0.5, 0.5, 1.5, 1.5)], [[0], [1]]),  # intersects and contains
            (
                "contains",
                [(-1, -1, 2, 2)],
                [[0, 0], [0, 1]],
            ),  # intersects and contains multiple
        ),
    )
    def test_query_bulk(self, predicate, test_geom, expected):
        """Tests the `query_bulk` method with valid
        inputs and valid predicates.
        """
        # pass through GeoSeries to have GeoPandas
        # determine if it should use shapely or pygeos geometry objects
        test_geom = geopandas.GeoSeries(
            [box(*geom) for geom in test_geom], index=range(len(test_geom))
        )
        res = self.df.sindex.query_bulk(test_geom, predicate=predicate)
        assert_array_equal(res, expected)

    @pytest.mark.parametrize(
        "test_geoms, expected_value",
        [
            # single empty geometry
            ([GeometryCollection()], [[], []]),
            # None should be skipped
            ([GeometryCollection(), None], [[], []]),
            ([None], [[], []]),
            ([None, box(-0.5, -0.5, 0.5, 0.5), None], [[1], [0]]),
        ],
    )
    def test_query_bulk_empty_geometry(self, test_geoms, expected_value):
        """Tests the `query_bulk` method with an empty geometry.
        """
        # pass through GeoSeries to have GeoPandas
        # determine if it should use shapely or pygeos geometry objects
        # note: for this test, test_geoms (note plural) is a list already
        test_geoms = geopandas.GeoSeries(test_geoms, index=range(len(test_geoms)))
        res = self.df.sindex.query_bulk(test_geoms)
        assert_array_equal(res, expected_value)

    def test_query_bulk_empty_input_array(self):
        """Tests the `query_bulk` method with an empty input array.
        """
        test_array = np.array([], dtype=object)
        expected_value = [[], []]
        res = self.df.sindex.query_bulk(test_array)
        assert_array_equal(res, expected_value)

    def test_query_bulk_invalid_input_geometry(self):
        """Tests the `query_bulk` method with invalid input for the `geometry` parameter.
        """
        test_array = "notanarray"
        with pytest.raises(TypeError):
            self.df.sindex.query_bulk(test_array)

    def test_query_bulk_invalid_predicate(self):
        """Tests the `query_bulk` method with invalid predicates.
        """
        test_geom_bounds = (-1, -1, -0.5, -0.5)
        test_predicate = "test"

        # pass through GeoSeries to have GeoPandas
        # determine if it should use shapely or pygeos geometry objects
        test_geom = geopandas.GeoSeries([box(*test_geom_bounds)], index=["0"])

        with pytest.raises(ValueError):
            self.df.sindex.query_bulk(test_geom.geometry, predicate=test_predicate)

    @pytest.mark.parametrize(
        "predicate, test_geom, expected",
        (
            (None, (-1, -1, -0.5, -0.5), [[], []]),
            ("intersects", (-1, -1, -0.5, -0.5), [[], []]),
            ("contains", (-1, -1, 1, 1), [[0], [0]]),
        ),
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
                    "https://github.com/geopandas/geopandas/issues/1337\n"
                    "Expected:\n {}\n".format(expected)
                    + "Got:\n {}\n".format(res.tolist())
                )
            raise e

    # --------------------------- misc tests ---------------------------- #

    def test_empty_tree_geometries(self):
        """Tests building sindex with interleaved empty geometries.
        """
        geoms = [Point(0, 0), None, Point(), Point(1, 1), Point()]
        df = geopandas.GeoDataFrame(geometry=geoms)
        assert df.sindex.query(Point(1, 1))[0] == 3

    def test_size(self):
        """Tests the `size` property."""
        assert self.df.sindex.size == self.expected_size

    def test_len(self):
        """Tests the `__len__` method of spatial indexes."""
        assert len(self.df.sindex) == self.expected_size

    def test_is_empty(self):
        """Tests the `is_empty` property."""
        # create empty tree
        cls_ = sindex.get_sindex_class()
        empty = geopandas.GeoSeries(dtype=object)
        tree = cls_(empty)
        assert tree.is_empty
        # create a non-empty tree
        non_empty = geopandas.GeoSeries([Point(0, 0)])
        tree = cls_(non_empty)
        assert not tree.is_empty

    @pytest.mark.parametrize(
        "predicate, expected_shape",
        [
            (None, (2, 396)),
            ("intersects", (2, 172)),
            ("within", (2, 172)),
            ("contains", (2, 0)),
            ("overlaps", (2, 0)),
            ("crosses", (2, 0)),
            ("touches", (2, 0)),
        ],
    )
    def test_integration_natural_earth(self, predicate, expected_shape):
        """Tests output sizes for the naturalearth datasets."""
        world = read_file(datasets.get_path("naturalearth_lowres"))
        capitals = read_file(datasets.get_path("naturalearth_cities"))

        res = world.sindex.query_bulk(capitals.geometry, predicate)
        assert res.shape == expected_shape
