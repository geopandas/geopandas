from math import sqrt

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
from geopandas import GeoDataFrame, GeoSeries, read_file, datasets

import pytest
import numpy as np
import pandas as pd

if compat.USE_SHAPELY_20:
    import shapely as mod
elif compat.USE_PYGEOS:
    import pygeos as mod


@pytest.mark.skip_no_sindex
class TestSeriesSindex:
    def test_has_sindex(self):
        """Test the has_sindex method."""
        t1 = Polygon([(0, 0), (1, 0), (1, 1)])
        t2 = Polygon([(0, 0), (1, 1), (0, 1)])

        d = GeoDataFrame({"geom": [t1, t2]}, geometry="geom")
        assert not d.has_sindex
        d.sindex
        assert d.has_sindex
        d.geometry.values._sindex = None
        assert not d.has_sindex
        d.sindex
        assert d.has_sindex

        s = GeoSeries([t1, t2])
        assert not s.has_sindex
        s.sindex
        assert s.has_sindex
        s.values._sindex = None
        assert not s.has_sindex
        s.sindex
        assert s.has_sindex

    def test_empty_geoseries(self):
        """Tests creating a spatial index from an empty GeoSeries."""
        s = GeoSeries(dtype=object)
        assert not s.sindex
        assert len(s.sindex) == 0

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
        assert not s.sindex
        assert len(s.sindex) == 0

    def test_polygons(self):
        t1 = Polygon([(0, 0), (1, 0), (1, 1)])
        t2 = Polygon([(0, 0), (1, 1), (0, 1)])
        sq = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        s = GeoSeries([t1, t2, sq])
        assert s.sindex.size == 3

    @pytest.mark.filterwarnings("ignore:The series.append method is deprecated")
    @pytest.mark.skipif(compat.PANDAS_GE_20, reason="append removed in pandas 2.0")
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
        assert s.values._sindex is None
        assert s.sindex.size == 1
        assert s.values._sindex is not None

    def test_rebuild_on_item_change(self):
        s = GeoSeries([Point(0, 0)])
        original_index = s.sindex
        s.iloc[0] = Point(0, 0)
        assert s.sindex is not original_index

    def test_rebuild_on_slice(self):
        s = GeoSeries([Point(0, 0), Point(0, 0)])
        original_index = s.sindex
        # Select a couple of rows
        sliced = s.iloc[:1]
        assert sliced.sindex is not original_index
        # Select all rows
        sliced = s.iloc[:]
        assert sliced.sindex is original_index
        # Select all rows and flip
        sliced = s.iloc[::-1]
        assert sliced.sindex is not original_index


@pytest.mark.skip_no_sindex
class TestFrameSindex:
    def setup_method(self):
        data = {
            "A": range(5),
            "B": range(-5, 0),
            "geom": [Point(x, y) for x, y in zip(range(5), range(5))],
        }
        self.df = GeoDataFrame(data, geometry="geom")

    def test_sindex(self):
        self.df.crs = "epsg:4326"
        assert self.df.sindex.size == 5
        hits = list(self.df.sindex.intersection((2.5, 2.5, 4, 4)))
        assert len(hits) == 2
        assert hits[0] == 3

    def test_lazy_build(self):
        assert self.df.geometry.values._sindex is None
        assert self.df.sindex.size == 5
        assert self.df.geometry.values._sindex is not None

    def test_sindex_rebuild_on_set_geometry(self):
        # First build the sindex
        assert self.df.sindex is not None
        original_index = self.df.sindex
        self.df.set_geometry(
            [Point(x, y) for x, y in zip(range(5, 10), range(5, 10))], inplace=True
        )
        assert self.df.sindex is not original_index

    def test_rebuild_on_row_slice(self):
        # Select a subset of rows rebuilds
        original_index = self.df.sindex
        sliced = self.df.iloc[:1]
        assert sliced.sindex is not original_index
        # Slicing all does not rebuild
        original_index = self.df.sindex
        sliced = self.df.iloc[:]
        assert sliced.sindex is original_index
        # Re-ordering rebuilds
        sliced = self.df.iloc[::-1]
        assert sliced.sindex is not original_index

    def test_rebuild_on_single_col_selection(self):
        """Selecting a single column should not rebuild the spatial index."""
        # Selecting geometry column preserves the index
        original_index = self.df.sindex
        geometry_col = self.df["geom"]
        assert geometry_col.sindex is original_index
        geometry_col = self.df.geometry
        assert geometry_col.sindex is original_index

    def test_rebuild_on_multiple_col_selection(self):
        """Selecting a subset of columns preserves the index."""
        original_index = self.df.sindex
        # Selecting a subset of columns preserves the index for pandas < 2.0
        # with pandas 2.0, the column is now copied, losing the index (although
        # with Copy-on-Write, this will again be preserved)
        subset1 = self.df[["geom", "A"]]
        if compat.PANDAS_GE_20 and not pd.options.mode.copy_on_write:
            assert subset1.sindex is not original_index
        else:
            assert subset1.sindex is original_index
        subset2 = self.df[["A", "geom"]]
        if compat.PANDAS_GE_20 and not pd.options.mode.copy_on_write:
            assert subset2.sindex is not original_index
        else:
            assert subset2.sindex is original_index

    def test_rebuild_on_update_inplace(self):
        gdf = self.df.copy()
        old_sindex = gdf.sindex
        # sorting in place
        gdf.sort_values("A", ascending=False, inplace=True)
        # spatial index should be invalidated
        assert not gdf.has_sindex
        new_sindex = gdf.sindex
        # and should be different
        assert new_sindex is not old_sindex

        # sorting should still have happened though
        assert gdf.index.tolist() == [4, 3, 2, 1, 0]

    def test_update_inplace_no_rebuild(self):
        gdf = self.df.copy()
        old_sindex = gdf.sindex
        gdf.rename(columns={"A": "AA"}, inplace=True)
        # a rename shouldn't invalidate the index
        assert gdf.has_sindex
        # and the "new" should be the same
        new_sindex = gdf.sindex
        assert old_sindex is new_sindex


# Skip to accommodate Shapely geometries being unhashable
@pytest.mark.skip
class TestJoinSindex:
    def setup_method(self):
        nybb_filename = geopandas.datasets.get_path("nybb")
        self.boros = read_file(nybb_filename)

    def test_merge_geo(self):
        # First check that we gets hits from the boros frame.
        tree = self.boros.sindex
        hits = tree.intersection((1012821.80, 229228.26))
        res = [self.boros.iloc[hit]["BoroName"] for hit in hits]
        assert res == ["Bronx", "Queens"]

        # Check that we only get the Bronx from this view.
        first = self.boros[self.boros["BoroCode"] < 3]
        tree = first.sindex
        hits = tree.intersection((1012821.80, 229228.26))
        res = [first.iloc[hit]["BoroName"] for hit in hits]
        assert res == ["Bronx"]

        # Check that we only get Queens from this view.
        second = self.boros[self.boros["BoroCode"] >= 3]
        tree = second.sindex
        hits = tree.intersection((1012821.80, 229228.26))
        res = ([second.iloc[hit]["BoroName"] for hit in hits],)
        assert res == ["Queens"]

        # Get both the Bronx and Queens again.
        merged = first.merge(second, how="outer")
        assert len(merged) == 5
        assert merged.sindex.size == 5
        tree = merged.sindex
        hits = tree.intersection((1012821.80, 229228.26))
        res = [merged.iloc[hit]["BoroName"] for hit in hits]
        assert res == ["Bronx", "Queens"]


@pytest.mark.skip_no_sindex
class TestPygeosInterface:
    def setup_method(self):
        data = {
            "geom": [Point(x, y) for x, y in zip(range(5), range(5))]
            + [box(10, 10, 20, 20)]  # include a box geometry
        }
        self.df = GeoDataFrame(data, geometry="geom")
        self.expected_size = len(data["geom"])

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
            (
                "contains",
                box(10, 10, 20, 20),
                [5],
            ),  # contains but does not contains_properly
            (
                "covers",
                box(-0.5, -0.5, 1, 1),
                [0, 1],
            ),  # covers (0, 0) and (1, 1)
            (
                "covers",
                box(0.001, 0.001, 0.99, 0.99),
                [],
            ),  # does not cover any
            (
                "covers",
                box(0, 0, 1, 1),
                [0, 1],
            ),  # covers but does not contain
            (
                "contains_properly",
                box(0, 0, 1, 1),
                [],
            ),  # intersects but does not contain
            (
                "contains_properly",
                box(0, 0, 1.001, 1.001),
                [1],
            ),  # intersects 2 and contains 1
            (
                "contains_properly",
                box(0.5, 0.5, 1.001, 1.001),
                [1],
            ),  # intersects 1 and contains 1
            (
                "contains_properly",
                box(0.5, 0.5, 1.5, 1.5),
                [1],
            ),  # intersects and contains
            (
                "contains_properly",
                box(-1, -1, 2, 2),
                [0, 1],
            ),  # intersects and contains multiple
            (
                "contains_properly",
                box(10, 10, 20, 20),
                [],
            ),  # contains but does not contains_properly
        ),
    )
    def test_query(self, predicate, test_geom, expected):
        """Tests the `query` method with valid inputs and valid predicates."""
        res = self.df.sindex.query(test_geom, predicate=predicate)
        assert_array_equal(res, expected)

    def test_query_invalid_geometry(self):
        """Tests the `query` method with invalid geometry."""
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
        """Tests the `query` method with empty geometry."""
        res = self.df.sindex.query(test_geom)
        assert_array_equal(res, expected_value)

    def test_query_invalid_predicate(self):
        """Tests the `query` method with invalid predicates."""
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

        test_geo = test_df.geometry.values[0]
        res = tree_df.sindex.query(test_geo, sort=sort)

        # asserting the same elements
        assert sorted(res) == sorted(expected)
        # asserting the exact array can fail if sort=False
        try:
            assert_array_equal(res, expected)
        except AssertionError as e:
            if sort is False:
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
            (
                "contains",
                [(0, 0, 1, 1)],
                [[], []],
            ),  # intersects and covers, but does not contain
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
            (
                "contains",
                [(10, 10, 20, 20)],
                [[0], [5]],
            ),  # contains but does not contains_properly
            ("touches", [(-1, -1, 0, 0)], [[0], [0]]),  # bbox intersects and touches
            (
                "touches",
                [(-0.5, -0.5, 1.5, 1.5)],
                [[], []],
            ),  # bbox intersects but geom does not touch
            (
                "covers",
                [(-0.5, -0.5, 1, 1)],
                [[0, 0], [0, 1]],
            ),  # covers (0, 0) and (1, 1)
            (
                "covers",
                [(0.001, 0.001, 0.99, 0.99)],
                [[], []],
            ),  # does not cover any
            (
                "covers",
                [(0, 0, 1, 1)],
                [[0, 0], [0, 1]],
            ),  # covers but does not contain
            (
                "contains_properly",
                [(0, 0, 1, 1)],
                [[], []],
            ),  # intersects but does not contain
            (
                "contains_properly",
                [(0, 0, 1.001, 1.001)],
                [[0], [1]],
            ),  # intersects 2 and contains 1
            (
                "contains_properly",
                [(0.5, 0.5, 1.001, 1.001)],
                [[0], [1]],
            ),  # intersects 1 and contains 1
            (
                "contains_properly",
                [(0.5, 0.5, 1.5, 1.5)],
                [[0], [1]],
            ),  # intersects and contains
            (
                "contains_properly",
                [(-1, -1, 2, 2)],
                [[0, 0], [0, 1]],
            ),  # intersects and contains multiple
            (
                "contains_properly",
                [(10, 10, 20, 20)],
                [[], []],
            ),  # contains but does not contains_properly
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
        res = self.df.sindex.query(test_geom, predicate=predicate)
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
        """Tests the `query_bulk` method with an empty geometry."""
        # pass through GeoSeries to have GeoPandas
        # determine if it should use shapely or pygeos geometry objects
        # note: for this test, test_geoms (note plural) is a list already
        test_geoms = geopandas.GeoSeries(test_geoms, index=range(len(test_geoms)))
        res = self.df.sindex.query(test_geoms)
        assert_array_equal(res, expected_value)

    def test_query_bulk_empty_input_array(self):
        """Tests the `query_bulk` method with an empty input array."""
        test_array = np.array([], dtype=object)
        expected_value = [[], []]
        res = self.df.sindex.query(test_array)
        assert_array_equal(res, expected_value)

    def test_query_bulk_invalid_input_geometry(self):
        """
        Tests the `query_bulk` method with invalid input for the `geometry` parameter.
        """
        test_array = "notanarray"
        with pytest.raises(TypeError):
            self.df.sindex.query(test_array)

    def test_query_bulk_invalid_predicate(self):
        """Tests the `query_bulk` method with invalid predicates."""
        test_geom_bounds = (-1, -1, -0.5, -0.5)
        test_predicate = "test"

        # pass through GeoSeries to have GeoPandas
        # determine if it should use shapely or pygeos geometry objects
        test_geom = geopandas.GeoSeries([box(*test_geom_bounds)], index=["0"])

        with pytest.raises(ValueError):
            self.df.sindex.query(test_geom.geometry, predicate=test_predicate)

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
        res = self.df.sindex.query(test_geom, predicate=predicate)
        assert_array_equal(res, expected)

        # test GeometryArray
        res = self.df.sindex.query(test_geom.geometry, predicate=predicate)
        assert_array_equal(res, expected)
        res = self.df.sindex.query(test_geom.geometry.values, predicate=predicate)
        assert_array_equal(res, expected)

        # test numpy array
        res = self.df.sindex.query(
            test_geom.geometry.values.to_numpy(), predicate=predicate
        )
        assert_array_equal(res, expected)
        res = self.df.sindex.query(
            test_geom.geometry.values.to_numpy(), predicate=predicate
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

        res = tree_df.sindex.query(test_df.geometry, sort=sort)

        # asserting the same elements
        assert sorted(res[0]) == sorted(expected[0])
        assert sorted(res[1]) == sorted(expected[1])
        # asserting the exact array can fail if sort=False
        try:
            assert_array_equal(res, expected)
        except AssertionError as e:
            if sort is False:
                pytest.xfail(
                    "rtree results are known to be unordered, see "
                    "https://github.com/geopandas/geopandas/issues/1337\n"
                    "Expected:\n {}\n".format(expected)
                    + "Got:\n {}\n".format(res.tolist())
                )
            raise e

    # ------------------------- `nearest` tests ------------------------- #
    @pytest.mark.skipif(
        compat.USE_PYGEOS or compat.USE_SHAPELY_20,
        reason=("RTree supports sindex.nearest with different behaviour"),
    )
    def test_rtree_nearest_warns(self):
        df = geopandas.GeoDataFrame({"geometry": []})
        with pytest.warns(
            FutureWarning, match="sindex.nearest using the rtree backend"
        ):
            df.sindex.nearest((0, 0, 1, 1), num_results=2)

    @pytest.mark.skipif(
        compat.USE_SHAPELY_20 or not (compat.USE_PYGEOS and not compat.PYGEOS_GE_010),
        reason=("PyGEOS < 0.10 does not support sindex.nearest"),
    )
    def test_pygeos_error(self):
        df = geopandas.GeoDataFrame({"geometry": []})
        with pytest.raises(NotImplementedError, match="requires pygeos >= 0.10"):
            df.sindex.nearest(None)

    @pytest.mark.skipif(
        not (compat.USE_SHAPELY_20 or (compat.USE_PYGEOS and compat.PYGEOS_GE_010)),
        reason=("PyGEOS >= 0.10 is required to test sindex.nearest"),
    )
    @pytest.mark.parametrize("return_all", [True, False])
    @pytest.mark.parametrize(
        "geometry,expected",
        [
            ([0.25, 0.25], [[0], [0]]),
            ([0.75, 0.75], [[0], [1]]),
        ],
    )
    def test_nearest_single(self, geometry, expected, return_all):
        geoms = mod.points(np.arange(10), np.arange(10))
        df = geopandas.GeoDataFrame({"geometry": geoms})

        p = Point(geometry)
        res = df.sindex.nearest(p, return_all=return_all)
        assert_array_equal(res, expected)

        p = mod.points(geometry)
        res = df.sindex.nearest(p, return_all=return_all)
        assert_array_equal(res, expected)

    @pytest.mark.skipif(
        not (compat.USE_SHAPELY_20 or (compat.USE_PYGEOS and compat.PYGEOS_GE_010)),
        reason=("PyGEOS >= 0.10 is required to test sindex.nearest"),
    )
    @pytest.mark.parametrize("return_all", [True, False])
    @pytest.mark.parametrize(
        "geometry,expected",
        [
            ([(1, 1), (0, 0)], [[0, 1], [1, 0]]),
            ([(1, 1), (0.25, 1)], [[0, 1], [1, 1]]),
        ],
    )
    def test_nearest_multi(self, geometry, expected, return_all):
        geoms = mod.points(np.arange(10), np.arange(10))
        df = geopandas.GeoDataFrame({"geometry": geoms})

        ps = [Point(p) for p in geometry]
        res = df.sindex.nearest(ps, return_all=return_all)
        assert_array_equal(res, expected)

        ps = mod.points(geometry)
        res = df.sindex.nearest(ps, return_all=return_all)
        assert_array_equal(res, expected)

        s = geopandas.GeoSeries(ps)
        res = df.sindex.nearest(s, return_all=return_all)
        assert_array_equal(res, expected)

        x, y = zip(*geometry)
        ga = geopandas.points_from_xy(x, y)
        res = df.sindex.nearest(ga, return_all=return_all)
        assert_array_equal(res, expected)

    @pytest.mark.skipif(
        not (compat.USE_SHAPELY_20 or (compat.USE_PYGEOS and compat.PYGEOS_GE_010)),
        reason=("PyGEOS >= 0.10 is required to test sindex.nearest"),
    )
    @pytest.mark.parametrize("return_all", [True, False])
    @pytest.mark.parametrize(
        "geometry,expected",
        [
            (None, [[], []]),
            ([None], [[], []]),
        ],
    )
    def test_nearest_none(self, geometry, expected, return_all):
        geoms = mod.points(np.arange(10), np.arange(10))
        df = geopandas.GeoDataFrame({"geometry": geoms})

        res = df.sindex.nearest(geometry, return_all=return_all)
        assert_array_equal(res, expected)

    @pytest.mark.skipif(
        not (compat.USE_SHAPELY_20 or (compat.USE_PYGEOS and compat.PYGEOS_GE_010)),
        reason=("PyGEOS >= 0.10 is required to test sindex.nearest"),
    )
    @pytest.mark.parametrize("return_distance", [True, False])
    @pytest.mark.parametrize(
        "return_all,max_distance,expected",
        [
            (True, None, ([[0, 0, 1], [0, 1, 5]], [sqrt(0.5), sqrt(0.5), sqrt(50)])),
            (False, None, ([[0, 1], [0, 5]], [sqrt(0.5), sqrt(50)])),
            (True, 1, ([[0, 0], [0, 1]], [sqrt(0.5), sqrt(0.5)])),
            (False, 1, ([[0], [0]], [sqrt(0.5)])),
        ],
    )
    def test_nearest_max_distance(
        self, expected, max_distance, return_all, return_distance
    ):
        geoms = mod.points(np.arange(10), np.arange(10))
        df = geopandas.GeoDataFrame({"geometry": geoms})

        ps = [Point(0.5, 0.5), Point(0, 10)]
        res = df.sindex.nearest(
            ps,
            return_all=return_all,
            max_distance=max_distance,
            return_distance=return_distance,
        )
        if return_distance:
            assert_array_equal(res[0], expected[0])
            assert_array_equal(res[1], expected[1])
        else:
            assert_array_equal(res, expected[0])

    @pytest.mark.skipif(
        not (compat.USE_SHAPELY_20),
        reason=(
            "shapely >= 2.0 is required to test sindex.nearest with parameter exclusive"
        ),
    )
    @pytest.mark.parametrize("return_distance", [True, False])
    @pytest.mark.parametrize(
        "return_all,max_distance,exclusive,expected",
        [
            (False, None, False, ([[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]], 5 * [0])),
            (False, None, True, ([[0, 1, 2, 3, 4], [1, 0, 1, 2, 3]], 5 * [sqrt(2)])),
            (True, None, False, ([[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]], 5 * [0])),
            (
                True,
                None,
                True,
                ([[0, 1, 1, 2, 2, 3, 3, 4], [1, 0, 2, 1, 3, 2, 4, 3]], 8 * [sqrt(2)]),
            ),
            (False, 1.1, True, ([[1, 2, 5], [5, 5, 1]], 3 * [1])),
            (True, 1.1, True, ([[1, 2, 5, 5], [5, 5, 1, 2]], 4 * [1])),
        ],
    )
    def test_nearest_exclusive(
        self, expected, max_distance, return_all, return_distance, exclusive
    ):
        geoms = mod.points(np.arange(5), np.arange(5))
        if max_distance:
            # add a non grid point
            geoms = np.append(geoms, [Point(1, 2)])

        df = geopandas.GeoDataFrame({"geometry": geoms})

        ps = geoms
        res = df.sindex.nearest(
            ps,
            return_all=return_all,
            max_distance=max_distance,
            return_distance=return_distance,
            exclusive=exclusive,
        )
        if return_distance:
            assert_array_equal(res[0], expected[0])
            assert_array_equal(res[1], expected[1])
        else:
            assert_array_equal(res, expected[0])

    @pytest.mark.skipif(
        compat.USE_SHAPELY_20 or not (compat.USE_PYGEOS and not compat.PYGEOS_GE_010),
        reason="sindex.nearest exclusive parameter requires shapely >= 2.0",
    )
    def test_nearest_exclusive_unavailable(self):
        from shapely.geometry import Point

        geoms = [Point((x, y)) for (x, y) in zip(np.arange(5), np.arange(5))]
        df = geopandas.GeoDataFrame(geometry=geoms)

        with pytest.raises(NotImplementedError, match="requires shapely >= 2.0"):
            df.sindex.nearest(geoms, exclusive=True)

    # --------------------------- misc tests ---------------------------- #

    def test_empty_tree_geometries(self):
        """Tests building sindex with interleaved empty geometries."""
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
        empty = geopandas.GeoSeries([], dtype=object)
        assert empty.sindex.is_empty
        empty = geopandas.GeoSeries([None])
        assert empty.sindex.is_empty
        empty = geopandas.GeoSeries([Point()])
        assert empty.sindex.is_empty
        # create a non-empty tree
        non_empty = geopandas.GeoSeries([Point(0, 0)])
        assert not non_empty.sindex.is_empty

    @pytest.mark.parametrize(
        "predicate, expected_shape",
        [
            (None, (2, 471)),
            ("intersects", (2, 213)),
            ("within", (2, 213)),
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

        res = world.sindex.query(capitals.geometry, predicate)
        assert res.shape == expected_shape


@pytest.mark.skipif(not compat.HAS_RTREE, reason="no rtree installed")
def test_old_spatial_index_deprecated():
    t1 = Polygon([(0, 0), (1, 0), (1, 1)])
    t2 = Polygon([(0, 0), (1, 1), (0, 1)])

    stream = ((i, item.bounds, None) for i, item in enumerate([t1, t2]))

    with pytest.warns(FutureWarning):
        idx = geopandas.sindex.SpatialIndex(stream)

    assert list(idx.intersection((0, 0, 1, 1))) == [0, 1]
