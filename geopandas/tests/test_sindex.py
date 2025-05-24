from math import sqrt

import numpy as np

import shapely
from shapely.geometry import (
    GeometryCollection,
    LineString,
    MultiPolygon,
    Point,
    Polygon,
    box,
)

import geopandas
from geopandas import GeoDataFrame, GeoSeries, read_file
from geopandas import _compat as compat

import pytest
from numpy.testing import assert_array_equal

try:
    from scipy.sparse import coo_array  # noqa: F401

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
SCIPY_MARK = pytest.mark.skipif(not HAS_SCIPY, reason="scipy not installed")


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
        # with pandas 2.0, the column is now copied, losing the index. But
        # with pandas >= 3.0 and Copy-on-Write this is preserved again
        subset1 = self.df[["geom", "A"]]
        if not compat.PANDAS_GE_30:
            assert subset1.sindex is not original_index
        else:
            assert subset1.sindex is original_index
        subset2 = self.df[["A", "geom"]]
        if not compat.PANDAS_GE_30:
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


# Skip to accommodate Shapely geometries being unhashable # TODO unskip?
@pytest.mark.skip
@pytest.mark.usefixtures("_setup_class_nybb_filename")
class TestJoinSindex:
    def setup_method(self):
        self.boros = read_file(self.nybb_filename)

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


class TestShapelyInterface:
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
        with pytest.raises(TypeError):
            # we raise a useful TypeError
            self.df.sindex.intersection(test_geom)

    # ------------------------------ `query` tests ------------------------------ #
    @pytest.mark.parametrize(
        "output_format", ("indices", pytest.param("sparse", marks=SCIPY_MARK), "dense")
    )
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
    def test_query(self, predicate, test_geom, expected, output_format):
        """Tests the `query` method with valid inputs and valid predicates."""
        res = self.df.sindex.query(test_geom, predicate=predicate)
        assert_array_equal(res, expected)

        if output_format != "indices":
            dense = np.zeros(len(self.df), dtype=bool)
            dense[expected] = True

            res = self.df.sindex.query(
                test_geom, predicate=predicate, output_format=output_format
            )
            if output_format == "sparse":
                res = res.todense()
            assert_array_equal(res, dense)

    def test_query_invalid_geometry(self):
        """Tests the `query` method with invalid geometry."""
        with pytest.raises(TypeError):
            self.df.sindex.query("notavalidgeom")

    @pytest.mark.skipif(not compat.GEOS_GE_310, reason="Requires GEOS 3.10")
    @pytest.mark.parametrize(
        "distance, test_geom, expected",
        (
            # bounds don't intersect and not within distance=0
            (
                0,
                box(9.0, 9.0, 9.9, 9.9),
                [],
            ),
            # bounds don't intersect but is within distance=1
            (
                1,
                box(9.0, 9.0, 9.9, 9.9),
                [5],
            ),
            # within 1-D absolute distance in both axes, but not euclidean distance
            (
                0.5,
                Point(0.5, 0.5),
                [],
            ),
            # same as before but within euclidean distance
            (
                sqrt(2 * 0.5**2) + 1e-9,
                Point(0.5, 0.5),
                [0, 1],
            ),
            # less than euclidean distance between points, multi-object
            (
                sqrt(2) - 1e-9,
                [
                    Polygon([(0, 0), (1, 0), (1, 1)]),
                    Polygon([(1, 1), (2, 1), (2, 2)]),
                ],  # multi-object test
                [[0, 0, 1, 1], [0, 1, 1, 2]],
            ),
            # more than euclidean distance between points, multi-object
            (
                sqrt(2) + 1e-9,
                [
                    Polygon([(0, 0), (1, 0), (1, 1)]),
                    Polygon([(1, 1), (2, 1), (2, 2)]),
                ],
                [[0, 0, 0, 1, 1, 1, 1], [0, 1, 2, 0, 1, 2, 3]],
            ),
            # distance is array-like, broadcastable to geometry
            (
                [2, 10],
                [Point(0.5, 0.5), Point(1, 1)],
                [[0, 0, 1, 1, 1, 1, 1], [0, 1, 0, 1, 2, 3, 4]],
            ),
        ),
    )
    def test_query_dwithin(self, distance, test_geom, expected):
        """Tests the `query` method with predicates that require keyword arguments."""
        res = self.df.sindex.query(test_geom, predicate="dwithin", distance=distance)
        assert_array_equal(res, expected)

    @pytest.mark.skipif(not compat.GEOS_GE_310, reason="Requires GEOS 3.10")
    def test_dwithin_no_distance(self):
        """Tests the `query` method with keyword arguments that are
        invalid for certain predicates."""
        with pytest.raises(
            ValueError, match="'distance' parameter is required for 'dwithin' predicate"
        ):
            self.df.sindex.query(Point(0, 0), predicate="dwithin")

    @pytest.mark.parametrize(
        "predicate",
        [
            None,
            "contains",
            "contains_properly",
            "covered_by",
            "covers",
            "crosses",
            "intersects",
            "overlaps",
            "touches",
            "within",
        ],
    )
    def test_query_distance_invalid(self, predicate):
        """Tests the `query` method with keyword arguments that are
        invalid for certain predicates."""
        msg = "'distance' parameter is only supported in combination with 'dwithin'"
        with pytest.raises(ValueError, match=msg):
            self.df.sindex.query(Point(0, 0), predicate=predicate, distance=0)

    @pytest.mark.skipif(
        compat.GEOS_GE_310, reason="Test for 'dwithin'-incompatible versions of GEOS"
    )
    def test_dwithin_requirements(self):
        """Tests whether a ValueError is raised when trying to use dwithin with
        incompatible versions of shapely or pyGEOS
        """
        with pytest.raises(
            ValueError, match="predicate = 'dwithin' requires GEOS >= 3.10.0"
        ):
            self.df.sindex.query(Point(0, 0), predicate="dwithin", distance=0)

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

        test_geo = test_polys.values[0]
        res = tree_polys.sindex.query(test_geo, sort=sort)

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
                    f"Expected:\n {expected}\n"
                    f"Got:\n {res.tolist()}\n"
                )
            raise e

    def test_unsupported_output(self):
        with pytest.raises(ValueError, match="Invalid output_format: 'dataarray'."):
            test_geom = box(-1, -1, -0.5, -0.5)
            self.df.sindex.query(test_geom, output_format="dataarray")

    # ------------------------- `query_bulk` tests -------------------------- #
    @pytest.mark.parametrize(
        "output_format", ("indices", pytest.param("sparse", marks=SCIPY_MARK), "dense")
    )
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
    def test_query_bulk(self, predicate, test_geom, expected, output_format):
        """Tests the `query` method with valid
        inputs and valid predicates.
        """
        test_geoms = [box(*geom) for geom in test_geom]
        res = self.df.sindex.query(test_geoms, predicate=predicate)
        assert_array_equal(res, expected)

        if output_format != "indices":
            dense = np.zeros((len(self.df), len(test_geoms)), dtype=bool)
            tree, other = expected[::-1]
            dense[tree, other] = True

            res = self.df.sindex.query(
                test_geoms, predicate=predicate, output_format=output_format
            )
            if output_format == "sparse":
                res = res.todense()
            assert_array_equal(res, dense)

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
        """Tests the `query` method with an empty geometries."""
        res = self.df.sindex.query(test_geoms)
        assert_array_equal(res, expected_value)

    def test_query_bulk_empty_input_array(self):
        """Tests the `query` method with an empty input array."""
        test_array = np.array([], dtype=object)
        expected_value = [[], []]
        res = self.df.sindex.query(test_array)
        assert_array_equal(res, expected_value)

    def test_query_bulk_invalid_input_geometry(self):
        """
        Tests the `query` method with invalid input for the `geometry` parameter.
        """
        test_array = "notanarray"
        with pytest.raises(TypeError):
            self.df.sindex.query(test_array)

    def test_query_bulk_invalid_predicate(self):
        """Tests the `query` method with invalid predicates."""
        test_geom_bounds = (-1, -1, -0.5, -0.5)
        test_predicate = "test"

        with pytest.raises(ValueError):
            self.df.sindex.query([box(*test_geom_bounds)], predicate=test_predicate)

    @pytest.mark.parametrize(
        "predicate, test_geom, expected",
        (
            (None, (-1, -1, -0.5, -0.5), [[], []]),
            ("intersects", (-1, -1, -0.5, -0.5), [[], []]),
            ("contains", (-1, -1, 1, 1), [[0], [0]]),
        ),
    )
    def test_query_bulk_input_type(self, predicate, test_geom, expected):
        """Tests that query can accept a GeoSeries, GeometryArray or
        numpy array.
        """
        # pass through GeoSeries to test input type
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
        """Check that results from `query` don't depend
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

        res = tree_polys.sindex.query(test_polys, sort=sort)

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
                    f"Expected:\n {expected}\n"
                    f"Got:\n {res.tolist()}\n"
                )
            raise e

    # ------------------------- `nearest` tests ------------------------- #
    @pytest.mark.parametrize("return_all", [True, False])
    @pytest.mark.parametrize(
        "geometry,expected",
        [
            ([0.25, 0.25], [[0], [0]]),
            ([0.75, 0.75], [[0], [1]]),
        ],
    )
    def test_nearest_single(self, geometry, expected, return_all):
        geoms = shapely.points(np.arange(10), np.arange(10))
        df = geopandas.GeoDataFrame({"geometry": geoms})

        p = Point(geometry)
        res = df.sindex.nearest(p, return_all=return_all)
        assert_array_equal(res, expected)

        p = shapely.points(geometry)
        res = df.sindex.nearest(p, return_all=return_all)
        assert_array_equal(res, expected)

    @pytest.mark.parametrize("return_all", [True, False])
    @pytest.mark.parametrize(
        "geometry,expected",
        [
            ([(1, 1), (0, 0)], [[0, 1], [1, 0]]),
            ([(1, 1), (0.25, 1)], [[0, 1], [1, 1]]),
        ],
    )
    def test_nearest_multi(self, geometry, expected, return_all):
        geoms = shapely.points(np.arange(10), np.arange(10))
        df = geopandas.GeoDataFrame({"geometry": geoms})

        ps = [Point(p) for p in geometry]
        res = df.sindex.nearest(ps, return_all=return_all)
        assert_array_equal(res, expected)

        ps = shapely.points(geometry)
        res = df.sindex.nearest(ps, return_all=return_all)
        assert_array_equal(res, expected)

        s = geopandas.GeoSeries(ps)
        res = df.sindex.nearest(s, return_all=return_all)
        assert_array_equal(res, expected)

        x, y = zip(*geometry)
        ga = geopandas.points_from_xy(x, y)
        res = df.sindex.nearest(ga, return_all=return_all)
        assert_array_equal(res, expected)

    @pytest.mark.parametrize("return_all", [True, False])
    @pytest.mark.parametrize(
        "geometry,expected",
        [
            (None, [[], []]),
            ([None], [[], []]),
        ],
    )
    def test_nearest_none(self, geometry, expected, return_all):
        geoms = shapely.points(np.arange(10), np.arange(10))
        df = geopandas.GeoDataFrame({"geometry": geoms})

        res = df.sindex.nearest(geometry, return_all=return_all)
        assert_array_equal(res, expected)

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
        geoms = shapely.points(np.arange(10), np.arange(10))
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
        geoms = shapely.points(np.arange(5), np.arange(5))
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
    def test_integration_natural_earth(
        self, predicate, expected_shape, naturalearth_lowres, naturalearth_cities
    ):
        """Tests output sizes for the naturalearth datasets."""
        world = read_file(naturalearth_lowres)
        capitals = read_file(naturalearth_cities)

        res = world.sindex.query(capitals.geometry, predicate)
        assert res.shape == expected_shape
