"""Tests for the clip module."""

import warnings
from packaging.version import Version

import numpy as np
import pandas as pd

import shapely
from shapely.geometry import (
    Polygon,
    Point,
    LineString,
    LinearRing,
    GeometryCollection,
    MultiPoint,
    box,
)

import geopandas
from geopandas import GeoDataFrame, GeoSeries, clip

from geopandas.testing import assert_geodataframe_equal, assert_geoseries_equal
import pytest

from geopandas.tools.clip import _mask_is_list_like_rectangle

pytestmark = pytest.mark.skip_no_sindex
pandas_133 = Version(pd.__version__) == Version("1.3.3")
mask_variants_single_rectangle = [
    "single_rectangle_gdf",
    "single_rectangle_gdf_list_bounds",
    "single_rectangle_gdf_tuple_bounds",
    "single_rectangle_gdf_array_bounds",
]
mask_variants_large_rectangle = [
    "larger_single_rectangle_gdf",
    "larger_single_rectangle_gdf_bounds",
]


@pytest.fixture
def point_gdf():
    """Create a point GeoDataFrame."""
    pts = np.array([[2, 2], [3, 4], [9, 8], [-12, -15]])
    gdf = GeoDataFrame([Point(xy) for xy in pts], columns=["geometry"], crs="EPSG:3857")
    return gdf


@pytest.fixture
def pointsoutside_nooverlap_gdf():
    """Create a point GeoDataFrame. Its points are all outside the single
    rectangle, and its bounds are outside the single rectangle's."""
    pts = np.array([[5, 15], [15, 15], [15, 20]])
    gdf = GeoDataFrame([Point(xy) for xy in pts], columns=["geometry"], crs="EPSG:3857")
    return gdf


@pytest.fixture
def pointsoutside_overlap_gdf():
    """Create a point GeoDataFrame. Its points are all outside the single
    rectangle, and its bounds are overlapping the single rectangle's."""
    pts = np.array([[5, 15], [15, 15], [15, 5]])
    gdf = GeoDataFrame([Point(xy) for xy in pts], columns=["geometry"], crs="EPSG:3857")
    return gdf


@pytest.fixture
def single_rectangle_gdf():
    """Create a single rectangle for clipping."""
    poly_inters = Polygon([(0, 0), (0, 10), (10, 10), (10, 0), (0, 0)])
    gdf = GeoDataFrame([1], geometry=[poly_inters], crs="EPSG:3857")
    gdf["attr2"] = "site-boundary"
    return gdf


@pytest.fixture
def single_rectangle_gdf_tuple_bounds(single_rectangle_gdf):
    """Bounds of the created single rectangle"""
    return tuple(single_rectangle_gdf.total_bounds)


@pytest.fixture
def single_rectangle_gdf_list_bounds(single_rectangle_gdf):
    """Bounds of the created single rectangle"""
    return list(single_rectangle_gdf.total_bounds)


@pytest.fixture
def single_rectangle_gdf_array_bounds(single_rectangle_gdf):
    """Bounds of the created single rectangle"""
    return single_rectangle_gdf.total_bounds


@pytest.fixture
def larger_single_rectangle_gdf():
    """Create a slightly larger rectangle for clipping.
    The smaller single rectangle is used to test the edge case where slivers
    are returned when you clip polygons. This fixture is larger which
    eliminates the slivers in the clip return.
    """
    poly_inters = Polygon([(-5, -5), (-5, 15), (15, 15), (15, -5), (-5, -5)])
    gdf = GeoDataFrame([1], geometry=[poly_inters], crs="EPSG:3857")
    gdf["attr2"] = ["study area"]
    return gdf


@pytest.fixture
def larger_single_rectangle_gdf_bounds(larger_single_rectangle_gdf):
    """Bounds of the created single rectangle"""
    return tuple(larger_single_rectangle_gdf.total_bounds)


@pytest.fixture
def buffered_locations(point_gdf):
    """Buffer points to create a multi-polygon."""
    buffered_locs = point_gdf
    buffered_locs["geometry"] = buffered_locs.buffer(4)
    buffered_locs["type"] = "plot"
    return buffered_locs


@pytest.fixture
def donut_geometry(buffered_locations, single_rectangle_gdf):
    """Make a geometry with a hole in the middle (a donut)."""
    donut = geopandas.overlay(
        buffered_locations, single_rectangle_gdf, how="symmetric_difference"
    )
    return donut


@pytest.fixture
def two_line_gdf():
    """Create Line Objects For Testing"""
    linea = LineString([(1, 1), (2, 2), (3, 2), (5, 3)])
    lineb = LineString([(3, 4), (5, 7), (12, 2), (10, 5), (9, 7.5)])
    gdf = GeoDataFrame([1, 2], geometry=[linea, lineb], crs="EPSG:3857")
    return gdf


@pytest.fixture
def multi_poly_gdf(donut_geometry):
    """Create a multi-polygon GeoDataFrame."""
    multi_poly = donut_geometry.unary_union
    out_df = GeoDataFrame(geometry=GeoSeries(multi_poly), crs="EPSG:3857")
    out_df["attr"] = ["pool"]
    return out_df


@pytest.fixture
def multi_line(two_line_gdf):
    """Create a multi-line GeoDataFrame.
    This GDF has one multiline and one regular line."""
    # Create a single and multi line object
    multiline_feat = two_line_gdf.unary_union
    linec = LineString([(2, 1), (3, 1), (4, 1), (5, 2)])
    out_df = GeoDataFrame(geometry=GeoSeries([multiline_feat, linec]), crs="EPSG:3857")
    out_df["attr"] = ["road", "stream"]
    return out_df


@pytest.fixture
def multi_point(point_gdf):
    """Create a multi-point GeoDataFrame."""
    multi_point = point_gdf.unary_union
    out_df = GeoDataFrame(
        geometry=GeoSeries(
            [multi_point, Point(2, 5), Point(-11, -14), Point(-10, -12)]
        ),
        crs="EPSG:3857",
    )
    out_df["attr"] = ["tree", "another tree", "shrub", "berries"]
    return out_df


@pytest.fixture
def mixed_gdf():
    """Create a Mixed Polygon and LineString For Testing"""
    point = Point(2, 3)
    line = LineString([(1, 1), (2, 2), (3, 2), (5, 3), (12, 1)])
    poly = Polygon([(3, 4), (5, 2), (12, 2), (10, 5), (9, 7.5)])
    ring = LinearRing([(1, 1), (2, 2), (3, 2), (5, 3), (12, 1)])
    gdf = GeoDataFrame(
        [1, 2, 3, 4], geometry=[point, poly, line, ring], crs="EPSG:3857"
    )
    return gdf


@pytest.fixture
def geomcol_gdf():
    """Create a Mixed Polygon and LineString For Testing"""
    point = Point(2, 3)
    poly = Polygon([(3, 4), (5, 2), (12, 2), (10, 5), (9, 7.5)])
    coll = GeometryCollection([point, poly])
    gdf = GeoDataFrame([1], geometry=[coll], crs="EPSG:3857")
    return gdf


@pytest.fixture
def sliver_line():
    """Create a line that will create a point when clipped."""
    linea = LineString([(10, 5), (13, 5), (15, 5)])
    lineb = LineString([(1, 1), (2, 2), (3, 2), (5, 3), (12, 1)])
    gdf = GeoDataFrame([1, 2], geometry=[linea, lineb], crs="EPSG:3857")
    return gdf


def test_not_gdf(single_rectangle_gdf):
    """Non-GeoDataFrame inputs raise attribute errors."""
    with pytest.raises(TypeError):
        clip((2, 3), single_rectangle_gdf)
    with pytest.raises(TypeError):
        clip(single_rectangle_gdf, "foobar")
    with pytest.raises(TypeError):
        clip(single_rectangle_gdf, (1, 2, 3))
    with pytest.raises(TypeError):
        clip(single_rectangle_gdf, (1, 2, 3, 4, 5))


def test_non_overlapping_geoms():
    """Test that a bounding box returns empty if the extents don't overlap"""
    unit_box = Polygon([(0, 0), (0, 1), (1, 1), (1, 0), (0, 0)])
    unit_gdf = GeoDataFrame([1], geometry=[unit_box], crs="EPSG:3857")
    non_overlapping_gdf = unit_gdf.copy()
    non_overlapping_gdf = non_overlapping_gdf.geometry.apply(
        lambda x: shapely.affinity.translate(x, xoff=20)
    )
    out = clip(unit_gdf, non_overlapping_gdf)
    assert_geodataframe_equal(out, unit_gdf.iloc[:0])
    out2 = clip(unit_gdf.geometry, non_overlapping_gdf)
    assert_geoseries_equal(out2, GeoSeries(crs=unit_gdf.crs))


@pytest.mark.parametrize("mask_fixture_name", mask_variants_single_rectangle)
class TestClipWithSingleRectangleGdf:
    @pytest.fixture
    def mask(self, mask_fixture_name, request):
        return request.getfixturevalue(mask_fixture_name)

    def test_returns_gdf(self, point_gdf, mask):
        """Test that function returns a GeoDataFrame (or GDF-like) object."""
        out = clip(point_gdf, mask)
        assert isinstance(out, GeoDataFrame)

    def test_returns_series(self, point_gdf, mask):
        """Test that function returns a GeoSeries if GeoSeries is passed."""
        out = clip(point_gdf.geometry, mask)
        assert isinstance(out, GeoSeries)

    def test_clip_points(self, point_gdf, mask):
        """Test clipping a points GDF with a generic polygon geometry."""
        clip_pts = clip(point_gdf, mask)
        pts = np.array([[2, 2], [3, 4], [9, 8]])
        exp = GeoDataFrame(
            [Point(xy) for xy in pts], columns=["geometry"], crs="EPSG:3857"
        )
        assert_geodataframe_equal(clip_pts, exp)

    def test_clip_points_geom_col_rename(self, point_gdf, mask):
        """Test clipping a points GDF with a generic polygon geometry."""
        point_gdf_geom_col_rename = point_gdf.rename_geometry("geometry2")
        clip_pts = clip(point_gdf_geom_col_rename, mask)
        pts = np.array([[2, 2], [3, 4], [9, 8]])
        exp = GeoDataFrame(
            [Point(xy) for xy in pts],
            columns=["geometry2"],
            crs="EPSG:3857",
            geometry="geometry2",
        )
        assert_geodataframe_equal(clip_pts, exp)

    def test_clip_poly(self, buffered_locations, mask):
        """Test clipping a polygon GDF with a generic polygon geometry."""
        clipped_poly = clip(buffered_locations, mask)
        assert len(clipped_poly.geometry) == 3
        assert all(clipped_poly.geom_type == "Polygon")

    def test_clip_poly_geom_col_rename(self, buffered_locations, mask):
        """Test clipping a polygon GDF with a generic polygon geometry."""

        poly_gdf_geom_col_rename = buffered_locations.rename_geometry("geometry2")
        clipped_poly = clip(poly_gdf_geom_col_rename, mask)
        assert len(clipped_poly.geometry) == 3
        assert "geometry" not in clipped_poly.keys()
        assert "geometry2" in clipped_poly.keys()

    def test_clip_poly_series(self, buffered_locations, mask):
        """Test clipping a polygon GDF with a generic polygon geometry."""
        clipped_poly = clip(buffered_locations.geometry, mask)
        assert len(clipped_poly) == 3
        assert all(clipped_poly.geom_type == "Polygon")

    @pytest.mark.xfail(pandas_133, reason="Regression in pandas 1.3.3 (GH #2101)")
    def test_clip_multipoly_keep_geom_type(self, multi_poly_gdf, mask):
        """Test a multi poly object where the return includes a sliver.
        Also the bounds of the object should == the bounds of the clip object
        if they fully overlap (as they do in these fixtures)."""
        clipped = clip(multi_poly_gdf, mask, keep_geom_type=True)
        expected_bounds = (
            mask if _mask_is_list_like_rectangle(mask) else mask.total_bounds
        )
        assert np.array_equal(clipped.total_bounds, expected_bounds)
        # Assert returned data is a not geometry collection
        assert (clipped.geom_type.isin(["Polygon", "MultiPolygon"])).all()

    def test_clip_multiline(self, multi_line, mask):
        """Test that clipping a multiline feature with a poly returns expected
        output."""
        clipped = clip(multi_line, mask)
        assert clipped.geom_type[0] == "MultiLineString"

    def test_clip_multipoint(self, multi_point, mask):
        """Clipping a multipoint feature with a polygon works as expected.
        should return a geodataframe with a single multi point feature"""
        clipped = clip(multi_point, mask)
        assert clipped.geom_type[0] == "MultiPoint"
        assert hasattr(clipped, "attr")
        # All points should intersect the clip geom
        assert len(clipped) == 2
        clipped_mutltipoint = MultiPoint(
            [
                Point(2, 2),
                Point(3, 4),
                Point(9, 8),
            ]
        )
        assert clipped.iloc[0].geometry.wkt == clipped_mutltipoint.wkt
        shape_for_points = (
            box(*mask) if _mask_is_list_like_rectangle(mask) else mask.unary_union
        )
        assert all(clipped.intersects(shape_for_points))

    def test_clip_lines(self, two_line_gdf, mask):
        """Test what happens when you give the clip_extent a line GDF."""
        clip_line = clip(two_line_gdf, mask)
        assert len(clip_line.geometry) == 2

    def test_mixed_geom(self, mixed_gdf, mask):
        """Test clipping a mixed GeoDataFrame"""
        clipped = clip(mixed_gdf, mask)
        assert (
            clipped.geom_type[0] == "Point"
            and clipped.geom_type[1] == "Polygon"
            and clipped.geom_type[2] == "LineString"
        )

    def test_mixed_series(self, mixed_gdf, mask):
        """Test clipping a mixed GeoSeries"""
        clipped = clip(mixed_gdf.geometry, mask)
        assert (
            clipped.geom_type[0] == "Point"
            and clipped.geom_type[1] == "Polygon"
            and clipped.geom_type[2] == "LineString"
        )

    def test_clip_warning_no_extra_geoms(self, buffered_locations, mask):
        """Test a user warning is provided if no new geometry types are found."""
        with pytest.warns(UserWarning):
            clip(buffered_locations, mask, True)
            warnings.warn(
                "keep_geom_type was called when no extra geometry types existed.",
                UserWarning,
            )

    def test_clip_with_line_extra_geom(self, sliver_line, mask):
        """When the output of a clipped line returns a geom collection,
        and keep_geom_type is True, no geometry collections should be returned."""
        clipped = clip(sliver_line, mask, keep_geom_type=True)
        assert len(clipped.geometry) == 1
        # Assert returned data is a not geometry collection
        assert not (clipped.geom_type == "GeometryCollection").any()

    def test_clip_no_box_overlap(self, pointsoutside_nooverlap_gdf, mask):
        """Test clip when intersection is empty and boxes do not overlap."""
        clipped = clip(pointsoutside_nooverlap_gdf, mask)
        assert len(clipped) == 0

    def test_clip_box_overlap(self, pointsoutside_overlap_gdf, mask):
        """Test clip when intersection is empty and boxes do overlap."""
        clipped = clip(pointsoutside_overlap_gdf, mask)
        assert len(clipped) == 0

    def test_warning_extra_geoms_mixed(self, mixed_gdf, mask):
        """Test the correct warnings are raised if keep_geom_type is
        called on a mixed GDF"""
        with pytest.warns(UserWarning):
            clip(mixed_gdf, mask, keep_geom_type=True)

    def test_warning_geomcoll(self, geomcol_gdf, mask):
        """Test the correct warnings are raised if keep_geom_type is
        called on a GDF with GeometryCollection"""
        with pytest.warns(UserWarning):
            clip(geomcol_gdf, mask, keep_geom_type=True)


def test_clip_line_keep_slivers(sliver_line, single_rectangle_gdf):
    """Test the correct output if a point is returned
    from a line only geometry type."""
    clipped = clip(sliver_line, single_rectangle_gdf)
    # Assert returned data is a geometry collection given sliver geoms
    assert "Point" == clipped.geom_type[0]
    assert "LineString" == clipped.geom_type[1]


@pytest.mark.xfail(pandas_133, reason="Regression in pandas 1.3.3 (GH #2101)")
def test_clip_multipoly_keep_slivers(multi_poly_gdf, single_rectangle_gdf):
    """Test a multi poly object where the return includes a sliver.
    Also the bounds of the object should == the bounds of the clip object
    if they fully overlap (as they do in these fixtures)."""
    clipped = clip(multi_poly_gdf, single_rectangle_gdf)
    assert np.array_equal(clipped.total_bounds, single_rectangle_gdf.total_bounds)
    # Assert returned data is a geometry collection given sliver geoms
    assert "GeometryCollection" in clipped.geom_type[0]


def test_warning_crs_mismatch(point_gdf, single_rectangle_gdf):
    with pytest.warns(UserWarning, match="CRS mismatch between the CRS"):
        clip(point_gdf, single_rectangle_gdf.to_crs(4326))


def test_clip_with_polygon(single_rectangle_gdf):
    """Test clip when using a shapely object"""
    polygon = Polygon([(0, 0), (5, 12), (10, 0), (0, 0)])
    clipped = clip(single_rectangle_gdf, polygon)
    exp_poly = polygon.intersection(
        Polygon([(0, 0), (0, 10), (10, 10), (10, 0), (0, 0)])
    )
    exp = GeoDataFrame([1], geometry=[exp_poly], crs="EPSG:3857")
    exp["attr2"] = "site-boundary"
    assert_geodataframe_equal(clipped, exp)


def test_clip_with_multipolygon(buffered_locations, single_rectangle_gdf):
    """Test clipping a polygon with a multipolygon."""
    multi = buffered_locations.dissolve(by="type").reset_index()
    clipped = clip(single_rectangle_gdf, multi)
    assert clipped.geom_type[0] == "Polygon"


@pytest.mark.parametrize(
    "mask_fixture_name",
    mask_variants_large_rectangle,
)
def test_clip_single_multipoly_no_extra_geoms(
    buffered_locations, mask_fixture_name, request
):
    """When clipping a multi-polygon feature, no additional geom types
    should be returned."""
    masks = request.getfixturevalue(mask_fixture_name)
    multi = buffered_locations.dissolve(by="type").reset_index()
    clipped = clip(multi, masks)
    assert clipped.geom_type[0] == "Polygon"


@pytest.mark.filterwarnings("ignore:All-NaN slice encountered")
@pytest.mark.parametrize(
    "mask",
    [
        Polygon(),
        (np.nan,) * 4,
        (np.nan, 0, np.nan, 1),
        GeoSeries([Polygon(), Polygon()], crs="EPSG:3857"),
        GeoSeries([Polygon(), Polygon()], crs="EPSG:3857").to_frame(),
        GeoSeries([], crs="EPSG:3857"),
        GeoSeries([], crs="EPSG:3857").to_frame(),
    ],
)
def test_clip_empty_mask(buffered_locations, mask):
    """Test that clipping with empty mask returns an empty result."""
    clipped = clip(buffered_locations, mask)
    assert_geodataframe_equal(
        clipped,
        GeoDataFrame([], columns=["geometry", "type"], crs="EPSG:3857"),
        check_index_type=False,
    )
    clipped = clip(buffered_locations.geometry, mask)
    assert_geoseries_equal(clipped, GeoSeries([], crs="EPSG:3857"))
