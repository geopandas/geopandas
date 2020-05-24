"""Tests for the clip module."""

import warnings

import numpy as np

import shapely
from shapely.geometry import Polygon, Point, LineString, LinearRing, GeometryCollection

import geopandas
from geopandas import GeoDataFrame, GeoSeries, clip

from geopandas.testing import assert_geodataframe_equal, assert_geoseries_equal
import pytest


pytestmark = pytest.mark.skipif(
    not geopandas.sindex.has_sindex(), reason="clip requires spatial index"
)


@pytest.fixture
def point_gdf():
    """Create a point GeoDataFrame."""
    pts = np.array([[2, 2], [3, 4], [9, 8], [-12, -15]])
    gdf = GeoDataFrame([Point(xy) for xy in pts], columns=["geometry"], crs="EPSG:4326")
    return gdf


@pytest.fixture
def pointsoutside_nooverlap_gdf():
    """Create a point GeoDataFrame. Its points are all outside the single
    rectangle, and its bounds are outside the single rectangle's."""
    pts = np.array([[5, 15], [15, 15], [15, 20]])
    gdf = GeoDataFrame([Point(xy) for xy in pts], columns=["geometry"], crs="EPSG:4326")
    return gdf


@pytest.fixture
def pointsoutside_overlap_gdf():
    """Create a point GeoDataFrame. Its points are all outside the single
    rectangle, and its bounds are overlapping the single rectangle's."""
    pts = np.array([[5, 15], [15, 15], [15, 5]])
    gdf = GeoDataFrame([Point(xy) for xy in pts], columns=["geometry"], crs="EPSG:4326")
    return gdf


@pytest.fixture
def single_rectangle_gdf():
    """Create a single rectangle for clipping."""
    poly_inters = Polygon([(0, 0), (0, 10), (10, 10), (10, 0), (0, 0)])
    gdf = GeoDataFrame([1], geometry=[poly_inters], crs="EPSG:4326")
    gdf["attr2"] = "site-boundary"
    return gdf


@pytest.fixture
def larger_single_rectangle_gdf():
    """Create a slightly larger rectangle for clipping.
    The smaller single rectangle is used to test the edge case where slivers
    are returned when you clip polygons. This fixture is larger which
    eliminates the slivers in the clip return.
    """
    poly_inters = Polygon([(-5, -5), (-5, 15), (15, 15), (15, -5), (-5, -5)])
    gdf = GeoDataFrame([1], geometry=[poly_inters], crs="EPSG:4326")
    gdf["attr2"] = ["study area"]
    return gdf


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
    gdf = GeoDataFrame([1, 2], geometry=[linea, lineb], crs="EPSG:4326")
    return gdf


@pytest.fixture
def multi_poly_gdf(donut_geometry):
    """Create a multi-polygon GeoDataFrame."""
    multi_poly = donut_geometry.unary_union
    out_df = GeoDataFrame(geometry=GeoSeries(multi_poly), crs="EPSG:4326")
    out_df["attr"] = ["pool"]
    return out_df


@pytest.fixture
def multi_line(two_line_gdf):
    """Create a multi-line GeoDataFrame.
    This GDF has one multiline and one regular line."""
    # Create a single and multi line object
    multiline_feat = two_line_gdf.unary_union
    linec = LineString([(2, 1), (3, 1), (4, 1), (5, 2)])
    out_df = GeoDataFrame(geometry=GeoSeries([multiline_feat, linec]), crs="EPSG:4326")
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
        crs="EPSG:4326",
    )
    out_df["attr"] = ["tree", "another tree", "shrub", "berries"]
    return out_df


@pytest.fixture
def mixed_gdf():
    """Create a Mixed Polygon and LineString For Testing"""
    point = Point([(2, 3), (11, 4), (7, 2), (8, 9), (1, 13)])
    line = LineString([(1, 1), (2, 2), (3, 2), (5, 3), (12, 1)])
    poly = Polygon([(3, 4), (5, 2), (12, 2), (10, 5), (9, 7.5)])
    ring = LinearRing([(1, 1), (2, 2), (3, 2), (5, 3), (12, 1)])
    gdf = GeoDataFrame(
        [1, 2, 3, 4], geometry=[point, poly, line, ring], crs="EPSG:4326"
    )
    return gdf


@pytest.fixture
def geomcol_gdf():
    """Create a Mixed Polygon and LineString For Testing"""
    point = Point([(2, 3), (11, 4), (7, 2), (8, 9), (1, 13)])
    poly = Polygon([(3, 4), (5, 2), (12, 2), (10, 5), (9, 7.5)])
    coll = GeometryCollection([point, poly])
    gdf = GeoDataFrame([1], geometry=[coll], crs="EPSG:4326")
    return gdf


@pytest.fixture
def sliver_line():
    """Create a line that will create a point when clipped."""
    linea = LineString([(10, 5), (13, 5), (15, 5)])
    lineb = LineString([(1, 1), (2, 2), (3, 2), (5, 3), (12, 1)])
    gdf = GeoDataFrame([1, 2], geometry=[linea, lineb], crs="EPSG:4326")
    return gdf


def test_not_gdf(single_rectangle_gdf):
    """Non-GeoDataFrame inputs raise attribute errors."""
    with pytest.raises(TypeError):
        clip((2, 3), single_rectangle_gdf)
    with pytest.raises(TypeError):
        clip(single_rectangle_gdf, (2, 3))


def test_returns_gdf(point_gdf, single_rectangle_gdf):
    """Test that function returns a GeoDataFrame (or GDF-like) object."""
    out = clip(point_gdf, single_rectangle_gdf)
    assert isinstance(out, GeoDataFrame)


def test_returns_series(point_gdf, single_rectangle_gdf):
    """Test that function returns a GeoSeries if GeoSeries is passed."""
    out = clip(point_gdf.geometry, single_rectangle_gdf)
    assert isinstance(out, GeoSeries)


def test_non_overlapping_geoms():
    """Test that a bounding box returns empty if the extents don't overlap"""
    unit_box = Polygon([(0, 0), (0, 1), (1, 1), (1, 0), (0, 0)])
    unit_gdf = GeoDataFrame([1], geometry=[unit_box], crs="EPSG:4326")
    non_overlapping_gdf = unit_gdf.copy()
    non_overlapping_gdf = non_overlapping_gdf.geometry.apply(
        lambda x: shapely.affinity.translate(x, xoff=20)
    )
    out = clip(unit_gdf, non_overlapping_gdf)
    assert_geodataframe_equal(out, unit_gdf.iloc[:0])
    out2 = clip(unit_gdf.geometry, non_overlapping_gdf)
    assert_geoseries_equal(out2, GeoSeries(crs=unit_gdf.crs))


def test_clip_points(point_gdf, single_rectangle_gdf):
    """Test clipping a points GDF with a generic polygon geometry."""
    clip_pts = clip(point_gdf, single_rectangle_gdf)
    pts = np.array([[2, 2], [3, 4], [9, 8]])
    exp = GeoDataFrame([Point(xy) for xy in pts], columns=["geometry"], crs="EPSG:4326")
    assert_geodataframe_equal(clip_pts, exp)


def test_clip_poly(buffered_locations, single_rectangle_gdf):
    """Test clipping a polygon GDF with a generic polygon geometry."""
    clipped_poly = clip(buffered_locations, single_rectangle_gdf)
    assert len(clipped_poly.geometry) == 3
    assert all(clipped_poly.geom_type == "Polygon")


def test_clip_poly_series(buffered_locations, single_rectangle_gdf):
    """Test clipping a polygon GDF with a generic polygon geometry."""
    clipped_poly = clip(buffered_locations.geometry, single_rectangle_gdf)
    assert len(clipped_poly) == 3
    assert all(clipped_poly.geom_type == "Polygon")


def test_clip_multipoly_keep_slivers(multi_poly_gdf, single_rectangle_gdf):
    """Test a multi poly object where the return includes a sliver.
    Also the bounds of the object should == the bounds of the clip object
    if they fully overlap (as they do in these fixtures)."""
    clipped = clip(multi_poly_gdf, single_rectangle_gdf)
    assert np.array_equal(clipped.total_bounds, single_rectangle_gdf.total_bounds)
    # Assert returned data is a geometry collection given sliver geoms
    assert "GeometryCollection" in clipped.geom_type[0]


def test_clip_multipoly_keep_geom_type(multi_poly_gdf, single_rectangle_gdf):
    """Test a multi poly object where the return includes a sliver.
    Also the bounds of the object should == the bounds of the clip object
    if they fully overlap (as they do in these fixtures)."""
    clipped = clip(multi_poly_gdf, single_rectangle_gdf, keep_geom_type=True)
    assert np.array_equal(clipped.total_bounds, single_rectangle_gdf.total_bounds)
    # Assert returned data is a not geometry collection
    assert (clipped.geom_type == "Polygon").any()


def test_clip_single_multipoly_no_extra_geoms(
    buffered_locations, larger_single_rectangle_gdf
):
    """When clipping a multi-polygon feature, no additional geom types
    should be returned."""
    multi = buffered_locations.dissolve(by="type").reset_index()
    clipped = clip(multi, larger_single_rectangle_gdf)
    assert clipped.geom_type[0] == "Polygon"


def test_clip_multiline(multi_line, single_rectangle_gdf):
    """Test that clipping a multiline feature with a poly returns expected output."""
    clipped = clip(multi_line, single_rectangle_gdf)
    assert clipped.geom_type[0] == "MultiLineString"


def test_clip_multipoint(single_rectangle_gdf, multi_point):
    """Clipping a multipoint feature with a polygon works as expected.
    should return a geodataframe with a single multi point feature"""
    clipped = clip(multi_point, single_rectangle_gdf)
    assert clipped.geom_type[0] == "MultiPoint"
    assert hasattr(clipped, "attr")
    # All points should intersect the clip geom
    assert all(clipped.intersects(single_rectangle_gdf.unary_union))


def test_clip_lines(two_line_gdf, single_rectangle_gdf):
    """Test what happens when you give the clip_extent a line GDF."""
    clip_line = clip(two_line_gdf, single_rectangle_gdf)
    assert len(clip_line.geometry) == 2


def test_clip_with_multipolygon(buffered_locations, single_rectangle_gdf):
    """Test clipping a polygon with a multipolygon."""
    multi = buffered_locations.dissolve(by="type").reset_index()
    clipped = clip(single_rectangle_gdf, multi)
    assert clipped.geom_type[0] == "Polygon"


def test_mixed_geom(mixed_gdf, single_rectangle_gdf):
    """Test clipping a mixed GeoDataFrame"""
    clipped = clip(mixed_gdf, single_rectangle_gdf)
    assert (
        clipped.geom_type[0] == "Point"
        and clipped.geom_type[1] == "Polygon"
        and clipped.geom_type[2] == "LineString"
    )


def test_mixed_series(mixed_gdf, single_rectangle_gdf):
    """Test clipping a mixed GeoSeries"""
    clipped = clip(mixed_gdf.geometry, single_rectangle_gdf)
    assert (
        clipped.geom_type[0] == "Point"
        and clipped.geom_type[1] == "Polygon"
        and clipped.geom_type[2] == "LineString"
    )


def test_clip_warning_no_extra_geoms(buffered_locations, single_rectangle_gdf):
    """Test a user warning is provided if no new geometry types are found."""
    with pytest.warns(UserWarning):
        clip(buffered_locations, single_rectangle_gdf, True)
        warnings.warn(
            "keep_geom_type was called when no extra geometry types existed.",
            UserWarning,
        )


def test_clip_with_polygon(single_rectangle_gdf):
    """Test clip when using a shapely object"""
    polygon = Polygon([(0, 0), (5, 12), (10, 0), (0, 0)])
    clipped = clip(single_rectangle_gdf, polygon)
    exp_poly = polygon.intersection(
        Polygon([(0, 0), (0, 10), (10, 10), (10, 0), (0, 0)])
    )
    exp = GeoDataFrame([1], geometry=[exp_poly], crs="EPSG:4326")
    exp["attr2"] = "site-boundary"
    assert_geodataframe_equal(clipped, exp)


def test_clip_with_line_extra_geom(single_rectangle_gdf, sliver_line):
    """When the output of a clipped line returns a geom collection,
    and keep_geom_type is True, no geometry collections should be returned."""
    clipped = clip(sliver_line, single_rectangle_gdf, keep_geom_type=True)
    assert len(clipped.geometry) == 1
    # Assert returned data is a not geometry collection
    assert not (clipped.geom_type == "GeometryCollection").any()


def test_clip_line_keep_slivers(single_rectangle_gdf, sliver_line):
    """Test the correct output if a point is returned
    from a line only geometry type."""
    clipped = clip(sliver_line, single_rectangle_gdf)
    # Assert returned data is a geometry collection given sliver geoms
    assert "Point" == clipped.geom_type[0]
    assert "LineString" == clipped.geom_type[1]


def test_clip_no_box_overlap(pointsoutside_nooverlap_gdf, single_rectangle_gdf):
    """Test clip when intersection is empty and boxes do not overlap."""
    clipped = clip(pointsoutside_nooverlap_gdf, single_rectangle_gdf)
    assert len(clipped) == 0


def test_clip_box_overlap(pointsoutside_overlap_gdf, single_rectangle_gdf):
    """Test clip when intersection is emtpy and boxes do overlap."""
    clipped = clip(pointsoutside_overlap_gdf, single_rectangle_gdf)
    assert len(clipped) == 0


def test_warning_extra_geoms_mixed(single_rectangle_gdf, mixed_gdf):
    """Test the correct warnings are raised if keep_geom_type is
    called on a mixed GDF"""
    with pytest.warns(UserWarning):
        clip(mixed_gdf, single_rectangle_gdf, keep_geom_type=True)


def test_warning_geomcoll(single_rectangle_gdf, geomcol_gdf):
    """Test the correct warnings are raised if keep_geom_type is
    called on a GDF with GeometryCollection"""
    with pytest.warns(UserWarning):
        clip(geomcol_gdf, single_rectangle_gdf, keep_geom_type=True)


def test_warning_crs_mismatch(point_gdf, single_rectangle_gdf):
    with pytest.warns(UserWarning, match="CRS mismatch between the CRS"):
        clip(point_gdf, single_rectangle_gdf.to_crs(3857))
