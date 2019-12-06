"""Tests for the clip module."""


import pytest
import numpy as np
from shapely.geometry import Polygon, Point, LineString
import shapely
import geopandas as gpd
from geopandas import GeoDataFrame
import warnings


@pytest.fixture
def point_gdf():
    """Create a point GeoDataFrame."""
    pts = np.array([[2, 2], [3, 4], [9, 8], [-12, -15]])
    gdf = GeoDataFrame(
        [Point(xy) for xy in pts], columns=["geometry"], crs={"init": "epsg:4326"},
    )
    return gdf


@pytest.fixture
def single_rectangle_gdf():
    """Create a single rectangle for clipping."""
    poly_inters = Polygon([(0, 0), (0, 10), (10, 10), (10, 0), (0, 0)])
    gdf = GeoDataFrame([1], geometry=[poly_inters], crs={"init": "epsg:4326"})
    gdf["attr2"] = "site-boundary"
    return gdf


@pytest.fixture
def larger_single_rectangle_gdf():
    """Create a slightly larger rectangle for clipping.
     The smaller single rectangle is used to test the edge case where slivers
     are returned when you clip polygons. This fixture is larger which
     eliminates the slivers in the clip return."""
    poly_inters = Polygon([(-5, -5), (-5, 15), (15, 15), (15, -5), (-5, -5)])
    gdf = GeoDataFrame([1], geometry=[poly_inters], crs={"init": "epsg:4326"})
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
    donut = gpd.overlay(
        buffered_locations, single_rectangle_gdf, how="symmetric_difference"
    )
    return donut


@pytest.fixture
def two_line_gdf():
    """Create Line Objects For Testing"""
    linea = LineString([(1, 1), (2, 2), (3, 2), (5, 3)])
    lineb = LineString([(3, 4), (5, 7), (12, 2), (10, 5), (9, 7.5)])
    gdf = GeoDataFrame([1, 2], geometry=[linea, lineb], crs={"init": "epsg:4326"})
    return gdf


@pytest.fixture
def multi_poly_gdf(donut_geometry):
    """Create a multi-polygon GeoDataFrame."""
    multi_poly = donut_geometry.unary_union
    out_df = GeoDataFrame(geometry=gpd.GeoSeries(multi_poly), crs={"init": "epsg:4326"})
    out_df["attr"] = ["pool"]
    return out_df


@pytest.fixture
def multi_line(two_line_gdf):
    """Create a multi-line GeoDataFrame.
    This GDF has one multiline and one regular line."""
    # Create a single and multi line object
    multiline_feat = two_line_gdf.unary_union
    linec = LineString([(2, 1), (3, 1), (4, 1), (5, 2)])
    out_df = GeoDataFrame(
        geometry=gpd.GeoSeries([multiline_feat, linec]), crs={"init": "epsg:4326"},
    )
    out_df["attr"] = ["road", "stream"]
    return out_df


@pytest.fixture
def multi_point(point_gdf):
    """Create a multi-point GeoDataFrame."""
    multi_point = point_gdf.unary_union
    out_df = GeoDataFrame(
        geometry=gpd.GeoSeries(
            [multi_point, Point(2, 5), Point(-11, -14), Point(-10, -12)]
        ),
        crs={"init": "epsg:4326"},
    )
    out_df["attr"] = ["tree", "another tree", "shrub", "berries"]
    return out_df


@pytest.fixture
def mixed_gdf():
    """Create a Mixed Polygon and LineString For Testing"""
    point = Point([(2, 3), (11, 4), (7, 2), (8, 9), (1, 13)])
    line = LineString([(1, 1), (2, 2), (3, 2), (5, 3), (12, 1)])
    poly = Polygon([(3, 4), (5, 2), (12, 2), (10, 5), (9, 7.5)])
    gdf = GeoDataFrame(
        [1, 2, 3], geometry=[point, line, poly], crs={"init": "epsg:4326"}
    )
    return gdf


@pytest.fixture
def sliver_line():
    """Create a line that will create a point when clipped."""
    linea = LineString([(10, 5), (13, 5), (15, 5)])
    lineb = LineString([(1, 1), (2, 2), (3, 2), (5, 3), (12, 1)])
    gdf = GeoDataFrame([1, 2], geometry=[linea, lineb], crs={"init": "epsg:4326"})
    return gdf


def test_not_gdf(single_rectangle_gdf):
    """Non-GeoDataFrame inputs raise attribute errors."""
    with pytest.raises(TypeError):
        gpd.clip((2, 3), single_rectangle_gdf)
    with pytest.raises(TypeError):
        gpd.clip(single_rectangle_gdf, (2, 3))


def test_returns_gdf(point_gdf, single_rectangle_gdf):
    """Test that function returns a GeoDataFrame (or GDF-like) object."""
    out = gpd.clip(point_gdf, single_rectangle_gdf)
    assert isinstance(out, GeoDataFrame)


def test_non_overlapping_geoms():
    """Test that a bounding box returns error if the extents don't overlap"""
    unit_box = Polygon([(0, 0), (0, 1), (1, 1), (1, 0), (0, 0)])
    unit_gdf = GeoDataFrame([1], geometry=[unit_box], crs={"init": "epsg:4326"})
    non_overlapping_gdf = unit_gdf.copy()
    non_overlapping_gdf = non_overlapping_gdf.geometry.apply(
        lambda x: shapely.affinity.translate(x, xoff=20)
    )
    with pytest.raises(ValueError):
        gpd.clip(unit_gdf, non_overlapping_gdf)


def test_input_gdfs(single_rectangle_gdf):
    """Test that function fails if not provided with 2 GDFs."""
    with pytest.raises(TypeError):
        gpd.clip(list(), single_rectangle_gdf)
    with pytest.raises(TypeError):
        gpd.clip(single_rectangle_gdf, list())


def test_clip_points(point_gdf, single_rectangle_gdf):
    """Test clipping a points GDF with a generic polygon geometry."""
    clip_pts = gpd.clip(point_gdf, single_rectangle_gdf)
    assert len(clip_pts.geometry) == 3 and clip_pts.geom_type[1] == "Point"


def test_clip_poly(buffered_locations, single_rectangle_gdf):
    """Test clipping a polygon GDF with a generic polygon geometry."""
    clipped_poly = gpd.clip(buffered_locations, single_rectangle_gdf)
    assert len(clipped_poly.geometry) == 3
    assert all(clipped_poly.geom_type == "Polygon")


def test_clip_multipoly_keep_slivers(multi_poly_gdf, single_rectangle_gdf):
    """Test a multi poly object where the return includes a sliver.
    Also the bounds of the object should == the bounds of the clip object
    if they fully overlap (as they do in these fixtures)."""
    with pytest.warns(UserWarning):
        clip = gpd.clip(multi_poly_gdf, single_rectangle_gdf)
        warnings.warn("A geometry collection has been returned. ", UserWarning)
        assert hasattr(clip, "geometry")
        assert np.array_equal(clip.total_bounds, single_rectangle_gdf.total_bounds)
        # Assert returned data is a geometry collection given sliver geoms
        assert "GeometryCollection" in clip.geom_type[0]


def test_clip_multipoly_drop_slivers(multi_poly_gdf, single_rectangle_gdf):
    """Test a multi poly object where the return includes a sliver.
    Also the bounds of the object should == the bounds of the clip object
    if they fully overlap (as they do in these fixtures)."""
    clip = gpd.clip(multi_poly_gdf, single_rectangle_gdf, drop_slivers=True)
    assert hasattr(clip, "geometry")
    assert np.array_equal(clip.total_bounds, single_rectangle_gdf.total_bounds)
    # Assert returned data is a not geometry collection
    assert (clip.geom_type == "Polygon").any()


def test_clip_multipoly_warning(multi_poly_gdf, single_rectangle_gdf):
    """Test that a user warning is provided when clip outputs a sliver."""
    with pytest.warns(UserWarning):
        gpd.clip(multi_poly_gdf, single_rectangle_gdf)
        warnings.warn("A geometry collection has been returned. ", UserWarning)


def test_clip_single_multipoly_no_slivers(
    buffered_locations, larger_single_rectangle_gdf
):
    """Test clipping a multi poly with another poly no sliver
    shapes are returned in this clip operation."""
    multi = buffered_locations.dissolve(by="type").reset_index()
    clip = gpd.clip(multi, larger_single_rectangle_gdf)

    assert hasattr(clip, "geometry") and clip.geom_type[0] == "Polygon"


def test_clip_multiline(multi_line, single_rectangle_gdf):
    """Test that clipping a multiline feature with a poly returns expected output."""
    clip = gpd.clip(multi_line, single_rectangle_gdf)
    assert hasattr(clip, "geometry") and clip.geom_type[0] == "MultiLineString"


def test_clip_multipoint(single_rectangle_gdf, multi_point):
    """Clipping a multipoint feature with a polygon works as expected.
    should return a geodataframe with a single multi point feature"""
    clip = gpd.clip(multi_point, single_rectangle_gdf)
    assert hasattr(clip, "geometry") and clip.geom_type[0] == "MultiPoint"
    assert hasattr(clip, "attr")
    # All points should intersect the clip geom
    assert all(clip.intersects(single_rectangle_gdf.unary_union))


def test_clip_lines(two_line_gdf, single_rectangle_gdf):
    """Test what happens when you give the clip_extent a line GDF."""
    clip_line = gpd.clip(two_line_gdf, single_rectangle_gdf)
    assert len(clip_line.geometry) == 2


def test_clip_with_multipolygon(buffered_locations, single_rectangle_gdf):
    """Test clipping a polygon with a multipolygon."""
    multi = buffered_locations.dissolve(by="type").reset_index()
    clip = gpd.clip(single_rectangle_gdf, multi)
    assert hasattr(clip, "geometry") and clip.geom_type[0] == "Polygon"


def test_mixed_geom(mixed_gdf, single_rectangle_gdf):
    """Test clipping a mixed GeoDataFrame"""
    clip = gpd.clip(mixed_gdf, single_rectangle_gdf)
    assert (
        hasattr(clip, "geometry")
        and clip.geom_type[0] == "Point"
        and clip.geom_type[1] == "LineString"
        and clip.geom_type[2] == "Polygon"
    )


def test_clip_warning_no_slivers(buffered_locations, single_rectangle_gdf):
    """Test a warning is provided to the user if no slivers are found."""
    with pytest.warns(UserWarning):
        gpd.clip(buffered_locations, single_rectangle_gdf, True)
        warnings.warn("Drop slivers was called when no slivers existed.", UserWarning)


def test_clip_with_polygon(single_rectangle_gdf):
    """Test clip when using a shapely object"""
    polygon = Polygon([(0, 0), (5, 12), (10, 0), (0, 0)])
    clip = gpd.clip(single_rectangle_gdf, polygon)
    assert hasattr(clip, "geometry") and clip.geom_type[0] == "Polygon"


def test_clip_with_line_sliver(single_rectangle_gdf, sliver_line):
    """Test clipping a line so that there is a sliver drops the sliver."""
    clip = gpd.clip(sliver_line, single_rectangle_gdf, drop_slivers=True)
    assert hasattr(clip, "geometry")
    assert len(clip.geometry) == 1
    # Assert returned data is a not geometry collection
    assert (clip.geom_type == "LineString").any()


def test_clip_line_keep_slivers(single_rectangle_gdf, sliver_line):
    """Test the correct warnings are raised if a point sliver is returned"""
    with pytest.warns(UserWarning):
        clip = gpd.clip(sliver_line, single_rectangle_gdf)
        warnings.warn(
            "More geometry types were returned than were in the original ", UserWarning,
        )
        assert hasattr(clip, "geometry")
        # Assert returned data is a geometry collection given sliver geoms
        assert "Point" == clip.geom_type[0]
        assert "LineString" == clip.geom_type[1]


def test_warning_slivers_mixed(single_rectangle_gdf, mixed_gdf):
    """Test the correct warnings are raised if drop_slivers is called on a mixed GDF"""
    with pytest.warns(UserWarning):
        gpd.clip(mixed_gdf, single_rectangle_gdf)
        warnings.warn("Drop slivers was called on a mixed type GeoDataFrame.")
