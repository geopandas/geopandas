import shapely
from shapely import LineString, MultiPoint, Point, Polygon

from geopandas._geodesic import geodetic_distance

import pytest
from numpy.testing import assert_allclose

pyproj = pytest.importorskip("pyproj")


@pytest.fixture
def crs():
    return pyproj.CRS("EPSG:4326")


def test_point_point(crs):
    # ellipsoidal distance of 1 degree of latitude at the equator (WGS84)
    result = geodetic_distance(Point(5, 1), Point(5, 0), crs)
    assert_allclose(result, 110574.389, atol=1e-3)


@pytest.mark.parametrize(
    "geom1, geom2",
    [
        (MultiPoint([(0, 0), (5, 1)]), MultiPoint([(10, 10), (5, 0)])),
        (MultiPoint([(0, 0), (5, 1)]), Point(5, 0)),
        (Point(5, 0), MultiPoint([(0, 0), (5, 1)])),
    ],
)
def test_multipoint(crs, geom1, geom2):
    result = geodetic_distance(geom1, geom2, crs)
    expected = geodetic_distance(Point(5, 1), Point(5, 0), crs)
    assert_allclose(result, expected)


# cases taken from s2geography
cases = [
    ("point_distance_same_point", "POINT (0 0)", "POINT (0 0)", 0.0),
    ("point_distance_point", "POINT (0 0)", "POINT (0 1)", 111195.10117748393),
    (
        "point_distance_wraparound_lng",
        "POINT (0 0)",
        "POINT (360 361)",
        111195.10117748113,
    ),
    # Point Z x point Z
    (
        "point_distance_point_z",
        "POINT Z (0 0 1)",
        "POINT Z (0 1 2)",
        111195.10117748393,
    ),
    # Point x linestring (point on linestring) ----------
    ("point_distance_linestring_on", "POINT (0 0)", "LINESTRING (0 0, 0 1)", 0.0),
    # Point x linestring (point off linestring)
    (
        "point_distance_linestring_off",
        "POINT (1 0)",
        "LINESTRING (0 0, 0 1)",
        111195.10117748393,
    ),
    # Point x linestring (point on linestring) ----------
    ("linestring_distance_point_on", "LINESTRING (0 0, 0 1)", "POINT (0 0)", 0.0),
    # Point x linestring (point off linestring) ----------
    (
        "linestring_distance_point_off",
        "LINESTRING (0 0, 0 1)",
        "POINT (1 0)",
        111195.10117748393,
    ),
    # Point x polygon (point inside)
    (
        "point_distance_polygon_inside",
        "POINT (0.25 0.25)",
        "POLYGON ((0 0, 2 0, 0 2, 0 0))",
        0.0,
    ),
    # Point x polygon (point on boundary)
    (
        "point_distance_polygon_boundary",
        "POINT (0 0)",
        "POLYGON ((0 0, 2 0, 0 2, 0 0))",
        0.0,
    ),
    # Point x polygon (point outside)
    (
        "point_distance_polygon_outside",
        "POINT (-1 0)",
        "POLYGON ((0 0, 2 0, 0 2, 0 0))",
        111195.10117748393,
    ),
    # Z Point x polygon (point inside)
    (
        "point_z_distance_polygon_inside",
        "POINT Z (0.25 0.25 10)",
        "POLYGON Z ((0 0 12, 2 0 12, 0 2 12, 0 0 12))",
        0.0,
    ),
    # Z Point x polygon (point on boundary)
    (
        "point_z_distance_polygon_boundary",
        "POINT Z (0 0 10)",
        "POLYGON Z ((0 0 12, 2 0 12, 0 2 12, 0 0 12))",
        0.0,
    ),
    # Z Point x polygon (point outside)
    (
        "point_z_distance_polygon_outside",
        "POINT Z (-1 0 10)",
        "POLYGON Z ((0 0 12, 2 0 12, 0 2 12, 0 0 12))",
        111195.10117748393,
    ),
    # Linestring x polygon (linestring fully inside)
    (
        "linestring_distance_polygon_inside",
        "LINESTRING (0.25 0.25, 0.5 0.5)",
        "POLYGON ((0 0, 2 0, 0 2, 0 0))",
        0.0,
    ),
    # Polygon x linestring (linestring fully inside)
    (
        "polygon_distance_linestring_inside",
        "POLYGON ((0 0, 2 0, 0 2, 0 0))",
        "LINESTRING (0.25 0.25, 0.5 0.5)",
        0.0,
    ),
    # Linestring x polygon (linestring partially crosses boundary)
    (
        "linestring_distance_polygon_crossing",
        "LINESTRING (0.25 0.25, 3 3)",
        "POLYGON ((0 0, 2 0, 0 2, 0 0))",
        0.0,
    ),
    # Polygon x linestring (linestring partially crosses boundary)
    (
        "polygon_distance_linestring_crossing",
        "POLYGON ((0 0, 2 0, 0 2, 0 0))",
        "LINESTRING (0.25 0.25, 3 3)",
        0.0,
    ),
    # Linestring x polygon (linestring crosses through, neither vertex inside)
    (
        "linestring_distance_polygon_through",
        "LINESTRING (-1 0.5, 3 0.5)",
        "POLYGON ((0 0, 2 0, 0 2, 0 0))",
        0.0,
    ),
    # Polygon x linestring (linestring crosses through, neither vertex inside)
    (
        "polygon_distance_linestring_through",
        "POLYGON ((0 0, 2 0, 0 2, 0 0))",
        "LINESTRING (-1 0.5, 3 0.5)",
        0.0,
    ),
    # Linestring x polygon (linestring fully outside)
    (
        "linestring_distance_polygon_outside",
        "LINESTRING (3 3, 4 4)",
        "POLYGON ((0 0, 2 0, 0 2, 0 0))",
        314367.35908786188,
    ),
    # Polygon x linestring (linestring fully outside)
    (
        "polygon_distance_linestring_outside",
        "POLYGON ((0 0, 2 0, 0 2, 0 0))",
        "LINESTRING (3 3, 4 4)",
        314367.35908786188,
    ),
    # Polygon x polygon (one fully inside the other)
    (
        "polygon_distance_polygon_inside",
        "POLYGON ((0 0, 2 0, 0 2, 0 0))",
        "POLYGON ((0.1 0.1, 0.5 0.1, 0.1 0.5, 0.1 0.1))",
        0.0,
    ),
    # Polygon x polygon (one fully inside, reversed)
    (
        "polygon_distance_polygon_inside_rev",
        "POLYGON ((0.1 0.1, 0.5 0.1, 0.1 0.5, 0.1 0.1))",
        "POLYGON ((0 0, 2 0, 0 2, 0 0))",
        0.0,
    ),
    # Polygon x polygon (partially overlapping) ----------
    (
        "polygon_distance_polygon_crossing",
        "POLYGON ((0 0, 2 0, 0 2, 0 0))",
        "POLYGON ((1 0, 3 0, 1 2, 1 0))",
        0.0,
    ),
    # Polygon x polygon (partially overlapping, reversed)
    (
        "polygon_distance_polygon_crossing_rev",
        "POLYGON ((1 0, 3 0, 1 2, 1 0))",
        "POLYGON ((0 0, 2 0, 0 2, 0 0))",
        0.0,
    ),
    # Polygon x polygon (fully outside) ----------
    (
        "polygon_distance_polygon_outside",
        "POLYGON ((0 0, 1 0, 0 1, 0 0))",
        "POLYGON ((30 30, 31 30, 30 31, 30 30))",
        4520972.0955287321,
    ),
    # Polygon x polygon (fully outside, reversed) ----------
    (
        "polygon_distance_polygon_outside_rev",
        "POLYGON ((30 30, 31 30, 30 31, 30 30))",
        "POLYGON ((0 0, 1 0, 0 1, 0 0))",
        4520972.0955287321,
    ),
    # Polygon x polygon (north pole vs south pole) ----------
    (
        "polygon_distance_polygon_poles",
        "POLYGON ((-120 80, 0 80, 120 80, -120 80))",
        "POLYGON ((-120 -80, 0 -80, 120 -80, -120 -80))",
        17791216.188397426,
    ),
    # Polygon x polygon (north pole vs south pole, reversed) ----------
    (
        "polygon_distance_polygon_poles_rev",
        "POLYGON ((-120 -80, 0 -80, 120 -80, -120 -80))",
        "POLYGON ((-120 80, 0 80, 120 80, -120 80))",
        17791216.188397426,
    ),
    # Linestring x linestring (antipodal crossing) ----------
    (
        "linestring_distance_linestring_poles",
        "LINESTRING (-90 -80, 90 -80)",
        "LINESTRING (0 80, 180 80)",
        18446595.193179362,
    ),
    # Linestring x polygon (antipodal crossing) ----------
    (
        "linestring_distance_polygon_poles",
        "LINESTRING (-90 -80, 90 -80)",
        "POLYGON ((-120 90, 0 90, 120 90, -120 90))",
        18903167.200172286,
    ),
    # Polygon x linestring (antipodal crossing) ----------
    (
        "polygon_distance_linestring_poles",
        "POLYGON ((-120 90, 0 90, 120 90, -120 90))",
        "LINESTRING (-90 -80, 90 -80)",
        18903167.200172286,
    ),
    # Point x point (antipodal crossing) ----------
    ("point_distance_point_poles", "POINT (0 -90)", "POINT (0 90)", 20015118.21194711),
    # GEOMETRYCOLLECTION tests
    # TODO add
]


@pytest.mark.parametrize(
    "name, geom1, geom2, expected", cases, ids=[c[0] for c in cases]
)
def test_cases(request, name, geom1, geom2, expected, crs):
    if name == "point_distance_wraparound_lng":
        # TODO
        request.node.add_marker(
            pytest.mark.xfail(reason="wraparound longitude not implemented")
        )

    geom1 = shapely.from_wkt(geom1)
    geom2 = shapely.from_wkt(geom2)

    if isinstance(geom1, Polygon) or isinstance(geom2, Polygon):
        with pytest.raises(NotImplementedError):
            geodetic_distance(geom1, geom2, crs)
        return

    if isinstance(geom1, LineString) and isinstance(geom2, LineString):
        with pytest.raises(NotImplementedError):
            geodetic_distance(geom1, geom2, crs)
        return

    result = geodetic_distance(geom1, geom2, crs)
    assert_allclose(result, expected, rtol=0.006)
