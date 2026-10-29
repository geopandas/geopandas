import numpy as np

import shapely
from shapely import LineString, MultiLineString, MultiPoint, Point, Polygon

from geopandas._geodesic import (
    # _lonlat_to_vector,
    geodetic_distance,
    nearest_point_on_segment,
)

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


def test_point_linestring(crs):
    line = LineString([(0, 0), (0, 1), (1, 1), (1, 2)])
    result = geodetic_distance(Point(0, 0), line, crs)
    assert_allclose(result, 0.0)
    result = geodetic_distance(line, Point(0, 0), crs)
    assert_allclose(result, 0.0)


def test_point_multilinestring(crs):
    line = MultiLineString([[(0, 0), (0, 1)], [(1, 1), (1, 2), (2, 2)]])
    result = geodetic_distance(Point(2, 0), line, crs)
    expected = min(
        geodetic_distance(Point(2, 0), LineString([(0, 0), (0, 1)]), crs),
        geodetic_distance(Point(2, 0), LineString([(1, 1), (1, 2), (2, 2)]), crs),
    )
    assert_allclose(result, expected)

    result = geodetic_distance(line, Point(2, 0), crs)
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


def _nearest_point_on_segment_scalar(point, segment):
    point_lon, point_lat = point
    (lon1, lat1), (lon2, lat2) = segment
    lon, lat = nearest_point_on_segment(
        np.array([point_lon]),
        np.array([point_lat]),
        np.array([lon1]),
        np.array([lat1]),
        np.array([lon2]),
        np.array([lat2]),
    )
    return lon[0], lat[0]


@pytest.mark.parametrize(
    "point, segment, expected",
    [
        # projection lands inside the segment: straight down the meridian
        ((45, 30), ((0, 0), (90, 0)), (45, 0)),
        # projection beyond p2 -> p2
        ((100, 10), ((0, 0), (90, 0)), (90, 0)),
        # projection beyond p1 -> p1
        ((-10, 10), ((0, 0), (90, 0)), (0, 0)),
        # point equal to an endpoint
        ((10, 10), ((10, 10), (20, 20)), (10, 10)),
        # point on the segment itself
        ((30, 0), ((0, 0), (90, 0)), (30, 0)),
    ],
)
def test_nearest_point_on_segment__basic(point, segment, expected):
    lon, lat = _nearest_point_on_segment_scalar(point, segment)
    assert_allclose(lon, expected[0])
    assert_allclose(lat, expected[1])


# testing some specific corner cases for this part of the logic


def test_nearest_point_on_segment__zero_length_segment():
    # case: zero-length segments
    lon, lat = _nearest_point_on_segment_scalar((5.0, 5.0), ((3.0, 3.0), (3.0, 3.0)))
    assert_allclose(lon, 3.0)
    assert_allclose(lat, 3.0)


def test_nearest_point_on_segment__nan_input():
    # case: invalid input with NaN should not return valid-looking coordinates
    lon, lat = _nearest_point_on_segment_scalar((np.nan, 5.0), ((0.0, 0.0), (3.0, 3.0)))
    assert np.isnan(lon) and np.isnan(lat)
    lon, lat = _nearest_point_on_segment_scalar((5.0, 5.0), ((0.0, np.nan), (3.0, 3.0)))
    assert np.isnan(lon) and np.isnan(lat)
    lon, lat = _nearest_point_on_segment_scalar((5.0, 5.0), ((0.0, 0.0), (np.nan, 3.0)))
    assert np.isnan(lon) and np.isnan(lat)

    # # Example test case: point near the equator and a segment along the equator
    # point_lon = np.array([1.0])
    # point_lat = np.array([0.5])
    # lon1 = np.array([0.0])
    # lat1 = np.array([0.0])
    # lon2 = np.array([2.0])
    # lat2 = np.array([0.0])

    # nearest_lon, nearest_lat = nearest_point_on_segment(
    #     point_lon, point_lat, lon1, lat1, lon2, lat2)
    # assert_allclose(nearest_lon, np.array([1.0]))
    # assert_allclose(nearest_lat, np.array([0.0]))


def test_nearest_point_on_segment__endpoints_returned_unchanged():
    # endpoint passed through exactly as given, including non-normalised longitude
    lon, lat = _nearest_point_on_segment_scalar((250, 1), ((190, 0), (210, 0)))
    assert (lon, lat) == (210, 0)


def test_nearest_point_on_segment__projected_longitude_in_minus180_180():
    lon, lat = _nearest_point_on_segment_scalar((200, 1), ((190, 0), (210, 0)))
    assert_allclose(lon, -160)
    assert_allclose(lat, 0)


@pytest.mark.parametrize(
    "point, expected",
    [
        # exactly on the great circle, beyond p1
        ((-30, 0), (0, 0)),
        # exactly on the great circle, beyond p2
        ((120, 0), (90, 0)),
        # exactly on the perpendicular through p1
        ((0, -30), (0, 0)),
        # exactly on the perpendicular through p2
        ((90, 30), (90, 0)),
        # antipode of p1 -> p2 is closer
        ((180, 0), (90, 0)),
        # antipode of p2 -> p1 is closer
        ((-90, 0), (0, 0)),
    ],
)
def test_nearest_point_on_segment__boundaries(point, expected):
    lon, lat = _nearest_point_on_segment_scalar(point, ((0, 0), (90, 0)))
    assert_allclose(lon, expected[0])
    assert_allclose(lat, expected[1])


@pytest.mark.parametrize("point", [(-135, 0), (-135, -20), (-135, 40)])
def test_nearest_point_on_segment__antipode_of_midpoint_gives_an_endpoint(point):
    # both endpoints are equally far: either is correct
    lon, lat = _nearest_point_on_segment_scalar(point, ((0, 0), (90, 0)))
    if lon < 45:
        assert_allclose(lon, 0)
    else:
        assert_allclose(lon, 90)
    assert_allclose(lat, 0)


@pytest.mark.parametrize("point", [(0, 90), (45, 90), (170, 90), (0, -90), (180, -90)])
def test_nearest_point_on_segment__point_at_pole_of_segment_circle(point):
    # every point of the segment is 90 degrees away: any point on it is correct,
    # but the result must be a real point on the segment (not NaN, not (0, 0) garbage)
    lon, lat = _nearest_point_on_segment_scalar(point, ((10, 0), (80, 0)))
    assert_allclose(lat, 0)
    assert lon >= 10 and lon <= 80


# @pytest.mark.parametrize(
#     "point, segment",
#     [
#         ((90, 10), ((0, 0), (180, 0))),
#         ((-90, 10), ((0, 0), (180, 0))),
#         ((0, 80), ((0, 0), (180, 0))),
#         ((45, 0), ((0, 90), (0, -90))),      # pole to pole
#         ((10, 10), ((-30, 20), (150, -20))),
#     ],
# )
# def test_antipodal_endpoints_are_handled(point, segment):
#     # ill-posed: infinitely many great circles connect antipodal points, so no single
#     # answer is correct. Require a finite result that is no worse than an endpoint.
#     lon, lat = _nearest_point_on_segment_scalar(point, segment)
#     assert_no_worse_than_endpoints(point, segment, lon, lat)


# def assert_no_worse_than_endpoints(p, seg, lon, lat, tol_deg=1e-9):
#     # the answer can never be farther from p than the closer endpoint
#     assert np.isfinite(lon) and np.isfinite(lat)
#     d = _dist_deg(*p, lon, lat)
#     d_end = min(_dist_deg(*p, *seg[0]), _dist_deg(*p, *seg[1]))
#     assert d <= d_end + tol_deg


def test_nearest_point_on_segment__near_antipodal_segment():
    # 179.999 degrees longitude: unique great circle, must give the exact answer
    lon, lat = _nearest_point_on_segment_scalar((90, 10), ((0, 0), (179.999, 0)))
    assert_allclose(lon, 90)
    assert_allclose(lat, 0)


def test_nearest_point_on_segment__point_at_earth_pole_segment_on_meridian():
    lon, lat = _nearest_point_on_segment_scalar((0, 90), ((30, 10), (30, 50)))
    assert_allclose(lon, 30)
    assert_allclose(lat, 50)


@pytest.mark.parametrize("point_lon", [180, -180])
def test_nearest_point_on_segment__dateline_crossing(point_lon):
    # (179, 0) -> (-179, 0) is a 2 degree short segment across the dateline,
    # not a long 358 degrees line
    lon, lat = _nearest_point_on_segment_scalar((point_lon, 1), ((179, 0), (-179, 0)))
    assert_allclose(lon, point_lon)
    assert_allclose(lat, 0)


# def _ang(a, b):
#     return np.arctan2(np.linalg.norm(np.cross(a, b), axis=-1), (a * b).sum(-1))


# def test_matches_brute_force_on_realistic_segments():
#     # Note: this test is AI-generated
#     rng = np.random.default_rng(0)
#     n = 5000
#     lon1 = rng.uniform(-180, 180, n)
#     lat1 = rng.uniform(-80, 80, n)
#     lon2 = lon1 + rng.uniform(-2, 2, n)
#     lat2 = lat1 + rng.uniform(-2, 2, n)
#     plon = lon1 + rng.uniform(-5, 5, n)
#     plat = lat1 + rng.uniform(-5, 5, n)

#     rlon, rlat = nearest_point_on_segment(plon, plat, lon1, lat1, lon2, lat2)
#     v, r = _lonlat_to_vector(plon, plat), _lonlat_to_vector(rlon, rlat)
#     v1, v2 = _lonlat_to_vector(lon1, lat1), _lonlat_to_vector(lon2, lat2)

#     # result must lie on the segment's great circle and between the endpoints
#     c1 = np.cross(v1, v2)
#     c1 /= np.linalg.norm(c1, axis=-1, keepdims=True)
#     assert np.abs((r * c1).sum(-1)).max() < 1e-12
#     seg_len = _ang(v1, v2)
#     assert np.all(_ang(v1, r) + _ang(r, v2) - seg_len < 1e-12)

#     # and must be at least as close as any sampled point along the arc
#     s = np.linspace(0, 1, 2001)[None, :, None]
#     pts = v1[:, None, :] * (1 - s) + v2[:, None, :] * s
#     pts /= np.linalg.norm(pts, axis=-1, keepdims=True)
#     best = _ang(v[:, None, :], pts).min(1)
#     excess_m = (_ang(v, r) - best) * 6371e3
#     assert excess_m.max() < 1e-6
