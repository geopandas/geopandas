import random
import warnings

import numpy as np
import pandas as pd

import shapely
import shapely.affinity
import shapely.geometry
import shapely.wkb
import shapely.wkt
from shapely import geos_version
from shapely.geometry.base import CAP_STYLE, JOIN_STYLE

import geopandas
from geopandas._compat import HAS_PYPROJ
from geopandas.array import (
    GeometryArray,
    _check_crs,
    _crs_mismatch_warn,
    from_shapely,
    from_wkb,
    from_wkt,
    points_from_xy,
    to_wkb,
    to_wkt,
)

import pytest

triangle_no_missing = [
    shapely.geometry.Polygon([(random.random(), random.random()) for i in range(3)])
    for _ in range(10)
]
triangles = triangle_no_missing + [shapely.wkt.loads("POLYGON EMPTY"), None]
T = from_shapely(triangles)

points_no_missing = [
    shapely.geometry.Point(random.random(), random.random()) for _ in range(20)
]
points = points_no_missing + [None]
P = from_shapely(points)


def equal_geometries(result, expected):
    for r, e in zip(result, expected):
        if r is None or e is None:
            if not (r is None and e is None):
                return False
        elif not r.equals(e):
            return False
    return True


def test_points():
    x = np.arange(10).astype(np.float64)
    y = np.arange(10).astype(np.float64) ** 2

    points = points_from_xy(x, y)
    assert isinstance(points, GeometryArray)

    for i in range(10):
        assert isinstance(points[i], shapely.geometry.Point)
        assert points[i].x == x[i]
        assert points[i].y == y[i]


def test_points_from_xy():
    # testing the top-level interface

    # using DataFrame column
    df = pd.DataFrame([{"x": x, "y": x, "z": x} for x in range(10)])
    gs = [shapely.geometry.Point(x, x) for x in range(10)]
    gsz = [shapely.geometry.Point(x, x, x) for x in range(10)]
    geometry1 = geopandas.points_from_xy(df["x"], df["y"])
    geometry2 = geopandas.points_from_xy(df["x"], df["y"], df["z"])
    assert isinstance(geometry1, GeometryArray)
    assert isinstance(geometry2, GeometryArray)
    assert list(geometry1) == gs
    assert list(geometry2) == gsz

    # using Series or numpy arrays or lists
    for s in [pd.Series(range(10)), np.arange(10), list(range(10))]:
        geometry1 = geopandas.points_from_xy(s, s)
        geometry2 = geopandas.points_from_xy(s, s, s)
        assert isinstance(geometry1, GeometryArray)
        assert isinstance(geometry2, GeometryArray)
        assert list(geometry1) == gs
        assert list(geometry2) == gsz

    # using different lengths should throw error
    arr_10 = np.arange(10)
    arr_20 = np.arange(20)
    with pytest.raises(ValueError):
        geopandas.points_from_xy(x=arr_10, y=arr_20)
        geopandas.points_from_xy(x=arr_10, y=arr_10, z=arr_20)

    # Using incomplete arguments should throw error
    with pytest.raises(TypeError):
        geopandas.points_from_xy(x=s)
        geopandas.points_from_xy(y=s)
        geopandas.points_from_xy(z=s)


def test_from_shapely():
    assert isinstance(T, GeometryArray)
    assert equal_geometries(T, triangles)


def test_from_shapely_geo_interface():
    class Point:
        def __init__(self, x, y):
            self.x = x
            self.y = y

        @property
        def __geo_interface__(self):
            return {"type": "Point", "coordinates": (self.x, self.y)}

    result = from_shapely([Point(1.0, 2.0), Point(3.0, 4.0)])

    expected = from_shapely(
        [shapely.geometry.Point(1.0, 2.0), shapely.geometry.Point(3.0, 4.0)]
    )

    assert all(v.equals(t) for v, t in zip(result, expected))


def test_from_wkb():
    # list
    L_wkb = [p.wkb for p in points_no_missing]
    res = from_wkb(L_wkb)
    assert isinstance(res, GeometryArray)
    assert all(v.equals(t) for v, t in zip(res, points_no_missing))

    # array
    res = from_wkb(np.array(L_wkb, dtype=object))
    assert isinstance(res, GeometryArray)
    assert all(v.equals(t) for v, t in zip(res, points_no_missing))

    # missing values
    # TODO(shapely) does not support empty strings, np.nan, or pd.NA
    missing_values = [None]

    res = from_wkb(missing_values)
    np.testing.assert_array_equal(res, np.full(len(missing_values), None))

    # single MultiPolygon
    multi_poly = shapely.geometry.MultiPolygon(
        [shapely.geometry.box(0, 0, 1, 1), shapely.geometry.box(3, 3, 4, 4)]
    )
    res = from_wkb([multi_poly.wkb])
    assert res[0] == multi_poly


def test_from_wkb_hex():
    geometry_hex = ["0101000000CDCCCCCCCCCC1440CDCCCCCCCC0C4A40"]
    res = from_wkb(geometry_hex)
    assert isinstance(res, GeometryArray)

    # array
    res = from_wkb(np.array(geometry_hex, dtype=object))
    assert isinstance(res, GeometryArray)


def test_from_wkb_on_invalid():
    # Single point LineString hex WKB: invalid
    invalid_wkb_hex = "01020000000100000000000000000008400000000000000840"
    message = "point array must contain 0 or >1 elements"

    with pytest.raises(Exception, match=message):
        from_wkb([invalid_wkb_hex], on_invalid="raise")

    with pytest.warns(Warning, match=message):
        res = from_wkb([invalid_wkb_hex], on_invalid="warn")
    assert res == [None]

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = from_wkb([invalid_wkb_hex], on_invalid="ignore")
    assert res == [None]


def test_to_wkb():
    P = from_shapely(points_no_missing)
    res = to_wkb(P)
    exp = np.array([p.wkb for p in points_no_missing], dtype=object)
    assert isinstance(res, np.ndarray)
    np.testing.assert_array_equal(res, exp)

    res = to_wkb(P, hex=True)
    exp = np.array([p.wkb_hex for p in points_no_missing], dtype=object)
    assert isinstance(res, np.ndarray)
    np.testing.assert_array_equal(res, exp)

    # missing values
    a = from_shapely([None, points_no_missing[0]])
    res = to_wkb(a)
    assert res[0] is None


@pytest.mark.parametrize("string_type", ["str", "bytes"])
def test_from_wkt(string_type):
    if string_type == "str":
        f = str
    else:

        def f(x):
            return bytes(x, "utf8")

    # list
    L_wkt = [f(p.wkt) for p in points_no_missing]
    res = from_wkt(L_wkt)
    assert isinstance(res, GeometryArray)
    tol = 0.5 * 10 ** (-6)
    assert all(v.equals_exact(t, tolerance=tol) for v, t in zip(res, points_no_missing))
    assert all(v.equals_exact(t, tolerance=tol) for v, t in zip(res, points_no_missing))

    # array
    res = from_wkt(np.array(L_wkt, dtype=object))
    assert isinstance(res, GeometryArray)
    assert all(v.equals_exact(t, tolerance=tol) for v, t in zip(res, points_no_missing))

    # missing values
    # TODO(shapely) does not support empty strings, np.nan, or pd.NA
    missing_values = [None]

    res = from_wkt(missing_values)
    np.testing.assert_array_equal(res, np.full(len(missing_values), None))

    # single MultiPolygon
    multi_poly = shapely.geometry.MultiPolygon(
        [shapely.geometry.box(0, 0, 1, 1), shapely.geometry.box(3, 3, 4, 4)]
    )
    res = from_wkt([f(multi_poly.wkt)])
    assert res[0] == multi_poly


def test_from_wkt_on_invalid():
    # Single point LineString WKT: invalid
    invalid_wkt = "LINESTRING(0 0)"
    message = "point array must contain 0 or >1 elements"

    with pytest.raises(Exception, match=message):
        from_wkt([invalid_wkt], on_invalid="raise")

    with pytest.warns(Warning, match=message):
        res = from_wkt([invalid_wkt], on_invalid="warn")
    assert res == [None]

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = from_wkt([invalid_wkt], on_invalid="ignore")
    assert res == [None]


def test_to_wkt():
    P = from_shapely(points_no_missing)
    res = to_wkt(P, rounding_precision=-1)
    exp = np.array([p.wkt for p in points_no_missing], dtype=object)
    assert isinstance(res, np.ndarray)
    np.testing.assert_array_equal(res, exp)

    # missing values
    a = from_shapely([None, points_no_missing[0]])
    res = to_wkt(a)
    assert res[0] is None


def test_as_array():
    arr = from_shapely(points_no_missing)
    np_arr1 = np.asarray(arr)
    np_arr2 = arr.to_numpy()
    assert np_arr1[0] == arr[0]
    np.testing.assert_array_equal(np_arr1, np_arr2)


@pytest.mark.parametrize(
    "attr,args",
    [
        ("contains", ()),
        ("covers", ()),
        ("crosses", ()),
        ("disjoint", ()),
        ("geom_equals", ()),
        ("intersects", ()),
        ("overlaps", ()),
        ("touches", ()),
        ("within", ()),
        ("geom_equals_exact", (0.1,)),
        ("geom_almost_equals", (3,)),
    ],
)
# filters required for attr=geom_almost_equals only
@pytest.mark.filterwarnings(r"ignore:The \'geom_almost_equals\(\)\' method is deprecat")
@pytest.mark.filterwarnings(r"ignore:The \'almost_equals\(\)\' method is deprecated")
def test_predicates_vector_scalar(attr, args):
    na_value = False

    point = points[0]
    tri = triangles[0]

    for other in [point, tri, shapely.geometry.Polygon()]:
        result = getattr(T, attr)(other, *args)
        assert isinstance(result, np.ndarray)
        assert result.dtype == bool

        expected = [
            (
                getattr(tri, attr if "geom" not in attr else attr[5:])(other, *args)
                if tri is not None
                else na_value
            )
            for tri in triangles
        ]

        assert result.tolist() == expected

    # TODO other is missing


@pytest.mark.parametrize(
    "attr,args",
    [
        ("contains", ()),
        ("covers", ()),
        ("crosses", ()),
        ("disjoint", ()),
        ("geom_equals", ()),
        ("intersects", ()),
        ("overlaps", ()),
        ("touches", ()),
        ("within", ()),
        ("geom_equals_exact", (0.1,)),
        ("geom_almost_equals", (3,)),
    ],
)
# filters required for attr=geom_almost_equals only
@pytest.mark.filterwarnings(r"ignore:The \'geom_almost_equals\(\)\' method is deprecat")
@pytest.mark.filterwarnings(r"ignore:The \'almost_equals\(\)\' method is deprecated")
def test_predicates_vector_vector(attr, args):
    na_value = False
    empty_value = True if attr == "disjoint" else False

    A = (
        [shapely.geometry.Polygon(), None]
        + [
            shapely.geometry.Polygon(
                [(random.random(), random.random()) for i in range(3)]
            )
            for _ in range(100)
        ]
        + [None]
    )
    B = [
        shapely.geometry.Polygon([(random.random(), random.random()) for i in range(3)])
        for _ in range(100)
    ] + [shapely.geometry.Polygon(), None, None]

    vec_A = from_shapely(A)
    vec_B = from_shapely(B)

    result = getattr(vec_A, attr)(vec_B, *args)
    assert isinstance(result, np.ndarray)
    assert result.dtype == bool

    expected = []
    for a, b in zip(A, B):
        if a is None or b is None:
            expected.append(na_value)
        elif a.is_empty or b.is_empty:
            expected.append(empty_value)
        else:
            expected.append(
                getattr(a, attr if "geom" not in attr else attr[5:])(b, *args)
            )

    assert result.tolist() == expected


@pytest.mark.parametrize(
    "attr",
    [
        "boundary",
        "centroid",
        "convex_hull",
        "envelope",
        "exterior",
        # 'interiors',
    ],
)
def test_unary_geo(attr):
    na_value = None

    result = getattr(T, attr)
    expected = [getattr(t, attr) if t is not None else na_value for t in triangles]

    assert equal_geometries(result, expected)


@pytest.mark.parametrize("attr", ["representative_point"])
def test_unary_geo_callable(attr):
    na_value = None

    result = getattr(T, attr)()
    expected = [getattr(t, attr)() if t is not None else na_value for t in triangles]

    assert equal_geometries(result, expected)


@pytest.mark.parametrize(
    "attr", ["difference", "symmetric_difference", "union", "intersection"]
)
def test_binary_geo_vector(attr):
    na_value = None

    quads = [shapely.geometry.Polygon(), None]
    while len(quads) < 12:
        geom = shapely.geometry.Polygon(
            [(random.random(), random.random()) for i in range(4)]
        )
        if geom.is_valid:
            quads.append(geom)

    Q = from_shapely(quads)

    result = getattr(T, attr)(Q)
    expected = [
        getattr(t, attr)(q) if t is not None and q is not None else na_value
        for t, q in zip(triangles, quads)
    ]

    assert equal_geometries(result, expected)


@pytest.mark.parametrize(
    "attr", ["difference", "symmetric_difference", "union", "intersection"]
)
def test_binary_geo_scalar(attr):
    na_value = None

    quads = []
    while len(quads) < 1:
        geom = shapely.geometry.Polygon(
            [(random.random(), random.random()) for i in range(4)]
        )
        if geom.is_valid:
            quads.append(geom)

    q = quads[0]

    for other in [q, shapely.geometry.Polygon()]:
        result = getattr(T, attr)(other)
        expected = [
            getattr(t, attr)(other) if t is not None else na_value for t in triangles
        ]

    assert equal_geometries(result, expected)


@pytest.mark.parametrize(
    "attr",
    [
        "is_closed",
        "is_valid",
        "is_empty",
        "is_simple",
        "has_z",
        # for is_ring we raise a warning about the value for Polygon changing
        "is_ring",
    ],
)
def test_unary_predicates(attr):
    na_value = False
    if attr == "is_simple" and geos_version < (3, 8):
        # poly.is_simple raises an error for empty polygon for GEOS < 3.8
        with pytest.raises(Exception):  # noqa: B017
            T.is_simple
        vals = triangle_no_missing
        V = from_shapely(vals)
    else:
        vals = triangles
        V = T

    result = getattr(V, attr)

    if attr == "is_ring":
        expected = [
            getattr(t, attr) if t is not None and t.exterior is not None else na_value
            for t in vals
        ]
    else:
        expected = [getattr(t, attr) if t is not None else na_value for t in vals]

    assert result.tolist() == expected


def test_is_ring():
    g = [
        shapely.geometry.LinearRing([(0, 0), (1, 1), (1, -1)]),
        shapely.geometry.LineString([(0, 0), (1, 1), (1, -1)]),
        shapely.geometry.LineString([(0, 0), (1, 1), (1, -1), (0, 0)]),
        shapely.geometry.Polygon([(0, 0), (1, 1), (1, -1)]),
        shapely.wkt.loads("POLYGON EMPTY"),
        None,
    ]
    expected = [True, False, True, False, False, False]
    result = from_shapely(g).is_ring

    assert result.tolist() == expected


@pytest.mark.parametrize("attr", ["area", "length"])
def test_unary_float(attr):
    na_value = np.nan
    result = getattr(T, attr)
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.dtype("float64")
    expected = [getattr(t, attr) if t is not None else na_value for t in triangles]
    np.testing.assert_allclose(result, expected)


def test_geom_types():
    cat = T.geom_type
    # empty polygon has GeometryCollection type
    assert list(cat) == ["Polygon"] * (len(T) - 1) + [None]


def test_geom_types_null_mixed():
    geoms = [
        shapely.geometry.Polygon([(0, 0), (0, 1), (1, 1)]),
        None,
        shapely.geometry.Point(0, 1),
    ]

    G = from_shapely(geoms)
    cat = G.geom_type

    assert list(cat) == ["Polygon", None, "Point"]


def test_binary_distance():
    attr = "distance"
    na_value = np.nan
    # also use nan for empty

    # vector - vector
    result = P[: len(T)].distance(T[::-1])
    expected = [
        (
            getattr(p, attr)(t)
            if not ((t is None or t.is_empty) or (p is None or p.is_empty))
            else na_value
        )
        for t, p in zip(triangles[::-1], points)
    ]
    np.testing.assert_allclose(result, expected)

    # vector - scalar
    p = points[0]
    result = T.distance(p)
    expected = [
        getattr(t, attr)(p) if not (t is None or t.is_empty) else na_value
        for t in triangles
    ]
    np.testing.assert_allclose(result, expected)

    # other is empty
    result = T.distance(shapely.geometry.Polygon())
    expected = [na_value] * len(T)
    np.testing.assert_allclose(result, expected)
    # TODO other is None


def test_binary_relate():
    attr = "relate"
    na_value = None

    # vector - vector
    result = getattr(P[: len(T)], attr)(T[::-1])
    expected = [
        getattr(p, attr)(t) if t is not None and p is not None else na_value
        for t, p in zip(triangles[::-1], points)
    ]
    assert list(result) == expected

    # vector - scalar
    p = points[0]
    result = getattr(T, attr)(p)
    expected = [getattr(t, attr)(p) if t is not None else na_value for t in triangles]
    assert list(result) == expected


@pytest.mark.parametrize("normalized", [True, False])
def test_binary_project(normalized):
    na_value = np.nan
    lines = (
        [None]
        + [
            shapely.geometry.LineString(
                [(random.random(), random.random()) for _ in range(2)]
            )
            for _ in range(len(P) - 2)
        ]
        + [None]
    )
    L = from_shapely(lines)

    result = L.project(P, normalized=normalized)
    expected = [
        (
            line.project(p, normalized=normalized)
            if line is not None and p is not None
            else na_value
        )
        for p, line in zip(points, lines)
    ]
    np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize("cap_style", [CAP_STYLE.round, CAP_STYLE.square])
@pytest.mark.parametrize("join_style", [JOIN_STYLE.round, JOIN_STYLE.bevel])
@pytest.mark.parametrize("resolution", [16, 25])
def test_buffer(resolution, cap_style, join_style):
    na_value = None
    expected = [
        (
            p.buffer(
                0.1, resolution=resolution, cap_style=cap_style, join_style=join_style
            )
            if p is not None
            else na_value
        )
        for p in points
    ]
    result = P.buffer(
        0.1, resolution=resolution, cap_style=cap_style, join_style=join_style
    )
    assert equal_geometries(expected, result)

    dist = np.array([0.1] * len(P))
    result = P.buffer(
        dist, resolution=resolution, cap_style=cap_style, join_style=join_style
    )
    assert equal_geometries(expected, result)


def test_simplify():
    triangles = [
        shapely.geometry.Polygon(
            [(random.random(), random.random()) for i in range(3)]
        ).buffer(10)
        for _ in range(10)
    ]
    T = from_shapely(triangles)

    result = T.simplify(1)
    expected = [t.simplify(1) for t in triangles]
    assert all(a.equals(b) for a, b in zip(expected, result))


def test_unary_union():
    geoms = [
        shapely.geometry.Polygon([(0, 0), (0, 1), (1, 1)]),
        shapely.geometry.Polygon([(0, 0), (1, 0), (1, 1)]),
    ]
    G = from_shapely(geoms)
    with pytest.warns(
        DeprecationWarning, match="The 'unary_union' attribute is deprecated"
    ):
        u = G.unary_union()

    expected = shapely.geometry.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    assert u.equals(expected)
    assert u.equals(G.union_all())


def test_union_all():
    geoms = [
        shapely.geometry.Polygon([(0, 0), (0, 1), (1, 1)]),
        shapely.geometry.Polygon([(0, 0), (1, 0), (1, 1)]),
    ]
    G = from_shapely(geoms)
    u = G.union_all()

    expected = shapely.geometry.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    assert u.equals(expected)

    u_cov = G.union_all(method="coverage")
    assert u_cov.equals(expected)

    with pytest.raises(ValueError, match="Method 'invalid' not recognized."):
        G.union_all(method="invalid")


@pytest.mark.parametrize(
    "attr, arg",
    [
        ("affine_transform", ([0, 1, 1, 0, 0, 0],)),
        ("translate", ()),
        ("rotate", (10,)),
        ("scale", ()),
        ("skew", ()),
    ],
)
def test_affinity_methods(attr, arg):
    result = getattr(T, attr)(*arg)
    expected = [
        getattr(shapely.affinity, attr)(t, *arg) if not (t is None or t.is_empty) else t
        for t in triangles
    ]
    assert equal_geometries(result, expected)


# def test_coords():
#     L = T.exterior.coords
#     assert L == [tuple(t.exterior.coords) for t in triangles]


def test_coords_x_y():
    na_value = np.nan
    result = P.x
    expected = [p.x if p is not None else na_value for p in points]
    np.testing.assert_allclose(result, expected)

    result = P.y
    expected = [p.y if p is not None else na_value for p in points]
    np.testing.assert_allclose(result, expected)


def test_bounds():
    result = T.bounds
    expected = [
        t.bounds if not (t is None or t.is_empty) else [np.nan] * 4 for t in triangles
    ]
    np.testing.assert_allclose(result, expected)

    # additional check for one empty / missing
    for geom in [None, shapely.geometry.Polygon()]:
        E = from_shapely([geom])
        result = E.bounds
        assert result.ndim == 2
        assert result.dtype == "float64"
        np.testing.assert_allclose(result, np.array([[np.nan] * 4]))

    # empty array (https://github.com/geopandas/geopandas/issues/1195)
    E = from_shapely([])
    result = E.bounds
    assert result.shape == (0, 4)
    assert result.dtype == "float64"


def test_total_bounds():
    result = T.total_bounds
    bounds = np.array(
        [t.bounds if not (t is None or t.is_empty) else [np.nan] * 4 for t in triangles]
    )
    expected = np.array(
        [
            np.nanmin(bounds[:, 0]),  # minx
            np.nanmin(bounds[:, 1]),  # miny
            np.nanmax(bounds[:, 2]),  # maxx
            np.nanmax(bounds[:, 3]),  # maxy
        ]
    )
    np.testing.assert_allclose(result, expected)

    # additional check for empty array or one empty / missing
    for geoms in [[], [None], [shapely.geometry.Polygon()]]:
        E = from_shapely(geoms)
        result = E.total_bounds
        assert result.ndim == 1
        assert result.dtype == "float64"
        np.testing.assert_allclose(result, np.array([np.nan] * 4))


def test_getitem():
    points = [shapely.geometry.Point(i, i) for i in range(10)]
    P = from_shapely(points)

    P2 = P[P.area > 0.3]
    assert isinstance(P2, GeometryArray)

    P3 = P[[1, 3, 5]]
    assert len(P3) == 3
    assert isinstance(P3, GeometryArray)
    assert [p.x for p in P3] == [1, 3, 5]

    P4 = P[1::2]
    assert len(P4) == 5
    assert isinstance(P3, GeometryArray)
    assert [p.x for p in P4] == [1, 3, 5, 7, 9]

    P5 = P[1]
    assert isinstance(P5, shapely.geometry.Point)
    assert P5.equals(points[1])


@pytest.mark.parametrize(
    "item",
    [
        geopandas.GeoDataFrame(
            geometry=[shapely.geometry.Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])]
        ),
        geopandas.GeoSeries(
            [shapely.geometry.Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])]
        ),
        np.array([shapely.geometry.Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])]),
        [shapely.geometry.Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])],
        shapely.geometry.Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]),
    ],
)
def test_setitem(item):
    points = [shapely.geometry.Point(i, i) for i in range(10)]
    P = from_shapely(points)

    P[[0]] = item

    assert isinstance(P[0], shapely.geometry.Polygon)


def test_equality_ops():
    with pytest.raises(ValueError):
        _ = P[:5] == P[:7]

    a1 = from_shapely([points[1], points[2], points[3]])
    a2 = from_shapely([points[1], points[0], points[3]])

    res = a1 == a2
    assert res.tolist() == [True, False, True]

    res = a1 != a2
    assert res.tolist() == [False, True, False]

    # check the correct expansion of list-like geometry
    multi_poly = shapely.geometry.MultiPolygon(
        [shapely.geometry.box(0, 0, 1, 1), shapely.geometry.box(3, 3, 4, 4)]
    )
    a3 = from_shapely([points[1], points[2], points[3], multi_poly])

    res = a3 == multi_poly
    assert res.tolist() == [False, False, False, True]


def test_dir():
    assert "contains" in dir(P)
    assert "to_numpy" in dir(P)


def test_chaining():
    # contains will give False for empty / missing
    T = from_shapely(triangle_no_missing)
    assert T.contains(T.centroid).all()


def test_pickle():
    import pickle

    T2 = pickle.loads(pickle.dumps(T))
    # assert (T.data != T2.data).all()
    assert T2[-1] is None
    assert T2[-2].is_empty
    assert T[:-2].geom_equals(T2[:-2]).all()


def test_raise_on_bad_sizes():
    with pytest.raises(ValueError) as info:
        T.contains(P)

    assert "lengths" in str(info.value).lower()
    assert "12" in str(info.value)
    assert "21" in str(info.value)


def test_buffer_single_multipolygon():
    # https://github.com/geopandas/geopandas/issues/1130
    multi_poly = shapely.geometry.MultiPolygon(
        [shapely.geometry.box(0, 0, 1, 1), shapely.geometry.box(3, 3, 4, 4)]
    )
    arr = from_shapely([multi_poly])
    result = arr.buffer(1)
    expected = [multi_poly.buffer(1)]
    equal_geometries(result, expected)
    result = arr.buffer(np.array([1]))
    equal_geometries(result, expected)


def test_astype_multipolygon():
    # https://github.com/geopandas/geopandas/issues/1145
    multi_poly = shapely.geometry.MultiPolygon(
        [shapely.geometry.box(0, 0, 1, 1), shapely.geometry.box(3, 3, 4, 4)]
    )
    arr = from_shapely([multi_poly])
    result = arr.astype(str)
    assert isinstance(result[0], str)
    assert result[0] == multi_poly.wkt

    # astype(object) does not convert to string
    result = arr.astype(object)
    assert isinstance(result[0], shapely.geometry.base.BaseGeometry)

    # astype(np_dtype) honors the dtype
    result = arr.astype(np.dtype("U10"))
    assert result.dtype == np.dtype("U10")
    assert result[0] == multi_poly.wkt[:10]


@pytest.mark.skipif(not HAS_PYPROJ, reason="pyproj not installed")
def test_check_crs():
    t1 = T.copy()
    t1.crs = 4326
    assert _check_crs(t1, T) is False
    assert _check_crs(t1, t1) is True
    assert _check_crs(t1, T, allow_none=True) is True


@pytest.mark.skipif(not HAS_PYPROJ, reason="pyproj not installed")
def test_crs_mismatch_warn():
    t1 = T.copy()
    t2 = T.copy()
    t1.crs = 4326
    t2.crs = 3857

    # two different CRS
    with pytest.warns(UserWarning, match="CRS mismatch between the CRS"):
        _crs_mismatch_warn(t1, t2)

    # left None
    with pytest.warns(UserWarning, match="CRS mismatch between the CRS"):
        _crs_mismatch_warn(T, t2)

    # right None
    with pytest.warns(UserWarning, match="CRS mismatch between the CRS"):
        _crs_mismatch_warn(t1, T)


@pytest.mark.skipif(HAS_PYPROJ, reason="pyproj installed")
def test_missing_pyproj():
    with pytest.warns(UserWarning, match="Cannot set the CRS, falling back to None"):
        t = T.copy()
        t.crs = 4326
    assert t.crs is None


@pytest.mark.parametrize("NA", [None, np.nan])
def test_isna(NA):
    t1 = T.copy()
    t1[0] = NA
    assert t1[0] is None


def test_isna_pdNA():
    t1 = T.copy()
    t1[0] = pd.NA
    assert t1[0] is None


def test_shift_has_crs():
    t = T.copy()
    t.crs = 4326
    assert t.shift(1).crs == t.crs
    assert t.shift(0).crs == t.crs
    assert t.shift(-1).crs == t.crs


def test_unique_has_crs():
    t = T.copy()
    t.crs = 4326
    assert t.unique().crs == t.crs


@pytest.mark.skipif(HAS_PYPROJ, reason="pyproj installed")
def test_to_crs_pyproj_error():
    t = T.copy()
    t.crs = 4326
    with pytest.raises(
        ImportError, match="The 'pyproj' package is required for to_crs"
    ):
        t.to_crs(3857)


@pytest.mark.skipif(HAS_PYPROJ, reason="pyproj installed")
def test_estimate_utm_crs_pyproj_error():
    with pytest.raises(
        ImportError, match="The 'pyproj' package is required for estimate_utm_crs"
    ):
        T.estimate_utm_crs()


class TestEstimateUtmCrs:
    def setup_method(self):
        self.esb = shapely.geometry.Point(-73.9847, 40.7484)
        self.sol = shapely.geometry.Point(-74.0446, 40.6893)
        self.landmarks = from_shapely([self.esb, self.sol], crs="epsg:4326")

    def test_estimate_utm_crs__geographic(self):
        pyproj = pytest.importorskip("pyproj")

        assert self.landmarks.estimate_utm_crs() == pyproj.CRS("EPSG:32618")
        assert self.landmarks.estimate_utm_crs("NAD83") == pyproj.CRS("EPSG:26918")

    def test_estimate_utm_crs__projected(self):
        pyproj = pytest.importorskip("pyproj")

        assert self.landmarks.to_crs("EPSG:3857").estimate_utm_crs() == pyproj.CRS(
            "EPSG:32618"
        )

    def test_estimate_utm_crs__antimeridian(self):
        pyproj = pytest.importorskip("pyproj")

        antimeridian = from_shapely(
            [
                shapely.geometry.Point(1722483.900174921, 5228058.6143420935),
                shapely.geometry.Point(4624385.494808555, 8692574.544944234),
            ],
            crs="EPSG:3851",
        )
        assert antimeridian.estimate_utm_crs() == pyproj.CRS("EPSG:32760")

    def test_estimate_utm_crs__out_of_bounds(self):
        pytest.importorskip("pyproj")

        with pytest.raises(RuntimeError, match="Unable to determine UTM CRS"):
            from_shapely(
                [shapely.geometry.Polygon([(0, 90), (1, 90), (2, 90)])], crs="EPSG:4326"
            ).estimate_utm_crs()

    def test_estimate_utm_crs__missing_crs(self):
        pytest.importorskip("pyproj")

        with pytest.raises(RuntimeError, match="crs must be set"):
            from_shapely(
                [shapely.geometry.Polygon([(0, 90), (1, 90), (2, 90)])]
            ).estimate_utm_crs()
