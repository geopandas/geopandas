import random

import numpy as np
import pandas as pd
import six

import shapely
import shapely.affinity
import shapely.geometry
from shapely.geometry.base import CAP_STYLE, JOIN_STYLE
import shapely.wkb
from shapely._buildcfg import geos_version

import geopandas
from geopandas.array import (
    GeometryArray,
    from_shapely,
    from_wkb,
    from_wkt,
    points_from_xy,
    to_wkb,
    to_wkt,
    _check_crs,
    _crs_mismatch_warn,
)
import geopandas._compat as compat

import pytest

triangle_no_missing = [
    shapely.geometry.Polygon([(random.random(), random.random()) for i in range(3)])
    for _ in range(10)
]
triangles = triangle_no_missing + [shapely.geometry.Polygon(), None]
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
    x = np.arange(10).astype(np.float)
    y = np.arange(10).astype(np.float) ** 2

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
    # TODO(pygeos) does not support empty strings
    if compat.USE_PYGEOS:
        L_wkb.extend([None])
    else:
        L_wkb.extend([b"", None])
    res = from_wkb(L_wkb)
    assert res[-1] is None
    if not compat.USE_PYGEOS:
        assert res[-2] is None

    # single MultiPolygon
    multi_poly = shapely.geometry.MultiPolygon(
        [shapely.geometry.box(0, 0, 1, 1), shapely.geometry.box(3, 3, 4, 4)]
    )
    res = from_wkb([multi_poly.wkb])
    assert res[0] == multi_poly


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
        f = six.text_type
    else:
        if six.PY3:

            def f(x):
                return bytes(x, "utf8")

        else:

            def f(x):
                return x

    # list
    L_wkt = [f(p.wkt) for p in points_no_missing]
    res = from_wkt(L_wkt)
    assert isinstance(res, GeometryArray)
    assert all(v.almost_equals(t) for v, t in zip(res, points_no_missing))

    # array
    res = from_wkt(np.array(L_wkt, dtype=object))
    assert isinstance(res, GeometryArray)
    assert all(v.almost_equals(t) for v, t in zip(res, points_no_missing))

    # missing values
    # TODO(pygeos) does not support empty strings
    if compat.USE_PYGEOS:
        L_wkt.extend([None])
    else:
        L_wkt.extend([f(""), None])
    res = from_wkt(L_wkt)
    assert res[-1] is None
    if not compat.USE_PYGEOS:
        assert res[-2] is None

    # single MultiPolygon
    multi_poly = shapely.geometry.MultiPolygon(
        [shapely.geometry.box(0, 0, 1, 1), shapely.geometry.box(3, 3, 4, 4)]
    )
    res = from_wkt([f(multi_poly.wkt)])
    assert res[0] == multi_poly


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
def test_predicates_vector_scalar(attr, args):
    na_value = False

    point = points[0]
    tri = triangles[0]

    for other in [point, tri, shapely.geometry.Polygon()]:
        result = getattr(T, attr)(other, *args)
        assert isinstance(result, np.ndarray)
        assert result.dtype == bool

        expected = [
            getattr(tri, attr if "geom" not in attr else attr[5:])(other, *args)
            if tri is not None
            else na_value
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
    "attr,args", [("equals_exact", (0.1,)), ("almost_equals", (3,))]
)
def test_equals_deprecation(attr, args):
    point = points[0]
    tri = triangles[0]

    for other in [point, tri, shapely.geometry.Polygon()]:
        with pytest.warns(FutureWarning):
            result = getattr(T, attr)(other, *args)
        assert result.tolist() == getattr(T, "geom_" + attr)(other, *args).tolist()


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

    if attr == "boundary":
        # pygeos returns None for empty geometries
        if not compat.USE_PYGEOS:
            # boundary raises for empty geometry
            with pytest.raises(Exception):
                T.boundary

        values = triangle_no_missing + [None]
        A = from_shapely(values)
    else:
        values = triangles
        A = T

    result = getattr(A, attr)
    if attr == "exterior" and compat.USE_PYGEOS:
        # TODO(pygeos)
        # empty Polygon() has an exterior with shapely > 1.7, which gives
        # empty LinearRing instead of None,
        # but conversion to pygeos still results in empty GeometryCollection
        expected = [
            getattr(t, attr) if t is not None and not t.is_empty else na_value
            for t in values
        ]
    else:
        expected = [getattr(t, attr) if t is not None else na_value for t in values]

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
    "attr", ["is_closed", "is_valid", "is_empty", "is_simple", "has_z", "is_ring"]
)
def test_unary_predicates(attr):
    na_value = False
    if attr == "is_simple" and geos_version < (3, 8) and not compat.USE_PYGEOS:
        # poly.is_simple raises an error for empty polygon for GEOS < 3.8
        with pytest.raises(Exception):
            T.is_simple
        vals = triangle_no_missing
        V = from_shapely(vals)
    else:
        vals = triangles
        V = T

    result = getattr(V, attr)

    if attr == "is_simple" and (geos_version < (3, 8) or compat.USE_PYGEOS):
        # poly.is_simple raises an error for empty polygon for GEOS < 3.8
        # with shapely, pygeos always returns False for all GEOS versions
        # But even for Shapely with GEOS >= 3.8, empty GeometryCollection
        # returns True instead of False
        expected = [
            getattr(t, attr) if t is not None and not t.is_empty else na_value
            for t in vals
        ]
    elif attr == "is_ring":
        expected = [
            getattr(t.exterior, attr)
            if t is not None and t.exterior is not None
            else na_value
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
        shapely.geometry.Polygon(),
        None,
    ]
    expected = [True, False, True, True, False, False]

    result = from_shapely(g).is_ring

    assert result.tolist() == expected


@pytest.mark.parametrize("attr", ["area", "length"])
def test_unary_float(attr):
    na_value = np.nan
    result = getattr(T, attr)
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float
    expected = [getattr(t, attr) if t is not None else na_value for t in triangles]
    np.testing.assert_allclose(result, expected)


def test_geom_types():
    cat = T.geom_type
    # empty polygon has GeometryCollection type
    assert list(cat) == ["Polygon"] * (len(T) - 2) + ["GeometryCollection", None]


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
        getattr(p, attr)(t)
        if not ((t is None or t.is_empty) or (p is None or p.is_empty))
        else na_value
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
        l.project(p, normalized=normalized)
        if l is not None and p is not None
        else na_value
        for p, l in zip(points, lines)
    ]
    np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize("cap_style", [CAP_STYLE.round, CAP_STYLE.square])
@pytest.mark.parametrize("join_style", [JOIN_STYLE.round, JOIN_STYLE.bevel])
@pytest.mark.parametrize("resolution", [16, 25])
def test_buffer(resolution, cap_style, join_style):
    if compat.USE_PYGEOS:
        # TODO(pygeos) need to further investigate why this test fails
        if cap_style == 1 and join_style == 3:
            pytest.skip("failing TODO")

    na_value = None
    expected = [
        p.buffer(0.1, resolution=resolution, cap_style=cap_style, join_style=join_style)
        if p is not None
        else na_value
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
    u = G.unary_union()

    expected = shapely.geometry.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    assert u.equals(expected)


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


def test_equality_ops():
    with pytest.raises(ValueError):
        P[:5] == P[:7]

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
    assert "data" in dir(P)


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


def test_check_crs():
    t1 = T.copy()
    t1.crs = 4326
    assert _check_crs(t1, T) is False
    assert _check_crs(t1, t1) is True
    assert _check_crs(t1, T, allow_none=True) is True


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


@pytest.mark.parametrize("NA", [None, np.nan])
def test_isna(NA):
    t1 = T.copy()
    t1[0] = NA
    assert t1[0] is None


@pytest.mark.skipif(not compat.PANDAS_GE_10, reason="pd.NA introduced in pandas 1.0")
def test_isna_pdNA():
    t1 = T.copy()
    t1[0] = pd.NA
    assert t1[0] is None
