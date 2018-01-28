import random
import time

import numpy as np
import six

import shapely
from shapely.geometry.base import CAP_STYLE, JOIN_STYLE

from geopandas.array import GeometryArray
from geopandas.array import points_from_xy, from_shapely, from_wkb, from_wkt
from geopandas.vectorized import serialize, deserialize, cysjoin

import pytest


triangles = [
    shapely.geometry.Polygon([(random.random(), random.random()) for i in range(3)])
    for _ in range(10)
]
T = from_shapely(triangles)

points = [shapely.geometry.Point(random.random(), random.random()) for _ in range(20)]
P = from_shapely(points)


point = points[0]


def test_points():
    x = np.arange(10).astype(np.float)
    y = np.arange(10).astype(np.float) ** 2

    points = points_from_xy(x, y)
    assert (points.data != 0).all()

    for i in range(10):
        assert points[i].x == x[i]
        assert points[i].y == y[i]

    assert isinstance(points[0], shapely.geometry.Point)


def test_from_shapely():
    assert isinstance(T, GeometryArray)
    assert [v.equals(t) for v, t in zip(T, triangles)]
    # TODO: handle gc


def test_from_wkb():
    # list
    L_wkb = [p.wkb for p in points]
    res = from_wkb(L_wkb)
    assert isinstance(res, GeometryArray)
    assert all(v.equals(t) for v, t in zip(res, points))

    # array
    res = from_wkb(np.array(L_wkb, dtype=object))
    assert isinstance(res, GeometryArray)
    assert all(v.equals(t) for v, t in zip(res, points))

    # missing values
    L_wkb.extend([b"", None])
    res = from_wkb(L_wkb)
    assert res[-1] is None
    assert res[-2] is None


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
    L_wkt = [f(p.wkt) for p in points]
    res = from_wkt(L_wkt)
    assert isinstance(res, GeometryArray)
    assert all(v.almost_equals(t) for v, t in zip(res, points))

    # array
    res = from_wkt(np.array(L_wkt, dtype=object))
    assert isinstance(res, GeometryArray)
    assert all(v.almost_equals(t) for v, t in zip(res, points))

    # missing values
    L_wkt.extend([f(""), None])
    res = from_wkt(L_wkt)
    assert res[-1] is None
    assert res[-2] is None


@pytest.mark.parametrize(
    "attr,args",
    [
        ("contains", ()),
        ("covers", ()),
        ("crosses", ()),
        ("disjoint", ()),
        ("equals", ()),
        ("intersects", ()),
        ("overlaps", ()),
        ("touches", ()),
        ("within", ()),
        ("equals_exact", (0.1,)),
    ],
)
def test_vector_scalar_predicates(attr, args):
    triangles = [
        shapely.geometry.Polygon([(random.random(), random.random()) for i in range(3)])
        for _ in range(100)
    ]

    T = from_shapely(triangles)

    point = shapely.geometry.Point(random.random(), random.random())
    tri = triangles[0]

    for other in [point, tri]:
        result = getattr(T, attr)(other, *args)
        assert isinstance(result, np.ndarray)
        assert result.dtype == bool

        expected = [getattr(tri, attr)(other, *args) for tri in triangles]

        assert result.tolist() == expected


@pytest.mark.parametrize(
    "attr,args",
    [
        ("contains", ()),
        ("covers", ()),
        ("crosses", ()),
        ("disjoint", ()),
        ("equals", ()),
        ("intersects", ()),
        ("overlaps", ()),
        ("touches", ()),
        ("within", ()),
        ("equals_exact", (0.1,)),
    ],
)
def test_vector_vector_predicates(attr, args):
    A = [
        shapely.geometry.Polygon([(random.random(), random.random()) for i in range(3)])
        for _ in range(100)
    ]
    B = [
        shapely.geometry.Polygon([(random.random(), random.random()) for i in range(3)])
        for _ in range(100)
    ]

    vec_A = from_shapely(A)
    vec_B = from_shapely(B)

    result = getattr(vec_A, attr)(vec_B, *args)
    assert isinstance(result, np.ndarray)
    assert result.dtype == bool

    expected = [getattr(a, attr)(b, *args) for a, b in zip(A, B)]

    assert result.tolist() == expected


@pytest.mark.parametrize(
    "attr", ["boundary", "centroid", "convex_hull", "envelope", "exterior"]
)
def test_unary_geo(attr):

    result = getattr(T, attr)
    expected = [getattr(t, attr) for t in triangles]

    assert all([a.equals(b) for a, b in zip(result, expected)])


@pytest.mark.parametrize("attr", ["representative_point"])
def test_unary_geo_callable(attr):

    result = getattr(T, attr)()
    expected = [getattr(t, attr)() for t in triangles]

    assert all([a.equals(b) for a, b in zip(result, expected)])


@pytest.mark.parametrize(
    "attr", ["difference", "symmetric_difference", "union", "intersection"]
)
def test_vector_binary_geo(attr):
    quads = []
    while len(quads) < 10:
        geom = shapely.geometry.Polygon(
            [(random.random(), random.random()) for i in range(4)]
        )
        if geom.is_valid:
            quads.append(geom)

    Q = from_shapely(quads)

    result = getattr(T, attr)(Q)
    expected = [getattr(t, attr)(q) for t, q in zip(triangles, quads)]

    assert all([a.equals(b) for a, b in zip(result, expected)])


@pytest.mark.parametrize(
    "attr", ["difference", "symmetric_difference", "union", "intersection"]
)
def test_binary_geo(attr):
    quads = []
    while len(quads) < 1:
        geom = shapely.geometry.Polygon(
            [(random.random(), random.random()) for i in range(4)]
        )
        if geom.is_valid:
            quads.append(geom)

    q = quads[0]

    T = from_shapely(triangles)

    result = getattr(T, attr)(q)
    expected = [getattr(t, attr)(q) for t in triangles]

    assert all([a.equals(b) for a, b in zip(result, expected)])


@pytest.mark.parametrize(
    "attr",
    [
        #'is_closed',
        "is_valid",
        "is_empty",
        "is_simple",
        "has_z",
        "is_ring",
    ],
)
def test_unary_predicates(attr):
    result = getattr(T, attr)
    expected = [getattr(t, attr) for t in triangles]

    assert result.tolist() == expected


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


@pytest.mark.xfail(reason="We don't yet clean up memory well")
def test_clean_up_on_gc():
    psutil = pytest.importorskip("psutil")
    proc = psutil.Process()
    mem_1 = proc.memory_info().rss
    x = np.random.random(1000000)
    y = np.random.random(1000000)
    nb = x.nbytes
    mem_2 = proc.memory_info().rss - mem_1
    assert mem_2 >= x.nbytes + y.nbytes

    points = points_from_xy(x, y)
    mem_3 = proc.memory_info().rss - mem_1
    assert mem_3 >= (x.nbytes + y.nbytes) * 2

    del x, y
    import gc

    gc.collect()
    mem_4 = proc.memory_info().rss - mem_1
    assert mem_4 < mem_3

    points2 = points[::2]
    mem_5 = proc.memory_info().rss - mem_1
    assert mem_5 < mem_4 + nb  # at most a small increase

    del points
    import gc

    gc.collect()
    mem_6 = proc.memory_info().rss - mem_1
    assert mem_6 < mem_5 + nb  # still holding onto most of the data

    del points2
    import gc

    gc.collect()
    time.sleep(0.1)
    mem_7 = proc.memory_info().rss - mem_1
    assert mem_7 <= mem_5
    assert proc.memory_info().rss - mem_1 < nb


def test_dir():
    assert "contains" in dir(P)
    assert "data" in dir(P)


def test_chaining():
    assert T.contains(T.centroid).all()


@pytest.mark.parametrize("cap_style", [CAP_STYLE.round, CAP_STYLE.square])
@pytest.mark.parametrize("join_style", [JOIN_STYLE.round, JOIN_STYLE.bevel])
@pytest.mark.parametrize("resolution", [16, 25])
def test_buffer(resolution, cap_style, join_style):
    expected = [
        p.buffer(0.1, resolution=resolution, cap_style=cap_style, join_style=join_style)
        for p in points
    ]
    result = P.buffer(
        0.1, resolution=resolution, cap_style=cap_style, join_style=join_style
    )

    assert all(a.equals(b) for a, b in zip(expected, result))


@pytest.mark.parametrize("attr", ["area", "length"])
def test_vector_float(attr):
    triangles = [
        shapely.geometry.Polygon([(random.random(), random.random()) for i in range(3)])
        for _ in range(100)
    ]

    T = from_shapely(triangles)

    result = getattr(T, attr)
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float

    expected = [getattr(tri, attr) for tri in triangles]

    assert result.tolist() == expected


@pytest.mark.parametrize("attr", ["distance"])
def test_binary_vector_float(attr):
    result = getattr(P[: len(T)], attr)(T)
    expected = [getattr(t, attr)(p) for t, p in zip(triangles, points)]

    assert list(result) == expected


@pytest.mark.parametrize("attr", ["distance"])
def test_binary_float(attr):
    p = points[0]
    result = getattr(T, attr)(p)
    expected = [getattr(t, attr)(p) for t in triangles]

    assert list(result) == expected


def test_project():
    lines = [
        shapely.geometry.LineString(
            [(random.random(), random.random()) for _ in range(2)]
        )
        for _ in range(20)
    ]
    L = from_shapely(lines)

    result = L.project(P)
    expected = [l.project(p) for p, l in zip(points, lines)]


def test_serialize_deserialize():
    ba, sizes = serialize(T.data)
    vec2 = GeometryArray(deserialize(ba, sizes))

    assert (T.data != vec2.data).all()
    assert T.equals(vec2).all()


def test_pickle():
    import pickle

    T2 = pickle.loads(pickle.dumps(T))

    assert (T.data != T2.data).all()
    assert T.equals(T2).all()


@pytest.mark.parametrize(
    "predicate",
    [
        "contains",
        "covers",
        "crosses",
        "disjoint",
        "intersects",
        "overlaps",
        "touches",
        "within",
    ],
)
def test_sjoin(predicate):
    left, right = cysjoin(T.data, P.data, predicate)

    assert isinstance(left, np.ndarray)
    assert isinstance(right, np.ndarray)
    assert left.dtype == T.data.dtype
    assert right.dtype == T.data.dtype
    assert left.shape == right.shape
    (n,) = left.shape
    assert n < (len(T) * len(P))

    for i, j in zip(left, right):
        left = triangles[i]
        right = points[j]
        assert getattr(left, predicate)(right)


def test_raise_on_bad_sizes():
    with pytest.raises(ValueError) as info:
        T.contains(P)

    assert "lengths" in str(info.value).lower()
    assert "10" in str(info.value)
    assert "20" in str(info.value)


def test_types():
    cat = T.geom_type
    assert list(cat) == ["Polygon"] * len(T)


def test_null_mixed_types():
    geoms = [
        shapely.geometry.Polygon([(0, 0), (0, 1), (1, 1)]),
        None,
        shapely.geometry.Point(0, 1),
    ]

    G = from_shapely(geoms)

    cat = G.geom_type
    assert list(cat) == ["Polygon", np.nan, "Point"]


def test_unary_union():
    geoms = [
        shapely.geometry.Polygon([(0, 0), (0, 1), (1, 1)]),
        shapely.geometry.Polygon([(0, 0), (1, 0), (1, 1)]),
    ]
    G = from_shapely(geoms)
    u = G.unary_union

    expected = shapely.geometry.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    assert u.equals(expected)


def test_coords():
    L = T.exterior.coords
    assert L == [tuple(t.exterior.coords) for t in triangles]


def test_fill():
    p = shapely.geometry.Point(1, 2)
    P2 = P._fill([0, 3], p)
    assert P2[0].equals(p)
    assert P2[3].equals(p)
    with pytest.raises(TypeError) as info:
        P._fill([1, 2], 123)

    assert "123" in str(info.value)
