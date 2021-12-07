from shapely.geometry import Polygon, LineString, Point
import geopandas


def test_explain():
    s = geopandas.GeoSeries(
        [
            Polygon([(0, 0), (2, 2), (0, 2)]),
            Polygon([(0, 0), (2, 2), (0, 2), (2, 2)]),
            LineString([(0, 0), (2, 2)]),
            LineString([(2, 0), (0, 2)]),
            Point(0, 1),
        ],
    )
    try:
        result = s.explain_validity()
        assert result[1] != "Valid Geometry"
    except ImportError:
        print("shapely version must great than 1.8.0")


def test_make_valid():
    s = geopandas.GeoSeries(
        [
            Polygon([(0, 0), (2, 2), (0, 2)]),
            Polygon([(0, 0), (2, 2), (0, 2), (2, 2)]),
            LineString([(0, 0), (2, 2)]),
            LineString([(2, 0), (0, 2)]),
            Point(0, 1),
        ],
    )
    try:
        sn = s.make_valid()
        result = sn.explain_validity()
        assert result[1] == "Valid Geometry"
    except ImportError:
        print("shapely version must great than 1.8.0")
