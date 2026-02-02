import numpy as np

from shapely.geometry import MultiPolygon, Point, Polygon

from geopandas import GeoDataFrame, GeoSeries
from geopandas.tools.make_grid import make_grid

import pytest
from geopandas.testing import assert_geoseries_equal


@pytest.fixture
def square():
    """Square polygon with side length 2"""
    return Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])


@pytest.fixture
def multi_polygon():
    polygon_1 = Polygon([(0.1, 0.1), (0.9, 0), (0.9, 0.9), (0, 0.1)])
    polygon_2 = Polygon([(1.1, 1.1), (1.9, 1.1), (1.9, 1.9)])
    return MultiPolygon([polygon_1, polygon_2])


@pytest.fixture
def geo_series():
    polygon_1 = Polygon([(0.1, 0.1), (0.9, 0), (0.9, 0.9), (0, 0.1)])
    polygon_2 = Polygon([(1.1, 1.1), (1.9, 1.1), (1.9, 1.9)])
    point_1 = Point((1.5, 0.4))
    return GeoSeries([polygon_1, polygon_2, point_1], crs="EPSG:4326")


@pytest.fixture
def geo_dataframe():
    polygon_1 = Polygon([(0.1, 0.1), (0.9, 0), (0.9, 0.9), (0, 0.1)])
    polygon_2 = Polygon([(1.1, 1.1), (1.9, 1.1), (1.9, 1.9)])
    point_1 = Point((1.5, 0.4))

    d = {
        "col1": ["poly1", "poly2", "point1"],
        "geometry": [polygon_1, polygon_2, point_1],
    }
    return GeoDataFrame(d, crs="EPSG:4326")


@pytest.fixture
def neg_square():
    return Polygon([(0, 0), (-2, 0), (-2, -2), (0, -2)])


@pytest.fixture
def excotic_polygon():
    return Polygon(
        [(0, 0), (0.75, 0), (1, 0.5), (1.25, 0), (2, 0), (2, 0.9), (1, 2), (0, 2)]
    )


class TestBasicChecks:
    def test_inputs_neg_cell_size(self, square):
        with pytest.raises(ValueError):
            make_grid(square, -3)

    def test_inputs_polygon(self):
        polygon = Point(0.0, 0.0)
        with pytest.raises(TypeError):
            make_grid(polygon, 1)

    def test_inputs_empty_polygon(self):
        with pytest.raises(ValueError):
            empty_polygon = Polygon()
            make_grid(empty_polygon, 1)

    def test_inputs_empty_gdf(self):
        with pytest.raises(ValueError):
            empty_gdf = GeoDataFrame()
            make_grid(empty_gdf, 1)

    def test_inputs_empty_gs(self):
        with pytest.raises(ValueError):
            empty_gs = GeoSeries()
            make_grid(empty_gs, 1)

    def test_inputs_cell_type(self, square):
        with pytest.raises(ValueError):
            make_grid(square, 1, cell_type="circle")

    def test_inputs_what(self, square):
        with pytest.raises(ValueError):
            make_grid(square, 1, what="corner")


class TestMakeGridSquare:
    def test_square_centers(self, square):
        cell_size = 1
        out = make_grid(
            square, cell_size, what="centers", cell_type="square", intersect=False
        )
        exp_out = GeoSeries(
            [Point(0.5, 0.5), Point(0.5, 1.5), Point(1.5, 0.5), Point(1.5, 1.5)]
        )
        assert_geoseries_equal(out, exp_out)

    def test_neg_square_centers(self, neg_square):
        cell_size = 1
        out = make_grid(
            neg_square, cell_size, what="centers", cell_type="square", intersect=False
        )
        exp_out = GeoSeries(
            [
                Point(-1.5, -1.5),
                Point(-1.5, -0.5),
                Point(-0.5, -1.5),
                Point(-0.5, -0.5),
            ]
        )
        assert_geoseries_equal(out, exp_out)

    def test_square_centers_multipolygon(self, multi_polygon):
        cell_size = 1
        out = make_grid(
            multi_polygon,
            cell_size,
            what="centers",
            cell_type="square",
            intersect=False,
        )
        exp_out = GeoSeries(
            [Point(0.5, 0.5), Point(0.5, 1.5), Point(1.5, 0.5), Point(1.5, 1.5)]
        )
        assert_geoseries_equal(out, exp_out)

    def test_square_centers_multipolygon_intersect(self, multi_polygon):
        cell_size = 1
        out = make_grid(multi_polygon, cell_size, what="centers", cell_type="square")
        exp_out = GeoSeries([Point(0.5, 0.5), Point(1.5, 1.5)])
        assert_geoseries_equal(out, exp_out)

    def test_exotic_square_centers(self, exotic_polygon):
        cell_size = 1
        out = make_grid(exotic_polygon, cell_size, what="centers", cell_type="square")
        exp_out = GeoSeries([Point(0.5, 0.5), Point(0.5, 1.5), Point(1.5, 0.5)])
        assert_geoseries_equal(out, exp_out)

    def test_square_centers_geoseries(self, geo_series):
        cell_size = 1
        out = make_grid(geo_series, cell_size, what="centers", cell_type="square")
        exp_out = GeoSeries([Point(0.5, 0.5), Point(1.5, 1.5)], crs="EPSG:4326")
        assert_geoseries_equal(out, exp_out)

    def test_square_centers_geodataframe(self, geo_dataframe):
        cell_size = 1
        out = make_grid(geo_dataframe, cell_size, what="centers", cell_type="square")
        exp_out = GeoSeries([Point(0.5, 0.5), Point(1.5, 1.5)], crs="EPSG:4326")
        assert_geoseries_equal(out, exp_out)

    def test_square_corners(self, square):
        cell_size = 2
        out = make_grid(
            square, cell_size, what="corners", cell_type="square", intersect=False
        )
        exp_out = GeoSeries([Point(0, 0), Point(0, 2), Point(2, 0), Point(2, 2)])
        assert_geoseries_equal(out, exp_out)

    def test_square_polygons(self, square):
        cell_size = 1
        out = make_grid(
            square, cell_size, what="polygons", cell_type="square", intersect=False
        )
        exp_out = GeoSeries(
            [
                Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                Polygon([(0, 1), (1, 1), (1, 2), (0, 2)]),
                Polygon([(1, 0), (2, 0), (2, 1), (1, 1)]),
                Polygon([(1, 1), (2, 1), (2, 2), (1, 2)]),
            ]
        )
        assert_geoseries_equal(out, exp_out)

    def test_square_polygons_multipolygon(self, multi_polygon):
        cell_size = 1
        out = make_grid(multi_polygon, cell_size, what="polygons", cell_type="square")
        exp_out = GeoSeries(
            [
                Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                Polygon([(1, 1), (2, 1), (2, 2), (1, 2)]),
            ]
        )
        assert_geoseries_equal(out, exp_out)

    def test_square_polygons_geoseries(self, geo_series):
        cell_size = 1
        out = make_grid(geo_series, cell_size, what="polygons", cell_type="square")
        exp_out = GeoSeries(
            [
                Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                Polygon([(1, 0), (2, 0), (2, 1), (1, 1)]),
                Polygon([(1, 1), (2, 1), (2, 2), (1, 2)]),
            ],
            crs="EPSG:4326",
        )
        assert_geoseries_equal(out, exp_out)

    def test_square_polygons_geodataframe(self, geo_dataframe):
        cell_size = 1
        out = make_grid(geo_dataframe, cell_size, what="polygons", cell_type="square")
        exp_out = GeoSeries(
            [
                Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                Polygon([(1, 0), (2, 0), (2, 1), (1, 1)]),
                Polygon([(1, 1), (2, 1), (2, 2), (1, 2)]),
            ],
            crs="EPSG:4326",
        )
        assert_geoseries_equal(out, exp_out)

    def test_square_cellsize_too_large(self, square):
        cell_size = 5
        with pytest.warns(UserWarning):
            out = make_grid(
                square, cell_size, what="centers", cell_type="square", intersect=False
            )
        exp_out = GeoSeries([Point(2.5, 2.5)])
        assert_geoseries_equal(out, exp_out)

    def test_square_cellsize_too_large_intersect(self, square):
        cell_size = 5
        with pytest.warns(UserWarning):
            out = make_grid(square, cell_size, what="centers", cell_type="square")
        exp_out = GeoSeries()
        assert_geoseries_equal(out, exp_out)

    def test_square_centers_offset(self, square):
        cell_size = 1
        out = make_grid(
            square,
            cell_size,
            what="centers",
            cell_type="square",
            offset=(0.1, 0.1),
            intersect=False,
        )
        exp_out = GeoSeries(
            [Point(0.6, 0.6), Point(0.6, 1.6), Point(1.6, 0.6), Point(1.6, 1.6)]
        )
        assert_geoseries_equal(out, exp_out)


class TestMakeGridHexagon:
    def test_hexagon_centers(self, square):
        cell_size = 1
        out = make_grid(
            square, cell_size, what="centers", cell_type="hexagon", intersect=False
        )

        exp_out = GeoSeries(
            [
                Point(0.5, -0.5 * np.sqrt(3)),
                Point(2, 0),
                Point(0.5, 0.5 * np.sqrt(3)),
                Point(2, np.sqrt(3)),
                Point(0.5, 1.5 * np.sqrt(3)),
                Point(2, 2 * np.sqrt(3)),
            ]
        )
        assert_geoseries_equal(out, exp_out, check_less_precise=True)

    def test_hexagon_centers_offset(self, square):
        cell_size = 1
        out = make_grid(
            square,
            cell_size,
            what="centers",
            cell_type="hexagon",
            offset=(0.1, 0.1),
            intersect=False,
        )
        exp_out = GeoSeries(
            [
                Point(0.5, -0.5 * np.sqrt(3)),
                Point(2, 0),
                Point(0.5, 0.5 * np.sqrt(3)),
                Point(2, np.sqrt(3)),
                Point(0.5, 1.5 * np.sqrt(3)),
                Point(2, 2 * np.sqrt(3)),
            ]
        ).translate(xoff=0.1, yoff=0.1)

        assert_geoseries_equal(out, exp_out, check_less_precise=True)

    def test_hexagon_centers_intersect(self, square):
        cell_size = 1
        out = make_grid(square, cell_size, what="centers", cell_type="hexagon")
        exp_out = GeoSeries(
            [Point(2, 0), Point(0.5, np.sqrt(3) / 2), Point(2, np.sqrt(3))]
        )
        assert_geoseries_equal(out, exp_out, check_less_precise=True)

    def test_hexagon_corners_cs_larger_bb(self, square):
        cell_size = 3
        with pytest.warns(UserWarning):
            out = make_grid(
                square, cell_size, what="corners", cell_type="hexagon", intersect=False
            )
        exp_out = GeoSeries(
            [
                Point(-1.5, -1.5 * np.sqrt(3)),
                Point(4.5, -1.5 * np.sqrt(3)),
                Point(0, 0),
                Point(3, 0),
                Point(-1.5, 1.5 * np.sqrt(3)),
                Point(4.5, 1.5 * np.sqrt(3)),
                Point(0, 3 * np.sqrt(3)),
                Point(3, 3 * np.sqrt(3)),
            ]
        )
        assert_geoseries_equal(out, exp_out, check_less_precise=True)

    def test_hexagon_corners_cs_eq_bb(self, square):
        cell_size = 2
        out = make_grid(
            square, cell_size, what="corners", cell_type="hexagon", intersect=False
        )
        exp_out = GeoSeries(
            [
                Point(-1, -np.sqrt(3)),
                Point(3, -np.sqrt(3)),
                Point(0, 0),
                Point(2, 0),
                Point(-1, np.sqrt(3)),
                Point(3, np.sqrt(3)),
                Point(0, 2 * np.sqrt(3)),
                Point(2, 2 * np.sqrt(3)),
                Point(-1, 3 * np.sqrt(3)),
                Point(3, 3 * np.sqrt(3)),
            ]
        )
        assert_geoseries_equal(out, exp_out, check_less_precise=True)

    def test_hexagon_corners_cs_smaller_bb(self, square):
        cell_size = 1
        out = make_grid(
            square, cell_size, what="corners", cell_type="hexagon", intersect=False
        )
        exp_out = GeoSeries(
            [
                Point(-0.5, -0.5 * np.sqrt(3)),
                Point(1.5, -0.5 * np.sqrt(3)),
                Point(2.5, -0.5 * np.sqrt(3)),
                Point(0, 0),
                Point(1, 0),
                Point(3, 0),
                Point(-0.5, 0.5 * np.sqrt(3)),
                Point(1.5, 0.5 * np.sqrt(3)),
                Point(2.5, 0.5 * np.sqrt(3)),
                Point(0, np.sqrt(3)),
                Point(1, np.sqrt(3)),
                Point(3, np.sqrt(3)),
                Point(-0.5, 1.5 * np.sqrt(3)),
                Point(1.5, 1.5 * np.sqrt(3)),
                Point(2.5, 1.5 * np.sqrt(3)),
                Point(0, 2 * np.sqrt(3)),
                Point(1, 2 * np.sqrt(3)),
                Point(3, 2 * np.sqrt(3)),
            ]
        )
        assert_geoseries_equal(out, exp_out, check_less_precise=True)

    def test_hexagon_corners_intersect(self, square):
        cell_size = 1
        out = make_grid(square, cell_size, what="corners", cell_type="hexagon")
        exp_out = GeoSeries(
            [
                Point(0, 0),
                Point(1, 0),
                Point(1.5, np.sqrt(3) / 2),
                Point(0, np.sqrt(3)),
                Point(1, np.sqrt(3)),
            ]
        )
        assert_geoseries_equal(out, exp_out, check_less_precise=True)

    def test_hexagon_polygons(self, square):
        cell_size = 3
        with pytest.warn(UserWarning):
            out = make_grid(
                square, cell_size, what="polygons", cell_type="hexagon", intersect=False
            )
        exp_out = GeoSeries(
            [
                Polygon(
                    [
                        (0, 0),
                        (3, 0),
                        (4.5, 1.5 * np.sqrt(3)),
                        (3, 3 * np.sqrt(3)),
                        (0, 3 * np.sqrt(3)),
                        (-1.5, 1.5 * np.sqrt(3)),
                    ]
                ),
            ]
        )

        assert_geoseries_equal(out, exp_out, check_less_precise=True)

    def test_hexagon_polygons_intersect(self, square):
        cell_size = 1.5
        out = make_grid(square, cell_size, what="polygons", cell_type="hexagon")
        exp_out = GeoSeries(
            [
                Polygon(
                    [
                        (0, 0),
                        (1.5, 0),
                        (2.25, 1.5 * np.sqrt(3) / 2),
                        (1.5, 1.5 * np.sqrt(3)),
                        (0, 1.5 * np.sqrt(3)),
                        (-0.75, 1.5 * np.sqrt(3) / 2),
                    ]
                ),
                Polygon(
                    [
                        (2.25, -1.5 * np.sqrt(3) / 2),
                        (3.75, -1.5 * np.sqrt(3) / 2),
                        (4.5, 0),
                        (3.75, 1.5 * np.sqrt(3) / 2),
                        (2.25, 1.5 * np.sqrt(3) / 2),
                        (1.5, 0),
                    ]
                ),
                Polygon(
                    [
                        (2.25, 1.5 * np.sqrt(3) / 2),
                        (3.75, 1.5 * np.sqrt(3) / 2),
                        (4.5, 1.5 * np.sqrt(3)),
                        (3.75, 2.25 * np.sqrt(3)),
                        (2.25, 2.25 * np.sqrt(3)),
                        (1.5, 1.5 * np.sqrt(3)),
                    ]
                ),
            ]
        )

        assert_geoseries_equal(out, exp_out, check_less_precise=True)
