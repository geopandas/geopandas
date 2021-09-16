from distutils.version import LooseVersion
from geopandas.tools.make_grid import make_grid

import numpy as np

from shapely import geometry

import geopandas
from geopandas import GeoDataFrame, GeoSeries
from geopandas.testing import assert_geoseries_equal

import pytest


@pytest.fixture
def square():
    """Square polygon with side length 2"""
    return geometry.Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])


@pytest.fixture
def neg_square():
    return geometry.Polygon([(0, 0), (-2, 0), (-2, -2), (0, -2)])


@pytest.fixture
def excotic_polygon():
    return geometry.Polygon(
        [(0, 0), (0.75, 0), (1, 0.5), (1.25, 0), (2, 0), (2, 0.9), (1, 2), (0, 2)]
    )


@pytest.fixture
def empty_polygon():
    return geometry.Polygon()


class TestBasicChecks:
    def test_inputs_cell_size(self):
        polygon = geometry.Polygon([(0, 0), (1, 1), (1, 0)])
        cell_size = -3
        with pytest.raises(ValueError):
            make_grid(polygon, cell_size)

    def test_inputs_polygon(self):
        polygon = geometry.Point(0.0, 0.0)
        with pytest.raises(ValueError):
            make_grid(polygon, 1)

    def test_inputs_empty_polygon(self, empty_polygon):
        with pytest.raises(ValueError):
            make_grid(empty_polygon, 1)


# TODO create shape with non rect boarders, test empty polygon


class TestMakeGridSquare:
    def test_square_centers(self, square):
        cell_size = 1
        out = make_grid(square, cell_size, what="centers", cell_type="square")
        exp_out = GeoSeries(
            [
                geometry.Point(0.5, 0.5),
                geometry.Point(0.5, 1.5),
                geometry.Point(1.5, 0.5),
                geometry.Point(1.5, 1.5),
            ]
        )
        assert_geoseries_equal(out, exp_out)

    def test_neg_square_centers(self, neg_square):
        cell_size = 1
        out = make_grid(neg_square, cell_size, what="centers", cell_type="square")
        exp_out = GeoSeries(
            [
                geometry.Point(-1.5, -1.5),
                geometry.Point(-1.5, -0.5),
                geometry.Point(-0.5, -1.5),
                geometry.Point(-0.5, -0.5),
            ]
        )
        assert_geoseries_equal(out, exp_out)

    def test_exotic_square_centers(self, excotic_polygon):
        cell_size = 1
        out = make_grid(excotic_polygon, cell_size, what="centers", cell_type="square")
        exp_out = GeoSeries(
            [
                geometry.Point(0.5, 0.5),
                geometry.Point(0.5, 1.5),
                geometry.Point(1.5, 0.5),
            ]
        )
        assert_geoseries_equal(out, exp_out)

    def test_square_corners(self, square):
        cell_size = 2
        out = make_grid(square, cell_size, what="corners", cell_type="square")
        exp_out = GeoSeries(
            [
                geometry.Point(0, 0),
                geometry.Point(0, 2),
                geometry.Point(2, 0),
                geometry.Point(2, 2),
            ]
        )
        assert_geoseries_equal(out, exp_out)

    def test_square_polygons(self, square):
        cell_size = 1
        out = make_grid(square, cell_size, what="polygons", cell_type="square")
        exp_out = GeoSeries(
            [
                geometry.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                geometry.Polygon([(0, 1), (1, 1), (1, 2), (0, 2)]),
                geometry.Polygon([(1, 0), (2, 0), (2, 1), (1, 1)]),
                geometry.Polygon([(1, 1), (2, 1), (2, 2), (1, 2)]),
            ]
        )
        assert_geoseries_equal(out, exp_out)

    def test_square_cellsize_too_large(self, square):
        cell_size = 5
        out = make_grid(square, cell_size, what="centers", cell_type="square")
        exp_out = GeoSeries([])
        assert_geoseries_equal(out, exp_out)


class TestMakeGridHexagon:
    def test_hexagon_centers(self, square):
        cell_size = 1
        out = make_grid(square, cell_size, what="centers", cell_type="hexagon")
        exp_out = GeoSeries(
            [
                geometry.Point(2, 0),
                geometry.Point(0.5, np.sqrt(3) / 2),
                geometry.Point(2, np.sqrt(3)),
            ]
        )
        assert_geoseries_equal(out, exp_out, check_less_precise=True)

    def test_hexagon_corners(self, square):
        cell_size = 1
        out = make_grid(square, cell_size, what="corners", cell_type="hexagon")
        exp_out = GeoSeries(
            [
                geometry.Point(0, 0),
                geometry.Point(1, 0),
                geometry.Point(1.5, np.sqrt(3) / 2),
                geometry.Point(0, np.sqrt(3)),
                geometry.Point(1, np.sqrt(3)),
            ]
        )
        assert_geoseries_equal(out, exp_out, check_less_precise=True)

    # TODO Add polygon Test
    def test_hexagon_polygons(self, square):
        cell_size = 1.5
        out = make_grid(square, cell_size, what="polygons", cell_type="hexagon")
        exp_out = GeoSeries(
            [
                geometry.Polygon(
                    [
                        (0, 0),
                        (1.5, 0),
                        (2.25, 1.5 * np.sqrt(3) / 2),
                        (1.5, 1.5 * np.sqrt(3)),
                        (0, 1.5 * np.sqrt(3)),
                        (-0.75, 1.5 * np.sqrt(3) / 2),
                    ]
                ),
                geometry.Polygon(
                    [
                        (2.25, -1.5 * np.sqrt(3) / 2),
                        (3.75, -1.5 * np.sqrt(3) / 2),
                        (4.5, 0),
                        (3.75, 1.5 * np.sqrt(3) / 2),
                        (2.25, 1.5 * np.sqrt(3) / 2),
                        (1.5, 0),
                    ]
                ),
                geometry.Polygon(
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
