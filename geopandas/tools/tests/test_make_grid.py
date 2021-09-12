from distutils.version import LooseVersion
from geopandas.tools.make_grid import make_grid

import numpy as np

from shapely import geometry

import geopandas
from geopandas import GeoDataFrame, GeoSeries
from geopandas.testing import assert_geodataframe_equal, assert_geoseries_equal

import pytest


@pytest.fixture
def square():
    """Make a simple square polygon with side length 2"""
    return geometry.Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])


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
        print(out)
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


"""
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
        assert_geoseries_equal(out, exp_out)
"""
