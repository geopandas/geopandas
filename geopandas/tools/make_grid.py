import warnings

import pandas as pd
import numpy as np

from shapely import geometry
from shapely.geometry.polygon import Polygon

from geopandas import GeoDataFrame, GeoSeries, points_from_xy
from geopandas.array import from_shapely


# Delete again
import matplotlib.pyplot as plt


def make_grid(
    polygon, cell_size, offset=(0, 0), crs=None, what="polygons", cell_type="square",
):
    # TODO Consider adding `n` as a parameter to stick with R implementation.
    # However, n introduces redundancy.

    # Run basic checks
    _basic_checks(polygon, cell_size, offset, what, cell_type)

    bounds = np.array(polygon.bounds)

    if cell_type == "square":

        x_coords_corn = np.arange(bounds[0], bounds[2] + cell_size, cell_size)
        y_coords_corn = np.arange(bounds[1], bounds[3] + cell_size, cell_size)
        xv, yv = np.meshgrid(x_coords_corn, y_coords_corn)
        sq_corners_np = np.array([xv, yv]).T.reshape(-1, 2)

        if what == "corners":
            sq_corners = points_from_xy(sq_corners_np[:, 0], sq_corners_np[:, 1])

            return GeoSeries(sq_corners[sq_corners.intersects(polygon)])

        elif what == "centers":
            sq_centers_np = sq_corners_np + cell_size / 2
            sq_centers = points_from_xy(sq_centers_np[:, 0], sq_centers_np[:, 1])

            return GeoSeries(sq_centers[sq_centers.intersects(polygon)])

        else:
            sq_coords = np.concatenate(
                (
                    np.array([xv[:-1, :-1], yv[:-1, :-1]]).T.reshape(-1, 1, 2),
                    np.array([xv[1:, 1:], yv[:-1, :-1]]).T.reshape(-1, 1, 2),
                    np.array([xv[1:, 1:], yv[1:, 1:]]).T.reshape(-1, 1, 2),
                    np.array([xv[:-1, :-1], yv[1:, 1:]]).T.reshape(-1, 1, 2),
                ),
                axis=1,
            )

            sq_polygons = [geometry.Polygon(sq_set) for sq_set in sq_coords]

            sq_polygons_np = np.array(sq_polygons)
            sq_polygons = from_shapely(sq_polygons_np)

            return GeoSeries(sq_polygons[sq_polygons.intersects(polygon)])


def _basic_checks(polygon, cell_size, offset, what, cell_type):
    """Checks the validity of make_grid input parameters.

    `cell_size` must be larger than 0.
    `what` and `cell_type` must be a valid option.

    Parameters
    ------------
    polygon : geometry.Polygon
    cell_size : float
    offset : tuple
    what : str, one of "centers", "corners", "polygons"
        type of return
    cell_type : str, one of "square", "hexagon"
        grid type
    """

    # TODO Check for cell_size larger than poylgon boounds
    # TODO Check for offset beyond polygon bounds

    if not isinstance(polygon, geometry.Polygon):
        raise ValueError(
            "'polygon' should be shapely.geometry.Polygon, got {}".format(type(polygon))
        )

    if cell_size <= 0:
        raise ValueError("'cell_size' should be positive, got {}".format(cell_size))

    allowed_what = ["centers", "corners", "polygons"]
    if what not in allowed_what:
        raise ValueError(
            '`what` was "{}" but is expected to be in {}'.format(what, allowed_what)
        )

    allowed_cell_type = ["square", "hexagon"]
    if cell_type not in allowed_cell_type:
        raise ValueError(
            '`cell_type` was "{}" but is expected to be in {}'.format(
                cell_type, allowed_cell_type
            )
        )
