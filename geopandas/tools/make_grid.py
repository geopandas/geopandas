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

    elif cell_type == "hexagon":

        # Create rectangular meshgrid containing both centers and cornes
        # of the hexagonal grid
        dx = cell_size
        dy = cell_size * np.sqrt(3) / 2

        x_coords = np.arange(bounds[0], bounds[2] + 2 * cell_size, dx)
        y_coords = np.arange(
            bounds[1] - np.sqrt(3) / 2 * cell_size,
            bounds[3] + np.sqrt(3) * cell_size,
            dy,
        )
        xv, yv = np.meshgrid(x_coords, y_coords)

        # Shift every second row to transform the rectangular into a hexagonal grid
        xv[::2, :] = xv[::2, :] - dx / 2

        # Create mask to select only the centers
        mask_center = np.zeros_like(xv, dtype=bool)
        mask_center[1::2, 2::3] = True
        mask_center[2::2, 1::3] = True

        if what == "centers":
            hex_centers_np = np.array([xv[mask_center], yv[mask_center]]).T.reshape(
                -1, 2
            )
            hex_centers = points_from_xy(hex_centers_np[:, 0], hex_centers_np[:, 1])
            return GeoSeries(hex_centers[hex_centers.intersects(polygon)])

        elif what == "corners":
            # The inverted center mask is the corner mask
            mask_corners = np.invert(mask_center)
            hex_corners_np = np.array([xv[mask_corners], yv[mask_corners]]).T.reshape(
                -1, 2
            )
            hex_corners = points_from_xy(hex_corners_np[:, 0], hex_corners_np[:, 1])
            return GeoSeries(hex_corners[hex_corners.intersects(polygon)])

        else:
            hex_coords_a = _hex_polygon_corners(xv, yv, (0, 1))
            hex_coords_b = _hex_polygon_corners(xv, yv, (2, 0))

            hex_coords = np.concatenate((hex_coords_a, hex_coords_b), axis=0)
            hex_polygons = [geometry.Polygon(hex_set) for hex_set in hex_coords]

            hex_polygons_np = np.array(hex_polygons)
            hex_polygons = from_shapely(hex_polygons_np)

            return GeoSeries(hex_polygons[hex_polygons.intersects(polygon)])


def _hex_polygon_corners(xv, yv, index_0=(0, 0)):
    i_x0 = index_0[0]
    i_y0 = index_0[1]

    # Determine the maximum index such that only hexagons
    # with 6 corners are selected.
    if (i_y0 % 2) == 0:
        x_corr = 1
    else:
        x_corr = 0

    # Middle right corner of the hexagon
    n_poly_x = xv[(i_y0 + 1) :: 2, (i_x0 + 2 - x_corr) :: 3].shape[1]
    max_i_x = i_x0 + n_poly_x * 3 - 1
    # Top right corner of the hexagon
    n_poly_y = xv[(i_y0 + 2) :: 2, (i_x0 + 1) :: 3].shape[0]
    max_i_y = i_y0 + n_poly_y * 2

    hex_coords = np.concatenate(
        (
            # Bottom left corner of the hexagon
            np.array(
                [
                    xv[i_y0:max_i_y:2, i_x0:max_i_x:3],
                    yv[i_y0:max_i_y:2, i_x0:max_i_x:3],
                ]
            ).T.reshape(-1, 1, 2),
            # Bottom right corner of the hexagon
            np.array(
                [
                    xv[i_y0:max_i_y:2, (i_x0 + 1) :: 3],
                    yv[i_y0:max_i_y:2, (i_x0 + 1) :: 3],
                ]
            ).T.reshape(-1, 1, 2),
            # Middle right corner of the hexagon
            np.array(
                [
                    xv[(i_y0 + 1) : max_i_y : 2, (i_x0 + 2 - x_corr) :: 3],
                    yv[(i_y0 + 1) : max_i_y : 2, (i_x0 + 2 - x_corr) :: 3],
                ]
            ).T.reshape(-1, 1, 2),
            # Top right corner of the hexagon
            np.array(
                [
                    xv[(i_y0 + 2) :: 2, (i_x0 + 1) :: 3],
                    yv[(i_y0 + 2) :: 2, (i_x0 + 1) :: 3],
                ]
            ).T.reshape(-1, 1, 2),
            # Top left corner of the hexagon
            np.array(
                [
                    xv[(i_y0 + 2) :: 2, i_x0:max_i_x:3],
                    yv[(i_y0 + 2) :: 2, i_x0:max_i_x:3],
                ]
            ).T.reshape(-1, 1, 2),
            # Middle left corner of the hexagon
            np.array(
                [
                    xv[(i_y0 + 1) : max_i_y : 2, (i_x0 - x_corr) : max_i_x : 3],
                    yv[(i_y0 + 1) : max_i_y : 2, (i_x0 - x_corr) : max_i_x : 3],
                ]
            ).T.reshape(-1, 1, 2),
        ),
        axis=1,
    )

    return hex_coords


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
