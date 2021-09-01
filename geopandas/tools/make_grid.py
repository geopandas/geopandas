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
    polygon,
    cell_size,
    offset=(0, 0),
    crs=None,
    what="polygons",
    cell_type="square",
):
    # TODO Consider adding `n` as a parameter to stick with R implementation.
    # However, n introduces redundancy.

    # Run basic checks
    _basic_checks(polygon, cell_size, offset, what, cell_type)


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
