import numpy as np

from shapely import geometry

from geopandas import GeoSeries, points_from_xy
from geopandas.array import from_shapely


def make_grid(
    polygon, cell_size, offset=(0, 0), crs=None, what="polygons", cell_type="square",
):
    #  TODO Also considere geoderies and geodataframes as possible inputs.
    """Provides the grid-cell centers, corners, or polygons of a square or hexagonal grid.

        Parameters
        ------------
        polygon : geometry.Polygon
            Polygon within its boundaries the grid is created
        cell_size : float
            Side length of the square or the hexagon
        offset : tuple
            x, y shift of the grid realtive to its original position.
        what : str, one of "centers", "corners", "polygons"
            return type
        cell_type : str, one of "square", "hexagon"
            grid type

        Returns
        -------
        GeoSeries
            The returned GeoSeries contains the grid-cell centers, corners, or
            polygons.

        Examples
        --------
        Cover a polygon (the South American continent) with a square grid:
        >>> world = geopandas.read_file(
        ...     geopandas.datasets.get_path('naturalearth_lowres'))
        >>> south_america = world[world['continent'] == "South America"]
        # TODO Finalize example

        """

    # TODO Consider adding `n` as a parameter to stick with R implementation.
    # However, n introduces redundancy.
    # TODO Enable GeoSerie and GeoDataFrame as input

    # Run basic checks
    _basic_checks(polygon, cell_size, offset, what, cell_type)

    computed_geometry = None

    bounds = np.array(polygon.bounds)

    if cell_type == "square":
        x_coords_corn = np.arange(
            bounds[0] + offset[0], bounds[2] + cell_size + offset[0], cell_size
        )
        y_coords_corn = np.arange(
            bounds[1] + offset[1], bounds[3] + cell_size + offset[1], cell_size
        )
        xv, yv = np.meshgrid(x_coords_corn, y_coords_corn)

        if what == "corners":
            sq_corners_np = np.array([xv, yv]).T.reshape(-1, 2)
            computed_geometry = points_from_xy(sq_corners_np[:, 0], sq_corners_np[:, 1])

        elif what == "centers":
            sq_centers_np = (
                np.array([xv[:-1, :-1], yv[:-1, :-1]]).T.reshape(-1, 2) + cell_size / 2
            )
            computed_geometry = points_from_xy(sq_centers_np[:, 0], sq_centers_np[:, 1])

        elif what == "polygons":
            # Extracting specific corners of all square grid cells.
            bottom_left_corners = np.array([xv[:-1, :-1], yv[:-1, :-1]]).T.reshape(
                -1, 1, 2
            )
            bottom_right_corners = np.array([xv[1:, 1:], yv[:-1, :-1]]).T.reshape(
                -1, 1, 2
            )
            top_right_corners = np.array([xv[1:, 1:], yv[1:, 1:]]).T.reshape(-1, 1, 2)
            top_left_corners = np.array([xv[:-1, :-1], yv[1:, 1:]]).T.reshape(-1, 1, 2)

            sq_coords = np.concatenate(
                (
                    bottom_left_corners,
                    bottom_right_corners,
                    top_right_corners,
                    top_left_corners,
                ),
                axis=1,
            )
            # TODO Could a pygeos solution be used to replace below expression
            sq_polygons_np = np.array(
                [geometry.Polygon(sq_set) for sq_set in sq_coords]
            )

            computed_geometry = from_shapely(sq_polygons_np)

        else:
            raise ValueError("Invalid value for parameter `what`")

    elif cell_type == "hexagon":
        # Create rectangular meshgrid containing both centers and cornes
        # of the hexagonal grid
        dx = cell_size
        dy = cell_size * np.sqrt(3) / 2

        x_coords = np.arange(
            bounds[0] + offset[0], bounds[2] + 3 * cell_size + offset[0], dx
        )
        y_coords = np.arange(
            bounds[1] - np.sqrt(3) / 2 * cell_size + offset[1],
            bounds[3] + np.sqrt(3) * cell_size + offset[1],
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
            computed_geometry = points_from_xy(
                hex_centers_np[:, 0], hex_centers_np[:, 1]
            )

        elif what == "corners":
            # The inverted center mask is the corner mask
            mask_corners = np.invert(mask_center)
            hex_corners_np = np.array([xv[mask_corners], yv[mask_corners]]).T.reshape(
                -1, 2
            )
            computed_geometry = points_from_xy(
                hex_corners_np[:, 0], hex_corners_np[:, 1]
            )

        elif what == "polygons":
            hex_coords_a = _hex_polygon_corners(xv, yv, (0, 1))
            hex_coords_b = _hex_polygon_corners(xv, yv, (2, 0))

            hex_coords = np.concatenate((hex_coords_a, hex_coords_b), axis=0)
            hex_polygons = [geometry.Polygon(hex_set) for hex_set in hex_coords]

            hex_polygons_np = np.array(hex_polygons)
            computed_geometry = from_shapely(hex_polygons_np)

        else:
            raise ValueError("Invalid value for parameter `what`")
    else:
        raise ValueError("Invalid value for parameter `cell_type`")

    return GeoSeries(computed_geometry[computed_geometry.intersects(polygon)])


def _hex_polygon_corners(xv, yv, index_0=(0, 0)):
    """Groups hexagon corners that belong to the same grid cell.

    Parameters
    ------------
    xv : np.array
        meshgrid containing x-values of all centers and corners of the hexgon grid
    yv : np.array
        meshgrid containing y-values of all centers and corners of the hexgon grid
    index_0 : tuple
        indicates index of the meshgrid, where polygon corner grouping starts

    Returns
    -------
    np.array
        The array has the shape (n, 6, 2) where n is the number of grid cells,
        6 corresponds to the hexagon corners and 2 to the x,y coordinates.

    """

    # Input checks
    if xv.shape != yv.shape:
        raise ValueError(
            f"`xv` and `yv` must have same shape, got {xv.shape} and {yv.shape}, respectively"
        )

    if len(index_0) != 2:
        raise ValueError(f"`index_0` must have length 2, got {len(index_0)}")

    if index_0[0] < 0 or index_0[0] >= xv.shape[1]:
        raise ValueError("`index_0[0]` exceeds dimension of `xv`, `yv`")

    if index_0[1] < 0 or index_0[1] >= xv.shape[0]:
        raise ValueError("`index_0[1]` exceeds dimension of `xv`, `yv`")

    i_x0 = index_0[0]
    i_y0 = index_0[1]

    # Determine the maximum index such that only complete hexagons
    # with 6 corners are grouped.
    if (i_y0 % 2) == 0:
        x_corr = 1
    else:
        x_corr = 0

    # Labeling of hexagon corners assumes that hexagon stands on a flat side.
    # Middle right corner of the hexagon
    n_poly_x = xv[(i_y0 + 1) :: 2, (i_x0 + 2 - x_corr) :: 3].shape[1]
    max_i_x = i_x0 + n_poly_x * 3 - 1
    # Top right corner of the hexagon
    n_poly_y = xv[(i_y0 + 2) :: 2, (i_x0 + 1) :: 3].shape[0]
    max_i_y = i_y0 + n_poly_y * 2

    # Extracting specific corners of all hexagon grid cells.
    bottom_left_corners = np.array(
        [xv[i_y0:max_i_y:2, i_x0:max_i_x:3], yv[i_y0:max_i_y:2, i_x0:max_i_x:3],]
    ).T.reshape(-1, 1, 2)

    bottom_right_corners = np.array(
        [
            xv[i_y0:max_i_y:2, (i_x0 + 1) : max_i_x : 3],
            yv[i_y0:max_i_y:2, (i_x0 + 1) : max_i_x : 3],
        ]
    ).T.reshape(-1, 1, 2)

    middle_right_corners = np.array(
        [
            xv[(i_y0 + 1) : max_i_y : 2, (i_x0 + 2 - x_corr) :: 3],
            yv[(i_y0 + 1) : max_i_y : 2, (i_x0 + 2 - x_corr) :: 3],
        ]
    ).T.reshape(-1, 1, 2)

    top_right_corners = np.array(
        [
            xv[(i_y0 + 2) :: 2, (i_x0 + 1) : max_i_x : 3],
            yv[(i_y0 + 2) :: 2, (i_x0 + 1) : max_i_x : 3],
        ]
    ).T.reshape(-1, 1, 2)

    top_left_corners = np.array(
        [xv[(i_y0 + 2) :: 2, i_x0:max_i_x:3], yv[(i_y0 + 2) :: 2, i_x0:max_i_x:3],]
    ).T.reshape(-1, 1, 2)

    middle_left_corners = np.array(
        [
            xv[(i_y0 + 1) : max_i_y : 2, (i_x0 - x_corr) : max_i_x : 3],
            yv[(i_y0 + 1) : max_i_y : 2, (i_x0 - x_corr) : max_i_x : 3],
        ]
    ).T.reshape(-1, 1, 2)

    hex_coords = np.concatenate(
        (
            bottom_left_corners,
            bottom_right_corners,
            middle_right_corners,
            top_right_corners,
            top_left_corners,
            middle_left_corners,
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

    # TODO Check for offset beyond polygon bounds

    if not isinstance(polygon, geometry.Polygon):
        raise ValueError(
            f"`polygon` should be shapely.geometry.Polygon, got {type(polygon)}"
        )

    if cell_size <= 0:
        raise ValueError(f"`cell_size` should be positive, got {cell_size}")

    allowed_what = ["centers", "corners", "polygons"]
    if what not in allowed_what:
        raise ValueError(f'`what` was "{what}" but is expected to be in {allowed_what}')

    allowed_cell_type = ["square", "hexagon"]
    if cell_type not in allowed_cell_type:
        raise ValueError(
            f'`cell_type` was "{cell_type}" but is expected to be in {allowed_cell_type}'
        )
