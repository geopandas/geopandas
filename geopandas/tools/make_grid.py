import warnings

import numpy as np

from shapely.geometry import MultiPolygon, Polygon

from geopandas import GeoDataFrame, GeoSeries, points_from_xy
from geopandas.array import from_shapely


def make_grid(
    input_geometry,
    cell_size,
    cell_type="square",
    what="polygons",
    offset=(0, 0),
    intersect=True,
    flat_topped=True,
):
    """Provides the centers, corners, or polygons of a square or hexagonal grid.

    The output covers the area ot the `input_geometry`. The origin of the grid is
    at the lower left corner of the bouding box of the `input_geometry`. By default,
    the grid is intersected with the `input_geometry`. Automatic intersecting can
    be avoided by adjusting the `intersect` parameter to `False`.

    If there are multiple geometries in a GeoSeries/GeoDataFrame, the grid will
    be created over the total bounds of the GeoSeries/GeoDataFrame. Subsequently,
    the grid is intersected with the individual geometries.

    Parameters
    ------------
    input_geometry : (Multi)Polygon, GeoSeries, GeoDataFrame
        Polygon within its boundaries the grid is made.
    cell_size : float
        Side length of the square or hexagonal grid cell.
    cell_type : str, one of "square", "hexagon", default "square"
        Grid type that is returned.
    what : str, one of "centers", "corners", "polygons", default "polygons"
        Grid feature that is returned.
    offset : tuple
        x, y offset of the grid realtive to lower-left corner of the input
        geometry's bounding box.
    intersect : bool, default True
        If False, the grid is not intersected with the `input_geometry`.
    flat_topped : bool, default True
        If False, the orientation of the hexagonal cells are rotated by 90 degree
        such that a corner points upwards.

    Returns
    -------
    GeoSeries
        The returned GeoSeries contains the grid-cell centers, corners, or
        polygons.

    Examples
    --------
    >>> import geopandas
    >>> world = geopandas.read_file(
    ...     geopandas.datasets.get_path('naturalearth_lowres'))
    >>> uruguay = world[world["name"] == "Uruguay"]
    >>> sq_grid = geopandas.makegrid(uruguay,3)
    >>> sq_grid
    0    POLYGON ((-58.42707 -34.95265, -55.42707 -34.9...
    1    POLYGON ((-58.42707 -31.95265, -55.42707 -31.9...
    2    POLYGON ((-55.42707 -34.95265, -52.42707 -34.9...
    3    POLYGON ((-55.42707 -31.95265, -52.42707 -31.9...
    """

    # Run basic checks
    _basic_checks(input_geometry, cell_size, offset, what, cell_type, intersect)

    output_grid = None

    if isinstance(input_geometry, (GeoDataFrame, GeoSeries)):
        bounds = np.array(input_geometry.total_bounds)
    else:
        bounds = np.array(input_geometry.bounds)

    x_dist = bounds[2] - bounds[0] - offset[0]
    y_dist = bounds[3] - bounds[1] - offset[1]

    grid_origin_x = bounds[0] + offset[0]
    grid_origin_y = bounds[1] + offset[1]

    if cell_type == "square":
        # Set corner coordinates of square grid. Grid always ends at the right/upper
        # edge of the bounding box; also if an offset is defined.
        x_coords_corn = np.arange(
            bounds[0] + offset[0], bounds[2] + cell_size, cell_size
        )
        y_coords_corn = np.arange(
            bounds[1] + offset[1], bounds[3] + cell_size, cell_size
        )
        xv, yv = np.meshgrid(x_coords_corn, y_coords_corn)

        if what == "corners":
            sq_corners_np = np.array([xv, yv]).T.reshape(-1, 2)
            output_grid = points_from_xy(sq_corners_np[:, 0], sq_corners_np[:, 1])

        elif what == "centers":
            sq_centers_np = (
                np.array([xv[:-1, :-1], yv[:-1, :-1]]).T.reshape(-1, 2) + cell_size / 2
            )

            output_grid = points_from_xy(sq_centers_np[:, 0], sq_centers_np[:, 1])

        elif what == "polygons":
            # Extracting corners of all square-grid cells.
            bt_left_corners = np.array([xv[:-1, :-1], yv[:-1, :-1]]).T.reshape(-1, 1, 2)
            bt_right_corners = np.array([xv[1:, 1:], yv[:-1, :-1]]).T.reshape(-1, 1, 2)
            top_right_corners = np.array([xv[1:, 1:], yv[1:, 1:]]).T.reshape(-1, 1, 2)
            top_left_corners = np.array([xv[:-1, :-1], yv[1:, 1:]]).T.reshape(-1, 1, 2)

            sq_coords = np.concatenate(
                (
                    bt_left_corners,
                    bt_right_corners,
                    top_right_corners,
                    top_left_corners,
                ),
                axis=1,
            )
            sq_polygons_np = np.array([Polygon(sq_set) for sq_set in sq_coords])

            output_grid = from_shapely(sq_polygons_np)

        else:
            raise ValueError("Invalid value for parameter `what`")

    elif cell_type == "hexagon":
        if not flat_topped:
            y_dist = bounds[2] - bounds[0] - offset[0]
            x_dist = bounds[3] - bounds[1] - offset[1]

        # Create rectangular meshgrid containing both the centers and the cornes
        # of the hexagonal grid
        dx = cell_size
        dy = cell_size * np.sqrt(3) / 2

        # Determine number of grid points along x/y direction such that the area of
        # any bounding box is fully covered with polygons.
        n_dx = x_dist / dx
        n_grid_point_x = int((np.ceil((n_dx - 1) / 1.5) + 2) * 1.5)
        n_dy = y_dist / dy
        n_grid_point_y = int(n_dy) + 4

        x_coords = np.arange(0, n_grid_point_x).astype(float) * dx
        y_coords = np.arange(0, n_grid_point_y).astype(float) * dy - dy
        xv, yv = np.meshgrid(x_coords, y_coords)

        # Shift every second row to transform the rectangular into a hexagonal grid
        xv[::2, :] = xv[::2, :] - (dx / 2)

        mask_center = np.zeros_like(xv, dtype=bool)
        mask_center[1::2, 2::3] = True
        mask_center[::2, 1::3] = True

        if what == "centers":
            hex_centers_np = np.array([xv[mask_center], yv[mask_center]]).T.reshape(
                -1, 2
            )
            if not flat_topped:
                hex_centers_np = hex_centers_np[:, ::-1]

            output_grid = points_from_xy(
                hex_centers_np[:, 0] + grid_origin_x,
                hex_centers_np[:, 1] + grid_origin_y,
            )

        elif what == "corners":
            # The inverted center mask is the corner mask. Now consider all corners
            mask_corners = np.invert(mask_center)

            hex_corners_np = np.array([xv[mask_corners], yv[mask_corners]]).T.reshape(
                -1, 2
            )

            if not flat_topped:
                hex_corners_np = hex_corners_np[:, ::-1]

            output_grid = points_from_xy(
                hex_corners_np[:, 0] + grid_origin_x,
                hex_corners_np[:, 1] + grid_origin_y,
            )

        elif what == "polygons":
            hex_coords_a = _hex_polygon_corners(xv, yv, (0, 1))
            hex_coords_b = _hex_polygon_corners(xv, yv, (2, 0))

            hex_coords = np.concatenate((hex_coords_a, hex_coords_b), axis=0)

            if not flat_topped:
                hex_coords = hex_coords[:, :, ::-1]

            hex_coords[:, :, 0] += grid_origin_x
            hex_coords[:, :, 1] += grid_origin_y

            hex_polygons = [Polygon(hex_set) for hex_set in hex_coords]

            hex_polygons_np = np.array(hex_polygons)
            output_grid = from_shapely(hex_polygons_np)

        else:
            raise ValueError("Invalid value for parameter `what`")
    else:
        raise ValueError("Invalid value for parameter `cell_type`")

    output_grid = GeoSeries(output_grid)

    if isinstance(input_geometry, (GeoDataFrame, GeoSeries)):
        if input_geometry.crs is not None:
            output_grid = output_grid.set_crs(input_geometry.crs)

    if intersect:
        if isinstance(input_geometry, (GeoDataFrame, GeoSeries)):
            return output_grid[
                output_grid.intersects(input_geometry.unary_union)
            ].reset_index(drop=True)
        else:
            return output_grid[output_grid.intersects(input_geometry)].reset_index(
                drop=True
            )
    else:
        return output_grid


def _hex_polygon_corners(xv, yv, index_0=(0, 0)):
    """Helper function that groups hexagon corners that belong to the same grid cell

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
        "6" corresponds to the hexagon corners and "2" to the corresponding
        x,y coordinates.

    """

    # Input checks
    if xv.shape != yv.shape:
        raise ValueError(
            f"`xv` and `yv` must have same shape, got {xv.shape} and {yv.shape},"
            "respectively"
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

    # Labeling of hexagon corners assumes that hexagon stands on a flat side:
    # Middle right corner of the hexagon
    n_poly_x = xv[(i_y0 + 1) :: 2, (i_x0 + 2 - x_corr) :: 3].shape[1]
    max_i_x = i_x0 + n_poly_x * 3 - 1
    # Top right corner of the hexagon
    n_poly_y = xv[(i_y0 + 2) :: 2, (i_x0 + 1) :: 3].shape[0]
    max_i_y = i_y0 + n_poly_y * 2

    # Extracting specific corners of all hexagon grid cells.
    bottom_left_corners = np.array(
        [xv[i_y0:max_i_y:2, i_x0:max_i_x:3], yv[i_y0:max_i_y:2, i_x0:max_i_x:3]]
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
        [xv[(i_y0 + 2) :: 2, i_x0:max_i_x:3], yv[(i_y0 + 2) :: 2, i_x0:max_i_x:3]]
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


def _basic_checks(input_geometry, cell_size, offset, what, cell_type, intersect):
    """Checks the validity of make_grid input parameters.

    `cell_size` must be larger than 0.
    `what` and `cell_type` must be a valid option.

    Parameters
    ------------
    input_geometry : (Multi)Polygon, GeoSeries, GeoDataFrame
    cell_size : float
    offset : tuple
    what : str, one of "centers", "corners", "polygons"
        type of return
    cell_type : str, one of "square", "hexagon"
        grid type
    """

    if not isinstance(input_geometry, (GeoDataFrame, GeoSeries, Polygon, MultiPolygon)):
        raise TypeError(
            "`input_geometry` should be GeoDataFrame, GeoSeries or"
            f"(Multi)Polygon, got {type(input_geometry)}"
        )
    if isinstance(input_geometry, (GeoDataFrame, GeoSeries)):
        if input_geometry.empty:
            raise ValueError("`input_geometry` is empty")
    else:
        if input_geometry.is_empty:
            raise ValueError("`input_geometry` is empty")

    if cell_size <= 0:
        raise ValueError(f"`cell_size` should be positive, got {cell_size}")

    allowed_what = ["centers", "corners", "polygons"]
    if what not in allowed_what:
        raise ValueError(f'`what` was "{what}" but is expected to be in {allowed_what}')

    allowed_cell_type = ["square", "hexagon"]
    if cell_type not in allowed_cell_type:
        raise ValueError(
            f'`cell_type` was "{cell_type}" but is expected'
            f"to be in {allowed_cell_type}"
        )

    if not isinstance(intersect, bool):
        raise TypeError(f"`intersect` should be a bool, got {type(input_geometry)}")

    if isinstance(input_geometry, (GeoDataFrame, GeoSeries)):
        bounds = np.array(input_geometry.total_bounds)
    else:
        bounds = np.array(input_geometry.bounds)
    if (offset[0] > (bounds[2] - bounds[0])) and (offset[1] > (bounds[3] - bounds[1])):
        warnings.warn("`offset` is larger than input_geometry bounds")
    if (cell_size > (bounds[2] - bounds[0])) and (cell_size > (bounds[3] - bounds[1])):
        warnings.warn("`cell_size` is larger than input_geometry bounds")
