import warnings
import numpy as np

from shapely.geometry import Polygon, MultiPolygon

from geopandas import GeoSeries, GeoDataFrame, points_from_xy
from geopandas.array import from_shapely


def make_grid(
    input_geometry,
    cell_size,
    offset=(0, 0),
    crs=None,
    what="polygons",
    cell_type="square",
    intersect=False,
):
    """Provides the grid-cell centers, corners, or polygons of a square or hexagonal grid.

        Parameters
        ------------
        input_geometry : (Multi)Polygon, GeoSeries, GeoDataFrame
            Polygon within its boundaries the grid is created
        cell_size : float
            Side length of the individual square or the hexagon
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
        Cover a input_geometry (the South American continent) with a square grid:
        >>> world = geopandas.read_file(
        ...     geopandas.datasets.get_path('naturalearth_lowres'))
        >>> south_america = world[world['continent'] == "South America"]
        # TODO Finalize example

        """

    # Run basic checks
    _basic_checks(input_geometry, cell_size, offset, what, cell_type, intersect)

    output_grid = None

    if isinstance(input_geometry, (GeoDataFrame, GeoSeries)):
        bounds = np.array(input_geometry.total_bounds)
    else:
        # TODO Test if multipolygons also work
        bounds = np.array(input_geometry.bounds)

    x_dist = bounds[2] - bounds[0] - offset[0]
    y_dist = bounds[3] - bounds[1] - offset[1]

    if cell_type == "square":
        # Set corner coordinates: Also in case of an offse,
        # grid always ends at the right/upper end of the bounding box
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
            # Use int() to ensure that half decimal numbers are
            # rounded to the next higher int.
            n_cent_x = int(x_dist / cell_size + 0.5)
            n_cent_y = int(y_dist / cell_size + 0.5)

            sq_centers_np = (
                np.array(
                    [xv[:n_cent_y, :n_cent_x], yv[:n_cent_y, :n_cent_x]]
                ).T.reshape(-1, 2)
                + cell_size / 2
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
            # TODO Could a pygeos solution be used to replace below expression
            sq_polygons_np = np.array([Polygon(sq_set) for sq_set in sq_coords])

            output_grid = from_shapely(sq_polygons_np)

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
            output_grid = points_from_xy(hex_centers_np[:, 0], hex_centers_np[:, 1])

        elif what == "corners":
            # The inverted center mask is the corner mask
            mask_corners = np.invert(mask_center)
            hex_corners_np = np.array([xv[mask_corners], yv[mask_corners]]).T.reshape(
                -1, 2
            )
            output_grid = points_from_xy(hex_corners_np[:, 0], hex_corners_np[:, 1])

        elif what == "polygons":
            hex_coords_a = _hex_polygon_corners(xv, yv, (0, 1))
            hex_coords_b = _hex_polygon_corners(xv, yv, (2, 0))

            hex_coords = np.concatenate((hex_coords_a, hex_coords_b), axis=0)
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
            output_grid.set_crs(input_geometry.crs)

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
        6 corresponds to the hexagon corners and 2 to the x,y coordinates.

    """

    # TODO Check again the number of polygons. To many centers are returned.

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

    # Labeling of hexagon corners assumes that hexagon stands on a flat side.
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

