from warnings import warn

import numpy

from ..array import points_from_xy
from ..geodataframe import GeoDataFrame
from ..geoseries import GeoSeries


def make_grid(
    geometry,
    size=None,
    spacing=None,
    tile="square",
    clip=True,
):
    """
    Sample uniformly at random from a geometry or a GeoSeries.

    For polygons, this samples uniformly within the area of the polygon. For lines,
    this samples uniformly along the length of the linestring. For multi-part
    geometries, the weights of each part are selected according to their relevant
    attribute (area for Polygons, length for LineStrings), and then points are
    sampled from each part uniformly.

    Any other geometry type (e.g. Point, GeometryCollection) is ignored, and an
    empty MultiPoint geometry is returned.

    Parameters
    ----------
    geometry : shapely.Geometry, GeoSeries or GeoDataFrame
        The shape(s) covering the area in which to sample.
    size : int | tuple, optional
        The number points along each side of the
        grid. If a tuple is provided, then the first value must indicate the number
        of grid points on the x axis, and second value must indicate the number of
        grid points on the y axis. Either ``size`` or ``spacing`` is required but never
        both.
    spacing : int, optional
        The spacing of points. Indicates the distance between
        the points in the grid. Either ``size`` or ``spacing`` is required but never
        both.
    tile : {"square", "hex"}, default "square"
        A type of the grid to sample. ``"square"`` generates
        a square grid, while ``"hex"`` generates a hexagonal grid (also known as
        triangualar, depending on the perspective).
    clip : bool, default True
        Clip the sampled grid to fall within the ``geometry``. When ``False``,
        the resulting grid covers the bounds of the ``geometry``. Otherwise it is
        clipped to fall only within.

    Returns
    -------
    shapely.MultiPoint geometry containing the sampled points

    See also
    --------
    GeoSeries.sample_points

    Examples
    --------
    >>> from shapely.geometry import box
    >>> square = box(0,0,1,1)
    >>> make_grid(square, size=100) # doctest: +SKIP
    """
    if size is not None:
        if isinstance(size, float):
            raise TypeError("Grid sizes must be integers or tuples of integers")
        if isinstance(size, int):
            size = (size, size)
        if not isinstance(size, tuple):
            raise TypeError("Grid sizes must be integers or tuples of integers")
        if spacing is not None:
            raise ValueError(
                "Either size or spacing options can be provided, not both."
            )
    elif size is None and spacing is None:
        raise ValueError("Either size or spacing options must be provided.")
    else:  # size is None but spacing is known, so compute size in grid makers.
        pass

    if isinstance(geometry, (GeoSeries, GeoDataFrame)):
        bounds = geometry.total_bounds
    else:
        bounds = geometry.bounds
    if tile == "square":
        grid = _square_points(size=size, bounds=bounds, spacing=spacing)
        result = GeoSeries(points_from_xy(*grid.T, crs=getattr(geometry, "crs", None)))

    elif tile == "hex":
        grid = _hex_points(
            size=size,
            bounds=bounds,
            spacing=spacing,
        )
        result = GeoSeries(points_from_xy(*grid.T, crs=getattr(geometry, "crs", None)))
    else:
        raise ValueError(
            f'The tile option must be either "square" or "hex". Recieved "{tile}".'
        )

    # clip if not forbidden.
    if clip:
        if isinstance(geometry, (GeoSeries, GeoDataFrame)):
            mask = geometry.unary_union
            was_df = True
        else:
            mask = geometry
            was_df = False

        # if the mask is a polygon, clip if not forbidden
        if mask.geom_type in ("Polygon", "MultiPolygon"):
            result = result.clip(geometry, keep_geom_type=True)
        # otherwise, if explicitly requested and the mask is not valid, warn:
        elif clip:
            if was_df:
                warning_string = (
                    f"GeoSeries/GeoDataFrame with types"
                    f" ({', '.join(geometry.geometry.geom_type.unique())})"
                )

            else:
                warning_string = f"geometry of type {geometry.geom_type}"
            warn(
                f"clip only makes sense when gridding (Multi)Polygon geometries."
                f" Your input was a {warning_string} that resulted in a mask of "
                f"type {mask.geom_type}. You may need to use the .clip() method with "
                f"a specific mask on this grid to get the result you want."
            )
        # finally, if clip is None or the mask isn't usable, ignore it
        else:
            pass

    return result.reset_index(drop=True)


def _hex_points(size, bounds, spacing, flat=True):
    x_min, y_min, x_max, y_max = bounds
    x_range, y_range = x_max - x_min, y_max - y_min
    center = numpy.array([[x_min + x_range / 2, y_min + y_range / 2]])
    if spacing is None:
        x_res, y_res = size
        x_step, y_step = x_range / x_res, y_range / y_res
        spacing = (x_step + y_step) / 2

    # build a hexcircle big enough to cover the bounds
    bounds_diagonal = numpy.sqrt(x_range**2 + y_range**2)

    hex_radius = numpy.ceil(bounds_diagonal / 2 / spacing)
    hex_circle = _hexgrid_circle(hex_radius, flat=flat)

    hex_circle *= spacing
    hex_circle += center
    mask = (
        (hex_circle[:, 0] >= x_min)
        & (hex_circle[:, 0] <= x_max)
        & (hex_circle[:, 1] >= y_min)
        & (hex_circle[:, 1] <= y_max)
    )

    return hex_circle[mask]


def _hexgrid_circle(radius, flat=True):
    i = j = numpy.arange(radius * 2 + 1) - radius
    i_grid, j_grid = numpy.meshgrid(i, j)
    i_grid = i_grid.flatten()
    j_grid = j_grid.flatten()
    k_grid = -i_grid - j_grid

    all_locs = numpy.column_stack((i_grid, j_grid, k_grid))
    mask = numpy.all(numpy.abs(all_locs) <= radius, axis=1)

    rotation = numpy.array([[numpy.sqrt(3), numpy.sqrt(3) / 2], [0, 3 / 2]])
    rotation = numpy.fliplr(rotation)[::-1] if (not flat) else rotation

    final_points = (rotation @ all_locs[:, :-1].T).T[mask]

    return final_points / numpy.sqrt(3)


def _squaregrid_circle(radius, flat=True):
    i = j = numpy.arange(radius * 2 + 1) - radius
    i_grid, j_grid = numpy.meshgrid(i, j)
    i_grid = i_grid.flatten()
    j_grid = j_grid.flatten()

    all_locs = numpy.column_stack((i_grid, j_grid))

    rotation = numpy.eye(2) if flat else numpy.array([[1, 1], [-1, 1]])

    rotated = (rotation @ all_locs.T).T
    mask = (rotated**2).sum(axis=1) <= (radius**2)
    return rotated[mask]


def _square_points(size, bounds, spacing):
    x_min, y_min, x_max, y_max = bounds
    if spacing is not None:
        x_locs, y_locs = numpy.arange(x_min, x_max + spacing, spacing), numpy.arange(
            y_min, y_max + spacing, spacing
        )
    else:
        x_res, y_res = size
        x_locs, y_locs = numpy.linspace(x_min, x_max, x_res), numpy.linspace(
            y_min, y_max, y_res
        )

    x, y = numpy.meshgrid(x_locs, y_locs)

    return numpy.column_stack((x.flatten(), y.flatten()))
