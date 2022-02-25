from ..array import points_from_xy
from ..geoseries import GeoSeries
from ..geodataframe import GeoDataFrame
import numpy


def make_grid(
    geom=None,
    size=None,
    spacing=None,
    method="square",
    as_polygons=False,
    clip=True,
):
    if size is not None:
        if isinstance(size, float):
            raise TypeError("Grid sizes must be integers or tuples of integers")
        if isinstance(size, int):
            size = (size, size)
        if not isinstance(size, tuple):
            raise TypeError("Grid sizes must be integers or tuples of integers")
        if spacing is not None:
            ValueError("Either size or spacing options can be provided, not both.")
    elif size is None and spacing is None:
        size = (10, 10)
    else:  # size is None but spacing is known, so compute size in grid makers.
        pass

    if geom is None:
        from shapely.geometry import box

        geom = box(0, 0, 1, 1)
    if isinstance(geom, (GeoSeries, GeoDataFrame)):
        bounds = geom.total_bounds
    else:
        bounds = geom.bounds
    if method == "square":
        if as_polygons:
            grid = _square_mesh(size=size, bounds=bounds, spacing=spacing)
        else:
            grid = _square_points(size=size, bounds=bounds, spacing=spacing)
            result = GeoSeries(points_from_xy(*grid.T, crs=getattr(geom, "crs", None)))

    elif method == "hex":
        if as_polygons:
            grid = _hex_mesh(size=size, bounds=bounds)
        else:
            grid = _hex_points(
                size=size,
                bounds=bounds,
                spacing=spacing,
            )
            result = GeoSeries(points_from_xy(*grid.T, crs=getattr(geom, "crs", None)))
    else:
        raise NotImplementedError(
            f'Cannot build a grid using method="{method}". '
            f'Only "hex" and "square" grid methods are supported.'
        )
    if clip:
        if isinstance(geom, (GeoSeries, GeoDataFrame)):
            result = result.clip(geom.cascaded_union, keep_geom_type=True)
        else:
            result = result.clip(geom, keep_geom_type=True)

    return result


def _hex_mesh(size, bounds):
    raise NotImplementedError


def _square_mesh(size, bounds):
    raise NotImplementedError


def _hex_points(size, bounds, spacing):
    """
    A hexagonal grid is formed of an "inner" and an "outer" rectangular
    grid with side lengths 3,sqrt(3). The two grids are offset by (.5,.5).
    Let size = x_size,y_size. We build the grid in 0,x_size and 0,y_size,
    then translate & scale the grid to the target bounds.
    """
    x_min, y_min, x_max, y_max = bounds
    x_range, y_range = x_max - x_min, y_max - y_min

    if spacing is None:
        x_res, y_res = size
        x_step, y_step = x_range / x_res, y_range / y_res
        spacing = (x_step + y_step) / 2

    x_eff, y_eff = numpy.ceil(x_range / spacing).astype(int), numpy.ceil(
        y_range / spacing
    ).astype(int)

    if size is None:
        size = (x_res, y_res) = (x_eff, y_eff)  # not needed, but for safety

    map_scale = numpy.array([[x_max - x_min, y_max - y_min]])

    x_ids, y_ids = numpy.arange(x_eff), numpy.arange(y_eff)
    points = numpy.empty((0, 2))
    for inner in range(2):
        x_locs, y_locs = (
            3 * (x_ids[: -1 if ((x_res % 2) & inner) else None] + 0.5 * inner),
            numpy.sqrt(3)
            * (y_ids[: -1 if ((y_res % 2) & inner) else None] + 0.5 * inner),
        )
        x_grid, y_grid = numpy.meshgrid(x_locs, y_locs)
        subgrid_points = numpy.column_stack((x_grid.flatten(), y_grid.flatten()))
        points = numpy.row_stack((points, subgrid_points))
    grid_scale = points.max(axis=0) - points.min(axis=0)
    points *= map_scale / grid_scale
    points += numpy.array([[x_min, y_min]])
    return points


def _square_points(size, bounds, spacing):
    x_min, y_min, x_max, y_max = bounds
    if spacing is not None:
        size = (
            numpy.ceil((x_max - x_min) / spacing).astype(int) + 1,
            numpy.ceil((y_max - y_min) / spacing).astype(int) + 1,
        )

    x_res, y_res = size
    x, y = numpy.meshgrid(
        numpy.linspace(x_min, x_max, x_res), numpy.linspace(y_min, y_max, y_res)
    )

    return numpy.column_stack((x.flatten(), y.flatten()))
