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


def _hex_points(size, bounds, spacing, flat=True):
    x_min, y_min, x_max, y_max = bounds
    x_range, y_range = x_max - x_min, y_max - y_min
    center = numpy.array([[x_min + x_range / 2, y_min + y_range / 2]])

    if spacing is None:
        x_res, y_res = size
        x_step, y_step = x_range / x_res, y_range / y_res
        spacing = (x_step + y_step) / 2

    # build a hexcircle big enough to cover the bounds
    bounds_diagonal = numpy.sqrt(x_range ** 2 + y_range ** 2)

    hex_radius = numpy.ceil(bounds_diagonal / 2 / spacing)
    hex_circle = _hexgrid_circle(hex_radius, flat=flat)
    hex_circle /= hex_radius * numpy.sqrt(3)
    hex_inradius = numpy.sqrt(3) / 2
    theta = numpy.arctan(y_range / 2 / x_range / 2)
    phi = (numpy.pi) / 2 - theta
    chord_remainder = (y_range / 2) / numpy.tan(phi)

    rescaling_factor = chord_remainder + (x_range / 2)

    hex_circle *= rescaling_factor / hex_inradius
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

    return final_points


def _squaregrid_circle(radius, flat=True):
    i = j = numpy.arange(radius * 2 + 1) - radius
    i_grid, j_grid = numpy.meshgrid(i, j)
    i_grid = i_grid.flatten()
    j_grid = j_grid.flatten()

    all_locs = numpy.column_stack((i_grid, j_grid))

    rotation = numpy.eye(2) if flat else numpy.array([[1, 1], [-1, 1]])

    rotated = (rotation @ all_locs.T).T
    mask = (rotated ** 2).sum(axis=1) <= (radius ** 2)
    return rotated[mask]


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
