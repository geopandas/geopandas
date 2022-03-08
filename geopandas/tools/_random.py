import numpy
from ..array import points_from_xy
from ..geoseries import GeoSeries
from ..geodataframe import GeoDataFrame
from .._compat import import_optional_dependency
from .grids import _hexgrid_circle, _squaregrid_circle
from shapely import geometry


def uniform(geom, size=1, batch_size=None):
    """
    # TODO: Write full docstring

    Sample uniformly at random from a geometry.

    For points, return a null geometry.
    For lines, sample uniformly along the length of the linestring.
    For polygons, sample uniformly within the area of the linestring.

    Parameters
    ----------
    geom : a shapely geometry
    size : an integer denoting how many points to sample, or a tuple
        denoting how many points to sample, and how many tines to conduct sampling.
    batch_size: integer denoting how large each round of simulation and checking
        should be. Should be approximately on the order of the number of points
        requested to sample. Some complex shapes may be faster to sample if the
        batch size increases. Only useful for (Multi)Polygon geometries.

    Returns
    -------

    Examples
    --------

    """

    try:
        assert int(size) == size
        size = int(size)
    except AssertionError:
        raise TypeError(
            "Size must be an integer denoting the number of samples to draw."
        )

    if geom.type in ("Polygon", "MultiPolygon"):
        multipoints = _uniform_polygon(geom, size=size, batch_size=batch_size)

    elif geom.type in ("LineString", "MultiLineString"):
        multipoints = _uniform_line(geom, size=size)
    else:
        # TODO: Should we recurse through geometrycollections?
        multipoints = geometry.MultiPoint()

    return multipoints


def grid(
    geom=None,
    size=None,
    spacing=None,
    method="square",
    random_offset=True,
    random_rotation=True,
):
    if geom is None:
        geom = geometry.box(0, 0, 1, 1)
    if size is None:
        if spacing is None:
            size = (10, 10)
        else:
            ValueError("Either size or spacing options must be provided.")
    else:  # only process size if it's provided; otherwise, allow spacing to lead
        if spacing is not None:
            raise ValueError(
                "Either size or spacing options can be provided, not both."
            )
        if isinstance(size, int):
            size = (size, size)
        try:
            assert isinstance(size, (tuple, list))
            assert len(size) == 2
        except AssertionError:
            raise TypeError(
                "Size must be an integer denoting the size of one side of the grid, "
                " or a tuple of two integers denoting the grid dimensions."
            )
    if method not in ("square", "hex"):
        raise ValueError(
            f'The method option must be either "square" or "hex". Recieved {method}.'
        )

    if geom.type in ("Polygon", "MultiPolygon"):
        multipoints = _grid_polygon(
            geom,
            size=size,
            spacing=spacing,
            method=method,
            random_offset=random_offset,
            random_rotation=random_rotation,
        )

    elif geom.type in ("LineString", "MultiLineString"):
        multipoints = _grid_line(
            geom, size=size, spacing=spacing, method=method, random_offset=random_offset
        )
    else:
        # TODO: Should we recurse through geometrycollections?
        multipoints = geometry.MultiPoint()

    return multipoints


def _grid_polygon(
    geom,
    size=None,
    spacing=None,
    method="square",
    random_offset=True,
    random_rotation=True,
    **unused_kws,
):
    pygeos = import_optional_dependency(
        "pygeos", "pygeos is required to randomly sample spatial grids from polygons"
    )
    pg_geom = pygeos.from_shapely(geom)
    pg_mbc = pygeos.minimum_bounding_circle(pg_geom)
    target_center = pygeos.get_coordinates(pygeos.centroid(pg_geom))

    # get spacing
    if spacing is None:
        x_min, y_min, x_max, y_max = pygeos.bounds(pg_mbc)
        x_res, y_res = size
        x_range, y_range = x_max - x_min, y_max - y_min
        x_step, y_step = x_range / x_res, y_range / y_res
        spacing = (x_step + y_step) / 2

    mbc_bounds = pygeos.bounds(pg_mbc)
    pg_radius = (mbc_bounds[2] - mbc_bounds[0]) / 2
    grid_radius = numpy.ceil(pg_radius / spacing).astype(int)
    if method == "square":
        raw_grid = _squaregrid_circle(grid_radius)
    elif method == "hex":
        raw_grid = _hexgrid_circle(grid_radius)
    else:
        ValueError(f'Method must be either "square" or "hex". Recieved {method}.')

    raw_grid *= pg_radius / grid_radius

    if method == "square":
        displacement = (
            numpy.random.uniform(-spacing / 2, spacing / 2, size=(1, 2)) * random_offset
        )

    else:
        hex_displacement = numpy.random.uniform(-spacing, spacing, size=(2, 1))
        hex_rotation = numpy.array([[numpy.sqrt(3), numpy.sqrt(3) / 2], [0, 3 / 2]])
        displacement = (hex_rotation @ hex_displacement).T

    rotation = numpy.random.uniform(0, numpy.pi * 2) * random_rotation
    c_rot, s_rot = numpy.cos(rotation), numpy.sin(rotation)
    rotation_matrix = numpy.array([[c_rot, s_rot], [-s_rot, c_rot]])

    rot_grid = rotation_matrix @ raw_grid.T

    disp_grid = rot_grid.T + displacement + target_center

    x, y = disp_grid.T

    return GeoSeries(points_from_xy(x=x, y=y)).clip(geom).unary_union


def _grid_line(geom, size, spacing, random_offset=True, **unused_kws):
    pygeos = import_optional_dependency(
        "pygeos", "pygeos is required to randomly sample along LineString geometries"
    )

    geom = pygeos.from_shapely(geom)

    if size is not None:
        if isinstance(size, tuple):
            # TODO: document this behavior: if you grid-sample geoms *and* they're
            # mixed, anysize tuple will get truncated to its first element when
            # sampling lines.The most clear alternative is to divide LineStrings
            # into "horizontal" and "vertical" segments based on the angle the
            # segment makes with the bottom of the bounding box, and then sample
            # size[0] if "horizontal" and size[1] if "vertical." That's a lot
            # of assumption for an unclear benefit, so it's not done here.
            size = size[0]
        spacing = pygeos.length(geom) / size
    else:
        size = spacing * pygeos.length(geom)

    parts = pygeos.get_parts(geom)
    grid = []
    for part in parts:
        # this does exactly one pass because "part" is always
        # LineString after pygeos.get_parts
        segs = _split_line(part)
        remainder = numpy.random.uniform(0, spacing * random_offset)
        lengths = pygeos.length(segs)
        for i, seg in enumerate(segs):
            locs = numpy.arange(remainder, lengths[i], spacing)

            if len(locs) > 0:
                # get points
                points = pygeos.line_interpolate_point(seg, locs, normalized=False)
                grid.extend(points)

                # the remainder is the "overhang" onto the next segment,
                remainder = spacing - (lengths[i] - locs[-1])
            else:
                remainder -= lengths[i]
    return pygeos.to_shapely(pygeos.union_all(grid))


def _uniform_line(geom, size=1, **unused_kws):
    """
    Sample points from an input shapely linestring
    """
    pygeos = import_optional_dependency(
        "pygeos", "pygeos is required to randomly sample along LineString geometries"
    )
    geom = pygeos.from_shapely(geom)
    n_points = size
    splits = _split_line(geom)
    lengths = pygeos.length(splits)
    points_per_split = numpy.random.multinomial(n_points, pvals=lengths / lengths.sum())
    random_fracs = numpy.random.uniform(size=size).T

    points_to_assign = points_per_split
    fracs = random_fracs
    split_to_point = numpy.repeat(numpy.arange(len(splits)), points_to_assign)
    samples = pygeos.line_interpolate_point(
        splits[split_to_point], fracs, normalized=True
    )
    grouped_samples = (
        GeoDataFrame(geometry=samples, index=split_to_point).dissolve(level=0).geometry
    )
    return grouped_samples.unary_union


def _split_line(geom):
    """
    Split an input linestring into component sub-segments.
    """
    pygeos = import_optional_dependency(
        "pygeos", "pygeos is required to randomly sample along LineString geometries"
    )
    parts = pygeos.get_parts(geom)
    splits = numpy.empty((0,))
    for part in parts:
        points = pygeos.get_coordinates(part)
        substring_splits = pygeos.linestrings(list(zip(points[:-1], points[1:])))
        splits = numpy.hstack((splits, substring_splits))
    return splits


def _uniform_polygon(geom, size=1, batch_size=None, **unused_kws):
    n_points = size
    if batch_size is None:
        batch_size = n_points
    xmin, ymin, xmax, ymax = geom.bounds
    candidates = []
    while len(candidates) < n_points:
        batch = points_from_xy(
            x=numpy.random.uniform(xmin, xmax, size=batch_size),
            y=numpy.random.uniform(ymin, ymax, size=batch_size),
        )
        valid_samples = batch[batch.sindex.query(geom, predicate="contains")]
        candidates.extend(valid_samples)
    return GeoSeries(candidates[:n_points]).unary_union
