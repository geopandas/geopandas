import numpy
from ..array import points_from_xy, from_shapely
from ..geoseries import GeoSeries
from .._compat import SHAPELY_GE_20
from .grids import _hexgrid_circle, _squaregrid_circle
import shapely
from shapely import geometry


def uniform(geom, size=1, batch_size=None):
    """

    Sample uniformly at random from a geometry.

    For polygons, this samples uniformly within the area of the polygon. For lines,
    this samples uniformly along the length of the linestring. For multi-part
    geometries, the weights of each part are selected according to their relevant
    attribute (area for Polygons, length for LineStrings), and then points are
    sampled from each part uniformly.

    Any other geometry type (e.g. Point, GeometryCollection) are ignored, and an
    empty MultiPoint geometry is returned.

    Parameters
    ----------
    geom : any shapely.geometry.BaseGeometry type
        the shape that describes the area in which to sample.

    size : integer, tuple
        an integer denoting how many points to sample, or a tuple
        denoting how many points to sample, and how many tines to conduct sampling.
    batch_size: integer
        a number denoting how large each round of simulation and checking
        should be. Should be approximately on the order of the number of points
        requested to sample. Some complex shapes may be faster to sample if the
        batch size increases. Only useful for (Multi)Polygon geometries.

    Returns
    -------
    shapely.MultiPoint geometry containing the sampled points

    Examples
    --------
    >>> from shapely.geometry import box
    >>> square = box(0,0,1,1)
    >>> uniform(square, size=100, batch_size=2) # doctest: +SKIP
    """

    try:
        assert int(size) == size
        size = int(size)
    except (AssertionError, TypeError):
        raise TypeError(
            "Size must be an integer denoting the number of samples to draw."
        )
    if geom is None or geom.is_empty:
        multipoints = geometry.MultiPoint()

    else:
        if geom.geom_type in ("Polygon", "MultiPolygon"):
            multipoints = _uniform_polygon(geom, size=size, batch_size=batch_size)

        elif geom.geom_type in ("LineString", "MultiLineString"):
            multipoints = _uniform_line(geom, size=size)
        else:
            # TODO: Should we recurse through geometrycollections?
            multipoints = geometry.MultiPoint()

    return multipoints


def grid(
    geom=None,
    size=None,
    spacing=None,
    tile="square",
    random_offset=True,
    random_rotation=True,
):
    """
    Sample a grid from within a given shape, possibly with a random
    rotation and offset.

    Parameters
    ----------
    geom : any shapely.geometry.BaseGeometry type
        the shape that describes the area in which to sample. Can sample
        within polygons, along lines, or within/along their multi-part forms.
        Any other geometry type will be ignored, and a MultiPoint returned.
    size : integer, tuple
        an integer denoting how many points to sample, or a tuple
        denoting how many points to sample, and how many tines to conduct sampling.
    batch_size: integer
        a number denoting how large each round of simulation and checking
        should be. Should be approximately on the order of the number of points
        requested to sample. Some complex shapes may be faster to sample if the
        batch size increases. Only useful for (Multi)Polygon geometries.

    Returns
    -------
    shapely.MultiPoint geometry containing the sampled points

    Examples
    --------
    >>> from shapely.geometry import box
    >>> square = box(0,0,1,1)
    >>> grid(square, size=(8,8), tile='hex') # doctest: +SKIP
    """

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
            if geom.geom_type in ("Polygon", "MultiPolygon"):
                size = (size, size)
            else:
                size = (size, 1)
        try:
            assert isinstance(size, (tuple, list))
            assert int(size[0]) == size[0]
            assert int(size[1]) == size[1]
            assert len(size) == 2
        except AssertionError:
            raise TypeError(
                "Size must be an integer denoting the size of one side of the grid, "
                " or a tuple of two integers denoting the grid dimensions."
            )
    if tile not in ("square", "hex"):
        raise ValueError(
            f'The tile option must be either "square" or "hex". Recieved {tile}.'
        )

    if geom.geom_type in ("Polygon", "MultiPolygon"):
        multipoints = _grid_polygon(
            geom,
            size=size,
            spacing=spacing,
            tile=tile,
            random_offset=random_offset,
            random_rotation=random_rotation,
        )

    elif geom.geom_type in ("LineString", "MultiLineString"):
        multipoints = _grid_line(
            geom, size=size, spacing=spacing, tile=tile, random_offset=random_offset
        )
    else:
        # TODO: Should we recurse through geometrycollections?
        multipoints = geometry.MultiPoint()

    return multipoints


def _grid_polygon(
    geom,
    size=None,
    spacing=None,
    tile="square",
    random_offset=True,
    random_rotation=True,
    **unused_kws,
):
    """
    Sample points from within a polygon according to a given spacing.
    """

    # cast to GeoSeries to automatically select the geometry engine
    geom = GeoSeries([geom])
    mbc = geom.minimum_bounding_circle()
    target_center = geom.centroid.get_coordinates().values
    mbc_bounds = mbc.bounds.iloc[0].values

    # get spacing
    if spacing is None:
        x_min, y_min, x_max, y_max = mbc_bounds
        x_res, y_res = size
        x_range, y_range = x_max - x_min, y_max - y_min
        x_step, y_step = x_range / x_res, y_range / y_res
        spacing = (x_step + y_step) / 2

    radius = (mbc_bounds[2] - mbc_bounds[0]) / 2
    grid_radius = numpy.ceil(radius / spacing).astype(int)
    if tile == "square":
        raw_grid = _squaregrid_circle(grid_radius)
    elif tile == "hex":
        raw_grid = _hexgrid_circle(grid_radius)
    else:
        ValueError(f'tile must be either "square" or "hex". Recieved {tile}.')

    raw_grid *= radius / grid_radius

    if tile == "square":
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

    return GeoSeries(points_from_xy(x=x, y=y)).clip(geom.iloc[0]).unary_union


def _grid_line(geom, size, spacing, random_offset=True, **unused_kws):
    """
    Sample points from along a line according to a given spacing.
    """
    geom = GeoSeries([geom])

    if size is not None:
        if isinstance(size, tuple):
            # TODO: document this behavior: if you grid-sample geoms *and* they're
            # mixed, any tuple size specification will get converted into a flat size
            # when sampling lines. The most clear alternative is to divide LineStrings
            # into "horizontal" and "vertical" segments based on the angle the
            # segment makes with the bottom of the bounding box, and then sample
            # size[0] if "horizontal" and size[1] if "vertical." That's a lot
            # of assumption for an unclear benefit, so it's not done here.
            size = size[0] * size[1]
        spacing = geom.length.iloc[0] / size
    else:
        size = spacing * geom.length.iloc[0]

    parts = geom.explode(ignore_index=True)
    grid = []
    for part in parts:
        # this does exactly one pass because "part" is always
        # LineString after explode
        segs = _split_line(part)
        remainder = numpy.random.uniform(0, spacing * random_offset)
        lengths = GeoSeries(segs).length.values
        for i, seg in enumerate(segs):
            locs = numpy.arange(remainder, lengths[i], spacing)

            if len(locs) > 0:
                # get points
                points = from_shapely([seg]).interpolate(locs)

                grid.extend(points)

                # the remainder is the "overhang" onto the next segment,
                remainder = spacing - (lengths[i] - locs[-1])
            else:
                remainder -= lengths[i]

    return from_shapely(grid).unary_union()


def _uniform_line(geom, size=1, **unused_kws):
    """
    Sample points from an input shapely linestring
    """
    n_points = size
    splits = _split_line(geom)
    lengths = from_shapely(splits).length
    points_per_split = numpy.random.multinomial(n_points, pvals=lengths / lengths.sum())
    random_fracs = numpy.random.uniform(size=size).T

    points_to_assign = points_per_split
    fracs = random_fracs
    split_to_point = numpy.repeat(numpy.arange(len(splits)), points_to_assign)
    samples = from_shapely(splits[split_to_point]).interpolate(fracs, normalized=True)
    return samples.unary_union()


def _split_line(geom):
    """
    Split an input linestring into component sub-segments.
    """
    splits = numpy.empty((0,))
    points = GeoSeries([geom]).get_coordinates().values
    if SHAPELY_GE_20:
        substring_splits = shapely.linestrings(list(zip(points[:-1], points[1:])))
    else:
        substring_splits = numpy.array(
            [shapely.LineString(x) for x in list(zip(points[:-1], points[1:]))]
        )
    splits = numpy.hstack((splits, substring_splits))
    return splits


def _uniform_polygon(geom, size=1, batch_size=None, **unused_kws):
    """
    Sample uniformly from within a polygon using batched sampling.
    """
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