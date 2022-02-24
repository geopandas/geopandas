import numpy
from ..array import points_from_xy, GeometryArray, from_shapely
from ..geoseries import GeoSeries
from ..geodataframe import GeoDataFrame
import pygeos


def uniform(geom, size=(1, 1), batch_size=None, exact=False):
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
    batch_size: integer denoting how large each simulation round should be. Should
        be approximately on the order of the number of points requested to sample.
        Some complex shapes may be faster to sample if the batch size increases.
    exact: whether or not to force there to be exactly the requested number of
        samples within the shape.

    Returns
    -------

    Examples
    --------

    """
    if isinstance(size, int):
        size = (size, 1)
    try:
        assert isinstance(size, (tuple, list))
        assert len(size) == 2
    except AssertionError:
        raise TypeError(
            "Size must be an integer denoting the number of samples"
            " or a tuple of integers with the number of samples and "
            " the number of replications to simulate."
        )
    if geom.type in ("Polygon", "MultiPolygon"):
        multipoints = _uniform_polygon(
            geom, size=size, batch_size=batch_size, exact=exact
        )
        multipoints = from_shapely(multipoints)

    elif geom.type in ("LineString", "MultiLineString"):
        multipoints = _uniform_line(geom, size=size, batch_size=batch_size, exact=exact)
        multipoints = from_shapely(multipoints)
    else:
        # TODO: Should we recurse through geometrycollections?
        multipoints = pygeos.empty(size, geom_type=pygeos.GeometryType.MULTIPOINT)
    return GeoSeries(
        multipoints, index=[f"sample_{i}" for i in range(size[-1])], name="geometry"
    ).squeeze()


def grid():
    ...


def hex():
    ...


def _grid_line():
    ...


def _hex_line():
    raise NotImplementedError("Hex sampling along linestrings is not supported")


def _uniform_line(geom, size=(1, 1), batch_size=None, exact=False):
    """
    Sample points from an input shapely linestring
    """
    geom = pygeos.from_shapely(geom)
    n_points, n_reps = size
    splits = _split_line(geom)
    lengths = pygeos.length(splits)
    points_per_split = numpy.random.multinomial(
        n_points, pvals=lengths / lengths.sum(), size=n_reps
    )
    random_fracs = numpy.random.uniform(size=size).T
    output = []
    for rep in range(n_reps):
        points_to_assign = points_per_split[rep]
        fracs = random_fracs[rep]
        split_to_point = numpy.repeat(numpy.arange(len(splits)), points_to_assign)
        samples = pygeos.line_interpolate_point(
            splits[split_to_point], fracs, normalized=True
        )
        grouped_samples = (
            GeoDataFrame(geometry=samples, index=split_to_point)
            .dissolve(level=0)
            .geometry
        )
        output.append(grouped_samples.unary_union)
    return output


def _split_line(geom):
    """
    Split an input linestring into component sub-segments.
    """
    parts = pygeos.get_parts(geom)
    splits = numpy.empty((0,))
    for part in parts:
        points = pygeos.get_coordinates(part)
        substring_splits = pygeos.linestrings(points[:-1], points[1:])
        splits = numpy.hstack((splits, substring_splits))
    return splits


def _grid_polygon(geom, size=(1, 1), batch_size=None, exact=False):
    return make_grid(geom, size=size, method="square").unary_union


def _hex_polygon(geom, size=(1, 1), batch_size=None, exact=False):
    return make_grid(geom, size=size, method="hex").unary_union


def _uniform_polygon(geom, size=(1, 1), batch_size=None, exact=False):
    n_points, n_reps = size
    if batch_size is None:
        batch_size = n_points
    xmin, ymin, xmax, ymax = geom.bounds
    result = []
    for rep in range(n_reps):
        candidates = []
        while len(candidates) < n_points:
            batch = points_from_xy(
                x=numpy.random.uniform(xmin, xmax, size=batch_size),
                y=numpy.random.uniform(ymin, ymax, size=batch_size),
            )
            valid_samples = batch[batch.sindex.query(geom, predicate="contains")]
            candidates.extend(valid_samples)
        result.append(GeoSeries(candidates[:n_points]).unary_union)
    return result


def _null_geom(*args, **kwargs):
    return None
