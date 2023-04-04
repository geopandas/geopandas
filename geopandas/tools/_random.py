from warnings import warn

import numpy
import shapely
from shapely.geometry import MultiPoint

from ..array import from_shapely, points_from_xy
from ..geoseries import GeoSeries


def uniform(geom, size, seed=None, generator=None):
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

    size : integer
        an integer denoting how many points to sample

    Returns
    -------
    shapely.MultiPoint geometry containing the sampled points

    Examples
    --------
    >>> from shapely.geometry import box
    >>> square = box(0,0,1,1)
    >>> uniform(square, size=102) # doctest: +SKIP
    """
    if generator is None:
        generator = numpy.random.default_rng(seed=seed)

    if geom is None or geom.is_empty:
        multipoints = MultiPoint()

    else:
        if geom.geom_type in ("Polygon", "MultiPolygon"):
            multipoints = _uniform_polygon(geom, size=size, generator=generator)

        elif geom.geom_type in ("LineString", "MultiLineString"):
            multipoints = _uniform_line(geom, size=size, generator=generator)
        else:
            warn(
                f"Sampling is not supported for {geom.geom_type} geometry type.",
                UserWarning,
                stacklevel=8,
            )
            multipoints = MultiPoint()

    return multipoints


def _uniform_line(geom, size, generator):
    """
    Sample points from an input shapely linestring
    """
    n_points = size
    if geom.geom_type == "MultiLineString":
        splits = numpy.concatenate([_split_line(g) for g in geom.geoms])
    else:
        splits = _split_line(geom)
    lengths = from_shapely(splits).length
    points_per_split = generator.multinomial(n_points, pvals=lengths / lengths.sum())
    random_fracs = generator.uniform(size=size).T

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
    substring_splits = shapely.linestrings(list(zip(points[:-1], points[1:])))
    splits = numpy.hstack((splits, substring_splits))
    return splits


def _uniform_polygon(geom, size, generator):
    """
    Sample uniformly from within a polygon using batched sampling.
    """
    xmin, ymin, xmax, ymax = geom.bounds
    candidates = []
    while len(candidates) < size:
        batch = points_from_xy(
            x=generator.uniform(xmin, xmax, size=size),
            y=generator.uniform(ymin, ymax, size=size),
        )
        valid_samples = batch[batch.sindex.query(geom, predicate="contains")]
        candidates.extend(valid_samples)
    return GeoSeries(candidates[:size]).unary_union
