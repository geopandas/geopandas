from copyreg import dispatch_table
import numpy
from ..array import points_from_xy, GeometryArray
from ..geoseries import GeoSeries
from pandas import DataFrame
import pygeos


def uniform(geom, size=(1, 1), batch_size=None, exact=False):
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
        return _uniform_polygon(geom, size=size, batch_size=batch_size, exact=exact)
    elif geom.type == ("LineString", "MultiLineString"):
        raise NotImplementedError()
    else:
        raise NotImplementedError()


def grid():
    ...


def hex():
    ...


def _grid_line():
    ...


def _hex_line():
    raise NotImplementedError("Hex sampling along linestrings is not supported")


def _uniform_line():
    ...


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
    return GeoSeries(result)


def _null_geom(*args, **kwargs):
    return None
