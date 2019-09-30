from dask.dataframe.extensions import make_array_nonempty, make_scalar
import numpy as np
import shapely.geometry

from .array import GeometryDtype, from_shapely


@make_array_nonempty.register(GeometryDtype)
def _(dtype):
    a = np.array([shapely.geometry.Point(i, i) for i in range(2)], dtype=object)
    return from_shapely(a)


@make_scalar.register(GeometryDtype.type)
def _(x):
    return shapely.geometry.Point(0, 0)
