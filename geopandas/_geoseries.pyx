cpdef fake_func(x, y):
    return x + y


from warnings import warn

from shapely.geometry import MultiPoint, MultiLineString, MultiPolygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import cascaded_union, unary_union
import shapely.affinity as affinity

import numpy as np
import pandas as pd
from pandas import Series, DataFrame, MultiIndex

import geopandas as gpd
from .base import GeoPandasBase


def _series_op(this, other, op, **kwargs):
    return _cy_series_op(this, other, op, kwargs)

cdef _cy_series_op(this, other, op, kwargs):
    """Geometric operation that returns a pandas Series"""
    null_val = False if op != 'distance' else np.nan

    if isinstance(other, GeoPandasBase):
        this = this.geometry
        this, other = this.align(other.geometry)
        try:
            func = getattr(type(this.iloc[0]), op)
        except KeyError:
            return Series([], index=this.index)
        else:
            return Series([func(this_elem, other_elem, **kwargs)
                        if not this_elem.is_empty | other_elem.is_empty else null_val
                        for this_elem, other_elem in zip(this, other)],
                        index=this.index)
    else:
        return Series([getattr(s, op)(other, **kwargs) if s else null_val
                      for s in this.geometry], index=this.index)


@cython.boundscheck(False)
@cython.wraparound(False)
def contains_cy(array, geometry):

    cdef Py_ssize_t idx
    cdef unsigned int n = array.size
    cdef np.ndarray[np.uint8_t, ndim=1, cast=True] result = np.empty(n, dtype=np.uint8)

    cdef GEOSContextHandle_t geos_handle
    cdef GEOSPreparedGeometry *geom1
    cdef GEOSGeometry *geom2
    cdef uintptr_t geos_geom

    # TODO: take `op` like "contains" and replace `GEOSPreparedContains_r`
    #       below with a switch statement on op.

    # Prepare the geometry if it hasn't already been prepared.
    if not isinstance(geometry, shapely.prepared.PreparedGeometry):
        geometry = shapely.prepared.prep(geometry)

    geos_h = get_geos_context_handle()
    geom1 = geos_from_prepared(geometry)

    for idx in xrange(n):
        # Construct a coordinate sequence with our x, y values.
        geos_geom = array[idx]._geom
        geom2 = <GEOSGeometry *>geos_geom

        # Put the result of whether the point is "contained" by the
        # prepared geometry into the result array.
        result[idx] = <np.uint8_t> GEOSPreparedContains_r(geos_h, geom1, geom2)
        #GEOSGeom_destroy_r(geos_h, geom2)

    return result.view(dtype=np.bool)
