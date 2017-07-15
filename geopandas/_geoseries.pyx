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
