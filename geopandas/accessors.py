"""Accessors for accessing GeoPandas functionality via pandas extension dtypes."""

import pandas.api.extensions

from geopandas import GeoSeries
from geopandas.array import GeometryDtype


@pandas.api.extensions.register_series_accessor("geo")
class GeoSeriesAccessor:
    """Series.geo accessor to expose GeoSeries methods on pandas Series.

    Parameters
    ----------
    series : pandas.Series
        A Series with geometry dtype.
    """

    def __init__(self, series):
        if not isinstance(series.dtype, GeometryDtype):
            raise AttributeError("Can only use .geo accessor with GeometryDtype values")

        self._geoseries = GeoSeries(series)

    def __getattr__(self, name):
        return getattr(self._geoseries, name)
