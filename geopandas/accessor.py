from pandas.core.accessor import _register_accessor

from ._decorator import doc


@doc(_register_accessor, klass="GeoDataFrame")
def register_geodataframe_accessor(name: str):
    from geopandas import GeoDataFrame

    return _register_accessor(name, GeoDataFrame)


@doc(_register_accessor, klass="GeoSeries")
def register_geoseries_accessor(name: str):
    from geopandas import GeoSeries

    return _register_accessor(name, GeoSeries)
