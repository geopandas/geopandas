from textwrap import dedent

from pandas.core.accessor import _register_accessor

from ._decorator import doc


_doc_register_accessor = dedent(
    """
    Register a custom accessor on {klass} objects.

    Parameters
    ----------
    name : str
        Name under which the accessor should be registered. A warning is issued
        if this name conflicts with a preexisting attribute.

    Returns
    -------
    callable
        A class decorator.

    See Also
    --------
    register_geoseries_accessor
    register_geodataframe_accessor

    Notes
    -----
    When accessed, your accessor will be initialized with the geopandas object
    the user is interacting with. So the signature must be

    .. code-block:: python
        def __init__(self, geopandas_object):  # noqa: E999
            ...

    For consistency with geopandas methods, you should raise an
    ``AttributeError`` if the data passed to your accessor has an incorrect
    dtype.

    >>> import geopandas as gpd
    >>> gpd.GeoSeries().dt
    Traceback (most recent call last):
    ...
    AttributeError: Can only use .dt accessor with datetimelike values

    Examples
    --------
    In your library code::

        import geopandas as gpd

        from pygeos import count_coordinates, from_shapely


        @gpd.api.extensions.register_geoseries_accessor("coords")
        class CoordinateAccessor:
            def __init__(self, gpd_obj):
                self._obj = gpd_obj

            @property
            def count_coordinates(self):
                # Counts the number of coordinate pairs in geometry

                func = lambda x: count_coordinates(from_shapely(x))
                return self._obj.apply(func)

    Back in an interactive IPython session:

    .. code-block:: ipython

        In [1]: s = gpd.GeoSeries.from_wkt(["POINT (1 1)", None])
        In [2]: s.coords.count_coordinates
        Out[2]:
        0    1
        1    0
        dtype: int64
    """
)


@doc(_doc_register_accessor, klass="GeoDataFrame")
def register_geodataframe_accessor(name: str):
    from geopandas import GeoDataFrame

    return _register_accessor(name, GeoDataFrame)


@doc(_doc_register_accessor, klass="GeoSeries")
def register_geoseries_accessor(name: str):
    from geopandas import GeoSeries

    return _register_accessor(name, GeoSeries)
