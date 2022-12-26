from textwrap import dedent

from pandas.core.accessor import _register_accessor

from ._decorator import doc


@doc(klass="GeoSeries")
def register_geoseries_accessor(name: str):
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

    For consistency with geopandas methods, you should raise an ``AttributeError``
    if the data passed to your accessor has an incorrect dtype.

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


        @gpd.api.extensions.register_geodataframe_accessor("coords")
        @gpd.api.extensions.register_geoseries_accessor("coords")
        class CoordinateAccessor:
            def __init__(self, gpd_obj):
                self._obj = gpd_obj

            @property
            def count_coordinates(self):
                # Counts the number of coordinate pairs in geometry

                func = lambda x: count_coordinates(from_shapely(x))
                return self._obj.geometry.apply(func)

    Back in an interactive IPython session:

    .. code-block:: ipython

        In [1]: s = gpd.GeoSeries.from_wkt(["POINT (1 1)", None])
        In [2]: s
        Out[2]:
        0    POINT (1.00000 1.00000)
        1                       None
        dtype: geometry

        In [3]: s.coords.count_coordinates
        Out[3]:
        0    1
        1    0
        dtype: int64

        In [4]: d = s.to_frame("geometry")
        In [5]: d
        Out[5]:
                        geometry
        0  POINT (1.00000 1.00000)
        1                     None

        In [6]: d.coords.countdinates
        Out[6]:
        0    1
        1    0
        Name: geometry, dtype: int64
    """
    from geopandas import GeoSeries

    return _register_accessor(name, GeoSeries)


@doc(register_geoseries_accessor, klass="GeoSeries")
def register_geodataframe_accessor(name: str):
    from geopandas import GeoDataFrame

    return _register_accessor(name, GeoDataFrame)
