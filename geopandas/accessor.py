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

        from __future__ import annotations

        from dataclasses import dataclass

        import pandas as pd


        @register_geodataframe_accessor("gtype")
        @register_geoseries_accessor("gtype")
        @dataclass
        class GeoAccessor:

            _obj: gpd.GeoSeries | gpd.GeoDataFrame

            @property
            def is_point(self) -> pd.Series:
                # Return a boolean Series denoting whether each geometry is a point.

                return self._obj.geometry.geom_type == "Point"

    Back in an interactive IPython session:

    .. code-block:: ipython

        In [1]: import geopandas as gpd

        In [2]: s = gpd.GeoSeries.from_wkt(["POINT (0 0)", "POINT (1 1)", None])

        In [3]: s
        Out[3]:
        0    POINT (0.00000 0.00000)
        1    POINT (1.00000 1.00000)
        2                       None
        dtype: geometry

        In [4]: s.gtype.is_point
        Out[4]:
        0     True
        1     True
        2    False
        dtype: bool

        In [5]: d = s.to_frame("geometry")
        Out[5]:
                          geometry
        0  POINT (0.00000 0.00000)
        1  POINT (1.00000 1.00000)
        2                     None

        In [6]: d.gtype.is_point
        Out[6]:
        0     True
        1     True
        2    False
        dtype: bool
    """
    from geopandas import GeoSeries

    return _register_accessor(name, GeoSeries)


@doc(register_geoseries_accessor, klass="GeoSeries")
def register_geodataframe_accessor(name: str):
    from geopandas import GeoDataFrame

    return _register_accessor(name, GeoDataFrame)
