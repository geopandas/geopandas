import json
import warnings

import numpy as np
import pandas as pd
from pandas import Series, MultiIndex
from pandas.core.internals import SingleBlockManager

from pyproj import CRS, Transformer
from shapely.geometry.base import BaseGeometry

from geopandas.base import GeoPandasBase, _delegate_property
from geopandas.plotting import plot_series

from .array import GeometryArray, GeometryDtype, from_shapely
from .base import is_geometry_type
from . import _vectorized as vectorized
from ._compat import ignore_shapely2_warnings


_SERIES_WARNING_MSG = """\
    You are passing non-geometry data to the GeoSeries constructor. Currently,
    it falls back to returning a pandas Series. But in the future, we will start
    to raise a TypeError instead."""


def _geoseries_constructor_with_fallback(data=None, index=None, crs=None, **kwargs):
    """
    A flexible constructor for GeoSeries._constructor, which needs to be able
    to fall back to a Series (if a certain operation does not produce
    geometries)
    """
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=_SERIES_WARNING_MSG,
                category=FutureWarning,
                module="geopandas[.*]",
            )
            return GeoSeries(data=data, index=index, crs=crs, **kwargs)
    except TypeError:
        return Series(data=data, index=index, **kwargs)


def inherit_doc(cls):
    """
    A decorator adding a docstring from an existing method.
    """

    def decorator(decorated):
        original_method = getattr(cls, decorated.__name__, None)
        if original_method:
            doc = original_method.__doc__ or ""
        else:
            doc = ""

        decorated.__doc__ = doc
        return decorated

    return decorator


class GeoSeries(GeoPandasBase, Series):
    """
    A Series object designed to store shapely geometry objects.

    Parameters
    ----------
    data : array-like, dict, scalar value
        The geometries to store in the GeoSeries.
    index : array-like or Index
        The index for the GeoSeries.
    crs : value (optional)
        Coordinate Reference System of the geometry objects. Can be anything accepted by
        :meth:`pyproj.CRS.from_user_input() <pyproj.crs.CRS.from_user_input>`,
        such as an authority string (eg "EPSG:4326") or a WKT string.

    kwargs
        Additional arguments passed to the Series constructor,
         e.g. ``name``.

    Examples
    --------

    >>> from shapely.geometry import Point
    >>> s = geopandas.GeoSeries([Point(1, 1), Point(2, 2), Point(3, 3)])
    >>> s
    0    POINT (1.00000 1.00000)
    1    POINT (2.00000 2.00000)
    2    POINT (3.00000 3.00000)
    dtype: geometry

    >>> s = geopandas.GeoSeries(
    ...     [Point(1, 1), Point(2, 2), Point(3, 3)], crs="EPSG:3857"
    ... )
    >>> s.crs  # doctest: +SKIP
    <Projected CRS: EPSG:3857>
    Name: WGS 84 / Pseudo-Mercator
    Axis Info [cartesian]:
    - X[east]: Easting (metre)
    - Y[north]: Northing (metre)
    Area of Use:
    - name: World - 85°S to 85°N
    - bounds: (-180.0, -85.06, 180.0, 85.06)
    Coordinate Operation:
    - name: Popular Visualisation Pseudo-Mercator
    - method: Popular Visualisation Pseudo Mercator
    Datum: World Geodetic System 1984
    - Ellipsoid: WGS 84
    - Prime Meridian: Greenwich

    >>> s = geopandas.GeoSeries(
    ...    [Point(1, 1), Point(2, 2), Point(3, 3)], index=["a", "b", "c"], crs=4326
    ... )
    >>> s
    a    POINT (1.00000 1.00000)
    b    POINT (2.00000 2.00000)
    c    POINT (3.00000 3.00000)
    dtype: geometry

    >>> s.crs
    <Geographic 2D CRS: EPSG:4326>
    Name: WGS 84
    Axis Info [ellipsoidal]:
    - Lat[north]: Geodetic latitude (degree)
    - Lon[east]: Geodetic longitude (degree)
    Area of Use:
    - name: World
    - bounds: (-180.0, -90.0, 180.0, 90.0)
    Datum: World Geodetic System 1984
    - Ellipsoid: WGS 84
    - Prime Meridian: Greenwich

    See Also
    --------
    GeoDataFrame
    pandas.Series

    """

    _metadata = ["name"]

    def __new__(cls, data=None, index=None, crs=None, **kwargs):
        # we need to use __new__ because we want to return Series instance
        # instead of GeoSeries instance in case of non-geometry data

        if hasattr(data, "crs") and crs:
            if not data.crs:
                # make a copy to avoid setting CRS to passed GeometryArray
                data = data.copy()
            else:
                if not data.crs == crs:
                    warnings.warn(
                        "CRS mismatch between CRS of the passed geometries "
                        "and 'crs'. Use 'GeoDataFrame.set_crs(crs, "
                        "allow_override=True)' to overwrite CRS or "
                        "'GeoSeries.to_crs(crs)' to reproject geometries. "
                        "CRS mismatch will raise an error in the future versions "
                        "of GeoPandas.",
                        FutureWarning,
                        stacklevel=2,
                    )
                    # TODO: raise error in 0.9 or 0.10.

        if isinstance(data, SingleBlockManager):
            if isinstance(data.blocks[0].dtype, GeometryDtype):
                if data.blocks[0].ndim == 2:
                    # bug in pandas 0.23 where in certain indexing operations
                    # (such as .loc) a 2D ExtensionBlock (still with 1D values
                    # is created) which results in other failures
                    # bug in pandas <= 0.25.0 when len(values) == 1
                    #   (https://github.com/pandas-dev/pandas/issues/27785)
                    from pandas.core.internals import ExtensionBlock

                    values = data.blocks[0].values
                    block = ExtensionBlock(values, slice(0, len(values), 1), ndim=1)
                    data = SingleBlockManager([block], data.axes[0], fastpath=True)
                self = super(GeoSeries, cls).__new__(cls)
                super(GeoSeries, self).__init__(data, index=index, **kwargs)
                self.crs = getattr(self.values, "crs", crs)
                return self
            warnings.warn(_SERIES_WARNING_MSG, FutureWarning, stacklevel=2)
            return Series(data, index=index, **kwargs)

        if isinstance(data, BaseGeometry):
            # fix problem for scalar geometries passed, ensure the list of
            # scalars is of correct length if index is specified
            n = len(index) if index is not None else 1
            data = [data] * n

        name = kwargs.pop("name", None)

        if not is_geometry_type(data):
            # if data is None and dtype is specified (eg from empty overlay
            # test), specifying dtype raises an error:
            # https://github.com/pandas-dev/pandas/issues/26469
            kwargs.pop("dtype", None)
            # Use Series constructor to handle input data
            with ignore_shapely2_warnings():
                s = pd.Series(data, index=index, name=name, **kwargs)
            # prevent trying to convert non-geometry objects
            if s.dtype != object:
                if s.empty:
                    s = s.astype(object)
                else:
                    warnings.warn(_SERIES_WARNING_MSG, FutureWarning, stacklevel=2)
                    return s
            # try to convert to GeometryArray, if fails return plain Series
            try:
                data = from_shapely(s.values, crs)
            except TypeError:
                warnings.warn(_SERIES_WARNING_MSG, FutureWarning, stacklevel=2)
                return s
            index = s.index
            name = s.name

        self = super(GeoSeries, cls).__new__(cls)
        super(GeoSeries, self).__init__(data, index=index, name=name, **kwargs)

        if not self.crs:
            self.crs = crs
        return self

    def __init__(self, *args, **kwargs):
        # need to overwrite Series init to prevent calling it for GeoSeries
        # (doesn't know crs, all work is already done above)
        pass

    def append(self, *args, **kwargs):
        return self._wrapped_pandas_method("append", *args, **kwargs)

    @property
    def geometry(self):
        return self

    @property
    def x(self):
        """Return the x location of point geometries in a GeoSeries

        Returns
        -------
        pandas.Series

        Examples
        --------

        >>> from shapely.geometry import Point
        >>> s = geopandas.GeoSeries([Point(1, 1), Point(2, 2), Point(3, 3)])
        >>> s.x
        0    1.0
        1    2.0
        2    3.0
        dtype: float64

        See Also
        --------

        GeoSeries.y

        """
        return _delegate_property("x", self)

    @property
    def y(self):
        """Return the y location of point geometries in a GeoSeries

        Returns
        -------
        pandas.Series

        Examples
        --------

        >>> from shapely.geometry import Point
        >>> s = geopandas.GeoSeries([Point(1, 1), Point(2, 2), Point(3, 3)])
        >>> s.y
        0    1.0
        1    2.0
        2    3.0
        dtype: float64

        See Also
        --------

        GeoSeries.x

        """
        return _delegate_property("y", self)

    @classmethod
    def from_file(cls, filename, **kwargs):
        """Alternate constructor to create a ``GeoSeries`` from a file.

        Can load a ``GeoSeries`` from a file from any format recognized by
        `fiona`. See http://fiona.readthedocs.io/en/latest/manual.html for details.
        From a file with attributes loads only geometry column. Note that to do
        that, GeoPandas first loads the whole GeoDataFrame.

        Parameters
        ----------
        filename : str
            File path or file handle to read from. Depending on which kwargs
            are included, the content of filename may vary. See
            http://fiona.readthedocs.io/en/latest/README.html#usage for usage details.
        kwargs : key-word arguments
            These arguments are passed to fiona.open, and can be used to
            access multi-layer data, data stored within archives (zip files),
            etc.

        Examples
        --------

        >>> path = geopandas.datasets.get_path('nybb')
        >>> s = geopandas.GeoSeries.from_file(path)
        >>> s
        0    MULTIPOLYGON (((970217.022 145643.332, 970227....
        1    MULTIPOLYGON (((1029606.077 156073.814, 102957...
        2    MULTIPOLYGON (((1021176.479 151374.797, 102100...
        3    MULTIPOLYGON (((981219.056 188655.316, 980940....
        4    MULTIPOLYGON (((1012821.806 229228.265, 101278...
        Name: geometry, dtype: geometry
        """
        from geopandas import GeoDataFrame

        df = GeoDataFrame.from_file(filename, **kwargs)

        return GeoSeries(df.geometry, crs=df.crs)

    @property
    def __geo_interface__(self):
        """Returns a ``GeoSeries`` as a python feature collection.

        Implements the `geo_interface`. The returned python data structure
        represents the ``GeoSeries`` as a GeoJSON-like ``FeatureCollection``.
        Note that the features will have an empty ``properties`` dict as they
        don't have associated attributes (geometry only).

        Examples
        --------

        >>> from shapely.geometry import Point
        >>> s = geopandas.GeoSeries([Point(1, 1), Point(2, 2), Point(3, 3)])
        >>> s.__geo_interface__
        {'type': 'FeatureCollection', 'features': [{'id': '0', 'type': 'Feature', \
'properties': {}, 'geometry': {'type': 'Point', 'coordinates': (1.0, 1.0)}, \
'bbox': (1.0, 1.0, 1.0, 1.0)}, {'id': '1', 'type': 'Feature', \
'properties': {}, 'geometry': {'type': 'Point', 'coordinates': (2.0, 2.0)}, \
'bbox': (2.0, 2.0, 2.0, 2.0)}, {'id': '2', 'type': 'Feature', 'properties': \
{}, 'geometry': {'type': 'Point', 'coordinates': (3.0, 3.0)}, 'bbox': (3.0, \
3.0, 3.0, 3.0)}], 'bbox': (1.0, 1.0, 3.0, 3.0)}
        """
        from geopandas import GeoDataFrame

        return GeoDataFrame({"geometry": self}).__geo_interface__

    def to_file(self, filename, driver="ESRI Shapefile", index=None, **kwargs):
        """Write the ``GeoSeries`` to a file.

        By default, an ESRI shapefile is written, but any OGR data source
        supported by Fiona can be written.

        Parameters
        ----------
        filename : string
            File path or file handle to write to.
        driver : string, default: 'ESRI Shapefile'
            The OGR format driver used to write the vector file.
        index : bool, default None
            If True, write index into one or more columns (for MultiIndex).
            Default None writes the index into one or more columns only if
            the index is named, is a MultiIndex, or has a non-integer data
            type. If False, no index is written.

            .. versionadded:: 0.7
                Previously the index was not written.

        Notes
        -----
        The extra keyword arguments ``**kwargs`` are passed to fiona.open and
        can be used to write to multi-layer data, store data within archives
        (zip files), etc.

        See Also
        --------
        GeoDataFrame.to_file

        Examples
        --------

        >>> s.to_file('series.shp')  # doctest: +SKIP

        >>> s.to_file('series.gpkg', driver='GPKG', layer='name1')  # doctest: +SKIP

        >>> s.to_file('series.geojson', driver='GeoJSON')  # doctest: +SKIP
        """
        from geopandas import GeoDataFrame

        data = GeoDataFrame({"geometry": self}, index=self.index)
        data.crs = self.crs
        data.to_file(filename, driver, index=index, **kwargs)

    #
    # Implement pandas methods
    #

    @property
    def _constructor(self):
        return _geoseries_constructor_with_fallback

    @property
    def _constructor_expanddim(self):
        from geopandas import GeoDataFrame

        return GeoDataFrame

    def _wrapped_pandas_method(self, mtd, *args, **kwargs):
        """Wrap a generic pandas method to ensure it returns a GeoSeries"""
        val = getattr(super(GeoSeries, self), mtd)(*args, **kwargs)
        if type(val) == Series:
            val.__class__ = GeoSeries
            val.crs = self.crs
        return val

    def __getitem__(self, key):
        return self._wrapped_pandas_method("__getitem__", key)

    @inherit_doc(pd.Series)
    def sort_index(self, *args, **kwargs):
        return self._wrapped_pandas_method("sort_index", *args, **kwargs)

    @inherit_doc(pd.Series)
    def take(self, *args, **kwargs):
        return self._wrapped_pandas_method("take", *args, **kwargs)

    @inherit_doc(pd.Series)
    def select(self, *args, **kwargs):
        return self._wrapped_pandas_method("select", *args, **kwargs)

    @inherit_doc(pd.Series)
    def apply(self, func, convert_dtype=True, args=(), **kwargs):
        result = super().apply(func, convert_dtype=convert_dtype, args=args, **kwargs)
        if isinstance(result, GeoSeries):
            if self.crs is not None:
                result.set_crs(self.crs, inplace=True)
        return result

    def __finalize__(self, other, method=None, **kwargs):
        """ propagate metadata from other to self """
        # NOTE: backported from pandas master (upcoming v0.13)
        for name in self._metadata:
            object.__setattr__(self, name, getattr(other, name, None))
        return self

    def isna(self):
        """
        Detect missing values.

        Historically, NA values in a GeoSeries could be represented by
        empty geometric objects, in addition to standard representations
        such as None and np.nan. This behaviour is changed in version 0.6.0,
        and now only actual missing values return True. To detect empty
        geometries, use ``GeoSeries.is_empty`` instead.

        Returns
        -------
        A boolean pandas Series of the same size as the GeoSeries,
        True where a value is NA.

        Examples
        --------

        >>> from shapely.geometry import Polygon
        >>> s = geopandas.GeoSeries(
        ...     [Polygon([(0, 0), (1, 1), (0, 1)]), None, Polygon([])]
        ... )
        >>> s
        0    POLYGON ((0.00000 0.00000, 1.00000 1.00000, 0....
        1                                                 None
        2                             GEOMETRYCOLLECTION EMPTY
        dtype: geometry
        >>> s.isna()
        0    False
        1     True
        2    False
        dtype: bool

        See Also
        --------
        GeoSeries.notna : inverse of isna
        GeoSeries.is_empty : detect empty geometries
        """
        if self.is_empty.any():
            warnings.warn(
                "GeoSeries.isna() previously returned True for both missing (None) "
                "and empty geometries. Now, it only returns True for missing values. "
                "Since the calling GeoSeries contains empty geometries, the result "
                "has changed compared to previous versions of GeoPandas.\n"
                "Given a GeoSeries 's', you can use 's.is_empty | s.isna()' to get "
                "back the old behaviour.\n\n"
                "To further ignore this warning, you can do: \n"
                "import warnings; warnings.filterwarnings('ignore', 'GeoSeries.isna', "
                "UserWarning)",
                UserWarning,
                stacklevel=2,
            )

        return super(GeoSeries, self).isna()

    def isnull(self):
        """Alias for `isna` method. See `isna` for more detail."""
        return self.isna()

    def notna(self):
        """
        Detect non-missing values.

        Historically, NA values in a GeoSeries could be represented by
        empty geometric objects, in addition to standard representations
        such as None and np.nan. This behaviour is changed in version 0.6.0,
        and now only actual missing values return False. To detect empty
        geometries, use ``~GeoSeries.is_empty`` instead.

        Returns
        -------
        A boolean pandas Series of the same size as the GeoSeries,
        False where a value is NA.

        Examples
        --------

        >>> from shapely.geometry import Polygon
        >>> s = geopandas.GeoSeries(
        ...     [Polygon([(0, 0), (1, 1), (0, 1)]), None, Polygon([])]
        ... )
        >>> s
        0    POLYGON ((0.00000 0.00000, 1.00000 1.00000, 0....
        1                                                 None
        2                             GEOMETRYCOLLECTION EMPTY
        dtype: geometry
        >>> s.notna()
        0     True
        1    False
        2     True
        dtype: bool

        See Also
        --------
        GeoSeries.isna : inverse of notna
        GeoSeries.is_empty : detect empty geometries
        """
        if self.is_empty.any():
            warnings.warn(
                "GeoSeries.notna() previously returned False for both missing (None) "
                "and empty geometries. Now, it only returns False for missing values. "
                "Since the calling GeoSeries contains empty geometries, the result "
                "has changed compared to previous versions of GeoPandas.\n"
                "Given a GeoSeries 's', you can use '~s.is_empty & s.notna()' to get "
                "back the old behaviour.\n\n"
                "To further ignore this warning, you can do: \n"
                "import warnings; warnings.filterwarnings('ignore', "
                "'GeoSeries.notna', UserWarning)",
                UserWarning,
                stacklevel=2,
            )
        return super(GeoSeries, self).notna()

    def notnull(self):
        """Alias for `notna` method. See `notna` for more detail."""
        return self.notna()

    def fillna(self, value=None, method=None, inplace=False, **kwargs):
        """Fill NA values with a geometry (empty polygon by default).

        "method" is currently not implemented for pandas <= 0.12.

        Examples
        --------

        >>> from shapely.geometry import Polygon
        >>> s = geopandas.GeoSeries(
        ...     [
        ...         Polygon([(0, 0), (1, 1), (0, 1)]),
        ...         None,
        ...         Polygon([(0, 0), (-1, 1), (0, -1)]),
        ...     ]
        ... )
        >>> s
        0    POLYGON ((0.00000 0.00000, 1.00000 1.00000, 0....
        1                                                 None
        2    POLYGON ((0.00000 0.00000, -1.00000 1.00000, 0...
        dtype: geometry

        >>> s.fillna()
        0    POLYGON ((0.00000 0.00000, 1.00000 1.00000, 0....
        1                             GEOMETRYCOLLECTION EMPTY
        2    POLYGON ((0.00000 0.00000, -1.00000 1.00000, 0...
        dtype: geometry

        >>> s.fillna(Polygon([(0, 1), (2, 1), (1, 2)]))
        0    POLYGON ((0.00000 0.00000, 1.00000 1.00000, 0....
        1    POLYGON ((0.00000 1.00000, 2.00000 1.00000, 1....
        2    POLYGON ((0.00000 0.00000, -1.00000 1.00000, 0...
        dtype: geometry
        """
        if value is None:
            value = BaseGeometry()
        return super(GeoSeries, self).fillna(
            value=value, method=method, inplace=inplace, **kwargs
        )

    def __contains__(self, other):
        """Allow tests of the form "geom in s"

        Tests whether a GeoSeries contains a geometry.

        Note: This is not the same as the geometric method "contains".
        """
        if isinstance(other, BaseGeometry):
            return np.any(self.geom_equals(other))
        else:
            return False

    def plot(self, *args, **kwargs):
        """Generate a plot of the geometries in the ``GeoSeries``.

        Wraps the ``plot_series()`` function, and documentation is copied from
        there.
        """
        return plot_series(self, *args, **kwargs)

    plot.__doc__ = plot_series.__doc__

    def explode(self):
        """
        Explode multi-part geometries into multiple single geometries.

        Single rows can become multiple rows.
        This is analogous to PostGIS's ST_Dump(). The 'path' index is the
        second level of the returned MultiIndex

        Returns
        ------
        A GeoSeries with a MultiIndex. The levels of the MultiIndex are the
        original index and a zero-based integer index that counts the
        number of single geometries within a multi-part geometry.

        Examples
        --------
        >>> from shapely.geometry import MultiPoint
        >>> s = geopandas.GeoSeries(
        ...     [MultiPoint([(0, 0), (1, 1)]), MultiPoint([(2, 2), (3, 3), (4, 4)])]
        ... )
        >>> s
        0        MULTIPOINT (0.00000 0.00000, 1.00000 1.00000)
        1    MULTIPOINT (2.00000 2.00000, 3.00000 3.00000, ...
        dtype: geometry

        >>> s.explode()
        0  0    POINT (0.00000 0.00000)
           1    POINT (1.00000 1.00000)
        1  0    POINT (2.00000 2.00000)
           1    POINT (3.00000 3.00000)
           2    POINT (4.00000 4.00000)
        dtype: geometry

        See also
        --------
        GeoDataFrame.explode

        """
        index = []
        geometries = []
        for idx, s in self.geometry.iteritems():
            if s.type.startswith("Multi") or s.type == "GeometryCollection":
                geoms = s.geoms
                idxs = [(idx, i) for i in range(len(geoms))]
            else:
                geoms = [s]
                idxs = [(idx, 0)]
            index.extend(idxs)
            geometries.extend(geoms)
        index = MultiIndex.from_tuples(index, names=self.index.names + [None])
        return GeoSeries(geometries, index=index, crs=self.crs).__finalize__(self)

    #
    # Additional methods
    #

    def set_crs(self, crs=None, epsg=None, inplace=False, allow_override=False):
        """
        Set the Coordinate Reference System (CRS) of a ``GeoSeries``.

        NOTE: The underlying geometries are not transformed to this CRS. To
        transform the geometries to a new CRS, use the ``to_crs`` method.

        Parameters
        ----------
        crs : pyproj.CRS, optional if `epsg` is specified
            The value can be anything accepted
            by :meth:`pyproj.CRS.from_user_input() <pyproj.crs.CRS.from_user_input>`,
            such as an authority string (eg "EPSG:4326") or a WKT string.
        epsg : int, optional if `crs` is specified
            EPSG code specifying the projection.
        inplace : bool, default False
            If True, the CRS of the GeoSeries will be changed in place
            (while still returning the result) instead of making a copy of
            the GeoSeries.
        allow_override : bool, default False
            If the the GeoSeries already has a CRS, allow to replace the
            existing CRS, even when both are not equal.

        Returns
        -------
        GeoSeries

        Examples
        --------
        >>> from shapely.geometry import Point
        >>> s = geopandas.GeoSeries([Point(1, 1), Point(2, 2), Point(3, 3)])
        >>> s
        0    POINT (1.00000 1.00000)
        1    POINT (2.00000 2.00000)
        2    POINT (3.00000 3.00000)
        dtype: geometry

        Setting CRS to a GeoSeries without one:

        >>> s.crs is None
        True

        >>> s = s.set_crs('epsg:3857')
        >>> s.crs  # doctest: +SKIP
        <Projected CRS: EPSG:3857>
        Name: WGS 84 / Pseudo-Mercator
        Axis Info [cartesian]:
        - X[east]: Easting (metre)
        - Y[north]: Northing (metre)
        Area of Use:
        - name: World - 85°S to 85°N
        - bounds: (-180.0, -85.06, 180.0, 85.06)
        Coordinate Operation:
        - name: Popular Visualisation Pseudo-Mercator
        - method: Popular Visualisation Pseudo Mercator
        Datum: World Geodetic System 1984
        - Ellipsoid: WGS 84
        - Prime Meridian: Greenwich

        Overriding existing CRS:

        >>> s = s.set_crs(4326, allow_override=True)

        Without ``allow_override=True``, ``set_crs`` returns an error if you try to
        override CRS.

        """
        if crs is not None:
            crs = CRS.from_user_input(crs)
        elif epsg is not None:
            crs = CRS.from_epsg(epsg)
        else:
            raise ValueError("Must pass either crs or epsg.")

        if not allow_override and self.crs is not None and not self.crs == crs:
            raise ValueError(
                "The GeoSeries already has a CRS which is not equal to the passed "
                "CRS. Specify 'allow_override=True' to allow replacing the existing "
                "CRS without doing any transformation. If you actually want to "
                "transform the geometries, use 'GeoSeries.to_crs' instead."
            )
        if not inplace:
            result = self.copy()
        else:
            result = self
        result.crs = crs
        return result

    def to_crs(self, crs=None, epsg=None):
        """Returns a ``GeoSeries`` with all geometries transformed to a new
        coordinate reference system.

        Transform all geometries in a GeoSeries to a different coordinate
        reference system.  The ``crs`` attribute on the current GeoSeries must
        be set.  Either ``crs`` or ``epsg`` may be specified for output.

        This method will transform all points in all objects.  It has no notion
        or projecting entire geometries.  All segments joining points are
        assumed to be lines in the current projection, not geodesics.  Objects
        crossing the dateline (or other projection boundary) will have
        undesirable behavior.

        Parameters
        ----------
        crs : pyproj.CRS, optional if `epsg` is specified
            The value can be anything accepted
            by :meth:`pyproj.CRS.from_user_input() <pyproj.crs.CRS.from_user_input>`,
            such as an authority string (eg "EPSG:4326") or a WKT string.
        epsg : int, optional if `crs` is specified
            EPSG code specifying output projection.

        Returns
        -------
        GeoSeries

        Examples
        --------
        >>> from shapely.geometry import Point
        >>> s = geopandas.GeoSeries([Point(1, 1), Point(2, 2), Point(3, 3)], crs=4326)
        >>> s
        0    POINT (1.00000 1.00000)
        1    POINT (2.00000 2.00000)
        2    POINT (3.00000 3.00000)
        dtype: geometry
        >>> s.crs  # doctest: +SKIP
        <Geographic 2D CRS: EPSG:4326>
        Name: WGS 84
        Axis Info [ellipsoidal]:
        - Lat[north]: Geodetic latitude (degree)
        - Lon[east]: Geodetic longitude (degree)
        Area of Use:
        - name: World
        - bounds: (-180.0, -90.0, 180.0, 90.0)
        Datum: World Geodetic System 1984
        - Ellipsoid: WGS 84
        - Prime Meridian: Greenwich

        >>> s = s.to_crs(3857)
        >>> s
        0    POINT (111319.491 111325.143)
        1    POINT (222638.982 222684.209)
        2    POINT (333958.472 334111.171)
        dtype: geometry
        >>> s.crs  # doctest: +SKIP
        <Projected CRS: EPSG:3857>
        Name: WGS 84 / Pseudo-Mercator
        Axis Info [cartesian]:
        - X[east]: Easting (metre)
        - Y[north]: Northing (metre)
        Area of Use:
        - name: World - 85°S to 85°N
        - bounds: (-180.0, -85.06, 180.0, 85.06)
        Coordinate Operation:
        - name: Popular Visualisation Pseudo-Mercator
        - method: Popular Visualisation Pseudo Mercator
        Datum: World Geodetic System 1984
        - Ellipsoid: WGS 84
        - Prime Meridian: Greenwich

        """
        if self.crs is None:
            raise ValueError(
                "Cannot transform naive geometries.  "
                "Please set a crs on the object first."
            )
        if crs is not None:
            crs = CRS.from_user_input(crs)
        elif epsg is not None:
            crs = CRS.from_epsg(epsg)
        else:
            raise ValueError("Must pass either crs or epsg.")

        # skip if the input CRS and output CRS are the exact same
        if self.crs.is_exact_same(crs):
            return self

        transformer = Transformer.from_crs(self.crs, crs, always_xy=True)

        new_data = vectorized.transform(self.values.data, transformer.transform)
        return GeoSeries(
            GeometryArray(new_data), crs=crs, index=self.index, name=self.name
        )

    def estimate_utm_crs(self, datum_name="WGS 84"):
        """Returns the estimated UTM CRS based on the bounds of the dataset.

        .. versionadded:: 0.9

        .. note:: Requires pyproj 3+

        Parameters
        ----------
        datum_name : str, optional
            The name of the datum to use in the query. Default is WGS 84.

        Returns
        -------
        pyproj.CRS

        Examples
        --------
        >>> world = geopandas.read_file(
        ...     geopandas.datasets.get_path("naturalearth_lowres")
        ... )
        >>> germany = world.loc[world.name == "Germany"]
        >>> germany.geometry.estimate_utm_crs()  # doctest: +SKIP
        <Projected CRS: EPSG:32632>
        Name: WGS 84 / UTM zone 32N
        Axis Info [cartesian]:
        - E[east]: Easting (metre)
        - N[north]: Northing (metre)
        Area of Use:
        - name: World - N hemisphere - 6°E to 12°E - by country
        - bounds: (6.0, 0.0, 12.0, 84.0)
        Coordinate Operation:
        - name: UTM zone 32N
        - method: Transverse Mercator
        Datum: World Geodetic System 1984
        - Ellipsoid: WGS 84
        - Prime Meridian: Greenwich
        """
        try:
            from pyproj.aoi import AreaOfInterest
            from pyproj.database import query_utm_crs_info
        except ImportError:
            raise RuntimeError("pyproj 3+ required for estimate_utm_crs.")

        if not self.crs:
            raise RuntimeError("crs must be set to estimate UTM CRS.")

        minx, miny, maxx, maxy = self.total_bounds
        # ensure using geographic coordinates
        if not self.crs.is_geographic:
            lon, lat = Transformer.from_crs(
                self.crs, "EPSG:4326", always_xy=True
            ).transform((minx, maxx, minx, maxx), (miny, miny, maxy, maxy))
            x_center = np.mean(lon)
            y_center = np.mean(lat)
        else:
            x_center = np.mean([minx, maxx])
            y_center = np.mean([miny, maxy])

        utm_crs_list = query_utm_crs_info(
            datum_name=datum_name,
            area_of_interest=AreaOfInterest(
                west_lon_degree=x_center,
                south_lat_degree=y_center,
                east_lon_degree=x_center,
                north_lat_degree=y_center,
            ),
        )
        try:
            return CRS.from_epsg(utm_crs_list[0].code)
        except IndexError:
            raise RuntimeError("Unable to determine UTM CRS")

    def to_json(self, **kwargs):
        """
        Returns a GeoJSON string representation of the GeoSeries.

        Parameters
        ----------
        *kwargs* that will be passed to json.dumps().

        Returns
        -------
        JSON string

        Examples
        --------
        >>> from shapely.geometry import Point
        >>> s = geopandas.GeoSeries([Point(1, 1), Point(2, 2), Point(3, 3)])
        >>> s
        0    POINT (1.00000 1.00000)
        1    POINT (2.00000 2.00000)
        2    POINT (3.00000 3.00000)
        dtype: geometry

        >>> s.to_json()
        '{"type": "FeatureCollection", "features": [{"id": "0", "type": "Feature", "pr\
operties": {}, "geometry": {"type": "Point", "coordinates": [1.0, 1.0]}, "bbox": [1.0,\
 1.0, 1.0, 1.0]}, {"id": "1", "type": "Feature", "properties": {}, "geometry": {"type"\
: "Point", "coordinates": [2.0, 2.0]}, "bbox": [2.0, 2.0, 2.0, 2.0]}, {"id": "2", "typ\
e": "Feature", "properties": {}, "geometry": {"type": "Point", "coordinates": [3.0, 3.\
0]}, "bbox": [3.0, 3.0, 3.0, 3.0]}], "bbox": [1.0, 1.0, 3.0, 3.0]}'
        """
        return json.dumps(self.__geo_interface__, **kwargs)

    #
    # Implement standard operators for GeoSeries
    #

    def __xor__(self, other):
        """Implement ^ operator as for builtin set type"""
        warnings.warn(
            "'^' operator will be deprecated. Use the 'symmetric_difference' "
            "method instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.symmetric_difference(other)

    def __or__(self, other):
        """Implement | operator as for builtin set type"""
        warnings.warn(
            "'|' operator will be deprecated. Use the 'union' method instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.union(other)

    def __and__(self, other):
        """Implement & operator as for builtin set type"""
        warnings.warn(
            "'&' operator will be deprecated. Use the 'intersection' method instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.intersection(other)

    def __sub__(self, other):
        """Implement - operator as for builtin set type"""
        warnings.warn(
            "'-' operator will be deprecated. Use the 'difference' method instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.difference(other)
