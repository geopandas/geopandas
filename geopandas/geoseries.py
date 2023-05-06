import json
import warnings

import numpy as np
import pandas as pd
from pandas import Series, MultiIndex, DataFrame
from pandas.core.internals import SingleBlockManager

from pyproj import CRS
import shapely
from shapely.geometry.base import BaseGeometry
from shapely.geometry import GeometryCollection

from geopandas.base import GeoPandasBase, _delegate_property
from geopandas.plotting import plot_series
from geopandas.explore import _explore_geoseries
import geopandas

from . import _compat as compat
from ._decorator import doc
from .array import (
    GeometryDtype,
    from_shapely,
    from_wkb,
    from_wkt,
    points_from_xy,
    to_wkb,
    to_wkt,
)
from .base import is_geometry_type


def _geoseries_constructor_with_fallback(data=None, index=None, crs=None, **kwargs):
    """
    A flexible constructor for GeoSeries._constructor, which needs to be able
    to fall back to a Series (if a certain operation does not produce
    geometries)
    """
    try:
        return GeoSeries(data=data, index=index, crs=crs, **kwargs)
    except TypeError:
        return Series(data=data, index=index, **kwargs)


def _geoseries_expanddim(data=None, *args, **kwargs):
    from geopandas import GeoDataFrame

    # pd.Series._constructor_expanddim == pd.DataFrame
    df = pd.DataFrame(data, *args, **kwargs)
    geo_col_name = None
    if isinstance(data, GeoSeries):
        # pandas default column name is 0, keep convention
        geo_col_name = data.name if data.name is not None else 0

    if df.shape[1] == 1:
        geo_col_name = df.columns[0]

    if (df.dtypes == "geometry").sum() > 0:
        if geo_col_name is None or not is_geometry_type(df[geo_col_name]):
            df = GeoDataFrame(df)
            df._geometry_column_name = None
        else:
            df = df.set_geometry(geo_col_name)

    return df


# pd.concat (pandas/core/reshape/concat.py) requires this for the
# concatenation of series since pandas 1.1
# (https://github.com/pandas-dev/pandas/commit/f9e4c8c84bcef987973f2624cc2932394c171c8c)
_geoseries_expanddim._get_axis_number = DataFrame._get_axis_number


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
    - name: World.
    - bounds: (-180.0, -90.0, 180.0, 90.0)
    Datum: World Geodetic System 1984 ensemble
    - Ellipsoid: WGS 84
    - Prime Meridian: Greenwich

    See Also
    --------
    GeoDataFrame
    pandas.Series

    """

    _metadata = ["name"]

    def __init__(self, data=None, index=None, crs=None, **kwargs):
        if hasattr(data, "crs") and crs:
            if not data.crs:
                # make a copy to avoid setting CRS to passed GeometryArray
                data = data.copy()
            else:
                if not data.crs == crs:
                    raise ValueError(
                        "CRS mismatch between CRS of the passed geometries "
                        "and 'crs'. Use 'GeoSeries.set_crs(crs, "
                        "allow_override=True)' to overwrite CRS or "
                        "'GeoSeries.to_crs(crs)' to reproject geometries. "
                    )

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
            else:
                raise TypeError(
                    "Non geometry data passed to GeoSeries constructor, "
                    f"received data of dtype '{data.blocks[0].dtype}'"
                )

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
            with compat.ignore_shapely2_warnings():
                # suppress additional warning from pandas for empty data
                # (will always give object dtype instead of float dtype in the future,
                # making the `if s.empty: s = s.astype(object)` below unnecessary)
                empty_msg = "The default dtype for empty Series"
                warnings.filterwarnings("ignore", empty_msg, DeprecationWarning)
                warnings.filterwarnings("ignore", empty_msg, FutureWarning)
                s = pd.Series(data, index=index, name=name, **kwargs)
            # prevent trying to convert non-geometry objects
            if s.dtype != object:
                if (s.empty and s.dtype == "float64") or data is None:
                    # pd.Series with empty data gives float64 for older pandas versions
                    s = s.astype(object)
                else:
                    raise TypeError(
                        "Non geometry data passed to GeoSeries constructor, "
                        f"received data of dtype '{s.dtype}'"
                    )
            # try to convert to GeometryArray, if fails return plain Series
            try:
                data = from_shapely(s.values, crs)
            except TypeError:
                raise TypeError(
                    "Non geometry data passed to GeoSeries constructor, "
                    f"received data of dtype '{s.dtype}'"
                )
            index = s.index
            name = s.name

        super().__init__(data, index=index, name=name, **kwargs)
        if not self.crs:
            self.crs = crs

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
        GeoSeries.z

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
        GeoSeries.z

        """
        return _delegate_property("y", self)

    @property
    def z(self):
        """Return the z location of point geometries in a GeoSeries

        Returns
        -------
        pandas.Series

        Examples
        --------

        >>> from shapely.geometry import Point
        >>> s = geopandas.GeoSeries([Point(1, 1, 1), Point(2, 2, 2), Point(3, 3, 3)])
        >>> s.z
        0    1.0
        1    2.0
        2    3.0
        dtype: float64

        See Also
        --------

        GeoSeries.x
        GeoSeries.y

        """
        return _delegate_property("z", self)

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
        >>> import geodatasets
        >>> path = geodatasets.get_path('nybb')
        >>> s = geopandas.GeoSeries.from_file(path)
        >>> s
        0    MULTIPOLYGON (((970217.022 145643.332, 970227....
        1    MULTIPOLYGON (((1029606.077 156073.814, 102957...
        2    MULTIPOLYGON (((1021176.479 151374.797, 102100...
        3    MULTIPOLYGON (((981219.056 188655.316, 980940....
        4    MULTIPOLYGON (((1012821.806 229228.265, 101278...
        Name: geometry, dtype: geometry

        See Also
        --------
        read_file : read file to GeoDataFame
        """
        from geopandas import GeoDataFrame

        df = GeoDataFrame.from_file(filename, **kwargs)

        return GeoSeries(df.geometry, crs=df.crs)

    @classmethod
    def from_wkb(cls, data, index=None, crs=None, **kwargs):
        """
        Alternate constructor to create a ``GeoSeries``
        from a list or array of WKB objects

        Parameters
        ----------
        data : array-like or Series
            Series, list or array of WKB objects
        index : array-like or Index
            The index for the GeoSeries.
        crs : value, optional
            Coordinate Reference System of the geometry objects. Can be anything
            accepted by
            :meth:`pyproj.CRS.from_user_input() <pyproj.crs.CRS.from_user_input>`,
            such as an authority string (eg "EPSG:4326") or a WKT string.
        kwargs
            Additional arguments passed to the Series constructor,
            e.g. ``name``.

        Returns
        -------
        GeoSeries

        See Also
        --------
        GeoSeries.from_wkt

        """
        return cls._from_wkb_or_wkb(from_wkb, data, index=index, crs=crs, **kwargs)

    @classmethod
    def from_wkt(cls, data, index=None, crs=None, **kwargs):
        """
        Alternate constructor to create a ``GeoSeries``
        from a list or array of WKT objects

        Parameters
        ----------
        data : array-like, Series
            Series, list, or array of WKT objects
        index : array-like or Index
            The index for the GeoSeries.
        crs : value, optional
            Coordinate Reference System of the geometry objects. Can be anything
            accepted by
            :meth:`pyproj.CRS.from_user_input() <pyproj.crs.CRS.from_user_input>`,
            such as an authority string (eg "EPSG:4326") or a WKT string.
        kwargs
            Additional arguments passed to the Series constructor,
            e.g. ``name``.

        Returns
        -------
        GeoSeries

        See Also
        --------
        GeoSeries.from_wkb

        Examples
        --------

        >>> wkts = [
        ... 'POINT (1 1)',
        ... 'POINT (2 2)',
        ... 'POINT (3 3)',
        ... ]
        >>> s = geopandas.GeoSeries.from_wkt(wkts)
        >>> s
        0    POINT (1.00000 1.00000)
        1    POINT (2.00000 2.00000)
        2    POINT (3.00000 3.00000)
        dtype: geometry
        """
        return cls._from_wkb_or_wkb(from_wkt, data, index=index, crs=crs, **kwargs)

    @classmethod
    def from_xy(cls, x, y, z=None, index=None, crs=None, **kwargs):
        """
        Alternate constructor to create a :class:`~geopandas.GeoSeries` of Point
        geometries from lists or arrays of x, y(, z) coordinates

        In case of geographic coordinates, it is assumed that longitude is captured
        by ``x`` coordinates and latitude by ``y``.

        Parameters
        ----------
        x, y, z : iterable
        index : array-like or Index, optional
            The index for the GeoSeries. If not given and all coordinate inputs
            are Series with an equal index, that index is used.
        crs : value, optional
            Coordinate Reference System of the geometry objects. Can be anything
            accepted by
            :meth:`pyproj.CRS.from_user_input() <pyproj.crs.CRS.from_user_input>`,
            such as an authority string (eg "EPSG:4326") or a WKT string.
        **kwargs
            Additional arguments passed to the Series constructor,
            e.g. ``name``.

        Returns
        -------
        GeoSeries

        See Also
        --------
        GeoSeries.from_wkt
        points_from_xy

        Examples
        --------

        >>> x = [2.5, 5, -3.0]
        >>> y = [0.5, 1, 1.5]
        >>> s = geopandas.GeoSeries.from_xy(x, y, crs="EPSG:4326")
        >>> s
        0    POINT (2.50000 0.50000)
        1    POINT (5.00000 1.00000)
        2    POINT (-3.00000 1.50000)
        dtype: geometry
        """
        if index is None:
            if (
                isinstance(x, Series)
                and isinstance(y, Series)
                and x.index.equals(y.index)
                and (z is None or (isinstance(z, Series) and x.index.equals(z.index)))
            ):  # check if we can reuse index
                index = x.index
        return cls(points_from_xy(x, y, z, crs=crs), index=index, crs=crs, **kwargs)

    @classmethod
    def _from_wkb_or_wkb(
        cls, from_wkb_or_wkt_function, data, index=None, crs=None, **kwargs
    ):
        """Create a GeoSeries from either WKT or WKB values"""
        if isinstance(data, Series):
            if index is not None:
                data = data.reindex(index)
            else:
                index = data.index
            data = data.values
        return cls(from_wkb_or_wkt_function(data, crs=crs), index=index, **kwargs)

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

    def to_file(self, filename, driver=None, index=None, **kwargs):
        """Write the ``GeoSeries`` to a file.

        By default, an ESRI shapefile is written, but any OGR data source
        supported by Fiona can be written.

        Parameters
        ----------
        filename : string
            File path or file handle to write to. The path may specify a
            GDAL VSI scheme.
        driver : string, default None
            The OGR format driver used to write the vector file.
            If not specified, it attempts to infer it from the file extension.
            If no extension is specified, it saves ESRI Shapefile to a folder.
        index : bool, default None
            If True, write index into one or more columns (for MultiIndex).
            Default None writes the index into one or more columns only if
            the index is named, is a MultiIndex, or has a non-integer data
            type. If False, no index is written.

            .. versionadded:: 0.7
                Previously the index was not written.
        mode : string, default 'w'
            The write mode, 'w' to overwrite the existing file and 'a' to append.
            Not all drivers support appending. The drivers that support appending
            are listed in fiona.supported_drivers or
            https://github.com/Toblerity/Fiona/blob/master/fiona/drvsupport.py
        crs : pyproj.CRS, default None
            If specified, the CRS is passed to Fiona to
            better control how the file is written. If None, GeoPandas
            will determine the crs based on crs df attribute.
            The value can be anything accepted
            by :meth:`pyproj.CRS.from_user_input() <pyproj.crs.CRS.from_user_input>`,
            such as an authority string (eg "EPSG:4326") or a WKT string.
        engine : str, "fiona" or "pyogrio"
            The underlying library that is used to write the file. Currently, the
            supported options are "fiona" and "pyogrio". Defaults to "fiona" if
            installed, otherwise tries "pyogrio".
        **kwargs :
            Keyword args to be passed to the engine, and can be used to write
            to multi-layer data, store data within archives (zip files), etc.
            In case of the "fiona" engine, the keyword arguments are passed to
            fiona.open`. For more information on possible keywords, type:
            ``import fiona; help(fiona.open)``. In case of the "pyogrio" engine,
            the keyword arguments are passed to `pyogrio.write_dataframe`.

        See Also
        --------
        GeoDataFrame.to_file : write GeoDataFrame to file
        read_file : read file to GeoDataFame

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
        return _geoseries_expanddim

    def _wrapped_pandas_method(self, mtd, *args, **kwargs):
        """Wrap a generic pandas method to ensure it returns a GeoSeries"""
        val = getattr(super(), mtd)(*args, **kwargs)
        if type(val) == Series:
            val.__class__ = GeoSeries
            val.crs = self.crs
        return val

    def __getitem__(self, key):
        return self._wrapped_pandas_method("__getitem__", key)

    @doc(pd.Series)
    def sort_index(self, *args, **kwargs):
        return self._wrapped_pandas_method("sort_index", *args, **kwargs)

    @doc(pd.Series)
    def take(self, *args, **kwargs):
        return self._wrapped_pandas_method("take", *args, **kwargs)

    @doc(pd.Series)
    def select(self, *args, **kwargs):
        return self._wrapped_pandas_method("select", *args, **kwargs)

    @doc(pd.Series)
    def apply(self, func, convert_dtype=True, args=(), **kwargs):
        result = super().apply(func, convert_dtype=convert_dtype, args=args, **kwargs)
        if isinstance(result, GeoSeries):
            if self.crs is not None:
                result.set_crs(self.crs, inplace=True)
        return result

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
        2                                        POLYGON EMPTY
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
        return super().isna()

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
        2                                        POLYGON EMPTY
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
        return super().notna()

    def notnull(self):
        """Alias for `notna` method. See `notna` for more detail."""
        return self.notna()

    def fillna(self, value=None, method=None, inplace=False, **kwargs):
        """
        Fill NA values with geometry (or geometries).

        ``method`` is currently not implemented.

        Parameters
        ----------
        value : shapely geometry or GeoSeries, default None
            If None is passed, NA values will be filled with GEOMETRYCOLLECTION EMPTY.
            If a shapely geometry object is passed, it will be
            used to fill all missing values. If a ``GeoSeries`` or ``GeometryArray``
            are passed, missing values will be filled based on the corresponding index
            locations. If pd.NA or np.nan are passed, values will be filled with
            ``None`` (not GEOMETRYCOLLECTION EMPTY).

        Returns
        -------
        GeoSeries

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

        Filled with an empty polygon.

        >>> s.fillna()
        0    POLYGON ((0.00000 0.00000, 1.00000 1.00000, 0....
        1                             GEOMETRYCOLLECTION EMPTY
        2    POLYGON ((0.00000 0.00000, -1.00000 1.00000, 0...
        dtype: geometry

        Filled with a specific polygon.

        >>> s.fillna(Polygon([(0, 1), (2, 1), (1, 2)]))
        0    POLYGON ((0.00000 0.00000, 1.00000 1.00000, 0....
        1    POLYGON ((0.00000 1.00000, 2.00000 1.00000, 1....
        2    POLYGON ((0.00000 0.00000, -1.00000 1.00000, 0...
        dtype: geometry

        Filled with another GeoSeries.

        >>> from shapely.geometry import Point
        >>> s_fill = geopandas.GeoSeries(
        ...     [
        ...         Point(0, 0),
        ...         Point(1, 1),
        ...         Point(2, 2),
        ...     ]
        ... )
        >>> s.fillna(s_fill)
        0    POLYGON ((0.00000 0.00000, 1.00000 1.00000, 0....
        1                              POINT (1.00000 1.00000)
        2    POLYGON ((0.00000 0.00000, -1.00000 1.00000, 0...
        dtype: geometry

        See Also
        --------
        GeoSeries.isna : detect missing values
        """
        if value is None:
            value = GeometryCollection() if compat.SHAPELY_GE_20 else BaseGeometry()
        return super().fillna(value=value, method=method, inplace=inplace, **kwargs)

    def __contains__(self, other):
        """Allow tests of the form "geom in s"

        Tests whether a GeoSeries contains a geometry.

        Note: This is not the same as the geometric method "contains".
        """
        if isinstance(other, BaseGeometry):
            return np.any(self.geom_equals(other))
        else:
            return False

    @doc(plot_series)
    def plot(self, *args, **kwargs):
        return plot_series(self, *args, **kwargs)

    @doc(_explore_geoseries)
    def explore(self, *args, **kwargs):
        """Interactive map based on folium/leaflet.js"""
        return _explore_geoseries(self, *args, **kwargs)

    def explode(self, ignore_index=False, index_parts=None):
        """
        Explode multi-part geometries into multiple single geometries.

        Single rows can become multiple rows.
        This is analogous to PostGIS's ST_Dump(). The 'path' index is the
        second level of the returned MultiIndex

        Parameters
        ----------
        ignore_index : bool, default False
            If True, the resulting index will be labelled 0, 1, …, n - 1,
            ignoring `index_parts`.
        index_parts : boolean, default True
            If True, the resulting index will be a multi-index (original
            index with an additional level indicating the multiple
            geometries: a new zero-based index for each single part geometry
            per multi-part geometry).

        Returns
        -------
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

        >>> s.explode(index_parts=True)
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
        from .base import _get_index_for_parts

        if index_parts is None and not ignore_index:
            warnings.warn(
                "Currently, index_parts defaults to True, but in the future, "
                "it will default to False to be consistent with Pandas. "
                "Use `index_parts=True` to keep the current behavior and True/False "
                "to silence the warning.",
                FutureWarning,
                stacklevel=2,
            )
            index_parts = True

        if compat.USE_SHAPELY_20 or (compat.USE_PYGEOS and compat.PYGEOS_GE_09):
            if compat.USE_SHAPELY_20:
                geometries, outer_idx = shapely.get_parts(
                    self.values._data, return_index=True
                )
            else:
                import pygeos  # noqa

                geometries, outer_idx = pygeos.get_parts(
                    self.values._data, return_index=True
                )

            index = _get_index_for_parts(
                self.index,
                outer_idx,
                ignore_index=ignore_index,
                index_parts=index_parts,
            )

            return GeoSeries(geometries, index=index, crs=self.crs).__finalize__(self)

        # else PyGEOS is not available or version <= 0.8

        index = []
        geometries = []
        for idx, s in self.geometry.items():
            if s.geom_type.startswith("Multi") or s.geom_type == "GeometryCollection":
                geoms = s.geoms
                idxs = [(idx, i) for i in range(len(geoms))]
            else:
                geoms = [s]
                idxs = [(idx, 0)]
            index.extend(idxs)
            geometries.extend(geoms)

        if ignore_index:
            index = range(len(geometries))

        elif index_parts:
            # if self.index is a MultiIndex then index is a list of nested tuples
            if isinstance(self.index, MultiIndex):
                index = [tuple(outer) + (inner,) for outer, inner in index]
            index = MultiIndex.from_tuples(index, names=self.index.names + [None])

        else:
            index = [idx for idx, _ in index]

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

        See Also
        --------
        GeoSeries.to_crs : re-project to another CRS

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
        of projecting entire geometries.  All segments joining points are
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

        See Also
        --------
        GeoSeries.set_crs : assign CRS

        """
        return GeoSeries(
            self.values.to_crs(crs=crs, epsg=epsg), index=self.index, name=self.name
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
        >>> import geodatasets
        >>> df = geopandas.read_file(
        ...     geodatasets.get_path("geoda.chicago_health")
        ... )
        >>> df.geometry.estimate_utm_crs()  # doctest: +SKIP
        <Derived Projected CRS: EPSG:32616>
        Name: WGS 84 / UTM zone 16N
        Axis Info [cartesian]:
        - E[east]: Easting (metre)
        - N[north]: Northing (metre)
        Area of Use:
        - name: Between 90°W and 84°W, northern hemisphere between equator and 84°N, ...
        - bounds: (-90.0, 0.0, -84.0, 84.0)
        Coordinate Operation:
        - name: UTM zone 16N
        - method: Transverse Mercator
        Datum: World Geodetic System 1984 ensemble
        - Ellipsoid: WGS 84
        - Prime Meridian: Greenwich
        """
        return self.values.estimate_utm_crs(datum_name)

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

        See Also
        --------
        GeoSeries.to_file : write GeoSeries to file
        """
        return json.dumps(self.__geo_interface__, **kwargs)

    def to_wkb(self, hex=False, **kwargs):
        """
        Convert GeoSeries geometries to WKB

        Parameters
        ----------
        hex : bool
            If true, export the WKB as a hexadecimal string.
            The default is to return a binary bytes object.
        kwargs
            Additional keyword args will be passed to
            :func:`shapely.to_wkb` if shapely >= 2 is installed or
            :func:`pygeos.to_wkb` if pygeos is installed.

        Returns
        -------
        Series
            WKB representations of the geometries

        See also
        --------
        GeoSeries.to_wkt
        """
        return Series(to_wkb(self.array, hex=hex, **kwargs), index=self.index)

    def to_wkt(self, **kwargs):
        """
        Convert GeoSeries geometries to WKT

        Parameters
        ----------
        kwargs
            Keyword args will be passed to :func:`pygeos.to_wkt`
            if pygeos is installed.

        Returns
        -------
        Series
            WKT representations of the geometries

        Examples
        --------
        >>> from shapely.geometry import Point
        >>> s = geopandas.GeoSeries([Point(1, 1), Point(2, 2), Point(3, 3)])
        >>> s
        0    POINT (1.00000 1.00000)
        1    POINT (2.00000 2.00000)
        2    POINT (3.00000 3.00000)
        dtype: geometry

        >>> s.to_wkt()
        0    POINT (1 1)
        1    POINT (2 2)
        2    POINT (3 3)
        dtype: object

        See also
        --------
        GeoSeries.to_wkb
        """
        return Series(to_wkt(self.array, **kwargs), index=self.index)

    #
    # Implement standard operators for GeoSeries
    #

    def __xor__(self, other):
        """Implement ^ operator as for builtin set type"""
        warnings.warn(
            "'^' operator will be deprecated. Use the 'symmetric_difference' "
            "method instead.",
            FutureWarning,
            stacklevel=2,
        )
        return self.symmetric_difference(other)

    def __or__(self, other):
        """Implement | operator as for builtin set type"""
        warnings.warn(
            "'|' operator will be deprecated. Use the 'union' method instead.",
            FutureWarning,
            stacklevel=2,
        )
        return self.union(other)

    def __and__(self, other):
        """Implement & operator as for builtin set type"""
        warnings.warn(
            "'&' operator will be deprecated. Use the 'intersection' method instead.",
            FutureWarning,
            stacklevel=2,
        )
        return self.intersection(other)

    def __sub__(self, other):
        """Implement - operator as for builtin set type"""
        warnings.warn(
            "'-' operator will be deprecated. Use the 'difference' method instead.",
            FutureWarning,
            stacklevel=2,
        )
        return self.difference(other)

    def clip(self, mask, keep_geom_type=False):
        """Clip points, lines, or polygon geometries to the mask extent.

        Both layers must be in the same Coordinate Reference System (CRS).
        The GeoSeries will be clipped to the full extent of the `mask` object.

        If there are multiple polygons in mask, data from the GeoSeries will be
        clipped to the total boundary of all polygons in mask.

        Parameters
        ----------
        mask : GeoDataFrame, GeoSeries, (Multi)Polygon, list-like
            Polygon vector layer used to clip `gdf`.
            The mask's geometry is dissolved into one geometric feature
            and intersected with GeoSeries.
            If the mask is list-like with four elements ``(minx, miny, maxx, maxy)``,
            ``clip`` will use a faster rectangle clipping
            (:meth:`~GeoSeries.clip_by_rect`), possibly leading to slightly different
            results.
        keep_geom_type : boolean, default False
            If True, return only geometries of original type in case of intersection
            resulting in multiple geometry types or GeometryCollections.
            If False, return all resulting geometries (potentially mixed-types).

        Returns
        -------
        GeoSeries
            Vector data (points, lines, polygons) from `gdf` clipped to
            polygon boundary from mask.

        See also
        --------
        clip : top-level function for clip

        Examples
        --------
        Clip points (grocery stores) with polygons (the Near West Side community):

        >>> import geodatasets
        >>> chicago = geopandas.read_file(
        ...     geodatasets.get_path("geoda.chicago_health")
        ... )
        >>> near_west_side = chicago[chicago["community"] == "NEAR WEST SIDE"]
        >>> groceries = geopandas.read_file(
        ...     geodatasets.get_path("geoda.groceries")
        ... ).to_crs(chicago.crs)
        >>> groceries.shape
        (148, 8)

        >>> nws_groceries = groceries.geometry.clip(near_west_side)
        >>> nws_groceries.shape
        (7,)
        """
        return geopandas.clip(self, mask=mask, keep_geom_type=keep_geom_type)
