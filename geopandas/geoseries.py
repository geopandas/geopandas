from __future__ import annotations

import typing
import warnings
from typing import Any

import numpy as np
import pandas as pd
from pandas import Series
from pandas.core.internals import SingleBlockManager

import shapely
from shapely.geometry import GeometryCollection
from shapely.geometry.base import BaseGeometry

import geopandas
from geopandas.base import GeoPandasBase, _delegate_property
from geopandas.explore import _explore_geoseries
from geopandas.plotting import plot_series

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

if typing.TYPE_CHECKING:
    import os
    from collections.abc import Callable


def _geoseries_constructor_with_fallback(
    data=None, index=None, crs: Any | None = None, **kwargs
):
    """A flexible constructor for GeoSeries._constructor, which needs to be able
    to fall back to a Series (if a certain operation does not produce
    geometries).
    """  # noqa: D401
    try:
        return GeoSeries(data=data, index=index, crs=crs, **kwargs)
    except TypeError:
        return Series(data=data, index=index, **kwargs)


def _expanddim_logic(df):
    """Shared logic for _constructor_expanddim and _constructor_from_mgr_expanddim."""
    from geopandas import GeoDataFrame

    if (df.dtypes == "geometry").sum() > 0:
        if df.shape[1] == 1:
            geo_col_name = df.columns[0]
        else:
            geo_col_name = None

        if geo_col_name is None or not is_geometry_type(df[geo_col_name]):
            df = GeoDataFrame(df)
            df._geometry_column_name = None
        else:
            df = df.set_geometry(geo_col_name)

    return df


def _geoseries_expanddim(data=None, *args, **kwargs):
    # pd.Series._constructor_expanddim == pd.DataFrame, we start
    # with that then specialize.
    df = pd.DataFrame(data, *args, **kwargs)
    return _expanddim_logic(df)


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
    0    POINT (1 1)
    1    POINT (2 2)
    2    POINT (3 3)
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
    a    POINT (1 1)
    b    POINT (2 2)
    c    POINT (3 3)
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

    def __init__(self, data=None, index=None, crs: Any | None = None, **kwargs):
        if (
            hasattr(data, "crs")
            or (isinstance(data, pd.Series) and hasattr(data.array, "crs"))
        ) and crs:
            data_crs = data.crs if hasattr(data, "crs") else data.array.crs
            if not data_crs:
                # make a copy to avoid setting CRS to passed GeometryArray
                data = data.copy()
            else:
                if not data_crs == crs:
                    raise ValueError(
                        "CRS mismatch between CRS of the passed geometries "
                        "and 'crs'. Use 'GeoSeries.set_crs(crs, "
                        "allow_override=True)' to overwrite CRS or "
                        "'GeoSeries.to_crs(crs)' to reproject geometries. "
                    )

        if isinstance(data, SingleBlockManager):
            if not isinstance(data.blocks[0].dtype, GeometryDtype):
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
            with warnings.catch_warnings():
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
            # extract object-dtype numpy array from pandas Series; with CoW this
            # gives a read-only array, so we try to set the flag back to writeable
            data = s.to_numpy()
            try:
                data.flags.writeable = True
            except ValueError:
                pass
            # try to convert to GeometryArray
            try:
                data = from_shapely(data, crs)
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

    @GeoPandasBase.crs.setter
    def crs(self, value):
        if self.crs is not None:
            warnings.warn(
                "Overriding the CRS of a GeoSeries that already has CRS. "
                "This unsafe behavior will be deprecated in future versions. "
                "Use GeoSeries.set_crs method instead.",
                stacklevel=2,
                category=DeprecationWarning,
            )
        self.geometry.values.crs = value

    @property
    def geometry(self) -> GeoSeries:
        return self

    @property
    def x(self) -> Series:
        """Return the x location of point geometries in a GeoSeries.

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
    def y(self) -> Series:
        """Return the y location of point geometries in a GeoSeries.

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
        GeoSeries.m

        """
        return _delegate_property("y", self)

    @property
    def z(self) -> Series:
        """Return the z location of point geometries in a GeoSeries.

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
        GeoSeries.m

        """
        return _delegate_property("z", self)

    @property
    def m(self) -> Series:
        """Return the m coordinate of point geometries in a GeoSeries.

        Requires Shapely >= 2.1.

        .. versionadded:: 1.1.0

        Returns
        -------
        pandas.Series

        Examples
        --------
        >>> from shapely.geometry import Point
        >>> s = geopandas.GeoSeries.from_wkt(
        ...     [
        ...         "POINT M (2 3 5)",
        ...         "POINT M (1 2 3)",
        ...     ]
        ... )
        >>> s
        0    POINT M (2 3 5)
        1    POINT M (1 2 3)
        dtype: geometry

        >>> s.m
        0    5.0
        1    3.0
        dtype: float64

        See Also
        --------
        GeoSeries.x
        GeoSeries.y
        GeoSeries.z

        """
        return _delegate_property("m", self)

    @classmethod
    def from_file(cls, filename: os.PathLike | typing.IO, **kwargs) -> GeoSeries:
        """Alternate constructor to create a ``GeoSeries`` from a file.

        Can load a ``GeoSeries`` from a file from any format recognized by
        `pyogrio`. See http://pyogrio.readthedocs.io/ for details.
        From a file with attributes loads only geometry column. Note that to do
        that, GeoPandas first loads the whole GeoDataFrame.

        Parameters
        ----------
        filename : str
            File path or file handle to read from. Depending on which kwargs
            are included, the content of filename may vary. See
            :func:`pyogrio.read_dataframe` for usage details.
        kwargs : key-word arguments
            These arguments are passed to :func:`pyogrio.read_dataframe`, and can be
            used to access multi-layer data, data stored within archives (zip files),
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
        read_file : read file to GeoDataFrame
        """
        from geopandas import GeoDataFrame

        df = GeoDataFrame.from_file(filename, **kwargs)

        return GeoSeries(df.geometry, crs=df.crs)

    @classmethod
    def from_wkb(
        cls, data, index=None, crs: Any | None = None, on_invalid="raise", **kwargs
    ) -> GeoSeries:
        r"""Alternate constructor to create a ``GeoSeries``
        from a list or array of WKB objects.

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
        on_invalid: {"raise", "warn", "ignore"}, default "raise"
            - raise: an exception will be raised if a WKB input geometry is invalid.
            - warn: a warning will be raised and invalid WKB geometries will be returned
              as None.
            - ignore: invalid WKB geometries will be returned as None without a warning.
            - fix: an effort is made to fix invalid input geometries (e.g. close
              unclosed rings). If this is not possible, they are returned as ``None``
              without a warning. Requires GEOS >= 3.11 and shapely >= 2.1.

        kwargs
            Additional arguments passed to the Series constructor,
            e.g. ``name``.

        Returns
        -------
        GeoSeries

        See Also
        --------
        GeoSeries.from_wkt

        Examples
        --------
        >>> wkbs = [
        ... (
        ...     b"\x01\x01\x00\x00\x00\x00\x00\x00\x00"
        ...     b"\x00\x00\xf0?\x00\x00\x00\x00\x00\x00\xf0?"
        ... ),
        ... (
        ...     b"\x01\x01\x00\x00\x00\x00\x00\x00\x00"
        ...     b"\x00\x00\x00@\x00\x00\x00\x00\x00\x00\x00@"
        ... ),
        ... (
        ...    b"\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00"
        ...    b"\x00\x08@\x00\x00\x00\x00\x00\x00\x08@"
        ... ),
        ... ]
        >>> s = geopandas.GeoSeries.from_wkb(wkbs)
        >>> s
        0    POINT (1 1)
        1    POINT (2 2)
        2    POINT (3 3)
        dtype: geometry
        """
        return cls._from_wkb_or_wkt(
            from_wkb, data, index=index, crs=crs, on_invalid=on_invalid, **kwargs
        )

    @classmethod
    def from_wkt(
        cls, data, index=None, crs: Any | None = None, on_invalid="raise", **kwargs
    ) -> GeoSeries:
        """Alternate constructor to create a ``GeoSeries``
        from a list or array of WKT objects.

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
        on_invalid : {"raise", "warn", "ignore"}, default "raise"
            - raise: an exception will be raised if a WKT input geometry is invalid.
            - warn: a warning will be raised and invalid WKT geometries will be
              returned as ``None``.
            - ignore: invalid WKT geometries will be returned as ``None`` without a
              warning.
            - fix: an effort is made to fix invalid input geometries (e.g. close
              unclosed rings). If this is not possible, they are returned as ``None``
              without a warning. Requires GEOS >= 3.11 and shapely >= 2.1.

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
        0    POINT (1 1)
        1    POINT (2 2)
        2    POINT (3 3)
        dtype: geometry
        """
        return cls._from_wkb_or_wkt(
            from_wkt, data, index=index, crs=crs, on_invalid=on_invalid, **kwargs
        )

    @classmethod
    def from_xy(cls, x, y, z=None, index=None, crs=None, **kwargs) -> GeoSeries:
        """Alternate constructor to create a :class:`~geopandas.GeoSeries` of Point
        geometries from lists or arrays of x, y(, z) coordinates.

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
        0    POINT (2.5 0.5)
        1    POINT (5 1)
        2    POINT (-3 1.5)
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
    def _from_wkb_or_wkt(
        cls,
        from_wkb_or_wkt_function: Callable,
        data,
        index=None,
        crs: Any | None = None,
        on_invalid: str = "raise",
        **kwargs,
    ) -> GeoSeries:
        """Create a GeoSeries from either WKT or WKB values."""
        if isinstance(data, Series):
            if index is not None:
                data = data.reindex(index)
            else:
                index = data.index
            data = data.values
        return cls(
            from_wkb_or_wkt_function(data, crs=crs, on_invalid=on_invalid),
            index=index,
            **kwargs,
        )

    @classmethod
    def from_arrow(cls, arr, **kwargs) -> GeoSeries:
        """
        Construct a GeoSeries from a Arrow array object with a GeoArrow
        extension type.

        See https://geoarrow.org/ for details on the GeoArrow specification.

        This functions accepts any Arrow array object implementing
        the `Arrow PyCapsule Protocol`_ (i.e. having an ``__arrow_c_array__``
        method).

        .. _Arrow PyCapsule Protocol: https://arrow.apache.org/docs/format/CDataInterface/PyCapsuleInterface.html

        .. versionadded:: 1.0

        Parameters
        ----------
        arr : pyarrow.Array, Arrow array
            Any array object implementing the Arrow PyCapsule Protocol
            (i.e. has an ``__arrow_c_array__`` or ``__arrow_c_stream__``
            method). The type of the array should be one of the
            geoarrow geometry types.
        **kwargs
            Other parameters passed to the GeoSeries constructor.

        Returns
        -------
        GeoSeries

        """
        from geopandas.io._geoarrow import arrow_to_geometry_array

        return cls(arrow_to_geometry_array(arr), **kwargs)

    @property
    def __geo_interface__(self) -> dict:
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

    def to_file(
        self,
        filename: os.PathLike | typing.IO,
        driver: str | None = None,
        index: bool | None = None,
        **kwargs,
    ):
        """Write the ``GeoSeries`` to a file.

        By default, an ESRI shapefile is written, but any OGR data source
        supported by Pyogrio or Fiona can be written.

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
            such as an authority string (eg "EPSG:4326") or a WKT string. The keyword
            is not supported for the "pyogrio" engine.
        engine : str, "pyogrio" or "fiona"
            The underlying library that is used to write the file. Currently, the
            supported options are "pyogrio" and "fiona". Defaults to "pyogrio" if
            installed, otherwise tries "fiona".
        **kwargs :
            Keyword args to be passed to the engine, and can be used to write
            to multi-layer data, store data within archives (zip files), etc.
            In case of the "pyogrio" engine, the keyword arguments are passed to
            `pyogrio.write_dataframe`. In case of the "fiona" engine, the keyword
            arguments are passed to fiona.open`. For more information on possible
            keywords, type: ``import pyogrio; help(pyogrio.write_dataframe)``.

        See Also
        --------
        GeoDataFrame.to_file : write GeoDataFrame to file
        read_file : read file to GeoDataFrame

        Examples
        --------
        >>> s.to_file('series.shp')  # doctest: +SKIP

        >>> s.to_file('series.gpkg', driver='GPKG', layer='name1')  # doctest: +SKIP

        >>> s.to_file('series.geojson', driver='GeoJSON')  # doctest: +SKIP
        """
        from geopandas import GeoDataFrame

        data = GeoDataFrame({"geometry": self}, index=self.index)
        data.to_file(filename, driver, index=index, **kwargs)

    #
    # Implement pandas methods
    #

    @property
    def _constructor(self):
        return _geoseries_constructor_with_fallback

    def _constructor_from_mgr(self, mgr, axes):
        assert isinstance(mgr, SingleBlockManager)

        if not isinstance(mgr.blocks[0].dtype, GeometryDtype):
            return Series._from_mgr(mgr, axes)

        return GeoSeries._from_mgr(mgr, axes)

    @property
    def _constructor_expanddim(self):
        return _geoseries_expanddim

    def _constructor_expanddim_from_mgr(self, mgr, axes):
        df = pd.DataFrame._from_mgr(mgr, axes)
        return _expanddim_logic(df)

    def _wrapped_pandas_method(self, mtd, *args, **kwargs):
        """Wrap a generic pandas method to ensure it returns a GeoSeries."""
        val = getattr(super(), mtd)(*args, **kwargs)
        if type(val) is Series:
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
    def apply(self, func, convert_dtype: bool | None = None, args=(), **kwargs):
        if convert_dtype is not None:
            kwargs["convert_dtype"] = convert_dtype
        else:
            # if compat.PANDAS_GE_21 don't pass through, use pandas default
            # of true to avoid internally triggering the pandas warning
            if not compat.PANDAS_GE_21:
                kwargs["convert_dtype"] = True

        # to avoid warning
        result = super().apply(func, args=args, **kwargs)
        if isinstance(result, GeoSeries):
            if self.crs is not None:
                result.set_crs(self.crs, inplace=True)
        return result

    def isna(self) -> Series:
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
        0    POLYGON ((0 0, 1 1, 0 1, 0 0))
        1                              None
        2                     POLYGON EMPTY
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

    def isnull(self) -> Series:
        """Alias for `isna` method. See `isna` for more detail."""
        return self.isna()

    def notna(self) -> Series:
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
        0    POLYGON ((0 0, 1 1, 0 1, 0 0))
        1                              None
        2                     POLYGON EMPTY
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

    def notnull(self) -> Series:
        """Alias for `notna` method. See `notna` for more detail."""
        return self.notna()

    def fillna(self, value=None, inplace: bool = False, limit=None, **kwargs):
        """
        Fill NA values with geometry (or geometries).

        Parameters
        ----------
        value : shapely geometry or GeoSeries, default None
            If None is passed, NA values will be filled with GEOMETRYCOLLECTION EMPTY.
            If a shapely geometry object is passed, it will be
            used to fill all missing values. If a ``GeoSeries`` or ``GeometryArray``
            are passed, missing values will be filled based on the corresponding index
            locations. If pd.NA or np.nan are passed, values will be filled with
            ``None`` (not GEOMETRYCOLLECTION EMPTY).
        limit : int, default None
            This is the maximum number of entries along the entire axis
            where NaNs will be filled. Must be greater than 0 if not None.

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
        0      POLYGON ((0 0, 1 1, 0 1, 0 0))
        1                                None
        2    POLYGON ((0 0, -1 1, 0 -1, 0 0))
        dtype: geometry

        Filled with an empty polygon.

        >>> s.fillna()
        0      POLYGON ((0 0, 1 1, 0 1, 0 0))
        1            GEOMETRYCOLLECTION EMPTY
        2    POLYGON ((0 0, -1 1, 0 -1, 0 0))
        dtype: geometry

        Filled with a specific polygon.

        >>> s.fillna(Polygon([(0, 1), (2, 1), (1, 2)]))
        0      POLYGON ((0 0, 1 1, 0 1, 0 0))
        1      POLYGON ((0 1, 2 1, 1 2, 0 1))
        2    POLYGON ((0 0, -1 1, 0 -1, 0 0))
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
        0      POLYGON ((0 0, 1 1, 0 1, 0 0))
        1                         POINT (1 1)
        2    POLYGON ((0 0, -1 1, 0 -1, 0 0))
        dtype: geometry

        See Also
        --------
        GeoSeries.isna : detect missing values
        """
        if value is None:
            value = GeometryCollection()
        return super().fillna(value=value, limit=limit, inplace=inplace, **kwargs)

    def __contains__(self, other) -> bool:
        """Allow tests of the form "geom in s".

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
        """Explore with an interactive map based on folium/leaflet.js."""
        return _explore_geoseries(self, *args, **kwargs)

    def explode(self, ignore_index=False, index_parts=False) -> GeoSeries:
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
        index_parts : boolean, default False
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
        0           MULTIPOINT ((0 0), (1 1))
        1    MULTIPOINT ((2 2), (3 3), (4 4))
        dtype: geometry

        >>> s.explode(index_parts=True)
        0  0    POINT (0 0)
           1    POINT (1 1)
        1  0    POINT (2 2)
           1    POINT (3 3)
           2    POINT (4 4)
        dtype: geometry

        See Also
        --------
        GeoDataFrame.explode

        """
        from .base import _get_index_for_parts

        geometries, outer_idx = shapely.get_parts(self.values._data, return_index=True)

        index = _get_index_for_parts(
            self.index,
            outer_idx,
            ignore_index=ignore_index,
            index_parts=index_parts,
        )

        return GeoSeries(geometries, index=index, crs=self.crs).__finalize__(self)

    #
    # Additional methods
    #
    @compat.requires_pyproj
    def set_crs(
        self,
        crs: Any | None = None,
        epsg: int | None = None,
        inplace: bool = False,
        allow_override: bool = False,
    ):
        """
        Set the Coordinate Reference System (CRS) of a ``GeoSeries``.

        Pass ``None`` to remove CRS from the ``GeoSeries``.

        Notes
        -----
        The underlying geometries are not transformed to this CRS. To
        transform the geometries to a new CRS, use the ``to_crs`` method.

        Parameters
        ----------
        crs : pyproj.CRS | None, optional
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
        0    POINT (1 1)
        1    POINT (2 2)
        2    POINT (3 3)
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
        from pyproj import CRS

        if crs is not None:
            crs = CRS.from_user_input(crs)
        elif epsg is not None:
            crs = CRS.from_epsg(epsg)

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
        result.array.crs = crs
        return result

    def to_crs(self, crs: Any | None = None, epsg: int | None = None) -> GeoSeries:
        """Return a ``GeoSeries`` with all geometries transformed to a new
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
        0    POINT (1 1)
        1    POINT (2 2)
        2    POINT (3 3)
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

    def estimate_utm_crs(self, datum_name: str = "WGS 84"):
        """Return the estimated UTM CRS based on the bounds of the dataset.

        .. versionadded:: 0.9

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

    def to_json(
        self,
        show_bbox: bool = True,
        drop_id: bool = False,
        to_wgs84: bool = False,
        **kwargs,
    ) -> str:
        """Return a GeoJSON string representation of the GeoSeries.

        Parameters
        ----------
        show_bbox : bool, optional, default: True
            Include bbox (bounds) in the geojson
        drop_id : bool, default: False
            Whether to retain the index of the GeoSeries as the id property
            in the generated GeoJSON. Default is False, but may want True
            if the index is just arbitrary row numbers.
        to_wgs84: bool, optional, default: False
            If the CRS is set on the active geometry column it is exported as
            WGS84 (EPSG:4326) to meet the `2016 GeoJSON specification
            <https://tools.ietf.org/html/rfc7946>`_.
            Set to True to force re-projection and set to False to ignore CRS. False by
            default.

        *kwargs* that will be passed to json.dumps().

        Returns
        -------
        JSON string

        Examples
        --------
        >>> from shapely.geometry import Point
        >>> s = geopandas.GeoSeries([Point(1, 1), Point(2, 2), Point(3, 3)])
        >>> s
        0    POINT (1 1)
        1    POINT (2 2)
        2    POINT (3 3)
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
        return self.to_frame("geometry").to_json(
            na="null", show_bbox=show_bbox, drop_id=drop_id, to_wgs84=to_wgs84, **kwargs
        )

    def to_wkb(self, hex: bool = False, **kwargs) -> Series:
        """Convert GeoSeries geometries to WKB.

        Parameters
        ----------
        hex : bool
            If true, export the WKB as a hexadecimal string.
            The default is to return a binary bytes object.
        kwargs
            Additional keyword args will be passed to
            :func:`shapely.to_wkb`.

        Returns
        -------
        Series
            WKB representations of the geometries

        See Also
        --------
        GeoSeries.to_wkt
        """
        return Series(to_wkb(self.array, hex=hex, **kwargs), index=self.index)

    def to_wkt(self, **kwargs) -> Series:
        """Convert GeoSeries geometries to WKT.

        Parameters
        ----------
        kwargs
            Keyword args will be passed to :func:`shapely.to_wkt`.

        Returns
        -------
        Series
            WKT representations of the geometries

        Examples
        --------
        >>> from shapely.geometry import Point
        >>> s = geopandas.GeoSeries([Point(1, 1), Point(2, 2), Point(3, 3)])
        >>> s
        0    POINT (1 1)
        1    POINT (2 2)
        2    POINT (3 3)
        dtype: geometry

        >>> s.to_wkt()
        0    POINT (1 1)
        1    POINT (2 2)
        2    POINT (3 3)
        dtype: object

        See Also
        --------
        GeoSeries.to_wkb
        """
        return Series(to_wkt(self.array, **kwargs), index=self.index)

    def to_arrow(self, geometry_encoding="WKB", interleaved=True, include_z=None):
        """Encode a GeoSeries to GeoArrow format.

        See https://geoarrow.org/ for details on the GeoArrow specification.

        This functions returns a generic Arrow array object implementing
        the `Arrow PyCapsule Protocol`_ (i.e. having an ``__arrow_c_array__``
        method). This object can then be consumed by your Arrow implementation
        of choice that supports this protocol.

        .. _Arrow PyCapsule Protocol: https://arrow.apache.org/docs/format/CDataInterface/PyCapsuleInterface.html

        .. versionadded:: 1.0

        Parameters
        ----------
        geometry_encoding : {'WKB', 'geoarrow' }, default 'WKB'
            The GeoArrow encoding to use for the data conversion.
        interleaved : bool, default True
            Only relevant for 'geoarrow' encoding. If True, the geometries'
            coordinates are interleaved in a single fixed size list array.
            If False, the coordinates are stored as separate arrays in a
            struct type.
        include_z : bool, default None
            Only relevant for 'geoarrow' encoding (for WKB, the dimensionality
            of the individial geometries is preserved).
            If False, return 2D geometries. If True, include the third dimension
            in the output (if a geometry has no third dimension, the z-coordinates
            will be NaN). By default, will infer the dimensionality from the
            input geometries. Note that this inference can be unreliable with
            empty geometries (for a guaranteed result, it is recommended to
            specify the keyword).

        Returns
        -------
        GeoArrowArray
            A generic Arrow array object with geometry data encoded to GeoArrow.

        Examples
        --------
        >>> from shapely.geometry import Point
        >>> gser = geopandas.GeoSeries([Point(1, 2), Point(2, 1)])
        >>> gser
        0    POINT (1 2)
        1    POINT (2 1)
        dtype: geometry

        >>> arrow_array = gser.to_arrow()
        >>> arrow_array
        <geopandas.io._geoarrow.GeoArrowArray object at ...>

        The returned array object needs to be consumed by a library implementing
        the Arrow PyCapsule Protocol. For example, wrapping the data as a
        pyarrow.Array (requires pyarrow >= 14.0):

        >>> import pyarrow as pa
        >>> array = pa.array(arrow_array)
        >>> array
        <pyarrow.lib.BinaryArray object at ...>
        [
          0101000000000000000000F03F0000000000000040,
          01010000000000000000000040000000000000F03F
        ]

        """
        from geopandas.io._geoarrow import (
            GeoArrowArray,
            construct_geometry_array,
            construct_wkb_array,
        )

        field_name = self.name if self.name is not None else ""

        if geometry_encoding.lower() == "geoarrow":
            field, geom_arr = construct_geometry_array(
                np.array(self.array),
                include_z=include_z,
                field_name=field_name,
                crs=self.crs,
                interleaved=interleaved,
            )
        elif geometry_encoding.lower() == "wkb":
            field, geom_arr = construct_wkb_array(
                np.asarray(self.array), field_name=field_name, crs=self.crs
            )
        else:
            raise ValueError(
                "Expected geometry encoding 'WKB' or 'geoarrow' "
                f"got {geometry_encoding}"
            )

        return GeoArrowArray(field, geom_arr)

    def clip(self, mask, keep_geom_type: bool = False, sort=False) -> GeoSeries:
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
        sort : boolean, default False
            If True, the order of rows in the clipped GeoSeries will be preserved
            at small performance cost.
            If False the order of rows in the clipped GeoSeries will be random.

        Returns
        -------
        GeoSeries
            Vector data (points, lines, polygons) from `gdf` clipped to
            polygon boundary from mask.

        See Also
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
        return geopandas.clip(self, mask=mask, keep_geom_type=keep_geom_type, sort=sort)
