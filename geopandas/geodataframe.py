import json
import warnings

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from pandas.core.accessor import CachedAccessor

from shapely.geometry import mapping, shape
from shapely.geometry.base import BaseGeometry

from pyproj import CRS

from geopandas.array import GeometryArray, GeometryDtype, from_shapely, to_wkb, to_wkt
from geopandas.base import GeoPandasBase, is_geometry_type
from geopandas.geoseries import GeoSeries
import geopandas.io
from geopandas.explore import _explore
from . import _compat as compat
from ._decorator import doc


DEFAULT_GEO_COLUMN_NAME = "geometry"


def _ensure_geometry(data, crs=None):
    """
    Ensure the data is of geometry dtype or converted to it.

    If input is a (Geo)Series, output is a GeoSeries, otherwise output
    is GeometryArray.

    If the input is a GeometryDtype with a set CRS, `crs` is ignored.
    """
    if is_geometry_type(data):
        if isinstance(data, Series):
            data = GeoSeries(data)
        if data.crs is None:
            data.crs = crs
        return data
    else:
        if isinstance(data, Series):
            out = from_shapely(np.asarray(data), crs=crs)
            return GeoSeries(out, index=data.index, name=data.name)
        else:
            out = from_shapely(data, crs=crs)
            return out


def _crs_mismatch_warning():
    # TODO: raise error in 0.9 or 0.10.
    warnings.warn(
        "CRS mismatch between CRS of the passed geometries "
        "and 'crs'. Use 'GeoDataFrame.set_crs(crs, "
        "allow_override=True)' to overwrite CRS or "
        "'GeoDataFrame.to_crs(crs)' to reproject geometries. "
        "CRS mismatch will raise an error in the future versions "
        "of GeoPandas.",
        FutureWarning,
        stacklevel=3,
    )


class GeoDataFrame(GeoPandasBase, DataFrame):
    """
    A GeoDataFrame object is a pandas.DataFrame that has a column
    with geometry. In addition to the standard DataFrame constructor arguments,
    GeoDataFrame also accepts the following keyword arguments:

    Parameters
    ----------
    crs : value (optional)
        Coordinate Reference System of the geometry objects. Can be anything accepted by
        :meth:`pyproj.CRS.from_user_input() <pyproj.crs.CRS.from_user_input>`,
        such as an authority string (eg "EPSG:4326") or a WKT string.
    geometry : str or array (optional)
        If str, column to use as geometry. If array, will be set as 'geometry'
        column on GeoDataFrame.

    Examples
    --------
    Constructing GeoDataFrame from a dictionary.

    >>> from shapely.geometry import Point
    >>> d = {'col1': ['name1', 'name2'], 'geometry': [Point(1, 2), Point(2, 1)]}
    >>> gdf = geopandas.GeoDataFrame(d, crs="EPSG:4326")
    >>> gdf
        col1                 geometry
    0  name1  POINT (1.00000 2.00000)
    1  name2  POINT (2.00000 1.00000)

    Notice that the inferred dtype of 'geometry' columns is geometry.

    >>> gdf.dtypes
    col1          object
    geometry    geometry
    dtype: object

    Constructing GeoDataFrame from a pandas DataFrame with a column of WKT geometries:

    >>> import pandas as pd
    >>> d = {'col1': ['name1', 'name2'], 'wkt': ['POINT (1 2)', 'POINT (2 1)']}
    >>> df = pd.DataFrame(d)
    >>> gs = geopandas.GeoSeries.from_wkt(df['wkt'])
    >>> gdf = geopandas.GeoDataFrame(df, geometry=gs, crs="EPSG:4326")
    >>> gdf
        col1          wkt                 geometry
    0  name1  POINT (1 2)  POINT (1.00000 2.00000)
    1  name2  POINT (2 1)  POINT (2.00000 1.00000)

    See also
    --------
    GeoSeries : Series object designed to store shapely geometry objects
    """

    _metadata = ["_crs", "_geometry_column_name"]

    _geometry_column_name = DEFAULT_GEO_COLUMN_NAME

    def __init__(self, data=None, *args, geometry=None, crs=None, **kwargs):
        with compat.ignore_shapely2_warnings():
            super().__init__(data, *args, **kwargs)

        # need to set this before calling self['geometry'], because
        # getitem accesses crs
        self._crs = CRS.from_user_input(crs) if crs else None

        # set_geometry ensures the geometry data have the proper dtype,
        # but is not called if `geometry=None` ('geometry' column present
        # in the data), so therefore need to ensure it here manually
        # but within a try/except because currently non-geometries are
        # allowed in that case
        # TODO do we want to raise / return normal DataFrame in this case?

        # if gdf passed in and geo_col is set, we use that for geometry
        if geometry is None and isinstance(data, GeoDataFrame):
            self._geometry_column_name = data._geometry_column_name
            if crs is not None and data.crs != crs:
                _crs_mismatch_warning()
                # TODO: raise error in 0.9 or 0.10.
            return

        if geometry is None and "geometry" in self.columns:
            # Check for multiple columns with name "geometry". If there are,
            # self["geometry"] is a gdf and constructor gets recursively recalled
            # by pandas internals trying to access this
            if (self.columns == "geometry").sum() > 1:
                raise ValueError(
                    "GeoDataFrame does not support multiple columns "
                    "using the geometry column name 'geometry'."
                )

            # only if we have actual geometry values -> call set_geometry
            index = self.index
            try:
                if (
                    hasattr(self["geometry"].values, "crs")
                    and self["geometry"].values.crs
                    and crs
                    and not self["geometry"].values.crs == crs
                ):
                    _crs_mismatch_warning()
                    # TODO: raise error in 0.9 or 0.10.
                self["geometry"] = _ensure_geometry(self["geometry"].values, crs)
            except TypeError:
                pass
            else:
                if self.index is not index:
                    # With pandas < 1.0 and an empty frame (no rows), the index
                    # gets reset to a default RangeIndex -> set back the original
                    # index if needed
                    self.index = index
                geometry = "geometry"

        if geometry is not None:
            if (
                hasattr(geometry, "crs")
                and geometry.crs
                and crs
                and not geometry.crs == crs
            ):
                _crs_mismatch_warning()
                # TODO: raise error in 0.9 or 0.10.
            self.set_geometry(geometry, inplace=True)

        if geometry is None and crs:
            warnings.warn(
                "Assigning CRS to a GeoDataFrame without a geometry column is now "
                "deprecated and will not be supported in the future.",
                FutureWarning,
                stacklevel=2,
            )

    def __setattr__(self, attr, val):
        # have to special case geometry b/c pandas tries to use as column...
        if attr == "geometry":
            object.__setattr__(self, attr, val)
        else:
            super().__setattr__(attr, val)

    def _get_geometry(self):
        if self._geometry_column_name not in self:
            raise AttributeError(
                "No geometry data set yet (expected in"
                " column '%s'.)" % self._geometry_column_name
            )
        return self[self._geometry_column_name]

    def _set_geometry(self, col):
        if not pd.api.types.is_list_like(col):
            raise ValueError("Must use a list-like to set the geometry property")
        self.set_geometry(col, inplace=True)

    geometry = property(
        fget=_get_geometry, fset=_set_geometry, doc="Geometry data for GeoDataFrame"
    )

    def set_geometry(self, col, drop=False, inplace=False, crs=None):
        """
        Set the GeoDataFrame geometry using either an existing column or
        the specified input. By default yields a new object.

        The original geometry column is replaced with the input.

        Parameters
        ----------
        col : column label or array
        drop : boolean, default False
            Delete column to be used as the new geometry
        inplace : boolean, default False
            Modify the GeoDataFrame in place (do not create a new object)
        crs : pyproj.CRS, optional
            Coordinate system to use. The value can be anything accepted
            by :meth:`pyproj.CRS.from_user_input() <pyproj.crs.CRS.from_user_input>`,
            such as an authority string (eg "EPSG:4326") or a WKT string.
            If passed, overrides both DataFrame and col's crs.
            Otherwise, tries to get crs from passed col values or DataFrame.

        Examples
        --------
        >>> from shapely.geometry import Point
        >>> d = {'col1': ['name1', 'name2'], 'geometry': [Point(1, 2), Point(2, 1)]}
        >>> gdf = geopandas.GeoDataFrame(d, crs="EPSG:4326")
        >>> gdf
            col1                 geometry
        0  name1  POINT (1.00000 2.00000)
        1  name2  POINT (2.00000 1.00000)

        Passing an array:

        >>> df1 = gdf.set_geometry([Point(0,0), Point(1,1)])
        >>> df1
            col1                 geometry
        0  name1  POINT (0.00000 0.00000)
        1  name2  POINT (1.00000 1.00000)

        Using existing column:

        >>> gdf["buffered"] = gdf.buffer(2)
        >>> df2 = gdf.set_geometry("buffered")
        >>> df2.geometry
        0    POLYGON ((3.00000 2.00000, 2.99037 1.80397, 2....
        1    POLYGON ((4.00000 1.00000, 3.99037 0.80397, 3....
        Name: buffered, dtype: geometry

        Returns
        -------
        GeoDataFrame

        See also
        --------
        GeoDataFrame.rename_geometry : rename an active geometry column
        """
        # Most of the code here is taken from DataFrame.set_index()
        if inplace:
            frame = self
        else:
            frame = self.copy()

        to_remove = None
        geo_column_name = self._geometry_column_name
        if isinstance(col, (Series, list, np.ndarray, GeometryArray)):
            level = col
        elif hasattr(col, "ndim") and col.ndim != 1:
            raise ValueError("Must pass array with one dimension only.")
        else:
            try:
                level = frame[col]
            except KeyError:
                raise ValueError("Unknown column %s" % col)
            except Exception:
                raise
            if isinstance(level, DataFrame):
                raise ValueError(
                    "GeoDataFrame does not support setting the geometry column where "
                    "the column name is shared by multiple columns."
                )

            if drop:
                to_remove = col
                geo_column_name = self._geometry_column_name
            else:
                geo_column_name = col

        if to_remove:
            del frame[to_remove]

        if not crs:
            level_crs = getattr(level, "crs", None)
            crs = level_crs if level_crs is not None else self._crs

        if isinstance(level, (GeoSeries, GeometryArray)) and level.crs != crs:
            # Avoids caching issues/crs sharing issues
            level = level.copy()
            level.crs = crs

        # Check that we are using a listlike of geometries
        level = _ensure_geometry(level, crs=crs)
        index = frame.index
        frame[geo_column_name] = level
        if frame.index is not index and len(frame.index) == len(index):
            # With pandas < 1.0 and an empty frame (no rows), the index gets reset
            # to a default RangeIndex -> set back the original index if needed
            frame.index = index
        frame._geometry_column_name = geo_column_name
        frame.crs = crs
        if not inplace:
            return frame

    def rename_geometry(self, col, inplace=False):
        """
        Renames the GeoDataFrame geometry column to
        the specified name. By default yields a new object.

        The original geometry column is replaced with the input.

        Parameters
        ----------
        col : new geometry column label
        inplace : boolean, default False
            Modify the GeoDataFrame in place (do not create a new object)

        Examples
        --------
        >>> from shapely.geometry import Point
        >>> d = {'col1': ['name1', 'name2'], 'geometry': [Point(1, 2), Point(2, 1)]}
        >>> df = geopandas.GeoDataFrame(d, crs="EPSG:4326")
        >>> df1 = df.rename_geometry('geom1')
        >>> df1.geometry.name
        'geom1'
        >>> df.rename_geometry('geom1', inplace=True)
        >>> df.geometry.name
        'geom1'

        Returns
        -------
        geodataframe : GeoDataFrame

        See also
        --------
        GeoDataFrame.set_geometry : set the active geometry
        """
        geometry_col = self.geometry.name
        if col in self.columns:
            raise ValueError(f"Column named {col} already exists")
        else:
            if not inplace:
                return self.rename(columns={geometry_col: col}).set_geometry(
                    col, inplace
                )
            self.rename(columns={geometry_col: col}, inplace=inplace)
            self.set_geometry(col, inplace=inplace)

    @property
    def crs(self):
        """
        The Coordinate Reference System (CRS) represented as a ``pyproj.CRS``
        object.

        Returns None if the CRS is not set, and to set the value it
        :getter: Returns a ``pyproj.CRS`` or None. When setting, the value
        can be anything accepted by
        :meth:`pyproj.CRS.from_user_input() <pyproj.crs.CRS.from_user_input>`,
        such as an authority string (eg "EPSG:4326") or a WKT string.

        Examples
        --------

        >>> gdf.crs  # doctest: +SKIP
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

        See also
        --------
        GeoDataFrame.set_crs : assign CRS
        GeoDataFrame.to_crs : re-project to another CRS

        """
        return self._crs

    @crs.setter
    def crs(self, value):
        """Sets the value of the crs"""
        if self._geometry_column_name not in self:
            warnings.warn(
                "Assigning CRS to a GeoDataFrame without a geometry column is now "
                "deprecated and will not be supported in the future.",
                FutureWarning,
                stacklevel=4,
            )
            self._crs = None if not value else CRS.from_user_input(value)
        else:
            if hasattr(self.geometry.values, "crs"):
                self.geometry.values.crs = value
                self._crs = self.geometry.values.crs
            else:
                # column called 'geometry' without geometry
                self._crs = None if not value else CRS.from_user_input(value)

    def __setstate__(self, state):
        # overriding DataFrame method for compat with older pickles (CRS handling)
        if isinstance(state, dict):
            if "_metadata" in state and "crs" in state["_metadata"]:
                metadata = state["_metadata"]
                metadata[metadata.index("crs")] = "_crs"
            if "crs" in state and "_crs" not in state:
                crs = state.pop("crs")
                state["_crs"] = CRS.from_user_input(crs) if crs is not None else crs

        super().__setstate__(state)

        # for some versions that didn't yet have CRS at array level -> crs is set
        # at GeoDataFrame level with '_crs' (and not 'crs'), so without propagating
        # to the GeoSeries/GeometryArray
        try:
            if self.crs is not None:
                if self.geometry.values.crs is None:
                    self.crs = self.crs
        except Exception:
            pass

    @classmethod
    def from_dict(cls, data, geometry=None, crs=None, **kwargs):
        """
        Construct GeoDataFrame from dict of array-like or dicts by
        overriding DataFrame.from_dict method with geometry and crs

        Parameters
        ----------
        data : dict
            Of the form {field : array-like} or {field : dict}.
        geometry : str or array (optional)
            If str, column to use as geometry. If array, will be set as 'geometry'
            column on GeoDataFrame.
        crs : str or dict (optional)
            Coordinate reference system to set on the resulting frame.
        kwargs : key-word arguments
            These arguments are passed to DataFrame.from_dict

        Returns
        -------
        GeoDataFrame

        """
        dataframe = super().from_dict(data, **kwargs)
        return GeoDataFrame(dataframe, geometry=geometry, crs=crs)

    @classmethod
    def from_file(cls, filename, **kwargs):
        """Alternate constructor to create a ``GeoDataFrame`` from a file.

        It is recommended to use :func:`geopandas.read_file` instead.

        Can load a ``GeoDataFrame`` from a file in any format recognized by
        `fiona`. See http://fiona.readthedocs.io/en/latest/manual.html for details.

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
        >>> gdf = geopandas.GeoDataFrame.from_file(path)
        >>> gdf  # doctest: +SKIP
           BoroCode       BoroName     Shape_Leng    Shape_Area                 \
                          geometry
        0         5  Staten Island  330470.010332  1.623820e+09  MULTIPOLYGON ((\
(970217.022 145643.332, 970227....
        1         4         Queens  896344.047763  3.045213e+09  MULTIPOLYGON ((\
(1029606.077 156073.814, 102957...
        2         3       Brooklyn  741080.523166  1.937479e+09  MULTIPOLYGON ((\
(1021176.479 151374.797, 102100...
        3         1      Manhattan  359299.096471  6.364715e+08  MULTIPOLYGON ((\
(981219.056 188655.316, 980940....
        4         2          Bronx  464392.991824  1.186925e+09  MULTIPOLYGON ((\
(1012821.806 229228.265, 101278...

        The recommended method of reading files is :func:`geopandas.read_file`:

        >>> gdf = geopandas.read_file(path)

        See also
        --------
        read_file : read file to GeoDataFame
        GeoDataFrame.to_file : write GeoDataFrame to file

        """
        return geopandas.io.file._read_file(filename, **kwargs)

    @classmethod
    def from_features(cls, features, crs=None, columns=None):
        """
        Alternate constructor to create GeoDataFrame from an iterable of
        features or a feature collection.

        Parameters
        ----------
        features
            - Iterable of features, where each element must be a feature
              dictionary or implement the __geo_interface__.
            - Feature collection, where the 'features' key contains an
              iterable of features.
            - Object holding a feature collection that implements the
              ``__geo_interface__``.
        crs : str or dict (optional)
            Coordinate reference system to set on the resulting frame.
        columns : list of column names, optional
            Optionally specify the column names to include in the output frame.
            This does not overwrite the property names of the input, but can
            ensure a consistent output format.

        Returns
        -------
        GeoDataFrame

        Notes
        -----
        For more information about the ``__geo_interface__``, see
        https://gist.github.com/sgillies/2217756

        Examples
        --------
        >>> feature_coll = {
        ...     "type": "FeatureCollection",
        ...     "features": [
        ...         {
        ...             "id": "0",
        ...             "type": "Feature",
        ...             "properties": {"col1": "name1"},
        ...             "geometry": {"type": "Point", "coordinates": (1.0, 2.0)},
        ...             "bbox": (1.0, 2.0, 1.0, 2.0),
        ...         },
        ...         {
        ...             "id": "1",
        ...             "type": "Feature",
        ...             "properties": {"col1": "name2"},
        ...             "geometry": {"type": "Point", "coordinates": (2.0, 1.0)},
        ...             "bbox": (2.0, 1.0, 2.0, 1.0),
        ...         },
        ...     ],
        ...     "bbox": (1.0, 1.0, 2.0, 2.0),
        ... }
        >>> df = geopandas.GeoDataFrame.from_features(feature_coll)
        >>> df
                          geometry   col1
        0  POINT (1.00000 2.00000)  name1
        1  POINT (2.00000 1.00000)  name2

        """
        # Handle feature collections
        if hasattr(features, "__geo_interface__"):
            fs = features.__geo_interface__
        else:
            fs = features

        if isinstance(fs, dict) and fs.get("type") == "FeatureCollection":
            features_lst = fs["features"]
        else:
            features_lst = features

        rows = []
        for feature in features_lst:
            # load geometry
            if hasattr(feature, "__geo_interface__"):
                feature = feature.__geo_interface__
            row = {
                "geometry": shape(feature["geometry"]) if feature["geometry"] else None
            }
            # load properties
            row.update(feature["properties"])
            rows.append(row)
        return GeoDataFrame(rows, columns=columns, crs=crs)

    @classmethod
    def from_postgis(
        cls,
        sql,
        con,
        geom_col="geom",
        crs=None,
        index_col=None,
        coerce_float=True,
        parse_dates=None,
        params=None,
        chunksize=None,
    ):
        """
        Alternate constructor to create a ``GeoDataFrame`` from a sql query
        containing a geometry column in WKB representation.

        Parameters
        ----------
        sql : string
        con : sqlalchemy.engine.Connection or sqlalchemy.engine.Engine
        geom_col : string, default 'geom'
            column name to convert to shapely geometries
        crs : optional
            Coordinate reference system to use for the returned GeoDataFrame
        index_col : string or list of strings, optional, default: None
            Column(s) to set as index(MultiIndex)
        coerce_float : boolean, default True
            Attempt to convert values of non-string, non-numeric objects (like
            decimal.Decimal) to floating point, useful for SQL result sets
        parse_dates : list or dict, default None
            - List of column names to parse as dates.
            - Dict of ``{column_name: format string}`` where format string is
              strftime compatible in case of parsing string times, or is one of
              (D, s, ns, ms, us) in case of parsing integer timestamps.
            - Dict of ``{column_name: arg dict}``, where the arg dict
              corresponds to the keyword arguments of
              :func:`pandas.to_datetime`. Especially useful with databases
              without native Datetime support, such as SQLite.
        params : list, tuple or dict, optional, default None
            List of parameters to pass to execute method.
        chunksize : int, default None
            If specified, return an iterator where chunksize is the number
            of rows to include in each chunk.

        Examples
        --------
        PostGIS

        >>> from sqlalchemy import create_engine  # doctest: +SKIP
        >>> db_connection_url = "postgresql://myusername:mypassword@myhost:5432/mydb"
        >>> con = create_engine(db_connection_url)  # doctest: +SKIP
        >>> sql = "SELECT geom, highway FROM roads"
        >>> df = geopandas.GeoDataFrame.from_postgis(sql, con)  # doctest: +SKIP

        SpatiaLite

        >>> sql = "SELECT ST_Binary(geom) AS geom, highway FROM roads"
        >>> df = geopandas.GeoDataFrame.from_postgis(sql, con)  # doctest: +SKIP

        The recommended method of reading from PostGIS is
        :func:`geopandas.read_postgis`:

        >>> df = geopandas.read_postgis(sql, con)  # doctest: +SKIP

        See also
        --------
        geopandas.read_postgis : read PostGIS database to GeoDataFrame
        """

        df = geopandas.io.sql._read_postgis(
            sql,
            con,
            geom_col=geom_col,
            crs=crs,
            index_col=index_col,
            coerce_float=coerce_float,
            parse_dates=parse_dates,
            params=params,
            chunksize=chunksize,
        )

        return df

    def to_json(self, na="null", show_bbox=False, drop_id=False, **kwargs):
        """
        Returns a GeoJSON representation of the ``GeoDataFrame`` as a string.

        Parameters
        ----------
        na : {'null', 'drop', 'keep'}, default 'null'
            Indicates how to output missing (NaN) values in the GeoDataFrame.
            See below.
        show_bbox : bool, optional, default: False
            Include bbox (bounds) in the geojson
        drop_id : bool, default: False
            Whether to retain the index of the GeoDataFrame as the id property
            in the generated GeoJSON. Default is False, but may want True
            if the index is just arbitrary row numbers.

        Notes
        -----
        The remaining *kwargs* are passed to json.dumps().

        Missing (NaN) values in the GeoDataFrame can be represented as follows:

        - ``null``: output the missing entries as JSON null.
        - ``drop``: remove the property from the feature. This applies to each
          feature individually so that features may have different properties.
        - ``keep``: output the missing entries as NaN.

        Examples
        --------

        >>> from shapely.geometry import Point
        >>> d = {'col1': ['name1', 'name2'], 'geometry': [Point(1, 2), Point(2, 1)]}
        >>> gdf = geopandas.GeoDataFrame(d, crs="EPSG:4326")
        >>> gdf
            col1                 geometry
        0  name1  POINT (1.00000 2.00000)
        1  name2  POINT (2.00000 1.00000)

        >>> gdf.to_json()
        '{"type": "FeatureCollection", "features": [{"id": "0", "type": "Feature", \
"properties": {"col1": "name1"}, "geometry": {"type": "Point", "coordinates": [1.0,\
 2.0]}}, {"id": "1", "type": "Feature", "properties": {"col1": "name2"}, "geometry"\
: {"type": "Point", "coordinates": [2.0, 1.0]}}]}'

        Alternatively, you can write GeoJSON to file:

        >>> gdf.to_file(path, driver="GeoJSON")  # doctest: +SKIP

        See also
        --------
        GeoDataFrame.to_file : write GeoDataFrame to file

        """
        return json.dumps(
            self._to_geo(na=na, show_bbox=show_bbox, drop_id=drop_id), **kwargs
        )

    @property
    def __geo_interface__(self):
        """Returns a ``GeoDataFrame`` as a python feature collection.

        Implements the `geo_interface`. The returned python data structure
        represents the ``GeoDataFrame`` as a GeoJSON-like
        ``FeatureCollection``.

        This differs from `_to_geo()` only in that it is a property with
        default args instead of a method

        Examples
        --------

        >>> from shapely.geometry import Point
        >>> d = {'col1': ['name1', 'name2'], 'geometry': [Point(1, 2), Point(2, 1)]}
        >>> gdf = geopandas.GeoDataFrame(d, crs="EPSG:4326")
        >>> gdf
            col1                 geometry
        0  name1  POINT (1.00000 2.00000)
        1  name2  POINT (2.00000 1.00000)

        >>> gdf.__geo_interface__
        {'type': 'FeatureCollection', 'features': [{'id': '0', 'type': 'Feature', \
'properties': {'col1': 'name1'}, 'geometry': {'type': 'Point', 'coordinates': (1.0\
, 2.0)}, 'bbox': (1.0, 2.0, 1.0, 2.0)}, {'id': '1', 'type': 'Feature', 'properties\
': {'col1': 'name2'}, 'geometry': {'type': 'Point', 'coordinates': (2.0, 1.0)}, 'b\
box': (2.0, 1.0, 2.0, 1.0)}], 'bbox': (1.0, 1.0, 2.0, 2.0)}


        """
        return self._to_geo(na="null", show_bbox=True, drop_id=False)

    def iterfeatures(self, na="null", show_bbox=False, drop_id=False):
        """
        Returns an iterator that yields feature dictionaries that comply with
        __geo_interface__

        Parameters
        ----------
        na : str, optional
            Options are {'null', 'drop', 'keep'}, default 'null'.
            Indicates how to output missing (NaN) values in the GeoDataFrame

            - null: output the missing entries as JSON null
            - drop: remove the property from the feature. This applies to each feature \
individually so that features may have different properties
            - keep: output the missing entries as NaN

        show_bbox : bool, optional
            Include bbox (bounds) in the geojson. Default False.
        drop_id : bool, default: False
            Whether to retain the index of the GeoDataFrame as the id property
            in the generated GeoJSON. Default is False, but may want True
            if the index is just arbitrary row numbers.

        Examples
        --------

        >>> from shapely.geometry import Point
        >>> d = {'col1': ['name1', 'name2'], 'geometry': [Point(1, 2), Point(2, 1)]}
        >>> gdf = geopandas.GeoDataFrame(d, crs="EPSG:4326")
        >>> gdf
            col1                 geometry
        0  name1  POINT (1.00000 2.00000)
        1  name2  POINT (2.00000 1.00000)

        >>> feature = next(gdf.iterfeatures())
        >>> feature
        {'id': '0', 'type': 'Feature', 'properties': {'col1': 'name1'}, 'geometry': {\
'type': 'Point', 'coordinates': (1.0, 2.0)}}
        """
        if na not in ["null", "drop", "keep"]:
            raise ValueError("Unknown na method {0}".format(na))

        if self._geometry_column_name not in self:
            raise AttributeError(
                "No geometry data set (expected in"
                " column '%s')." % self._geometry_column_name
            )

        ids = np.array(self.index, copy=False)
        geometries = np.array(self[self._geometry_column_name], copy=False)

        if not self.columns.is_unique:
            raise ValueError("GeoDataFrame cannot contain duplicated column names.")

        properties_cols = self.columns.difference([self._geometry_column_name])

        if len(properties_cols) > 0:
            # convert to object to get python scalars.
            properties = self[properties_cols].astype(object).values
            if na == "null":
                properties[pd.isnull(self[properties_cols]).values] = None

            for i, row in enumerate(properties):
                geom = geometries[i]

                if na == "drop":
                    properties_items = {
                        k: v for k, v in zip(properties_cols, row) if not pd.isnull(v)
                    }
                else:
                    properties_items = {k: v for k, v in zip(properties_cols, row)}

                if drop_id:
                    feature = {}
                else:
                    feature = {"id": str(ids[i])}

                feature["type"] = "Feature"
                feature["properties"] = properties_items
                feature["geometry"] = mapping(geom) if geom else None

                if show_bbox:
                    feature["bbox"] = geom.bounds if geom else None

                yield feature

        else:
            for fid, geom in zip(ids, geometries):

                if drop_id:
                    feature = {}
                else:
                    feature = {"id": str(fid)}

                feature["type"] = "Feature"
                feature["properties"] = {}
                feature["geometry"] = mapping(geom) if geom else None

                if show_bbox:
                    feature["bbox"] = geom.bounds if geom else None

                yield feature

    def _to_geo(self, **kwargs):
        """
        Returns a python feature collection (i.e. the geointerface)
        representation of the GeoDataFrame.

        """
        geo = {
            "type": "FeatureCollection",
            "features": list(self.iterfeatures(**kwargs)),
        }

        if kwargs.get("show_bbox", False):
            geo["bbox"] = tuple(self.total_bounds)

        return geo

    def to_wkb(self, hex=False, **kwargs):
        """
        Encode all geometry columns in the GeoDataFrame to WKB.

        Parameters
        ----------
        hex : bool
            If true, export the WKB as a hexadecimal string.
            The default is to return a binary bytes object.
        kwargs
            Additional keyword args will be passed to
            :func:`pygeos.to_wkb` if pygeos is installed.

        Returns
        -------
        DataFrame
            geometry columns are encoded to WKB
        """

        df = DataFrame(self.copy())

        # Encode all geometry columns to WKB
        for col in df.columns[df.dtypes == "geometry"]:
            df[col] = to_wkb(df[col].values, hex=hex, **kwargs)

        return df

    def to_wkt(self, **kwargs):
        """
        Encode all geometry columns in the GeoDataFrame to WKT.

        Parameters
        ----------
        kwargs
            Keyword args will be passed to :func:`pygeos.to_wkt`
            if pygeos is installed.

        Returns
        -------
        DataFrame
            geometry columns are encoded to WKT
        """

        df = DataFrame(self.copy())

        # Encode all geometry columns to WKT
        for col in df.columns[df.dtypes == "geometry"]:
            df[col] = to_wkt(df[col].values, **kwargs)

        return df

    def to_parquet(self, path, index=None, compression="snappy", **kwargs):
        """Write a GeoDataFrame to the Parquet format.

        Any geometry columns present are serialized to WKB format in the file.

        Requires 'pyarrow'.

        WARNING: this is an initial implementation of Parquet file support and
        associated metadata.  This is tracking version 0.1.0 of the metadata
        specification at:
        https://github.com/geopandas/geo-arrow-spec

        This metadata specification does not yet make stability promises.  As such,
        we do not yet recommend using this in a production setting unless you are
        able to rewrite your Parquet files.

        .. versionadded:: 0.8

        Parameters
        ----------
        path : str, path object
        index : bool, default None
            If ``True``, always include the dataframe's index(es) as columns
            in the file output.
            If ``False``, the index(es) will not be written to the file.
            If ``None``, the index(ex) will be included as columns in the file
            output except `RangeIndex` which is stored as metadata only.
        compression : {'snappy', 'gzip', 'brotli', None}, default 'snappy'
            Name of the compression to use. Use ``None`` for no compression.
        kwargs
            Additional keyword arguments passed to :func:`pyarrow.parquet.write_table`.

        Examples
        --------

        >>> gdf.to_parquet('data.parquet')  # doctest: +SKIP

        See also
        --------
        GeoDataFrame.to_feather : write GeoDataFrame to feather
        GeoDataFrame.to_file : write GeoDataFrame to file
        """

        from geopandas.io.arrow import _to_parquet

        _to_parquet(self, path, compression=compression, index=index, **kwargs)

    def to_feather(self, path, index=None, compression=None, **kwargs):
        """Write a GeoDataFrame to the Feather format.

        Any geometry columns present are serialized to WKB format in the file.

        Requires 'pyarrow' >= 0.17.

        WARNING: this is an initial implementation of Feather file support and
        associated metadata.  This is tracking version 0.1.0 of the metadata
        specification at:
        https://github.com/geopandas/geo-arrow-spec

        This metadata specification does not yet make stability promises.  As such,
        we do not yet recommend using this in a production setting unless you are
        able to rewrite your Feather files.

        .. versionadded:: 0.8

        Parameters
        ----------
        path : str, path object
        index : bool, default None
            If ``True``, always include the dataframe's index(es) as columns
            in the file output.
            If ``False``, the index(es) will not be written to the file.
            If ``None``, the index(ex) will be included as columns in the file
            output except `RangeIndex` which is stored as metadata only.
        compression : {'zstd', 'lz4', 'uncompressed'}, optional
            Name of the compression to use. Use ``"uncompressed"`` for no
            compression. By default uses LZ4 if available, otherwise uncompressed.
        kwargs
            Additional keyword arguments passed to to
            :func:`pyarrow.feather.write_feather`.

        Examples
        --------

        >>> gdf.to_feather('data.feather')  # doctest: +SKIP

        See also
        --------
        GeoDataFrame.to_parquet : write GeoDataFrame to parquet
        GeoDataFrame.to_file : write GeoDataFrame to file
        """

        from geopandas.io.arrow import _to_feather

        _to_feather(self, path, index=index, compression=compression, **kwargs)

    def to_file(self, filename, driver=None, schema=None, index=None, **kwargs):
        """Write the ``GeoDataFrame`` to a file.

        By default, an ESRI shapefile is written, but any OGR data source
        supported by Fiona can be written. A dictionary of supported OGR
        providers is available via:

        >>> import fiona
        >>> fiona.supported_drivers  # doctest: +SKIP

        Parameters
        ----------
        filename : string
            File path or file handle to write to.
        driver : string, default None
            The OGR format driver used to write the vector file.
            If not specified, it attempts to infer it from the file extension.
            If no extension is specified, it saves ESRI Shapefile to a folder.
        schema : dict, default: None
            If specified, the schema dictionary is passed to Fiona to
            better control how the file is written.
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

        The format drivers will attempt to detect the encoding of your data, but
        may fail. In this case, the proper encoding can be specified explicitly
        by using the encoding keyword parameter, e.g. ``encoding='utf-8'``.

        See Also
        --------
        GeoSeries.to_file
        GeoDataFrame.to_postgis : write GeoDataFrame to PostGIS database
        GeoDataFrame.to_parquet : write GeoDataFrame to parquet
        GeoDataFrame.to_feather : write GeoDataFrame to feather

        Examples
        --------

        >>> gdf.to_file('dataframe.shp')  # doctest: +SKIP

        >>> gdf.to_file('dataframe.gpkg', driver='GPKG', layer='name')  # doctest: +SKIP

        >>> gdf.to_file('dataframe.geojson', driver='GeoJSON')  # doctest: +SKIP

        With selected drivers you can also append to a file with `mode="a"`:

        >>> gdf.to_file('dataframe.shp', mode="a")  # doctest: +SKIP
        """
        from geopandas.io.file import _to_file

        _to_file(self, filename, driver, schema, index, **kwargs)

    def set_crs(self, crs=None, epsg=None, inplace=False, allow_override=False):
        """
        Set the Coordinate Reference System (CRS) of the ``GeoDataFrame``.

        If there are multiple geometry columns within the GeoDataFrame, only
        the CRS of the active geometry column is set.

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
            If True, the CRS of the GeoDataFrame will be changed in place
            (while still returning the result) instead of making a copy of
            the GeoDataFrame.
        allow_override : bool, default False
            If the the GeoDataFrame already has a CRS, allow to replace the
            existing CRS, even when both are not equal.

        Examples
        --------
        >>> from shapely.geometry import Point
        >>> d = {'col1': ['name1', 'name2'], 'geometry': [Point(1, 2), Point(2, 1)]}
        >>> gdf = geopandas.GeoDataFrame(d)
        >>> gdf
            col1                 geometry
        0  name1  POINT (1.00000 2.00000)
        1  name2  POINT (2.00000 1.00000)

        Setting CRS to a GeoDataFrame without one:

        >>> gdf.crs is None
        True

        >>> gdf = gdf.set_crs('epsg:3857')
        >>> gdf.crs  # doctest: +SKIP
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

        >>> gdf = gdf.set_crs(4326, allow_override=True)

        Without ``allow_override=True``, ``set_crs`` returns an error if you try to
        override CRS.

        See also
        --------
        GeoDataFrame.to_crs : re-project to another CRS

        """
        if not inplace:
            df = self.copy()
        else:
            df = self
        df.geometry = df.geometry.set_crs(
            crs=crs, epsg=epsg, allow_override=allow_override, inplace=True
        )
        return df

    def to_crs(self, crs=None, epsg=None, inplace=False):
        """Transform geometries to a new coordinate reference system.

        Transform all geometries in an active geometry column to a different coordinate
        reference system.  The ``crs`` attribute on the current GeoSeries must
        be set.  Either ``crs`` or ``epsg`` may be specified for output.

        This method will transform all points in all objects. It has no notion
        or projecting entire geometries.  All segments joining points are
        assumed to be lines in the current projection, not geodesics. Objects
        crossing the dateline (or other projection boundary) will have
        undesirable behavior.

        Parameters
        ----------
        crs : pyproj.CRS, optional if `epsg` is specified
            The value can be anything accepted by
            :meth:`pyproj.CRS.from_user_input() <pyproj.crs.CRS.from_user_input>`,
            such as an authority string (eg "EPSG:4326") or a WKT string.
        epsg : int, optional if `crs` is specified
            EPSG code specifying output projection.
        inplace : bool, optional, default: False
            Whether to return a new GeoDataFrame or do the transformation in
            place.

        Returns
        -------
        GeoDataFrame

        Examples
        --------
        >>> from shapely.geometry import Point
        >>> d = {'col1': ['name1', 'name2'], 'geometry': [Point(1, 2), Point(2, 1)]}
        >>> gdf = geopandas.GeoDataFrame(d, crs=4326)
        >>> gdf
            col1                 geometry
        0  name1  POINT (1.00000 2.00000)
        1  name2  POINT (2.00000 1.00000)
        >>> gdf.crs  # doctest: +SKIP
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

        >>> gdf = gdf.to_crs(3857)
        >>> gdf
            col1                       geometry
        0  name1  POINT (111319.491 222684.209)
        1  name2  POINT (222638.982 111325.143)
        >>> gdf.crs  # doctest: +SKIP
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

        See also
        --------
        GeoDataFrame.set_crs : assign CRS without re-projection
        """
        if inplace:
            df = self
        else:
            df = self.copy()
        geom = df.geometry.to_crs(crs=crs, epsg=epsg)
        df.geometry = geom
        df.crs = geom.crs
        if not inplace:
            return df

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
        >>> germany.estimate_utm_crs()  # doctest: +SKIP
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
        return self.geometry.estimate_utm_crs(datum_name=datum_name)

    def __getitem__(self, key):
        """
        If the result is a column containing only 'geometry', return a
        GeoSeries. If it's a DataFrame with a 'geometry' column, return a
        GeoDataFrame.
        """
        result = super().__getitem__(key)
        geo_col = self._geometry_column_name
        if isinstance(result, Series) and isinstance(result.dtype, GeometryDtype):
            result.__class__ = GeoSeries
        elif isinstance(result, DataFrame) and geo_col in result:
            result.__class__ = GeoDataFrame
            result._geometry_column_name = geo_col
        elif isinstance(result, DataFrame) and geo_col not in result:
            result.__class__ = DataFrame
        return result

    def __setitem__(self, key, value):
        """
        Overwritten to preserve CRS of GeometryArray in cases like
        df['geometry'] = [geom... for geom in df.geometry]
        """
        if not pd.api.types.is_list_like(key) and key == self._geometry_column_name:
            if pd.api.types.is_scalar(value) or isinstance(value, BaseGeometry):
                value = [value] * self.shape[0]
            try:
                value = _ensure_geometry(value, crs=self.crs)
                self._crs = value.crs
            except TypeError:
                warnings.warn("Geometry column does not contain geometry.")
        super().__setitem__(key, value)

    #
    # Implement pandas methods
    #

    def merge(self, *args, **kwargs):
        r"""Merge two ``GeoDataFrame`` objects with a database-style join.

        Returns a ``GeoDataFrame`` if a geometry column is present; otherwise,
        returns a pandas ``DataFrame``.

        Returns
        -------
        GeoDataFrame or DataFrame

        Notes
        -----
        The extra arguments ``*args`` and keyword arguments ``**kwargs`` are
        passed to DataFrame.merge.

        Reference
        ---------
        https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas\
        .DataFrame.merge.html

        """
        result = DataFrame.merge(self, *args, **kwargs)
        geo_col = self._geometry_column_name
        if isinstance(result, DataFrame) and geo_col in result:
            result.__class__ = GeoDataFrame
            result.crs = self.crs
            result._geometry_column_name = geo_col
        elif isinstance(result, DataFrame) and geo_col not in result:
            result.__class__ = DataFrame
        return result

    @doc(pd.DataFrame)
    def apply(self, func, axis=0, raw=False, result_type=None, args=(), **kwargs):
        result = super().apply(
            func, axis=axis, raw=raw, result_type=result_type, args=args, **kwargs
        )
        if (
            isinstance(result, GeoDataFrame)
            and self._geometry_column_name in result.columns
            and isinstance(result[self._geometry_column_name].dtype, GeometryDtype)
        ):
            # apply calls _constructor which resets geom col name to geometry
            result._geometry_column_name = self._geometry_column_name
            if self.crs is not None and result.crs is None:
                result.set_crs(self.crs, inplace=True)
        return result

    @property
    def _constructor(self):
        return GeoDataFrame

    def __finalize__(self, other, method=None, **kwargs):
        """propagate metadata from other to self"""
        self = super().__finalize__(other, method=method, **kwargs)

        # merge operation: using metadata of the left object
        if method == "merge":
            for name in self._metadata:
                object.__setattr__(self, name, getattr(other.left, name, None))
        # concat operation: using metadata of the first object
        elif method == "concat":
            for name in self._metadata:
                object.__setattr__(self, name, getattr(other.objs[0], name, None))

            if (self.columns == self._geometry_column_name).sum() > 1:
                raise ValueError(
                    "Concat operation has resulted in multiple columns using "
                    f"the geometry column name '{self._geometry_column_name}'.\n"
                    f"Please ensure this column from the first DataFrame is not "
                    f"repeated."
                )
        return self

    def dissolve(
        self,
        by=None,
        aggfunc="first",
        as_index=True,
        level=None,
        sort=True,
        observed=False,
        dropna=True,
    ):
        """
        Dissolve geometries within `groupby` into single observation.
        This is accomplished by applying the `unary_union` method
        to all geometries within a groupself.

        Observations associated with each `groupby` group will be aggregated
        using the `aggfunc`.

        Parameters
        ----------
        by : string, default None
            Column whose values define groups to be dissolved. If None,
            whole GeoDataFrame is considered a single group.
        aggfunc : function or string, default "first"
            Aggregation function for manipulation of data associated
            with each group. Passed to pandas `groupby.agg` method.
        as_index : boolean, default True
            If true, groupby columns become index of result.
        level : int or str or sequence of int or sequence of str, default None
            If the axis is a MultiIndex (hierarchical), group by a
            particular level or levels.

            .. versionadded:: 0.9.0
        sort : bool, default True
            Sort group keys. Get better performance by turning this off.
            Note this does not influence the order of observations within
            each group. Groupby preserves the order of rows within each group.

            .. versionadded:: 0.9.0
        observed : bool, default False
            This only applies if any of the groupers are Categoricals.
            If True: only show observed values for categorical groupers.
            If False: show all values for categorical groupers.

            .. versionadded:: 0.9.0
        dropna : bool, default True
            If True, and if group keys contain NA values, NA values
            together with row/column will be dropped. If False, NA
            values will also be treated as the key in groups.

            This parameter is not supported for pandas < 1.1.0.
            A warning will be emitted for earlier pandas versions
            if a non-default value is given for this parameter.

            .. versionadded:: 0.9.0

        Returns
        -------
        GeoDataFrame

        Examples
        --------
        >>> from shapely.geometry import Point
        >>> d = {
        ...     "col1": ["name1", "name2", "name1"],
        ...     "geometry": [Point(1, 2), Point(2, 1), Point(0, 1)],
        ... }
        >>> gdf = geopandas.GeoDataFrame(d, crs=4326)
        >>> gdf
            col1                 geometry
        0  name1  POINT (1.00000 2.00000)
        1  name2  POINT (2.00000 1.00000)
        2  name1  POINT (0.00000 1.00000)

        >>> dissolved = gdf.dissolve('col1')
        >>> dissolved  # doctest: +SKIP
                                                    geometry
        col1
        name1  MULTIPOINT (0.00000 1.00000, 1.00000 2.00000)
        name2                        POINT (2.00000 1.00000)

        See also
        --------
        GeoDataFrame.explode : explode muti-part geometries into single geometries

        """

        if by is None and level is None:
            by = np.zeros(len(self), dtype="int64")

        groupby_kwargs = dict(
            by=by, level=level, sort=sort, observed=observed, dropna=dropna
        )
        if not compat.PANDAS_GE_11:
            groupby_kwargs.pop("dropna")

            if not dropna:  # If they passed a non-default dropna value
                warnings.warn("dropna kwarg is not supported for pandas < 1.1.0")

        # Process non-spatial component
        data = self.drop(labels=self.geometry.name, axis=1)
        aggregated_data = data.groupby(**groupby_kwargs).agg(aggfunc)

        # Process spatial component
        def merge_geometries(block):
            merged_geom = block.unary_union
            return merged_geom

        g = self.groupby(group_keys=False, **groupby_kwargs)[self.geometry.name].agg(
            merge_geometries
        )

        # Aggregate
        aggregated_geometry = GeoDataFrame(g, geometry=self.geometry.name, crs=self.crs)
        # Recombine
        aggregated = aggregated_geometry.join(aggregated_data)

        # Reset if requested
        if not as_index:
            aggregated = aggregated.reset_index()

        return aggregated

    # overrides the pandas native explode method to break up features geometrically
    def explode(self, column=None, ignore_index=False, index_parts=None, **kwargs):
        """
        Explode muti-part geometries into multiple single geometries.

        Each row containing a multi-part geometry will be split into
        multiple rows with single geometries, thereby increasing the vertical
        size of the GeoDataFrame.

        .. note:: ignore_index requires pandas 1.1.0 or newer.

        Parameters
        ----------
        column : string, default None
            Column to explode. In the case of a geometry column, multi-part
            geometries are converted to single-part.
            If None, the active geometry column is used.
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
        GeoDataFrame
            Exploded geodataframe with each single geometry
            as a separate entry in the geodataframe.

        Examples
        --------

        >>> from shapely.geometry import MultiPoint
        >>> d = {
        ...     "col1": ["name1", "name2"],
        ...     "geometry": [
        ...         MultiPoint([(1, 2), (3, 4)]),
        ...         MultiPoint([(2, 1), (0, 0)]),
        ...     ],
        ... }
        >>> gdf = geopandas.GeoDataFrame(d, crs=4326)
        >>> gdf
            col1                                       geometry
        0  name1  MULTIPOINT (1.00000 2.00000, 3.00000 4.00000)
        1  name2  MULTIPOINT (2.00000 1.00000, 0.00000 0.00000)

        >>> exploded = gdf.explode(index_parts=True)
        >>> exploded
              col1                 geometry
        0 0  name1  POINT (1.00000 2.00000)
          1  name1  POINT (3.00000 4.00000)
        1 0  name2  POINT (2.00000 1.00000)
          1  name2  POINT (0.00000 0.00000)

        >>> exploded = gdf.explode(index_parts=False)
        >>> exploded
            col1                 geometry
        0  name1  POINT (1.00000 2.00000)
        0  name1  POINT (3.00000 4.00000)
        1  name2  POINT (2.00000 1.00000)
        1  name2  POINT (0.00000 0.00000)

        >>> exploded = gdf.explode(ignore_index=True)
        >>> exploded
            col1                 geometry
        0  name1  POINT (1.00000 2.00000)
        1  name1  POINT (3.00000 4.00000)
        2  name2  POINT (2.00000 1.00000)
        3  name2  POINT (0.00000 0.00000)

        See also
        --------
        GeoDataFrame.dissolve : dissolve geometries into a single observation.

        """

        # If no column is specified then default to the active geometry column
        if column is None:
            column = self.geometry.name
        # If the specified column is not a geometry dtype use pandas explode
        if not isinstance(self[column].dtype, GeometryDtype):
            if compat.PANDAS_GE_11:
                return super().explode(column, ignore_index=ignore_index, **kwargs)
            else:
                return super().explode(column, **kwargs)

        if index_parts is None:
            if not ignore_index:
                warnings.warn(
                    "Currently, index_parts defaults to True, but in the future, "
                    "it will default to False to be consistent with Pandas. "
                    "Use `index_parts=True` to keep the current behavior and "
                    "True/False to silence the warning.",
                    FutureWarning,
                    stacklevel=2,
                )
            index_parts = True

        df_copy = self.copy()

        level_str = f"level_{df_copy.index.nlevels}"

        if level_str in df_copy.columns:  # GH1393
            df_copy = df_copy.rename(columns={level_str: f"__{level_str}"})

        if index_parts:
            exploded_geom = df_copy.geometry.explode(index_parts=True)
            exploded_index = exploded_geom.index
            exploded_geom = exploded_geom.reset_index(level=-1, drop=True)
        else:
            exploded_geom = df_copy.geometry.explode(index_parts=True).reset_index(
                level=-1, drop=True
            )
            exploded_index = exploded_geom.index

        df = (
            df_copy.drop(df_copy._geometry_column_name, axis=1)
            .join(exploded_geom)
            .__finalize__(self)
        )

        if ignore_index:
            df.reset_index(inplace=True, drop=True)
        elif index_parts:
            # reset to MultiIndex, otherwise df index is only first level of
            # exploded GeoSeries index.
            df.set_index(exploded_index, inplace=True)
            df.index.names = list(self.index.names) + [None]
        else:
            df.set_index(exploded_index, inplace=True)
            df.index.names = self.index.names

        if f"__{level_str}" in df.columns:
            df = df.rename(columns={f"__{level_str}": level_str})

        geo_df = df.set_geometry(self._geometry_column_name)
        return geo_df

    # overrides the pandas astype method to ensure the correct return type
    def astype(self, dtype, copy=True, errors="raise", **kwargs):
        """
        Cast a pandas object to a specified dtype ``dtype``.

        Returns a GeoDataFrame when the geometry column is kept as geometries,
        otherwise returns a pandas DataFrame.

        See the pandas.DataFrame.astype docstring for more details.

        Returns
        -------
        GeoDataFrame or DataFrame
        """
        df = super().astype(dtype, copy=copy, errors=errors, **kwargs)

        try:
            geoms = df[self._geometry_column_name]
            if is_geometry_type(geoms):
                return geopandas.GeoDataFrame(df, geometry=self._geometry_column_name)
        except KeyError:
            pass
        # if the geometry column is converted to non-geometries or did not exist
        # do not return a GeoDataFrame
        return pd.DataFrame(df)

    def convert_dtypes(self, *args, **kwargs):
        """
        Convert columns to best possible dtypes using dtypes supporting ``pd.NA``.

        Always returns a GeoDataFrame as no conversions are applied to the
        geometry column.

        See the pandas.DataFrame.convert_dtypes docstring for more details.

        Returns
        -------
        GeoDataFrame

        """
        # Overridden to fix GH1870, that return type is not preserved always
        # (and where it was, geometry col was not)

        if not compat.PANDAS_GE_10:
            raise NotImplementedError(
                "GeoDataFrame.convert_dtypes requires pandas >= 1.0"
            )

        return GeoDataFrame(
            super().convert_dtypes(*args, **kwargs),
            geometry=self.geometry.name,
            crs=self.crs,
        )

    def to_postgis(
        self,
        name,
        con,
        schema=None,
        if_exists="fail",
        index=False,
        index_label=None,
        chunksize=None,
        dtype=None,
    ):
        """
        Upload GeoDataFrame into PostGIS database.

        This method requires SQLAlchemy and GeoAlchemy2, and a PostgreSQL
        Python driver (e.g. psycopg2) to be installed.

        Parameters
        ----------
        name : str
            Name of the target table.
        con : sqlalchemy.engine.Connection or sqlalchemy.engine.Engine
            Active connection to the PostGIS database.
        if_exists : {'fail', 'replace', 'append'}, default 'fail'
            How to behave if the table already exists:

            - fail: Raise a ValueError.
            - replace: Drop the table before inserting new values.
            - append: Insert new values to the existing table.
        schema : string, optional
            Specify the schema. If None, use default schema: 'public'.
        index : bool, default True
            Write DataFrame index as a column.
            Uses *index_label* as the column name in the table.
        index_label : string or sequence, default None
            Column label for index column(s).
            If None is given (default) and index is True,
            then the index names are used.
        chunksize : int, optional
            Rows will be written in batches of this size at a time.
            By default, all rows will be written at once.
        dtype : dict of column name to SQL type, default None
            Specifying the datatype for columns.
            The keys should be the column names and the values
            should be the SQLAlchemy types.

        Examples
        --------

        >>> from sqlalchemy import create_engine
        >>> engine = create_engine("postgresql://myusername:mypassword@myhost:5432\
/mydatabase")  # doctest: +SKIP
        >>> gdf.to_postgis("my_table", engine)  # doctest: +SKIP

        See also
        --------
        GeoDataFrame.to_file : write GeoDataFrame to file
        read_postgis : read PostGIS database to GeoDataFrame

        """
        geopandas.io.sql._write_postgis(
            self, name, con, schema, if_exists, index, index_label, chunksize, dtype
        )

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
        return self.geometry.symmetric_difference(other)

    def __or__(self, other):
        """Implement | operator as for builtin set type"""
        warnings.warn(
            "'|' operator will be deprecated. Use the 'union' method instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.geometry.union(other)

    def __and__(self, other):
        """Implement & operator as for builtin set type"""
        warnings.warn(
            "'&' operator will be deprecated. Use the 'intersection' method instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.geometry.intersection(other)

    def __sub__(self, other):
        """Implement - operator as for builtin set type"""
        warnings.warn(
            "'-' operator will be deprecated. Use the 'difference' method instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.geometry.difference(other)

    plot = CachedAccessor("plot", geopandas.plotting.GeoplotAccessor)

    @doc(_explore)
    def explore(self, *args, **kwargs):
        """Interactive map based on folium/leaflet.js"""
        return _explore(self, *args, **kwargs)

    def sjoin(self, df, *args, **kwargs):
        """Spatial join of two GeoDataFrames.

        See the User Guide page :doc:`../../user_guide/mergingdata` for details.

        Parameters
        ----------
        df : GeoDataFrame
        how : string, default 'inner'
            The type of join:

            * 'left': use keys from left_df; retain only left_df geometry column
            * 'right': use keys from right_df; retain only right_df geometry column
            * 'inner': use intersection of keys from both dfs; retain only
              left_df geometry column

        predicate : string, default 'intersects'
            Binary predicate. Valid values are determined by the spatial index used.
            You can check the valid values in left_df or right_df as
            ``left_df.sindex.valid_query_predicates`` or
            ``right_df.sindex.valid_query_predicates``
        lsuffix : string, default 'left'
            Suffix to apply to overlapping column names (left GeoDataFrame).
        rsuffix : string, default 'right'
            Suffix to apply to overlapping column names (right GeoDataFrame).

        Examples
        --------
        >>> countries = geopandas.read_file( \
    geopandas.datasets.get_path("naturalearth_lowres"))
        >>> cities = geopandas.read_file( \
    geopandas.datasets.get_path("naturalearth_cities"))
        >>> countries.head()  # doctest: +SKIP
            pop_est      continent                      name \
    iso_a3  gdp_md_est                                           geometry
        0     920938        Oceania                      Fiji    FJI      8374.0 \
    MULTIPOLYGON (((180.00000 -16.06713, 180.00000...
        1   53950935         Africa                  Tanzania    TZA    150600.0 \
    POLYGON ((33.90371 -0.95000, 34.07262 -1.05982...
        2     603253         Africa                 W. Sahara    ESH       906.5 \
    POLYGON ((-8.66559 27.65643, -8.66512 27.58948...
        3   35623680  North America                    Canada    CAN   1674000.0 \
    MULTIPOLYGON (((-122.84000 49.00000, -122.9742...
        4  326625791  North America  United States of America    USA  18560000.0 \
    MULTIPOLYGON (((-122.84000 49.00000, -120.0000...
        >>> cities.head()
                name                   geometry
        0  Vatican City  POINT (12.45339 41.90328)
        1    San Marino  POINT (12.44177 43.93610)
        2         Vaduz   POINT (9.51667 47.13372)
        3    Luxembourg   POINT (6.13000 49.61166)
        4       Palikir  POINT (158.14997 6.91664)

        >>> cities_w_country_data = cities.sjoin(countries)
        >>> cities_w_country_data.head()  # doctest: +SKIP
                name_left                   geometry  index_right   pop_est \
    continent name_right iso_a3  gdp_md_est
        0    Vatican City  POINT (12.45339 41.90328)          141  62137802 \
    Europe    Italy    ITA   2221000.0
        1    San Marino  POINT (12.44177 43.93610)          141  62137802 \
    Europe    Italy    ITA   2221000.0
        192          Rome  POINT (12.48131 41.89790)          141  62137802 \
    Europe    Italy    ITA   2221000.0
        2           Vaduz   POINT (9.51667 47.13372)          114   8754413 \
    Europe    Au    stria    AUT    416600.0
        184        Vienna  POINT (16.36469 48.20196)          114   8754413 \
    Europe    Austria    AUT    416600.0

        Notes
        ------
        Every operation in GeoPandas is planar, i.e. the potential third
        dimension is not taken into account.

        See also
        --------
        GeoDataFrame.sjoin_nearest : nearest neighbor join
        sjoin : equivalent top-level function
        """
        return geopandas.sjoin(left_df=self, right_df=df, *args, **kwargs)

    def sjoin_nearest(
        self,
        right,
        how="inner",
        max_distance=None,
        lsuffix="left",
        rsuffix="right",
        distance_col=None,
    ):
        """
        Spatial join of two GeoDataFrames based on the distance between their
        geometries.

        Results will include multiple output records for a single input record
        where there are multiple equidistant nearest or intersected neighbors.

        See the User Guide page
        https://geopandas.readthedocs.io/en/latest/docs/user_guide/mergingdata.html
        for more details.


        Parameters
        ----------
        right : GeoDataFrame
        how : string, default 'inner'
            The type of join:

            * 'left': use keys from left_df; retain only left_df geometry column
            * 'right': use keys from right_df; retain only right_df geometry column
            * 'inner': use intersection of keys from both dfs; retain only
              left_df geometry column

        max_distance : float, default None
            Maximum distance within which to query for nearest geometry.
            Must be greater than 0.
            The max_distance used to search for nearest items in the tree may have a
            significant impact on performance by reducing the number of input
            geometries that are evaluated for nearest items in the tree.
        lsuffix : string, default 'left'
            Suffix to apply to overlapping column names (left GeoDataFrame).
        rsuffix : string, default 'right'
            Suffix to apply to overlapping column names (right GeoDataFrame).
        distance_col : string, default None
            If set, save the distances computed between matching geometries under a
            column of this name in the joined GeoDataFrame.

        Examples
        --------
        >>> countries = geopandas.read_file(geopandas.datasets.get_\
path("naturalearth_lowres"))
        >>> cities = geopandas.read_file(geopandas.datasets.get_path("naturalearth_citi\
es"))
        >>> countries.head(2).name  # doctest: +SKIP
            pop_est      continent                      name \
    iso_a3  gdp_md_est                                           geometry
        0     920938        Oceania                      Fiji    FJI      8374.0  MULTI\
    POLYGON (((180.00000 -16.06713, 180.00000...
        1   53950935         Africa                  Tanzania    TZA    150600.0  POLYG\
    ON ((33.90371 -0.95000, 34.07262 -1.05982...
        >>> cities.head(2).name  # doctest: +SKIP
                name                   geometry
        0  Vatican City  POINT (12.45339 41.90328)
        1    San Marino  POINT (12.44177 43.93610)

        >>> cities_w_country_data = cities.sjoin_nearest(countries)
        >>> cities_w_country_data[['name_left', 'name_right']].head(2)  # doctest: +SKIP
                name_left                   geometry  index_right   pop_est continent n\
    ame_right iso_a3  gdp_md_est
        0    Vatican City  POINT (12.45339 41.90328)          141  62137802    Europe  \
        Italy    ITA   2221000.0
        1      San Marino  POINT (12.44177 43.93610)          141  62137802    Europe  \
        Italy    ITA   2221000.0

        To include the distances:

        >>> cities_w_country_data = cities.sjoin_nearest(countries, \
distance_col="distances")
        >>> cities_w_country_data[["name_left", "name_right", \
"distances"]].head(2)  # doctest: +SKIP
                name_left name_right distances
        0    Vatican City      Italy       0.0
        1      San Marino      Italy       0.0

        In the following example, we get multiple cities for Italy because all results
        are equidistant (in this case zero because they intersect).
        In fact, we get 3 results in total:

        >>> countries_w_city_data = cities.sjoin_nearest(countries, \
distance_col="distances", how="right")
        >>> italy_results = \
countries_w_city_data[countries_w_city_data["name_left"] == "Italy"]
        >>> italy_results  # doctest: +SKIP
            name_x        name_y
        141  Vatican City  Italy
        141    San Marino  Italy
        141          Rome  Italy

        See also
        --------
        GeoDataFrame.sjoin : binary predicate joins
        sjoin_nearest : equivalent top-level function

        Notes
        -----
        Since this join relies on distances, results will be innaccurate
        if your geometries are in a geographic CRS.

        Every operation in GeoPandas is planar, i.e. the potential third
        dimension is not taken into account.
        """
        return geopandas.sjoin_nearest(
            self,
            right,
            how=how,
            max_distance=max_distance,
            lsuffix=lsuffix,
            rsuffix=rsuffix,
            distance_col=distance_col,
        )

    def clip(self, mask, keep_geom_type=False):
        """Clip points, lines, or polygon geometries to the mask extent.

        Both layers must be in the same Coordinate Reference System (CRS).
        The GeoDataFrame will be clipped to the full extent of the `mask` object.

        If there are multiple polygons in mask, data from the GeoDataFrame will be
        clipped to the total boundary of all polygons in mask.

        Parameters
        ----------
        mask : GeoDataFrame, GeoSeries, (Multi)Polygon
            Polygon vector layer used to clip `gdf`.
            The mask's geometry is dissolved into one geometric feature
            and intersected with `gdf`.
        keep_geom_type : boolean, default False
            If True, return only geometries of original type in case of intersection
            resulting in multiple geometry types or GeometryCollections.
            If False, return all resulting geometries (potentially mixed types).

        Returns
        -------
        GeoDataFrame
            Vector data (points, lines, polygons) from `gdf` clipped to
            polygon boundary from mask.

        See also
        --------
        clip : equivalent top-level function

        Examples
        --------
        Clip points (global cities) with a polygon (the South American continent):

        >>> world = geopandas.read_file(
        ...     geopandas.datasets.get_path('naturalearth_lowres'))
        >>> south_america = world[world['continent'] == "South America"]
        >>> capitals = geopandas.read_file(
        ...     geopandas.datasets.get_path('naturalearth_cities'))
        >>> capitals.shape
        (202, 2)

        >>> sa_capitals = capitals.clip(south_america)
        >>> sa_capitals.shape
        (12, 2)
        """
        return geopandas.clip(self, mask=mask, keep_geom_type=keep_geom_type)

    def overlay(self, right, how="intersection", keep_geom_type=None, make_valid=True):
        """Perform spatial overlay between GeoDataFrames.

        Currently only supports data GeoDataFrames with uniform geometry types,
        i.e. containing only (Multi)Polygons, or only (Multi)Points, or a
        combination of (Multi)LineString and LinearRing shapes.
        Implements several methods that are all effectively subsets of the union.

        See the User Guide page :doc:`../../user_guide/set_operations` for details.

        Parameters
        ----------
        right : GeoDataFrame
        how : string
            Method of spatial overlay: 'intersection', 'union',
            'identity', 'symmetric_difference' or 'difference'.
        keep_geom_type : bool
            If True, return only geometries of the same geometry type the GeoDataFrame
            has, if False, return all resulting geometries. Default is None,
            which will set keep_geom_type to True but warn upon dropping
            geometries.
        make_valid : bool, default True
            If True, any invalid input geometries are corrected with a call to
            `buffer(0)`, if False, a `ValueError` is raised if any input geometries
            are invalid.

        Returns
        -------
        df : GeoDataFrame
            GeoDataFrame with new set of polygons and attributes
            resulting from the overlay

        Examples
        --------
        >>> from shapely.geometry import Polygon
        >>> polys1 = geopandas.GeoSeries([Polygon([(0,0), (2,0), (2,2), (0,2)]),
        ...                               Polygon([(2,2), (4,2), (4,4), (2,4)])])
        >>> polys2 = geopandas.GeoSeries([Polygon([(1,1), (3,1), (3,3), (1,3)]),
        ...                               Polygon([(3,3), (5,3), (5,5), (3,5)])])
        >>> df1 = geopandas.GeoDataFrame({'geometry': polys1, 'df1_data':[1,2]})
        >>> df2 = geopandas.GeoDataFrame({'geometry': polys2, 'df2_data':[1,2]})

        >>> df1.overlay(df2, how='union')
        df1_data  df2_data                                           geometry
        0       1.0       1.0  POLYGON ((2.00000 2.00000, 2.00000 1.00000, 1....
        1       2.0       1.0  POLYGON ((2.00000 2.00000, 2.00000 3.00000, 3....
        2       2.0       2.0  POLYGON ((4.00000 4.00000, 4.00000 3.00000, 3....
        3       1.0       NaN  POLYGON ((2.00000 0.00000, 0.00000 0.00000, 0....
        4       2.0       NaN  MULTIPOLYGON (((3.00000 3.00000, 4.00000 3.000...
        5       NaN       1.0  MULTIPOLYGON (((2.00000 2.00000, 3.00000 2.000...
        6       NaN       2.0  POLYGON ((3.00000 5.00000, 5.00000 5.00000, 5....

        >>> df1.overlay(df2, how='intersection')
        df1_data  df2_data                                           geometry
        0         1         1  POLYGON ((2.00000 2.00000, 2.00000 1.00000, 1....
        1         2         1  POLYGON ((2.00000 2.00000, 2.00000 3.00000, 3....
        2         2         2  POLYGON ((4.00000 4.00000, 4.00000 3.00000, 3....

        >>> df1.overlay(df2, how='symmetric_difference')
        df1_data  df2_data                                           geometry
        0       1.0       NaN  POLYGON ((2.00000 0.00000, 0.00000 0.00000, 0....
        1       2.0       NaN  MULTIPOLYGON (((3.00000 3.00000, 4.00000 3.000...
        2       NaN       1.0  MULTIPOLYGON (((2.00000 2.00000, 3.00000 2.000...
        3       NaN       2.0  POLYGON ((3.00000 5.00000, 5.00000 5.00000, 5....

        >>> df1.overlay(df2, how='difference')
                                                geometry  df1_data
        0  POLYGON ((2.00000 0.00000, 0.00000 0.00000, 0....         1
        1  MULTIPOLYGON (((3.00000 3.00000, 4.00000 3.000...         2

        >>> df1.overlay(df2, how='identity')
        df1_data  df2_data                                           geometry
        0       1.0       1.0  POLYGON ((2.00000 2.00000, 2.00000 1.00000, 1....
        1       2.0       1.0  POLYGON ((2.00000 2.00000, 2.00000 3.00000, 3....
        2       2.0       2.0  POLYGON ((4.00000 4.00000, 4.00000 3.00000, 3....
        3       1.0       NaN  POLYGON ((2.00000 0.00000, 0.00000 0.00000, 0....
        4       2.0       NaN  MULTIPOLYGON (((3.00000 3.00000, 4.00000 3.000...

        See also
        --------
        GeoDataFrame.sjoin : spatial join
        overlay : equivalent top-level function

        Notes
        ------
        Every operation in GeoPandas is planar, i.e. the potential third
        dimension is not taken into account.
        """
        return geopandas.overlay(
            self, right, how=how, keep_geom_type=keep_geom_type, make_valid=make_valid
        )


def _dataframe_set_geometry(self, col, drop=False, inplace=False, crs=None):
    if inplace:
        raise ValueError(
            "Can't do inplace setting when converting from DataFrame to GeoDataFrame"
        )
    gf = GeoDataFrame(self)
    # this will copy so that BlockManager gets copied
    return gf.set_geometry(col, drop=drop, inplace=False, crs=crs)


DataFrame.set_geometry = _dataframe_set_geometry
