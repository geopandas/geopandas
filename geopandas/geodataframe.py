import json
import warnings

import numpy as np
import pandas as pd
import shapely.errors
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


def _geodataframe_constructor_with_fallback(*args, **kwargs):
    """
    A flexible constructor for GeoDataFrame._constructor, which falls back
    to returning a DataFrame (if a certain operation does not preserve the
    geometry column)
    """
    df = GeoDataFrame(*args, **kwargs)
    geometry_cols_mask = df.dtypes == "geometry"
    if len(geometry_cols_mask) == 0 or geometry_cols_mask.sum() == 0:
        df = pd.DataFrame(df)

    return df


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
        if data.crs is None and crs is not None:
            # Avoids caching issues/crs sharing issues
            data = data.copy()
            data.crs = crs
        return data
    else:
        if isinstance(data, Series):
            out = from_shapely(np.asarray(data), crs=crs)
            return GeoSeries(out, index=data.index, name=data.name)
        else:
            out = from_shapely(data, crs=crs)
            return out


crs_mismatch_error = (
    "CRS mismatch between CRS of the passed geometries "
    "and 'crs'. Use 'GeoDataFrame.set_crs(crs, "
    "allow_override=True)' to overwrite CRS or "
    "'GeoDataFrame.to_crs(crs)' to reproject geometries. "
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

    _metadata = ["_geometry_column_name"]

    _internal_names = DataFrame._internal_names + ["geometry"]
    _internal_names_set = set(_internal_names)

    _geometry_column_name = None

    def __init__(self, data=None, *args, geometry=None, crs=None, **kwargs):
        with compat.ignore_shapely2_warnings():
            if (
                kwargs.get("copy") is None
                and isinstance(data, DataFrame)
                and not isinstance(data, GeoDataFrame)
            ):
                kwargs.update(copy=True)
            super().__init__(data, *args, **kwargs)

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
                raise ValueError(crs_mismatch_error)

        if (
            geometry is None
            and self.columns.nlevels == 1
            and "geometry" in self.columns
        ):
            # Check for multiple columns with name "geometry". If there are,
            # self["geometry"] is a gdf and constructor gets recursively recalled
            # by pandas internals trying to access this
            if (self.columns == "geometry").sum() > 1:
                raise ValueError(
                    "GeoDataFrame does not support multiple columns "
                    "using the geometry column name 'geometry'."
                )

            # only if we have actual geometry values -> call set_geometry
            try:
                if (
                    hasattr(self["geometry"].values, "crs")
                    and self["geometry"].values.crs
                    and crs
                    and not self["geometry"].values.crs == crs
                ):
                    raise ValueError(crs_mismatch_error)
                self["geometry"] = _ensure_geometry(self["geometry"].values, crs)
            except TypeError:
                pass
            else:
                geometry = "geometry"

        if geometry is not None:
            if (
                hasattr(geometry, "crs")
                and geometry.crs
                and crs
                and not geometry.crs == crs
            ):
                raise ValueError(crs_mismatch_error)

            self.set_geometry(geometry, inplace=True, crs=crs)

        if geometry is None and crs:
            raise ValueError(
                "Assigning CRS to a GeoDataFrame without a geometry column is not "
                "supported. Supply geometry using the 'geometry=' keyword argument, "
                "or by providing a DataFrame with column name 'geometry'",
            )

    def __setattr__(self, attr, val):
        # have to special case geometry b/c pandas tries to use as column...
        if attr == "geometry":
            object.__setattr__(self, attr, val)
        else:
            super().__setattr__(attr, val)

    def _get_geometry(self):
        if self._geometry_column_name not in self:
            if self._geometry_column_name is None:
                msg = (
                    "You are calling a geospatial method on the GeoDataFrame, "
                    "but the active geometry column to use has not been set. "
                )
            else:
                msg = (
                    "You are calling a geospatial method on the GeoDataFrame, "
                    f"but the active geometry column ('{self._geometry_column_name}') "
                    "is not present. "
                )
            geo_cols = list(self.columns[self.dtypes == "geometry"])
            if len(geo_cols) > 0:
                msg += (
                    f"\nThere are columns with geometry data type ({geo_cols}), and "
                    "you can either set one as the active geometry with "
                    'df.set_geometry("name") or access the column as a '
                    'GeoSeries (df["name"]) and call the method directly on it.'
                )
            else:
                msg += (
                    "\nThere are no existing columns with geometry data type. You can "
                    "add a geometry column as the active geometry column with "
                    "df.set_geometry. "
                )

            raise AttributeError(msg)
        return self[self._geometry_column_name]

    def _set_geometry(self, col):
        if not pd.api.types.is_list_like(col):
            raise ValueError("Must use a list-like to set the geometry property")
        self._persist_old_default_geometry_colname()
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
        if geo_column_name is None:
            geo_column_name = "geometry"
        if isinstance(col, (Series, list, np.ndarray, GeometryArray)):
            level = col
        elif hasattr(col, "ndim") and col.ndim > 1:
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
            else:
                geo_column_name = col

        if to_remove:
            del frame[to_remove]

        if not crs:
            crs = getattr(level, "crs", None)

        if isinstance(level, (GeoSeries, GeometryArray)) and level.crs != crs:
            # Avoids caching issues/crs sharing issues
            level = level.copy()
            level.crs = crs

        # Check that we are using a listlike of geometries
        level = _ensure_geometry(level, crs=crs)
        # update _geometry_column_name prior to assignment
        # to avoid default is None warning
        frame._geometry_column_name = geo_column_name
        frame[geo_column_name] = level

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
        try:
            return self.geometry.crs
        except AttributeError:
            raise AttributeError(
                "The CRS attribute of a GeoDataFrame without an active "
                "geometry column is not defined. Use GeoDataFrame.set_geometry "
                "to set the active geometry column."
            )

    @crs.setter
    def crs(self, value):
        """Sets the value of the crs"""
        if self._geometry_column_name is None:
            raise ValueError(
                "Assigning CRS to a GeoDataFrame without a geometry column is not "
                "supported. Use GeoDataFrame.set_geometry to set the active "
                "geometry column.",
            )

        if hasattr(self.geometry.values, "crs"):
            self.geometry.values.crs = value
        else:
            # column called 'geometry' without geometry
            raise ValueError(
                "Assigning CRS to a GeoDataFrame without an active geometry "
                "column is not supported. Use GeoDataFrame.set_geometry to set "
                "the active geometry column.",
            )

    def __setstate__(self, state):
        # overriding DataFrame method for compat with older pickles (CRS handling)
        crs = None
        if isinstance(state, dict):
            if "crs" in state and "_crs" not in state:
                crs = state.pop("crs", None)
            else:
                crs = state.pop("_crs", None)
            crs = CRS.from_user_input(crs) if crs is not None else crs

        super().__setstate__(state)

        # for some versions that didn't yet have CRS at array level -> crs is set
        # at GeoDataFrame level with '_crs' (and not 'crs'), so without propagating
        # to the GeoSeries/GeometryArray
        try:
            if crs is not None:
                if self.geometry.values.crs is None:
                    self.crs = crs
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
        dataframe = DataFrame.from_dict(data, **kwargs)
        return cls(dataframe, geometry=geometry, crs=crs)

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
        >>> import geodatasets
        >>> path = geodatasets.get_path('nybb')
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
            properties = feature["properties"]
            if properties is None:
                properties = {}
            row.update(properties)
            rows.append(row)
        return cls(rows, columns=columns, crs=crs)

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

    def to_json(
        self, na="null", show_bbox=False, drop_id=False, to_wgs84=False, **kwargs
    ):
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
        to_wgs84: bool, optional, default: False
            If the CRS is set on the active geometry column it is exported as
            WGS84 (EPSG:4326) to meet the `2016 GeoJSON specification
            <https://tools.ietf.org/html/rfc7946>`_.
            Set to True to force re-projection and set to False to ignore CRS. False by
            default.

        Notes
        -----
        The remaining *kwargs* are passed to json.dumps().

        Missing (NaN) values in the GeoDataFrame can be represented as follows:

        - ``null``: output the missing entries as JSON null.
        - ``drop``: remove the property from the feature. This applies to each
          feature individually so that features may have different properties.
        - ``keep``: output the missing entries as NaN.

        If the GeoDataFrame has a defined CRS, its definition will be included
        in the output unless it is equal to WGS84 (default GeoJSON CRS) or not
        possible to represent in the URN OGC format, or unless ``to_wgs84=True``
        is specified.

        Examples
        --------

        >>> from shapely.geometry import Point
        >>> d = {'col1': ['name1', 'name2'], 'geometry': [Point(1, 2), Point(2, 1)]}
        >>> gdf = geopandas.GeoDataFrame(d, crs="EPSG:3857")
        >>> gdf
            col1             geometry
        0  name1  POINT (1.000 2.000)
        1  name2  POINT (2.000 1.000)

        >>> gdf.to_json()
        '{"type": "FeatureCollection", "features": [{"id": "0", "type": "Feature", \
"properties": {"col1": "name1"}, "geometry": {"type": "Point", "coordinates": [1.0,\
 2.0]}}, {"id": "1", "type": "Feature", "properties": {"col1": "name2"}, "geometry"\
: {"type": "Point", "coordinates": [2.0, 1.0]}}], "crs": {"type": "name", "properti\
es": {"name": "urn:ogc:def:crs:EPSG::3857"}}}'

        Alternatively, you can write GeoJSON to file:

        >>> gdf.to_file(path, driver="GeoJSON")  # doctest: +SKIP

        See also
        --------
        GeoDataFrame.to_file : write GeoDataFrame to file

        """
        if to_wgs84:
            if self.crs:
                df = self.to_crs(epsg=4326)
            else:
                raise ValueError(
                    "CRS is not set. Cannot re-project to WGS84 (EPSG:4326)."
                )
        else:
            df = self

        geo = df._to_geo(na=na, show_bbox=show_bbox, drop_id=drop_id)

        # if the geometry is not in WGS84, include CRS in the JSON
        if df.crs is not None and not df.crs.equals("epsg:4326"):
            auth_crsdef = self.crs.to_authority()
            allowed_authorities = ["EDCS", "EPSG", "OGC", "SI", "UCUM"]

            if auth_crsdef is None or auth_crsdef[0] not in allowed_authorities:
                warnings.warn(
                    "GeoDataFrame's CRS is not representable in URN OGC "
                    "format. Resulting JSON will contain no CRS information.",
                    stacklevel=2,
                )
            else:
                authority, code = auth_crsdef
                ogc_crs = f"urn:ogc:def:crs:{authority}::{code}"
                geo["crs"] = {"type": "name", "properties": {"name": ogc_crs}}

        return json.dumps(geo, **kwargs)

    @property
    def __geo_interface__(self):
        """Returns a ``GeoDataFrame`` as a python feature collection.

        Implements the `geo_interface`. The returned python data structure
        represents the ``GeoDataFrame`` as a GeoJSON-like
        ``FeatureCollection``.

        This differs from `_to_geo()` only in that it is a property with
        default args instead of a method.

        CRS of the dataframe is not passed on to the output, unlike
        :meth:`~GeoDataFrame.to_json()`.

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

        properties_cols = self.columns.drop(self._geometry_column_name)

        if len(properties_cols) > 0:
            # convert to object to get python scalars.
            properties_cols = self[properties_cols]
            properties = properties_cols.astype(object)
            na_mask = pd.isna(properties_cols).values

            if na == "null":
                properties[na_mask] = None

            for i, row in enumerate(properties.values):
                geom = geometries[i]

                if na == "drop":
                    na_mask_row = na_mask[i]
                    properties_items = {
                        k: v
                        for k, v, na in zip(properties_cols, row, na_mask_row)
                        if not na
                    }
                else:
                    properties_items = dict(zip(properties_cols, row))

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
            :func:`shapely.to_wkb` if shapely >= 2 is installed or
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

    def to_parquet(
        self, path, index=None, compression="snappy", schema_version=None, **kwargs
    ):
        """Write a GeoDataFrame to the Parquet format.

        Any geometry columns present are serialized to WKB format in the file.

        Requires 'pyarrow'.

        WARNING: this is an early implementation of Parquet file support and
        associated metadata, the specification for which continues to evolve.
        This is tracking version 0.4.0 of the GeoParquet specification at:
        https://github.com/opengeospatial/geoparquet

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
        schema_version : {'0.1.0', '0.4.0', None}
            GeoParquet specification version; if not provided will default to
            latest supported version.
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

        # Accept engine keyword for compatibility with pandas.DataFrame.to_parquet
        # The only engine currently supported by GeoPandas is pyarrow, so no
        # other engine should be specified.
        engine = kwargs.pop("engine", "auto")
        if engine not in ("auto", "pyarrow"):
            raise ValueError(
                f"GeoPandas only supports using pyarrow as the engine for "
                f"to_parquet: {engine!r} passed instead."
            )

        from geopandas.io.arrow import _to_parquet

        _to_parquet(
            self,
            path,
            compression=compression,
            index=index,
            schema_version=schema_version,
            **kwargs,
        )

    def to_feather(
        self, path, index=None, compression=None, schema_version=None, **kwargs
    ):
        """Write a GeoDataFrame to the Feather format.

        Any geometry columns present are serialized to WKB format in the file.

        Requires 'pyarrow' >= 0.17.

        WARNING: this is an early implementation of Parquet file support and
        associated metadata, the specification for which continues to evolve.
        This is tracking version 0.4.0 of the GeoParquet specification at:
        https://github.com/opengeospatial/geoparquet

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
        schema_version : {'0.1.0', '0.4.0', None}
            GeoParquet specification version; if not provided will default to
            latest supported version.
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

        _to_feather(
            self,
            path,
            index=index,
            compression=compression,
            schema_version=schema_version,
            **kwargs,
        )

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
            File path or file handle to write to. The path may specify a
            GDAL VSI scheme.
        driver : string, default None
            The OGR format driver used to write the vector file.
            If not specified, it attempts to infer it from the file extension.
            If no extension is specified, it saves ESRI Shapefile to a folder.
        schema : dict, default None
            If specified, the schema dictionary is passed to Fiona to
            better control how the file is written. If None, GeoPandas
            will determine the schema based on each column's dtype.
            Not supported for the "pyogrio" engine.
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

        Notes
        -----
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

        Using the engine-specific keyword arguments it is possible to e.g. create a
        spatialite file with a custom layer name:

        >>> gdf.to_file(
        ...     'dataframe.sqlite', driver='SQLite', spatialite=True, layer='test'
        ... )  # doctest: +SKIP

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
        of projecting entire geometries.  All segments joining points are
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
        if not inplace:
            return df

    def estimate_utm_crs(self, datum_name="WGS 84"):
        """Returns the estimated UTM CRS based on the bounds of the dataset.

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
        >>> df.estimate_utm_crs()  # doctest: +SKIP
        <Derived Projected CRS: EPSG:32616>
        Name: WGS 84 / UTM zone 16N
        Axis Info [cartesian]:
        - E[east]: Easting (metre)
        - N[north]: Northing (metre)
        Area of Use:
        - name: Between 90°W and 84°W, northern hemisphere between equator and 84°N...
        - bounds: (-90.0, 0.0, -84.0, 84.0)
        Coordinate Operation:
        - name: UTM zone 16N
        - method: Transverse Mercator
        Datum: World Geodetic System 1984 ensemble
        - Ellipsoid: WGS 84
        - Prime Meridian: Greenwich
        """
        return self.geometry.estimate_utm_crs(datum_name=datum_name)

    def __getitem__(self, key):
        """
        If the result is a column containing only 'geometry', return a
        GeoSeries. If it's a DataFrame with any columns of GeometryDtype,
        return a GeoDataFrame.
        """
        result = super().__getitem__(key)
        # Custom logic to avoid waiting for pandas GH51895
        # result is not geometry dtype for multi-indexes
        if (
            pd.api.types.is_scalar(key)
            and key == ""
            and isinstance(self.columns, pd.MultiIndex)
            and isinstance(result, Series)
            and not is_geometry_type(result)
        ):
            loc = self.columns.get_loc(key)
            # squeeze stops multilevel columns from returning a gdf
            result = self.iloc[:, loc].squeeze(axis="columns")
        geo_col = self._geometry_column_name
        if isinstance(result, Series) and isinstance(result.dtype, GeometryDtype):
            result.__class__ = GeoSeries
        elif isinstance(result, DataFrame):
            if (result.dtypes == "geometry").sum() > 0:
                result.__class__ = GeoDataFrame
                if geo_col in result:
                    result._geometry_column_name = geo_col
            else:
                result.__class__ = DataFrame
        return result

    def _persist_old_default_geometry_colname(self):
        """Internal util to temporarily persist the default geometry column
        name of 'geometry' for backwards compatibility."""
        # self.columns check required to avoid this warning in __init__
        if self._geometry_column_name is None and "geometry" not in self.columns:
            msg = (
                "You are adding a column named 'geometry' to a GeoDataFrame "
                "constructed without an active geometry column. Currently, "
                "this automatically sets the active geometry column to 'geometry' "
                "but in the future that will no longer happen. Instead, either "
                "provide geometry to the GeoDataFrame constructor "
                "(GeoDataFrame(... geometry=GeoSeries()) or use "
                "`set_geometry('geometry')` "
                "to explicitly set the active geometry column."
            )
            warnings.warn(msg, category=FutureWarning, stacklevel=3)
            self._geometry_column_name = "geometry"

    def __setitem__(self, key, value):
        """
        Overwritten to preserve CRS of GeometryArray in cases like
        df['geometry'] = [geom... for geom in df.geometry]
        """

        if not pd.api.types.is_list_like(key) and (
            key == self._geometry_column_name
            or key == "geometry"
            and self._geometry_column_name is None
        ):
            if pd.api.types.is_scalar(value) or isinstance(value, BaseGeometry):
                value = [value] * self.shape[0]
            try:
                if self._geometry_column_name is not None:
                    crs = getattr(self, "crs", None)
                else:  # don't use getattr, because a col "crs" might exist
                    crs = None
                value = _ensure_geometry(value, crs=crs)
                if key == "geometry":
                    self._persist_old_default_geometry_colname()
            except TypeError:
                warnings.warn(
                    "Geometry column does not contain geometry.",
                    stacklevel=2,
                )
        super().__setitem__(key, value)

    #
    # Implement pandas methods
    #
    @doc(pd.DataFrame)
    def copy(self, deep=True):
        copied = super().copy(deep=deep)
        if type(copied) is pd.DataFrame:
            copied.__class__ = GeoDataFrame
            copied._geometry_column_name = self._geometry_column_name
        return copied

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
        See https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas\
.DataFrame.merge.html
        for more details.
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
        # pandas <1.4 re-attach last geometry col if lost
        if (
            not compat.PANDAS_GE_14
            and isinstance(result, GeoDataFrame)
            and result._geometry_column_name is None
        ):
            result._geometry_column_name = self._geometry_column_name
        # Reconstruct gdf if it was lost by apply
        if (
            isinstance(result, DataFrame)
            and self._geometry_column_name in result.columns
        ):
            # axis=1 apply will split GeometryDType to object, try and cast back
            try:
                result = result.set_geometry(self._geometry_column_name)
            except TypeError:
                pass
            else:
                if self.crs is not None and result.crs is None:
                    result.set_crs(self.crs, inplace=True)
        elif isinstance(result, Series) and result.dtype == "object":
            # Try reconstruct series GeometryDtype if lost by apply
            # If all none and object dtype assert list of nones is more likely
            # intended than list of null geometry.
            if not result.isna().all():
                try:
                    # not enough info about func to preserve CRS
                    result = _ensure_geometry(result)

                except (TypeError, shapely.errors.GeometryTypeError):
                    pass

        return result

    @property
    def _constructor(self):
        return _geodataframe_constructor_with_fallback

    @property
    def _constructor_sliced(self):
        def _geodataframe_constructor_sliced(*args, **kwargs):
            """
            A specialized (Geo)Series constructor which can fall back to a
            Series if a certain operation does not produce geometries:

            - We only return a GeoSeries if the data is actually of geometry
              dtype (and so we don't try to convert geometry objects such as
              the normal GeoSeries(..) constructor does with `_ensure_geometry`).
            - When we get here from obtaining a row or column from a
              GeoDataFrame, the goal is to only return a GeoSeries for a
              geometry column, and not return a GeoSeries for a row that happened
              to come from a DataFrame with only geometry dtype columns (and
              thus could have a geometry dtype). Therefore, we don't return a
              GeoSeries if we are sure we are in a row selection case (by
              checking the identity of the index)
            """
            srs = pd.Series(*args, **kwargs)
            is_row_proxy = srs.index is self.columns
            if is_geometry_type(srs) and not is_row_proxy:
                srs = GeoSeries(srs)
            return srs

        return _geodataframe_constructor_sliced

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
        elif method == "unstack":
            # unstack adds multiindex columns and reshapes data.
            # it never makes sense to retain geometry column
            self._geometry_column_name = None
            self._crs = None
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
        **kwargs,
    ):
        """
        Dissolve geometries within `groupby` into single observation.
        This is accomplished by applying the `unary_union` method
        to all geometries within a groupself.

        Observations associated with each `groupby` group will be aggregated
        using the `aggfunc`.

        Parameters
        ----------
        by : str or list-like, default None
            Column(s) whose values define the groups to be dissolved. If None,
            the entire GeoDataFrame is considered as a single group. If a list-like
            object is provided, the values in the list are treated as categorical
            labels, and polygons will be combined based on the equality of
            these categorical labels.
        aggfunc : function or string, default "first"
            Aggregation function for manipulation of data associated
            with each group. Passed to pandas `groupby.agg` method.
            Accepted combinations are:

            - function
            - string function name
            - list of functions and/or function names, e.g. [np.sum, 'mean']
            - dict of axis labels -> functions, function names or list of such.
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

            .. versionadded:: 0.9.0
        **kwargs :
            Keyword arguments to be passed to the pandas `DataFrameGroupby.agg` method
            which is used by `dissolve`. In particular, `numeric_only` may be
            supplied, which will be required in pandas 2.0 for certain aggfuncs.

            .. versionadded:: 0.13.0
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
        GeoDataFrame.explode : explode multi-part geometries into single geometries

        """

        if by is None and level is None:
            by = np.zeros(len(self), dtype="int64")

        groupby_kwargs = {
            "by": by,
            "level": level,
            "sort": sort,
            "observed": observed,
            "dropna": dropna,
        }

        # Process non-spatial component
        data = self.drop(labels=self.geometry.name, axis=1)
        with warnings.catch_warnings(record=True) as record:
            aggregated_data = data.groupby(**groupby_kwargs).agg(aggfunc, **kwargs)
        for w in record:
            if str(w.message).startswith("The default value of numeric_only"):
                msg = (
                    f"The default value of numeric_only in aggfunc='{aggfunc}' "
                    "within pandas.DataFrameGroupBy.agg used in dissolve is "
                    "deprecated. In pandas 2.0, numeric_only will default to False. "
                    "Either specify numeric_only as additional argument in dissolve() "
                    "or select only columns which should be valid for the function."
                )
                warnings.warn(msg, FutureWarning, stacklevel=2)
            else:
                # Only want to capture specific warning,
                # other warnings from pandas should be passed through
                # TODO this is not an ideal approach
                warnings.showwarning(
                    w.message, w.category, w.filename, w.lineno, w.file, w.line
                )

        aggregated_data.columns = aggregated_data.columns.to_flat_index()

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
        Explode multi-part geometries into multiple single geometries.

        Each row containing a multi-part geometry will be split into
        multiple rows with single geometries, thereby increasing the vertical
        size of the GeoDataFrame.

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
            return super().explode(column, ignore_index=ignore_index, **kwargs)

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

        exploded_geom = self.geometry.reset_index(drop=True).explode(index_parts=True)

        df = self.drop(self._geometry_column_name, axis=1).take(
            exploded_geom.index.droplevel(-1)
        )
        df[exploded_geom.name] = exploded_geom.values
        df = df.set_geometry(self._geometry_column_name).__finalize__(self)

        if ignore_index:
            df.reset_index(inplace=True, drop=True)
        elif index_parts:
            # reset to MultiIndex, otherwise df index is only first level of
            # exploded GeoSeries index.
            df = df.set_index(
                exploded_geom.index.droplevel(
                    list(range(exploded_geom.index.nlevels - 1))
                ),
                append=True,
            )

        return df

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

        It is also possible to use :meth:`~GeoDataFrame.to_file` to write to a database.
        Especially for file geodatabases like GeoPackage or SpatiaLite this can be
        easier.

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
        index : bool, default False
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
            FutureWarning,
            stacklevel=2,
        )
        return self.geometry.symmetric_difference(other)

    def __or__(self, other):
        """Implement | operator as for builtin set type"""
        warnings.warn(
            "'|' operator will be deprecated. Use the 'union' method instead.",
            FutureWarning,
            stacklevel=2,
        )
        return self.geometry.union(other)

    def __and__(self, other):
        """Implement & operator as for builtin set type"""
        warnings.warn(
            "'&' operator will be deprecated. Use the 'intersection' method instead.",
            FutureWarning,
            stacklevel=2,
        )
        return self.geometry.intersection(other)

    def __sub__(self, other):
        """Implement - operator as for builtin set type"""
        warnings.warn(
            "'-' operator will be deprecated. Use the 'difference' method instead.",
            FutureWarning,
            stacklevel=2,
        )
        return self.geometry.difference(other)

    plot = CachedAccessor("plot", geopandas.plotting.GeoplotAccessor)

    @doc(_explore)
    def explore(self, *args, **kwargs):
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
        >>> import geodatasets
        >>> chicago = geopandas.read_file(
        ...     geodatasets.get_path("geoda.chicago_commpop")
        ... )
        >>> groceries = geopandas.read_file(
        ...     geodatasets.get_path("geoda.groceries")
        ... ).to_crs(chicago.crs)

        >>> chicago.head()  # doctest: +SKIP
                 community  ...                                           geometry
        0          DOUGLAS  ...  MULTIPOLYGON (((-87.60914 41.84469, -87.60915 ...
        1          OAKLAND  ...  MULTIPOLYGON (((-87.59215 41.81693, -87.59231 ...
        2      FULLER PARK  ...  MULTIPOLYGON (((-87.62880 41.80189, -87.62879 ...
        3  GRAND BOULEVARD  ...  MULTIPOLYGON (((-87.60671 41.81681, -87.60670 ...
        4          KENWOOD  ...  MULTIPOLYGON (((-87.59215 41.81693, -87.59215 ...

        [5 rows x 9 columns]

        >>> groceries.head()  # doctest: +SKIP
           OBJECTID     Ycoord  ...  Category                         geometry
        0        16  41.973266  ...       NaN  MULTIPOINT (-87.65661 41.97321)
        1        18  41.696367  ...       NaN  MULTIPOINT (-87.68136 41.69713)
        2        22  41.868634  ...       NaN  MULTIPOINT (-87.63918 41.86847)
        3        23  41.877590  ...       new  MULTIPOINT (-87.65495 41.87783)
        4        27  41.737696  ...       NaN  MULTIPOINT (-87.62715 41.73623)
        [5 rows x 8 columns]

        >>> groceries_w_communities = groceries.sjoin(chicago)
        >>> groceries_w_communities[["OBJECTID", "community", "geometry"]].head()
             OBJECTID    community                         geometry
        0          16       UPTOWN  MULTIPOINT (-87.65661 41.97321)
        87        365       UPTOWN  MULTIPOINT (-87.65465 41.96138)
        90        373       UPTOWN  MULTIPOINT (-87.65598 41.96297)
        140       582       UPTOWN  MULTIPOINT (-87.67417 41.96977)
        1          18  MORGAN PARK  MULTIPOINT (-87.68136 41.69713)

        Notes
        -----
        Every operation in GeoPandas is planar, i.e. the potential third
        dimension is not taken into account.

        See also
        --------
        GeoDataFrame.sjoin_nearest : nearest neighbor join
        sjoin : equivalent top-level function
        """
        return geopandas.sjoin(left_df=self, right_df=df, *args, **kwargs)  # noqa: B026

    def sjoin_nearest(
        self,
        right,
        how="inner",
        max_distance=None,
        lsuffix="left",
        rsuffix="right",
        distance_col=None,
        exclusive=False,
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
        exclusive : bool, optional, default False
            If True, the nearest geometries that are equal to the input geometry
            will not be returned, default False.
            Requires Shapely >= 2.0

        Examples
        --------
        >>> import geodatasets
        >>> groceries = geopandas.read_file(
        ...     geodatasets.get_path("geoda.groceries")
        ... )
        >>> chicago = geopandas.read_file(
        ...     geodatasets.get_path("geoda.chicago_health")
        ... ).to_crs(groceries.crs)

        >>> chicago.head()  # doctest: +SKIP
            ComAreaID  ...                                           geometry
        0         35  ...  POLYGON ((-87.60914 41.84469, -87.60915 41.844...
        1         36  ...  POLYGON ((-87.59215 41.81693, -87.59231 41.816...
        2         37  ...  POLYGON ((-87.62880 41.80189, -87.62879 41.801...
        3         38  ...  POLYGON ((-87.60671 41.81681, -87.60670 41.816...
        4         39  ...  POLYGON ((-87.59215 41.81693, -87.59215 41.816...
        [5 rows x 87 columns]

        >>> groceries.head()  # doctest: +SKIP
            OBJECTID     Ycoord  ...  Category                         geometry
        0        16  41.973266  ...       NaN  MULTIPOINT (-87.65661 41.97321)
        1        18  41.696367  ...       NaN  MULTIPOINT (-87.68136 41.69713)
        2        22  41.868634  ...       NaN  MULTIPOINT (-87.63918 41.86847)
        3        23  41.877590  ...       new  MULTIPOINT (-87.65495 41.87783)
        4        27  41.737696  ...       NaN  MULTIPOINT (-87.62715 41.73623)
        [5 rows x 8 columns]

        >>> groceries_w_communities = groceries.sjoin_nearest(chicago)
        >>> groceries_w_communities[["Chain", "community", "geometry"]].head(2)
                     Chain community                              geometry
        0   VIET HOA PLAZA    UPTOWN  MULTIPOINT (1168268.672 1933554.350)
        87      JEWEL OSCO    UPTOWN  MULTIPOINT (1168837.980 1929246.962)


        To include the distances:

        >>> groceries_w_communities = groceries.sjoin_nearest(chicago, \
distance_col="distances")
        >>> groceries_w_communities[["Chain", "community", \
"distances"]].head(2)  # doctest: +SKIP
                     Chain community  distances
        0   VIET HOA PLAZA    UPTOWN        0.0
        87      JEWEL OSCO    UPTOWN        0.0

        In the following example, we get multiple groceries for Uptown because all
        results are equidistant (in this case zero because they intersect).
        In fact, we get 4 results in total:

        >>> chicago_w_groceries = groceries.sjoin_nearest(chicago, \
distance_col="distances", how="right")
        >>> uptown_results = \
chicago_w_groceries[chicago_w_groceries["community"] == "UPTOWN"]
        >>> uptown_results[["Chain", "community"]]  # doctest: +SKIP
                    Chain community
        30  VIET HOA PLAZA    UPTOWN
        30      JEWEL OSCO    UPTOWN
        30          TARGET    UPTOWN
        30       Mariano's    UPTOWN

        See also
        --------
        GeoDataFrame.sjoin : binary predicate joins
        sjoin_nearest : equivalent top-level function

        Notes
        -----
        Since this join relies on distances, results will be inaccurate
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
            exclusive=exclusive,
        )

    def clip(self, mask, keep_geom_type=False):
        """Clip points, lines, or polygon geometries to the mask extent.

        Both layers must be in the same Coordinate Reference System (CRS).
        The GeoDataFrame will be clipped to the full extent of the ``mask`` object.

        If there are multiple polygons in mask, data from the GeoDataFrame will be
        clipped to the total boundary of all polygons in mask.

        Parameters
        ----------
        mask : GeoDataFrame, GeoSeries, (Multi)Polygon, list-like
            Polygon vector layer used to clip the GeoDataFrame.
            The mask's geometry is dissolved into one geometric feature
            and intersected with GeoDataFrame.
            If the mask is list-like with four elements ``(minx, miny, maxx, maxy)``,
            ``clip`` will use a faster rectangle clipping
            (:meth:`~GeoSeries.clip_by_rect`), possibly leading to slightly different
            results.
        keep_geom_type : boolean, default False
            If True, return only geometries of original type in case of intersection
            resulting in multiple geometry types or GeometryCollections.
            If False, return all resulting geometries (potentially mixed types).

        Returns
        -------
        GeoDataFrame
            Vector data (points, lines, polygons) from the GeoDataFrame clipped to
            polygon boundary from mask.

        See also
        --------
        clip : equivalent top-level function

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

        >>> nws_groceries = groceries.clip(near_west_side)
        >>> nws_groceries.shape
        (7, 8)
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
        4       2.0       NaN  MULTIPOLYGON (((3.00000 4.00000, 3.00000 3.000...
        5       NaN       1.0  MULTIPOLYGON (((2.00000 3.00000, 2.00000 2.000...
        6       NaN       2.0  POLYGON ((3.00000 5.00000, 5.00000 5.00000, 5....

        >>> df1.overlay(df2, how='intersection')
        df1_data  df2_data                                           geometry
        0         1         1  POLYGON ((2.00000 2.00000, 2.00000 1.00000, 1....
        1         2         1  POLYGON ((2.00000 2.00000, 2.00000 3.00000, 3....
        2         2         2  POLYGON ((4.00000 4.00000, 4.00000 3.00000, 3....

        >>> df1.overlay(df2, how='symmetric_difference')
        df1_data  df2_data                                           geometry
        0       1.0       NaN  POLYGON ((2.00000 0.00000, 0.00000 0.00000, 0....
        1       2.0       NaN  MULTIPOLYGON (((3.00000 4.00000, 3.00000 3.000...
        2       NaN       1.0  MULTIPOLYGON (((2.00000 3.00000, 2.00000 2.000...
        3       NaN       2.0  POLYGON ((3.00000 5.00000, 5.00000 5.00000, 5....

        >>> df1.overlay(df2, how='difference')
                                                geometry  df1_data
        0  POLYGON ((2.00000 0.00000, 0.00000 0.00000, 0....         1
        1  MULTIPOLYGON (((3.00000 4.00000, 3.00000 3.000...         2

        >>> df1.overlay(df2, how='identity')
        df1_data  df2_data                                           geometry
        0       1.0       1.0  POLYGON ((2.00000 2.00000, 2.00000 1.00000, 1....
        1       2.0       1.0  POLYGON ((2.00000 2.00000, 2.00000 3.00000, 3....
        2       2.0       2.0  POLYGON ((4.00000 4.00000, 4.00000 3.00000, 3....
        3       1.0       NaN  POLYGON ((2.00000 0.00000, 0.00000 0.00000, 0....
        4       2.0       NaN  MULTIPOLYGON (((3.00000 4.00000, 3.00000 3.000...

        See also
        --------
        GeoDataFrame.sjoin : spatial join
        overlay : equivalent top-level function

        Notes
        -----
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
