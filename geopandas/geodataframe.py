import json
import warnings

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from shapely.geometry import mapping, shape
from shapely.geometry.base import BaseGeometry


from pyproj import CRS

from geopandas.array import GeometryArray, from_shapely, GeometryDtype
from geopandas.base import GeoPandasBase, is_geometry_type
from geopandas.geoseries import GeoSeries
import geopandas.io
from geopandas.plotting import plot_dataframe


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
            return GeoSeries(data)
        return data
    else:
        if isinstance(data, Series):
            out = from_shapely(np.asarray(data), crs=crs)
            return GeoSeries(out, index=data.index, name=data.name)
        else:
            out = from_shapely(data, crs=crs)
            return out


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
    >>> d = {'col1': ['name1', 'name2'], 'geometry': [Point(1,2), Point(2,1)]}
    >>> gdf = gpd.GeoDataFrame(d, crs="EPSG:4326")
    >>> gdf
        col1                 geometry
    0  name1  POINT (1.00000 2.00000)
    1  name2  POINT (2.00000 1.00000)

    Notice that the inferred dtype of 'geometry' columns is geometry.

    >>> gdf.dtypes
    col1          object
    geometry    geometry
    dtype: object
    """

    _metadata = ["_crs", "_geometry_column_name"]

    _geometry_column_name = DEFAULT_GEO_COLUMN_NAME

    def __init__(self, *args, **kwargs):
        crs = kwargs.pop("crs", None)
        geometry = kwargs.pop("geometry", None)
        super(GeoDataFrame, self).__init__(*args, **kwargs)

        # need to set this before calling self['geometry'], because
        # getitem accesses crs
        self._crs = crs if crs is not None else None

        # set_geometry ensures the geometry data have the proper dtype,
        # but is not called if `geometry=None` ('geometry' column present
        # in the data), so therefore need to ensure it here manually
        # but within a try/except because currently non-geometries are
        # allowed in that case
        # TODO do we want to raise / return normal DataFrame in this case?
        if geometry is None and "geometry" in self.columns:
            # only if we have actual geometry values -> call set_geometry
            index = self.index
            try:
                if (
                    hasattr(self["geometry"].values, "crs")
                    and self["geometry"].values.crs
                    and crs
                    and not self["geometry"].values.crs == crs
                ):
                    warnings.warn(
                        "CRS mismatch between CRS of the passed geometries "
                        "and 'crs'. Use 'GeoDataFrame.set_crs(crs, "
                        "allow_override=True)' to overwrite CRS or "
                        "'GeoDataFrame.to_crs(crs)' to reproject geometries. "
                        "CRS mismatch will raise an error in the future versions "
                        "of GeoPandas.",
                        FutureWarning,
                        stacklevel=2,
                    )
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
                warnings.warn(
                    "CRS mismatch between CRS of the passed geometries "
                    "and 'crs'. Use 'GeoDataFrame.set_crs(crs, "
                    "allow_override=True)' to overwrite CRS or "
                    "'GeoDataFrame.to_crs(crs)' to reproject geometries. "
                    "CRS mismatch will raise an error in the future versions "
                    "of GeoPandas.",
                    FutureWarning,
                    stacklevel=2,
                )
                # TODO: raise error in 0.9 or 0.10.
            self.set_geometry(geometry, inplace=True)
        self._invalidate_sindex()

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
            super(GeoDataFrame, self).__setattr__(attr, val)

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
        drop : boolean, default True
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
        >>> df1 = df.set_geometry([Point(0,0), Point(1,1), Point(2,2)])
        >>> df2 = df.set_geometry('geom1')

        Returns
        -------
        GeoDataFrame
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
                level = frame[col].values
            except KeyError:
                raise ValueError("Unknown column %s" % col)
            except Exception:
                raise
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
        frame._invalidate_sindex()
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
        >>> df1 = df.rename_geometry('geom1')
        >>> df1.geometry.name
        'geom1'
        >>> df.rename_geometry('geom1', inplace=True)
        >>> df.geometry.name
        'geom1'

        Returns
        -------
        geodataframe : GeoDataFrame
        """
        geometry_col = self.geometry.name
        if not inplace:
            return self.rename(columns={geometry_col: col}).set_geometry(col, inplace)
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
    def from_file(cls, filename, **kwargs):
        """Alternate constructor to create a ``GeoDataFrame`` from a file.

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
        >>> df = geopandas.GeoDataFrame.from_file('nybb.shp')
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
        con : DB connection object or SQLAlchemy engine
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
        >>> sql = "SELECT geom, highway FROM roads"
        SpatiaLite
        >>> sql = "SELECT ST_Binary(geom) AS geom, highway FROM roads"
        >>> df = geopandas.GeoDataFrame.from_postgis(sql, con)
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

    def to_json(self, na="null", show_bbox=False, **kwargs):
        """
        Returns a GeoJSON representation of the ``GeoDataFrame`` as a string.

        Parameters
        ----------
        na : {'null', 'drop', 'keep'}, default 'null'
            Indicates how to output missing (NaN) values in the GeoDataFrame.
            See below.
        show_bbox : bool, optional, default: False
            Include bbox (bounds) in the geojson

        Notes
        -----
        The remaining *kwargs* are passed to json.dumps().

        Missing (NaN) values in the GeoDataFrame can be represented as follows:

        - ``null``: output the missing entries as JSON null.
        - ``drop``: remove the property from the feature. This applies to each
          feature individually so that features may have different properties.
        - ``keep``: output the missing entries as NaN.
        """
        return json.dumps(self._to_geo(na=na, show_bbox=show_bbox), **kwargs)

    @property
    def __geo_interface__(self):
        """Returns a ``GeoDataFrame`` as a python feature collection.

        Implements the `geo_interface`. The returned python data structure
        represents the ``GeoDataFrame`` as a GeoJSON-like
        ``FeatureCollection``.

        This differs from `_to_geo()` only in that it is a property with
        default args instead of a method
        """
        return self._to_geo(na="null", show_bbox=True)

    def iterfeatures(self, na="null", show_bbox=False):
        """
        Returns an iterator that yields feature dictionaries that comply with
        __geo_interface__

        Parameters
        ----------
        na : {'null', 'drop', 'keep'}, default 'null'
            Indicates how to output missing (NaN) values in the GeoDataFrame
            * null: ouput the missing entries as JSON null
            * drop: remove the property from the feature. This applies to
                    each feature individually so that features may have
                    different properties
            * keep: output the missing entries as NaN

        show_bbox : include bbox (bounds) in the geojson. default False
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

                feature = {
                    "id": str(ids[i]),
                    "type": "Feature",
                    "properties": properties_items,
                    "geometry": mapping(geom) if geom else None,
                }

                if show_bbox:
                    feature["bbox"] = geom.bounds if geom else None
                yield feature

        else:
            for fid, geom in zip(ids, geometries):
                feature = {
                    "id": str(fid),
                    "type": "Feature",
                    "properties": {},
                    "geometry": mapping(geom) if geom else None,
                }
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
            Additional keyword arguments passed to to pyarrow.parquet.write_table().
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
            Additional keyword arguments passed to to pyarrow.feather.write_feather().
        """

        from geopandas.io.arrow import _to_feather

        _to_feather(self, path, index=index, compression=compression, **kwargs)

    def to_file(
        self, filename, driver="ESRI Shapefile", schema=None, index=None, **kwargs
    ):
        """Write the ``GeoDataFrame`` to a file.

        By default, an ESRI shapefile is written, but any OGR data source
        supported by Fiona can be written. A dictionary of supported OGR
        providers is available via:

        >>> import fiona
        >>> fiona.supported_drivers

        Parameters
        ----------
        filename : string
            File path or file handle to write to.
        driver : string, default: 'ESRI Shapefile'
            The OGR format driver used to write the vector file.
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

    def __getitem__(self, key):
        """
        If the result is a column containing only 'geometry', return a
        GeoSeries. If it's a DataFrame with a 'geometry' column, return a
        GeoDataFrame.
        """
        result = super(GeoDataFrame, self).__getitem__(key)
        geo_col = self._geometry_column_name
        if isinstance(result, Series) and isinstance(result.dtype, GeometryDtype):
            result.__class__ = GeoSeries
            result._invalidate_sindex()
        elif isinstance(result, DataFrame) and geo_col in result:
            result.__class__ = GeoDataFrame
            result._geometry_column_name = geo_col
            result._invalidate_sindex()
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
            except TypeError:
                warnings.warn("Geometry column does not contain geometry.")
        super(GeoDataFrame, self).__setitem__(key, value)

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
            result._invalidate_sindex()
        elif isinstance(result, DataFrame) and geo_col not in result:
            result.__class__ = DataFrame
        return result

    @property
    def _constructor(self):
        return GeoDataFrame

    def __finalize__(self, other, method=None, **kwargs):
        """propagate metadata from other to self """
        # merge operation: using metadata of the left object
        if method == "merge":
            for name in self._metadata:
                object.__setattr__(self, name, getattr(other.left, name, None))
        # concat operation: using metadata of the first object
        elif method == "concat":
            for name in self._metadata:
                object.__setattr__(self, name, getattr(other.objs[0], name, None))
        else:
            for name in self._metadata:
                object.__setattr__(self, name, getattr(other, name, None))

        return self

    def plot(self, *args, **kwargs):
        """Generate a plot of the geometries in the ``GeoDataFrame``.

        If the ``column`` parameter is given, colors plot according to values
        in that column, otherwise calls ``GeoSeries.plot()`` on the
        ``geometry`` column.

        Wraps the ``plot_dataframe()`` function, and documentation is copied
        from there.
        """
        return plot_dataframe(self, *args, **kwargs)

    plot.__doc__ = plot_dataframe.__doc__

    def dissolve(self, by=None, aggfunc="first", as_index=True):
        """
        Dissolve geometries within `groupby` into single observation.
        This is accomplished by applying the `unary_union` method
        to all geometries within a groupself.

        Observations associated with each `groupby` group will be aggregated
        using the `aggfunc`.

        Parameters
        ----------
        by : string, default None
            Column whose values define groups to be dissolved
        aggfunc : function or string, default "first"
            Aggregation function for manipulation of data associated
            with each group. Passed to pandas `groupby.agg` method.
        as_index : boolean, default True
            If true, groupby columns become index of result.

        Returns
        -------
        GeoDataFrame
        """

        # Process non-spatial component
        data = self.drop(labels=self.geometry.name, axis=1)
        aggregated_data = data.groupby(by=by).agg(aggfunc)

        # Process spatial component
        def merge_geometries(block):
            merged_geom = block.unary_union
            return merged_geom

        g = self.groupby(by=by, group_keys=False)[self.geometry.name].agg(
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

    # overrides GeoPandasBase method
    def explode(self):
        """
        Explode muti-part geometries into multiple single geometries.

        Each row containing a multi-part geometry will be split into
        multiple rows with single geometries, thereby increasing the vertical
        size of the GeoDataFrame.

        The index of the input geodataframe is no longer unique and is
        replaced with a multi-index (original index with additional level
        indicating the multiple geometries: a new zero-based index for each
        single part geometry per multi-part geometry).

        Returns
        -------
        GeoDataFrame
            Exploded geodataframe with each single geometry
            as a separate entry in the geodataframe.

        """
        df_copy = self.copy()

        if "level_1" in df_copy.columns:  # GH1393
            df_copy = df_copy.rename(columns={"level_1": "__level_1"})

        exploded_geom = df_copy.geometry.explode().reset_index(level=-1)
        exploded_index = exploded_geom.columns[0]

        df = pd.concat(
            [df_copy.drop(df_copy._geometry_column_name, axis=1), exploded_geom], axis=1
        )
        # reset to MultiIndex, otherwise df index is only first level of
        # exploded GeoSeries index.
        df.set_index(exploded_index, append=True, inplace=True)
        df.index.names = list(self.index.names) + [None]

        if "__level_1" in df.columns:
            df = df.rename(columns={"__level_1": "level_1"})

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
        df = super(GeoDataFrame, self).astype(dtype, copy=copy, errors=errors, **kwargs)

        try:
            geoms = df[self._geometry_column_name]
            if is_geometry_type(geoms):
                return geopandas.GeoDataFrame(df, geometry=self._geometry_column_name)
        except KeyError:
            pass
        # if the geometry column is converted to non-geometries or did not exist
        # do not return a GeoDataFrame
        return pd.DataFrame(df)

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
        con : sqlalchemy.engine.Engine
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
        >>> engine = create_engine("postgres://myusername:mypassword@myhost:5432\
/mydatabase";)
        >>> gdf.to_postgis("my_table", engine)
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


def _dataframe_set_geometry(self, col, drop=False, inplace=False, crs=None):
    if inplace:
        raise ValueError(
            "Can't do inplace setting when converting from DataFrame to GeoDataFrame"
        )
    gf = GeoDataFrame(self)
    # this will copy so that BlockManager gets copied
    return gf.set_geometry(col, drop=drop, inplace=False, crs=crs)


DataFrame.set_geometry = _dataframe_set_geometry
