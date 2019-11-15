import json

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from six import PY3, string_types

from shapely.geometry import mapping, shape

from geopandas.array import GeometryArray, from_shapely
from geopandas.base import GeoPandasBase, is_geometry_type
from geopandas.geoseries import GeoSeries
import geopandas.io
from geopandas.plotting import plot_dataframe

DEFAULT_GEO_COLUMN_NAME = "geometry"


def _ensure_geometry(data):
    """
    Ensure the data is of geometry dtype or converted to it.

    If input is a (Geo)Series, output is a GeoSeries, otherwise output
    is GeometryArray.
    """
    if is_geometry_type(data):
        if isinstance(data, Series):
            return GeoSeries(data)
        return data
    else:
        if isinstance(data, Series):
            out = from_shapely(np.asarray(data))
            return GeoSeries(out, index=data.index, name=data.name)
        else:
            out = from_shapely(data)
            return out


class GeoDataFrame(GeoPandasBase, DataFrame):
    """
    A GeoDataFrame object is a pandas.DataFrame that has a column
    with geometry. In addition to the standard DataFrame constructor arguments,
    GeoDataFrame also accepts the following keyword arguments:

    Parameters
    ----------
    crs : str (optional)
        Coordinate system
    geometry : str or array (optional)
        If str, column to use as geometry. If array, will be set as 'geometry'
        column on GeoDataFrame.
    """

    _metadata = ["crs", "_geometry_column_name"]

    _geometry_column_name = DEFAULT_GEO_COLUMN_NAME

    def __init__(self, *args, **kwargs):
        crs = kwargs.pop("crs", None)
        geometry = kwargs.pop("geometry", None)
        super(GeoDataFrame, self).__init__(*args, **kwargs)

        # need to set this before calling self['geometry'], because
        # getitem accesses crs
        self.crs = crs

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
                self["geometry"] = _ensure_geometry(self["geometry"].values)
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
            self.set_geometry(geometry, inplace=True)
        self._invalidate_sindex()

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
                " column '%s'." % self._geometry_column_name
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
        crs : str/result of fion.get_crs (optional)
            Coordinate system to use. If passed, overrides both DataFrame and
            col's crs. Otherwise, tries to get crs from passed col values or
            DataFrame.

        Examples
        --------
        >>> df1 = df.set_geometry([Point(0,0), Point(1,1), Point(2,2)])
        >>> df2 = df.set_geometry('geom1')

        Returns
        -------
        geodataframe : GeoDataFrame
        """
        # Most of the code here is taken from DataFrame.set_index()
        if inplace:
            frame = self
        else:
            frame = self.copy()

        if not crs:
            crs = getattr(col, "crs", self.crs)

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

        if isinstance(level, GeoSeries) and level.crs != crs:
            # Avoids caching issues/crs sharing issues
            level = level.copy()
            level.crs = crs

        # Check that we are using a listlike of geometries
        level = _ensure_geometry(level)
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
        return geopandas.io.file.read_file(filename, **kwargs)

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
        for f in features_lst:
            if hasattr(f, "__geo_interface__"):
                f = f.__geo_interface__
            else:
                f = f

            d = {"geometry": shape(f["geometry"]) if f["geometry"] else None}
            d.update(f["properties"])
            rows.append(d)
        df = GeoDataFrame(rows, columns=columns)
        df.crs = crs
        return df

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

        Examples
        --------
        >>> sql = "SELECT geom, highway FROM roads"
        SpatiaLite
        >>> sql = "SELECT ST_Binary(geom) AS geom, highway FROM roads"
        >>> df = geopandas.GeoDataFrame.from_postgis(sql, con)
        """

        df = geopandas.io.sql.read_postgis(
            sql,
            con,
            geom_col=geom_col,
            crs=crs,
            index_col=index_col,
            coerce_float=coerce_float,
            parse_dates=parse_dates,
            params=params,
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

    def to_file(self, filename, driver="ESRI Shapefile", schema=None, **kwargs):
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

        Notes
        -----
        The extra keyword arguments ``**kwargs`` are passed to fiona.open and
        can be used to write to multi-layer data, store data within archives
        (zip files), etc.
        """
        from geopandas.io.file import to_file

        to_file(self, filename, driver, schema, **kwargs)

    def to_crs(self, crs=None, epsg=None, inplace=False):
        """Transform geometries to a new coordinate reference system.

        Transform all geometries in a GeoSeries to a different coordinate
        reference system.  The ``crs`` attribute on the current GeoSeries must
        be set.  Either ``crs`` in string or dictionary form or an EPSG code
        may be specified for output.

        This method will transform all points in all objects.  It has no notion
        or projecting entire geometries.  All segments joining points are
        assumed to be lines in the current projection, not geodesics.  Objects
        crossing the dateline (or other projection boundary) will have
        undesirable behavior.

        Parameters
        ----------
        crs : dict or str
            Output projection parameters as string or in dictionary form.
        epsg : int
            EPSG code specifying output projection.
        inplace : bool, optional, default: False
            Whether to return a new GeoDataFrame or do the transformation in
            place.
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
        if isinstance(key, string_types) and key == geo_col:
            result.__class__ = GeoSeries
            result.crs = self.crs
            result._invalidate_sindex()
        elif isinstance(result, DataFrame) and geo_col in result:
            result.__class__ = GeoDataFrame
            result.crs = self.crs
            result._geometry_column_name = geo_col
            result._invalidate_sindex()
        elif isinstance(result, DataFrame) and geo_col not in result:
            result.__class__ = DataFrame
        return result

    #
    # Implement pandas methods
    #

    def merge(self, *args, **kwargs):
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

        exploded_geom = df_copy.geometry.explode().reset_index(level=-1)
        exploded_index = exploded_geom.columns[0]

        df = pd.concat(
            [df_copy.drop(df_copy._geometry_column_name, axis=1), exploded_geom], axis=1
        )
        # reset to MultiIndex, otherwise df index is only first level of
        # exploded GeoSeries index.
        df.set_index(exploded_index, append=True, inplace=True)
        df.index.names = list(self.index.names) + [None]
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


def _dataframe_set_geometry(self, col, drop=False, inplace=False, crs=None):
    if inplace:
        raise ValueError(
            "Can't do inplace setting when converting from DataFrame to GeoDataFrame"
        )
    gf = GeoDataFrame(self)
    # this will copy so that BlockManager gets copied
    return gf.set_geometry(col, drop=drop, inplace=False, crs=crs)


if PY3:
    DataFrame.set_geometry = _dataframe_set_geometry
else:
    import types

    DataFrame.set_geometry = types.MethodType(_dataframe_set_geometry, None, DataFrame)
