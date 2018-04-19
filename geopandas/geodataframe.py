from __future__ import absolute_import, division, print_function

from collections import Iterable
import json

import numpy as np
import pandas as pd
from pandas import DataFrame, Series, Index
from pandas.core.internals import BlockManager

from six import string_types, PY3

from shapely.geometry import mapping, shape

from .geoseries import GeoSeries
from .base import GeoPandasBase, _CoordinateIndexer, is_geometry_type
from .plotting import plot_dataframe
from .array import GeometryArray, from_shapely, _HAS_EXTENSION_ARRAY
from . import _block
from ._block import GeometryBlock


def coerce_to_geoseries(x, **kwargs):
    if isinstance(x, GeoSeries):
        return x
    if isinstance(x, GeometryArray):
        return GeoSeries(x, **kwargs)
    if isinstance(x, pd.Series):
        kwargs['name'] = kwargs.get('name', x.name)
        kwargs['index'] = kwargs.get('index', x.index)
        return GeoSeries(from_shapely(x.values), **kwargs)
    if isinstance(x, Iterable):
        return GeoSeries(from_shapely(list(x)), **kwargs)
    raise TypeError(type(x))


DEFAULT_GEO_COLUMN_NAME = 'geometry'


class GeoDataFrame(GeoPandasBase, DataFrame):
    """
    A GeoDataFrame object is a pandas.DataFrame that has a column
    with geometry. In addition to the standard DataFrame constructor arguments,
    GeoDataFrame also accepts the following keyword arguments:

    Keyword Arguments
    -----------------
    crs : str (optional)
        Coordinate system
    geometry : str or array (optional)
        If str, column to use as geometry. If array, will be set as 'geometry'
        column on GeoDataFrame.
    """

    # XXX: This will no longer be necessary in pandas 0.17
    _internal_names = ['_data', '_cacher', '_item_cache', '_cache',
                       'is_copy', '_subtyp', '_index',
                       '_default_kind', '_default_fill_value', '_metadata',
                       '__array_struct__', '__array_interface__']

    _metadata = ['crs', '_geometry_column_name']

    _geometry_column_name = DEFAULT_GEO_COLUMN_NAME

    def __init__(self, *args, **kwargs):
        crs = kwargs.pop('crs', None)

        geometry = kwargs.pop('geometry', 'geometry')
        if not isinstance(geometry, str):
            geometry = coerce_to_geoseries(geometry)
            if not geometry.name:
                geometry.name = 'geometry'

        if not args:
            if not isinstance(geometry, str):
                # ensure correct length of the empty frame
                n = len(geometry)
                kwargs['index'] = kwargs.pop('index', range(n))
            arg = []
        else:
            [arg] = args

        if isinstance(arg, BlockManager):
            super(GeoDataFrame, self).__init__(arg, **kwargs)
            self.crs = crs
            self._geometry_column_name = geometry
            return

        gs = {}

        if isinstance(arg, dict):
            arg = arg.copy()
            for i, (k, v) in list(enumerate(arg.items())):
                if isinstance(v, GeoSeries) or is_geometry_type(v):
                    if 'columns' in kwargs:
                        columns = list(kwargs['columns'])
                        columns.remove(k)
                        kwargs['columns'] = columns
                    gs[k] = arg.pop(k)
                    kwargs['index'] = v.index  # TODO: assumes consistent index

        if isinstance(arg, (dict, list, pd.Series, np.ndarray)):
            arg = pd.DataFrame(arg, **kwargs)

        assert isinstance(arg, pd.DataFrame)

        if not isinstance(geometry, str):
            gs[geometry.name] = coerce_to_geoseries(geometry)
            geometry = geometry.name
            if geometry in arg.columns:
                arg.drop(geometry, axis=1, inplace=True)

        elif geometry in arg.columns:
            arg = arg.copy()
            geom = arg.pop(geometry)
            geom = coerce_to_geoseries(geom, name=geometry)
            gs[geometry] = geom

        if not _HAS_EXTENSION_ARRAY:
            block_manager = _block.add_geometries(arg._data, gs)
            kwargs['columns'] = block_manager.axes[0]
            super(GeoDataFrame, self).__init__(block_manager, **kwargs)
        else:
            for k, geom in gs.items():
                arg[k] = geom
            super(GeoDataFrame, self).__init__(arg._data) #, **kwargs)

        self.crs = crs
        self._geometry_column_name = geometry

        self._invalidate_sindex()

    # Serialize metadata (will no longer be necessary in pandas 0.17+)
    # See https://github.com/pydata/pandas/pull/10557
    def __getstate__(self):
        geometry = self._geometry_array
        geometry_name = self._geometry_column_name
        data = pd.DataFrame(self.drop([self._geometry_column_name], axis=1))

        return dict(geometry=geometry, geometry_name=geometry_name, data=data,
                    crs=self.crs)

    def __setstate__(self, state):
        self.__init__(state['data'], geometry=state['geometry'], crs=state['crs'])
        self.rename(columns={'geometry': state['geometry_name']}).set_geometry(state['geometry_name'])

    def __setattr__(self, attr, val):
        # have to special case geometry b/c pandas tries to use as column...
        if attr == 'geometry':
            object.__setattr__(self, attr, val)
        else:
            super(GeoDataFrame, self).__setattr__(attr, val)

    def _get_geometry(self):
        if self._geometry_column_name not in self:
            raise AttributeError("No geometry data set yet (expected in"
                                 " column '%s'." % self._geometry_column_name)
        return self[self._geometry_column_name]

    def _set_geometry(self, col):
        # TODO: Use pandas' core.common.is_list_like() here.
        if not isinstance(col, (list, np.ndarray, Series)):
            raise ValueError("Must use a list-like to set the geometry"
                             " property")
        if isinstance(col, GeoSeries):
            self.set_geometry(col, inplace=True)
        elif isinstance(col, (list, np.ndarray, Series)):
            col = from_shapely(col)
            self.set_geometry(col, inplace=True)


    geometry = property(fget=_get_geometry, fset=_set_geometry,
                        doc="Geometry data for GeoDataFrame")

    def set_geometry(self, col, drop=False, inplace=False, crs=None):
        """
        Set the GeoDataFrame geometry using either an existing column or
        the specified input. By default yields a new object.

        The original geometry column is replaced with the input.  The geometry
        column will have the same name as the selected column.

        Parameters
        ----------
        keys : column label or array
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
        if inplace:
            frame = self
        else:
            frame = self.copy()

        if not crs:
            crs = getattr(col, 'crs', self.crs)

        to_remove = None
        geo_column_name = self._geometry_column_name
        if isinstance(col, (Series, list, np.ndarray, GeometryArray)):
            level = col
            to_remove = geo_column_name
            if isinstance(col, Series):
                if col.name:
                    geo_column_name = col.name
        elif hasattr(col, 'ndim') and col.ndim != 1:
            raise ValueError("Must pass array with one dimension only.")
        else:
            try:
                level = frame[col]._values
            except KeyError:
                raise ValueError("Unknown column %s" % col)
            except:
                raise
            if drop:
                to_remove = col
                geo_column_name = self._geometry_column_name
            else:
                geo_column_name = col

        if to_remove and to_remove in self.columns:
            del frame[to_remove]

        if geo_column_name in frame.columns:
            del frame[geo_column_name]

        if isinstance(level, GeoSeries) and level.crs != crs:
            # Avoids caching issues/crs sharing issues
            level = level.copy()
            level.crs = crs

        if isinstance(level, Series):
            level = GeoSeries(level)
            level, _ = level.align(frame, join='right')
        else:
            level = GeoSeries(level, index=frame.index)

        if not _HAS_EXTENSION_ARRAY:
            blk_mgr = _block.add_geometries(frame._data, {geo_column_name: level})

            # frame[geo_column_name] = level
            # frame._geometry_column_name = geo_column_name
            # frame.crs = crs
            # frame._invalidate_sindex()
            if inplace:
                frame._data = blk_mgr
                frame._geometry_column_name = geo_column_name
                frame.crs = crs
                frame._invalidate_sindex()
            else:
                frame = self._constructor(blk_mgr, geometry=geo_column_name,
                                          crs=crs)
                return frame

        frame[geo_column_name] = level._data._block.values
        frame._geometry_column_name = geo_column_name
        frame.crs = crs
        frame._invalidate_sindex()
        if not inplace:
            return frame

    @classmethod
    def from_file(cls, filename, **kwargs):
        """Alternate constructor to create a ``GeoDataFrame`` from a file.

        Can load a ``GeoDataFrame`` from a file in any format recognized by
        `fiona`. See http://toblerity.org/fiona/manual.html for details.

        Parameters
        ----------

        filename : str
            File path or file handle to read from. Depending on which kwargs
            are included, the content of filename may vary. See
            http://toblerity.org/fiona/README.html#usage for usage details.
        kwargs : key-word arguments
            These arguments are passed to fiona.open, and can be used to
            access multi-layer data, data stored within archives (zip files),
            etc.

        Examples
        --------

        >>> df = geopandas.GeoDataFrame.from_file('nybb.shp')
        """
        import geopandas.io
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

        if isinstance(fs, dict) and fs.get('type') == 'FeatureCollection':
            features_lst = fs['features']
        else:
            features_lst = features

        rows = []
        for f in features_lst:
            if hasattr(f, "__geo_interface__"):
                f = f.__geo_interface__
            else:
                f = f

            d = {'geometry': shape(f['geometry']) if f['geometry'] else None}
            d.update(f['properties'])
            rows.append(d)
        df = GeoDataFrame(rows, columns=columns)
        df.crs = crs
        return df

    @classmethod
    def from_postgis(cls, sql, con, geom_col='geom', crs=None, index_col=None,
                     coerce_float=True, params=None):
        """Alternate constructor to create a ``GeoDataFrame`` from a sql query
        containing a geometry column.

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
        params : list, tuple or dict, optional, default: None
            List of parameters to pass to execute method.

        Examples
        --------

        >>> sql = "SELECT geom, highway FROM roads;"
        >>> df = geopandas.GeoDataFrame.from_postgis(sql, con)
        """
        import geopandas.io
        return geopandas.io.sql.read_postgis(
            sql, con, geom_col, crs, index_col, coerce_float, params)

    def to_json(self, na='null', show_bbox=False, **kwargs):
        """Returns a GeoJSON representation of the ``GeoDataFrame`` as a string.

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
        return self._to_geo(na='null', show_bbox=True)

    def iterfeatures(self, na='null', show_bbox=False):
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
        def fill_none(row):
            """
            Takes in a Series, converts to a dictionary with null values
            set to None

            """
            na_keys = row.index[row.isnull()]
            d = row.to_dict()
            for k in na_keys:
                d[k] = None
            return d

        # na_methods must take in a Series and return dict
        na_methods = {'null': fill_none,
                      'drop': lambda row: row.dropna().to_dict(),
                      'keep': lambda row: row.to_dict()}

        if na not in na_methods:
            raise ValueError('Unknown na method {0}'.format(na))
        f = na_methods[na]

        for name, row in self.iterrows():
            properties = f(row)
            del properties[self._geometry_column_name]

            feature = {
                'id': str(name),
                'type': 'Feature',
                'properties': properties,
                'geometry': mapping(row[self._geometry_column_name])
                            if row[self._geometry_column_name] else None
            }

            if show_bbox:
                feature['bbox'] = row.geometry.bounds

            yield feature

    def iterrows(self):
        rows = super(GeoDataFrame, self).iterrows()
        for (index, series), geom in zip(rows, self.geometry._geometry_array):
            series.loc[self._geometry_column_name] = geom
            yield (index, series)

    def _to_geo(self, **kwargs):
        """
        Returns a python feature collection (i.e. the geointerface)
        representation of the GeoDataFrame.

        """
        geo = {'type': 'FeatureCollection',
               'features': list(self.iterfeatures(**kwargs))}

        if kwargs.get('show_bbox', False):
            geo['bbox'] = tuple(self.total_bounds)

        return geo

    def to_file(self, filename, driver="ESRI Shapefile", schema=None,
                **kwargs):
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
        if isinstance(key, string_types)\
                and ((key == geo_col)
                     or (result.ndim == 1
                         and isinstance(result._data._block, GeometryBlock))):
            if not key == geo_col:
                result = GeoSeries(result, crs=self.crs)
            else:
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

    def __setitem__(self, key, value):
        """If what is set is a GeoSeries, make sure the GeometryBlock
        is preserved"""
        if _HAS_EXTENSION_ARRAY:
            super(GeoDataFrame, self).__setitem__(key, value)
        else:
            if (isinstance(key, string_types)
                    and (isinstance(value, GeoSeries)
                        or isinstance(value, GeometryArray))):
                if isinstance(value, GeoSeries):
                    value = value.reindex(self.index)
                    value = value._geometry_array
                else:
                    if not len(value) == len(self):
                        raise ValueError("Length of values does not match length "
                                        "of index")

                _block.frame_set_geometry(self, key, value)

            else:
                super(GeoDataFrame, self).__setitem__(key, value)

    #
    # Implement pandas methods
    #

    def merge(self, *args, **kwargs):
        result = DataFrame.merge(self, *args, **kwargs)
        geo_col = self._geometry_column_name
        if isinstance(result, DataFrame) and geo_col in result:
            result.__class__ = GeoDataFrame
            result.crs = self.crs
            values = result[geo_col]._values
            if isinstance(values, np.ndarray):
                g = from_shapely(values)
            else:
                g = values
            result[geo_col] = list(g)
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
        if method == 'merge':
            for name in self._metadata:
                object.__setattr__(self, name, getattr(other.left, name, None))
        # concat operation: using metadata of the first object
        elif method == 'concat':
            for name in self._metadata:
                object.__setattr__(self, name, getattr(other.objs[0], name, None))
        else:
            for name in self._metadata:
                object.__setattr__(self, name, getattr(other, name, None))
        return self

    def copy(self, deep=True):
        """
        Make a copy of this GeoDataFrame object

        Parameters
        ----------
        deep : boolean, default True
            Make a deep copy, i.e. also copy data

        Returns
        -------
        copy : GeoDataFrame
        """
        # FIXME: this will likely be unnecessary in pandas >= 0.13
        data = self._data
        if deep:
            data = data.copy()
        out = GeoDataFrame(data).__finalize__(self)
        return out

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


    def dissolve(self, by=None, aggfunc='first', as_index=True):
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

        g = self.groupby(by=by, group_keys=False)[self.geometry.name].agg(merge_geometries)

        # Recombine
        aggregated = aggregated_data.set_geometry(g, crs=self.crs)

        # Reset if requested
        if not as_index:
            aggregated = aggregated.reset_index()

        return aggregated


def _dataframe_set_geometry(self, col, drop=False, inplace=False, crs=None):
    if inplace:
        raise ValueError("Can't do inplace setting when converting from"
                         " DataFrame to GeoDataFrame")
    return GeoDataFrame(self, geometry=col, crs=crs)


def concat(objs, axis=0, ignore_index=False):
    """Concatenate multiple GeoDataFrames or GeoSeries together

    **Expectations for axis=0**
    -  Expects all input types to be either GeoDataFrame or GeoSeries
    -  CRS values should be the same
    -  Geometry names should be the same

    **Expectations for axis=1**
    - Only one of the objects is a GeoDataFrame or GeoSeries

    """
    if axis == 0:
        return _concat_axis0(objs, ignore_index=ignore_index)
    elif axis == 1:
        return _concat_axis1(objs, ignore_index=ignore_index)
    else:
        raise ValueError("Invalid axis value")


def _concat_arrays(L, axis=0):
    if axis != 0:
        raise NotImplementedError("Can only concatenate geometries along axis=0")
    L = list(L)
    x = np.concatenate([ga.data for ga in L])
    return GeometryArray(x, base=set(L))


def _concat_axis0(L, **kwargs):

    types = set(map(type, L))
    if types != {GeoDataFrame} and types != {GeoSeries}:
        raise TypeError("Expected consistent types, got %s" % str(types))

    if not len(set(str(df.crs) for df in L)) == 1:
        raise ValueError("Expected all crs values to be the same")

    if isinstance(L[0], GeoDataFrame):
        if not len(set(df._geometry_column_name for df in L)) == 1:
            raise ValueError("Expected all geometry names to be the same")
        name = L[0]._geometry_column_name
    else:
        if not len(set(s.name for s in L)) == 1:
            raise ValueError("Expected all geometry names to be the same")
        name = L[0].name

    geometry = _concat_arrays(df._geometry_array for df in L)
    if isinstance(L[0], GeoDataFrame):
        L = [df.drop(name, axis=1) for df in L]
        new = pd.concat(L, **kwargs)
        new = new.set_geometry(GeoSeries(geometry, name=name, index=new.index),
                               crs=L[0].crs)
    else:
        index = pd.concat([pd.Series(index=s.index) for s in L],
                          **kwargs).index
        new = GeoSeries(geometry, index=index, name=name, crs=L[0].crs)
    return new


def _concat_axis1(objs, **kwargs):
    is_geo = [isinstance(obj, (GeoDataFrame, GeoSeries)) for obj in objs]

    if not sum(is_geo) == 1:
        raise ValueError("'concat' with axis=1 currently only supports "
                         "concatenating objects of which only one is a "
                         "GeoDataFrame or GeoSeries")

    geo_obj = objs[is_geo.index(True)]
    if isinstance(geo_obj, GeoSeries):
        geometry = geo_obj
        temp_obj = pd.Series(0, index=geo_obj.index, name=geo_obj.name)
    else:
        geometry = geo_obj.geometry
        temp_obj = geo_obj.copy()
        temp_obj[geometry.name] = 1

    objs[is_geo.index(True)] = temp_obj
    result = pd.concat(objs, axis=1, **kwargs)

    geometry_reindexed = geometry.reindex(result.index)

    result[geometry.name] = geometry_reindexed

    return result


if PY3:
    DataFrame.set_geometry = _dataframe_set_geometry
else:
    import types
    DataFrame.set_geometry = types.MethodType(_dataframe_set_geometry, None,
                                              DataFrame)


GeoDataFrame._create_indexer('cx', _CoordinateIndexer)
