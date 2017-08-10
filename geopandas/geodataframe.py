import json
import os
import sys

import numpy as np
from pandas import DataFrame, Series, Index
from shapely.geometry import mapping, shape
from shapely.geometry.base import BaseGeometry
from six import string_types, PY3

from geopandas.base import GeoPandasBase, _CoordinateIndexer
from geopandas.geoseries import GeoSeries
from geopandas.plotting import plot_dataframe
import geopandas.io


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
        geometry = kwargs.pop('geometry', None)
        super(GeoDataFrame, self).__init__(*args, **kwargs)
        self.crs = crs
        if geometry is not None:
            self.set_geometry(geometry, inplace=True)
        self._invalidate_sindex()

    # Serialize metadata (will no longer be necessary in pandas 0.17+)
    # See https://github.com/pydata/pandas/pull/10557
    def __getstate__(self):
        meta = dict((k, getattr(self, k, None)) for k in self._metadata)
        return dict(_data=self._data, _typ=self._typ,
                    _metadata=self._metadata, **meta)

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
        self.set_geometry(col, inplace=True)

    geometry = property(fget=_get_geometry, fset=_set_geometry,
                        doc="Geometry data for GeoDataFrame")

    def set_geometry(self, col, drop=False, inplace=False, crs=None):
        """
        Set the GeoDataFrame geometry using either an existing column or
        the specified input. By default yields a new object.

        The original geometry column is replaced with the input.

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
        # Most of the code here is taken from DataFrame.set_index()
        if inplace:
            frame = self
        else:
            frame = self.copy()

        if not crs:
            crs = getattr(col, 'crs', self.crs)

        to_remove = None
        geo_column_name = self._geometry_column_name
        if isinstance(col, (Series, list, np.ndarray)):
            level = col
        elif hasattr(col, 'ndim') and col.ndim != 1:
            raise ValueError("Must pass array with one dimension only.")
        else:
            try:
                level = frame[col].values
            except KeyError:
                raise ValueError("Unknown column %s" % col)
            except:
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
        if not all(isinstance(item, BaseGeometry) or not item for item in level):
            raise TypeError("Input geometry column must contain valid geometry objects.")
        frame[geo_column_name] = level
        frame._geometry_column_name = geo_column_name
        frame.crs = crs
        frame._invalidate_sindex()
        if not inplace:
            return frame

    @classmethod
    def from_file(cls, filename, **kwargs):
        """
        Alternate constructor to create a GeoDataFrame from a file.

        Example:
            df = geopandas.GeoDataFrame.from_file('nybb.shp')

        Wraps geopandas.read_file(). For additional help, see read_file()

        """
        return geopandas.io.file.read_file(filename, **kwargs)

    @classmethod
    def from_features(cls, features, crs=None):
        """
        Alternate constructor to create GeoDataFrame from an iterable of
        features. Each element must be a feature dictionary or implement
        the __geo_interface__.
        See: https://gist.github.com/sgillies/2217756

        """
        rows = []
        for f in features:
            if hasattr(f, "__geo_interface__"):
                f = f.__geo_interface__
            else:
                f = f

            d = {'geometry': shape(f['geometry']) if f['geometry'] else None}
            d.update(f['properties'])
            rows.append(d)
        df = GeoDataFrame.from_dict(rows)
        df.crs = crs
        return df

    @classmethod
    def from_postgis(cls, sql, con, geom_col='geom', crs=None, index_col=None,
                     coerce_float=True, params=None):
        """
        Alternate constructor to create a GeoDataFrame from a sql query
        containing a geometry column.

        Example:
            df = geopandas.GeoDataFrame.from_postgis(con,
                "SELECT geom, highway FROM roads;")

        Wraps geopandas.read_postgis(). For additional help, see read_postgis()

        """
        return geopandas.io.sql.read_postgis(sql, con, geom_col, crs, index_col,
                     coerce_float, params)

    def to_json(self, na='null', show_bbox=False, **kwargs):
        """
        Returns a GeoJSON string representation of the GeoDataFrame.

        Parameters
        ----------
        na : {'null', 'drop', 'keep'}, default 'null'
            Indicates how to output missing (NaN) values in the GeoDataFrame
            * null: output the missing entries as JSON null
            * drop: remove the property from the feature. This applies to
                    each feature individually so that features may have
                    different properties
            * keep: output the missing entries as NaN

        show_bbox : include bbox (bounds) in the geojson

        The remaining *kwargs* are passed to json.dumps().

        """
        return json.dumps(self._to_geo(na=na, show_bbox=show_bbox), **kwargs)

    @property
    def __geo_interface__(self):
        """
        Returns a python feature collection (i.e. the geointerface)
        representation of the GeoDataFrame.

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
        """
        Write this GeoDataFrame to an OGR data source

        A dictionary of supported OGR providers is available via:
        >>> import fiona
        >>> fiona.supported_drivers

        Parameters
        ----------
        filename : string
            File path or file handle to write to.
        driver : string, default 'ESRI Shapefile'
            The OGR format driver used to write the vector file.
        schema : dict, default None
            If specified, the schema dictionary is passed to Fiona to
            better control how the file is written.

        The *kwargs* are passed to fiona.open and can be used to write
        to multi-layer data, store data within archives (zip files), etc.
        """
        from geopandas.io.file import to_file
        to_file(self, filename, driver, schema, **kwargs)

    def to_crs(self, crs=None, epsg=None, inplace=False):
        """Transform geometries to a new coordinate reference system

        This method will transform all points in all objects.  It has
        no notion or projecting entire geometries.  All segments
        joining points are assumed to be lines in the current
        projection, not geodesics.  Objects crossing the dateline (or
        other projection boundary) will have undesirable behavior.

        `to_crs` passes the `crs` argument to the `Proj` function from the
        `pyproj` library (with the option `preserve_units=True`). It can
        therefore accept proj4 projections in any format
        supported by `Proj`, including dictionaries, or proj4 strings.

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
        return GeoDataFrame(data).__finalize__(self)

    def plot(self, *args, **kwargs):

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

        # Aggregate
        aggregated_geometry = GeoDataFrame(g, geometry=self.geometry.name, crs=self.crs)
        # Recombine
        aggregated = aggregated_geometry.join(aggregated_data)

        # Reset if requested
        if not as_index:
            aggregated = aggregated.reset_index()

        return aggregated

def _dataframe_set_geometry(self, col, drop=False, inplace=False, crs=None):
    if inplace:
        raise ValueError("Can't do inplace setting when converting from"
                         " DataFrame to GeoDataFrame")
    gf = GeoDataFrame(self)
    # this will copy so that BlockManager gets copied
    return gf.set_geometry(col, drop=drop, inplace=False, crs=crs)

if PY3:
    DataFrame.set_geometry = _dataframe_set_geometry
else:
    import types
    DataFrame.set_geometry = types.MethodType(_dataframe_set_geometry, None,
                                              DataFrame)


GeoDataFrame._create_indexer('cx', _CoordinateIndexer)
