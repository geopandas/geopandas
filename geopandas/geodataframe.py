try:
    from collections import OrderedDict
except ImportError:
    # Python 2.6
    from ordereddict import OrderedDict
from collections import defaultdict
import json
import os
import sys

import numpy as np
from pandas import DataFrame, Series
from shapely.geometry import mapping, shape
from shapely.geometry.base import BaseGeometry
from six import string_types
from six import string_types, iteritems

from geopandas import GeoSeries
from geopandas.base import GeoPandasBase
from geopandas.plotting import plot_dataframe
import geopandas.io


DEFAULT_GEO_COLUMN_NAME = 'geometry'
PY3 = sys.version_info[0] == 3


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
    _metadata = ['crs', '_geometry_column_name']
    _geometry_column_name = DEFAULT_GEO_COLUMN_NAME

    def __init__(self, *args, **kwargs):
        crs = kwargs.pop('crs', None)
        geometry = kwargs.pop('geometry', None)
        super(GeoDataFrame, self).__init__(*args, **kwargs)
        self.crs = crs
        if geometry is not None:
            self.set_geometry(geometry, inplace=True)

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
        geo_column_name = DEFAULT_GEO_COLUMN_NAME
        if isinstance(col, Series):
            level = col.values
        elif isinstance(col, (list, np.ndarray)):
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
                geo_column_name = DEFAULT_GEO_COLUMN_NAME
            else:
                geo_column_name = col

        if to_remove:
            del frame[to_remove]

        if isinstance(level, GeoSeries) and level.crs != crs:
            # Avoids caching issues/crs sharing issues
            level = level.copy()
            level.crs = crs

        # Check that we are using a listlike of geometries
        if not all(isinstance(item, BaseGeometry) for item in level):
            raise TypeError("Input geometry column must contain valid geometry objects.")
        frame[geo_column_name] = level
        frame._geometry_column_name = geo_column_name
        frame.crs = crs

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

        Note: This method does not attempt to align rows.  Properties that are
        not present in all features of the source file will not be properly
        aligned.  This should be fixed.

        """
        geoms = []
        columns = defaultdict(lambda: [])
        for f in features:
            if hasattr(f, "__geo_interface__"):
                f = f.__geo_interface__
            else:
                f = f

            geoms.append(shape(f['geometry']))
            for key, value in f['properties'].iteritems():
                columns[key].append(value)
        geom = GeoSeries(geoms)
        df = GeoDataFrame(columns)
        df['geometry'] = geom
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


    def to_json(self, na='null', **kwargs):
        """Returns a GeoJSON representation of the GeoDataFrame.

        Parameters
        ----------
        na : {'null', 'drop', 'keep'}, default 'null'
            Indicates how to output missing (NaN) values in the GeoDataFrame
            * null: ouput the missing entries as JSON null
            * drop: remove the property from the feature. This applies to
                    each feature individually so that features may have
                    different properties
            * keep: output the missing entries as NaN
        
        The remaining *kwargs* are passed to json.dumps().
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

        # na_methods must take in a Series and return dict-like
        na_methods = {'null': fill_none,
                      'drop': lambda row: row.dropna(),
                      'keep': lambda row: row}

        if na not in na_methods:
            raise ValueError('Unknown na method {}'.format(na))
        f = na_methods[na]

        def feature(i, row):
            row = f(row)
            return {
                'id': str(i),
                'type': 'Feature',
                'properties':
                    dict((k, v) for k, v in iteritems(row) if k != self._geometry_column_name),
                'geometry': mapping(row[self._geometry_column_name]) }

        return json.dumps(
            {'type': 'FeatureCollection',
             'features': [feature(i, row) for i, row in self.iterrows()]},
            **kwargs )
            
    def to_file(self, filename, driver="ESRI Shapefile", **kwargs):
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

        The *kwargs* are passed to fiona.open and can be used to write 
        to multi-layer data, store data within archives (zip files), etc.
        """
        import fiona
        def convert_type(in_type):
            if in_type == object:
                return 'str'
            return type(np.asscalar(np.zeros(1, in_type))).__name__
            
        def feature(i, row):
            return {
                'id': str(i),
                'type': 'Feature',
                'properties':
                    dict((k, v) for k, v in iteritems(row) if k != 'geometry'),
                'geometry': mapping(row['geometry']) }
        
        properties = OrderedDict([(col, convert_type(_type)) for col, _type 
            in zip(self.columns, self.dtypes) if col!='geometry'])
        # Need to check geom_types before we write to file... 
        # Some (most?) providers expect a single geometry type: 
        # Point, LineString, or Polygon
        geom_types = self['geometry'].geom_type.unique()
        from os.path import commonprefix # To find longest common prefix
        geom_type = commonprefix([g[::-1] for g in geom_types])[::-1]  # Reverse
        if geom_type == '': # No common suffix = mixed geometry types
            raise ValueError("Geometry column cannot contains mutiple "
                             "geometry types when writing to file.")
        schema = {'geometry': geom_type, 'properties': properties}
        filename = os.path.abspath(os.path.expanduser(filename))
        with fiona.open(filename, 'w', driver=driver, crs=self.crs, 
                        schema=schema, **kwargs) as c:
            for i, row in self.iterrows():
                c.write(feature(i, row))

    def to_crs(self, crs=None, epsg=None, inplace=False):
        """Transform geometries to a new coordinate reference system

        This method will transform all points in all objects.  It has
        no notion or projecting entire geometries.  All segments
        joining points are assumed to be lines in the current
        projection, not geodesics.  Objects crossing the dateline (or
        other projection boundary) will have undesirable behavior.
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
        elif isinstance(result, DataFrame) and geo_col in result:
            result.__class__ = GeoDataFrame
            result.crs = self.crs
            result._geometry_column_name = geo_col
        elif isinstance(result, DataFrame) and geo_col not in result:
            result.__class__ = DataFrame
            result.crs = self.crs
        return result

    #
    # Implement pandas methods
    #

    @property
    def _constructor(self):
        return GeoDataFrame

    def __finalize__(self, other, method=None, **kwargs):
        """ propagate metadata from other to self """
        # NOTE: backported from pandas master (upcoming v0.13)
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
