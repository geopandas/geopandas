from functools import partial
from warnings import warn

import numpy as np
from pandas import Series, DataFrame
from pandas.core.indexing import _NDFrameIndexer
from pandas.util.decorators import cache_readonly
import pyproj
from shapely.geometry import box, shape, Polygon, Point
from shapely.geometry.collection import GeometryCollection
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform

from geopandas.plotting import plot_series
from geopandas.base import GeoPandasBase


OLD_PANDAS = issubclass(Series, np.ndarray)

def _is_empty(x):
    try:
        return x.is_empty
    except:
        return False

def _convert_array_args(args):
    if len(args) == 1 and isinstance(args[0], BaseGeometry):
        args = ([args[0]],)
    return args

class _CoordinateIndexer(_NDFrameIndexer):
    """ Indexing by coordinate slices """
    def _getitem_tuple(self, tup):
        obj = self.obj
        xs, ys = tup
        # handle numeric values as x and/or y coordinate index
        if type(xs) is not slice:
            xs = slice(xs, xs)
        if type(ys) is not slice:
            ys = slice(ys, ys)
        # don't know how to handle step; should this raise?
        if xs.step is not None or ys.step is not None:
            warn("Ignoring step - full interval is used.")
        xmin, ymin, xmax, ymax = obj.total_bounds
        bbox = box(xs.start or xmin,
                   ys.start or ymin,
                   xs.stop or xmax,
                   ys.stop or ymax)
        idx = obj.intersects(bbox)
        return obj[idx]


class GeoSeries(GeoPandasBase, Series):
    """A Series object designed to store shapely geometry objects."""
    _metadata = ['name', 'crs']

    def __new__(cls, *args, **kwargs):
        kwargs.pop('crs', None)
        if OLD_PANDAS:
            args = _convert_array_args(args)
            arr = Series.__new__(cls, *args, **kwargs)
        else:
            arr = Series.__new__(cls)
        if type(arr) is GeoSeries:
            return arr
        else:
            return arr.view(GeoSeries)

    def __init__(self, *args, **kwargs):
        if not OLD_PANDAS:
            args = _convert_array_args(args)
        crs = kwargs.pop('crs', None)

        super(GeoSeries, self).__init__(*args, **kwargs)
        self.crs = crs
        self._generate_sindex()

    def append(self, *args, **kwargs):
        return self._wrapped_pandas_method('append', *args, **kwargs)

    @property
    def geometry(self):
        return self

    @classmethod
    def from_file(cls, filename, **kwargs):
        """
        Alternate constructor to create a GeoSeries from a file
        
        Parameters
        ----------
        
        filename : str
            File path or file handle to read from. Depending on which kwargs
            are included, the content of filename may vary, see:
            http://toblerity.github.io/fiona/README.html#usage
            for usage details.
        kwargs : key-word arguments
            These arguments are passed to fiona.open, and can be used to 
            access multi-layer data, data stored within archives (zip files),
            etc.
        
        """
        import fiona
        geoms = []
        with fiona.open(filename, **kwargs) as f:
            crs = f.crs
            for rec in f:
                geoms.append(shape(rec['geometry']))
        g = GeoSeries(geoms)
        g.crs = crs
        return g

    def to_file(self, filename, driver="ESRI Shapefile", **kwargs):
        from geopandas import GeoDataFrame
        data = GeoDataFrame({"geometry": self,
                          "id":self.index.values},
                          index=self.index)
        data.crs = self.crs
        data.to_file(filename, driver, **kwargs)
        
    #
    # Implement pandas methods
    #

    @property
    def _constructor(self):
        return GeoSeries

    def _wrapped_pandas_method(self, mtd, *args, **kwargs):
        """Wrap a generic pandas method to ensure it returns a GeoSeries"""
        val = getattr(super(GeoSeries, self), mtd)(*args, **kwargs)
        if type(val) == Series:
            val.__class__ = GeoSeries
            val.crs = self.crs
            val._generate_sindex()
        return val

    def __getitem__(self, key):
        return self._wrapped_pandas_method('__getitem__', key)

    def sort_index(self, *args, **kwargs):
        return self._wrapped_pandas_method('sort_index', *args, **kwargs)

    def take(self, *args, **kwargs):
        return self._wrapped_pandas_method('take', *args, **kwargs)

    def select(self, *args, **kwargs):
        return self._wrapped_pandas_method('select', *args, **kwargs)

    @property
    def _can_hold_na(self):
        return False

    def __finalize__(self, other, method=None, **kwargs):
        """ propagate metadata from other to self """
        # NOTE: backported from pandas master (upcoming v0.13)
        for name in self._metadata:
            object.__setattr__(self, name, getattr(other, name, None))
        return self

    def copy(self, order='C'):
        """
        Make a copy of this GeoSeries object

        Parameters
        ----------
        deep : boolean, default True
            Make a deep copy, i.e. also copy data

        Returns
        -------
        copy : GeoSeries
        """
        # FIXME: this will likely be unnecessary in pandas >= 0.13
        return GeoSeries(self.values.copy(order), index=self.index,
                      name=self.name).__finalize__(self)

    def isnull(self):
        """Null values in a GeoSeries are represented by empty geometric objects"""
        non_geo_null = super(GeoSeries, self).isnull()
        val = self.apply(_is_empty)
        return np.logical_or(non_geo_null, val)

    def fillna(self, value=None, method=None, inplace=False,
               **kwargs):
        """Fill NA/NaN values with a geometry (empty polygon by default).

        "method" is currently not implemented for pandas <= 0.12.
        """
        if value is None:
            value = Point()
        if not OLD_PANDAS:
            return super(GeoSeries, self).fillna(value=value, method=method,
                                                 inplace=inplace, **kwargs)
        else:
            # FIXME: this is an ugly way to support pandas <= 0.12
            if method is not None:
                raise NotImplementedError('Fill method is currently not implemented for GeoSeries')
            if isinstance(value, BaseGeometry):
                result = self.copy() if not inplace else self
                mask = self.isnull()
                result[mask] = value
                if not inplace:
                    return GeoSeries(result)
            else:
                raise ValueError('Non-geometric fill values not allowed for GeoSeries')

    def align(self, other, join='outer', level=None, copy=True,
              fill_value=None, **kwargs):
        if fill_value is None:
            fill_value = Point()
        left, right = super(GeoSeries, self).align(other, join=join,
                                                   level=level, copy=copy,
                                                   fill_value=fill_value,
                                                   **kwargs)
        if isinstance(other, GeoSeries):
            return GeoSeries(left), GeoSeries(right)
        else: # It is probably a Series, let's keep it that way
            return GeoSeries(left), right


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
        return plot_series(self, *args, **kwargs)

    #
    # Additional methods
    #

    def to_crs(self, crs=None, epsg=None):
        """Transform geometries to a new coordinate reference system

        This method will transform all points in all objects.  It has
        no notion or projecting entire geometries.  All segments
        joining points are assumed to be lines in the current
        projection, not geodesics.  Objects crossing the dateline (or
        other projection boundary) will have undesirable behavior.
        """
        from fiona.crs import from_epsg
        if self.crs is None:
            raise ValueError('Cannot transform naive geometries.  '
                             'Please set a crs on the object first.')
        if crs is None:
            try:
                crs = from_epsg(epsg)
            except TypeError:
                raise TypeError('Must set either crs or epsg for output.')
        proj_in = pyproj.Proj(preserve_units=True, **self.crs)
        proj_out = pyproj.Proj(preserve_units=True, **crs)
        project = partial(pyproj.transform, proj_in, proj_out)
        result = self.apply(lambda geom: transform(project, geom))
        result.__class__ = GeoSeries
        result.crs = crs
        return result

    #
    # Implement standard operators for GeoSeries
    #

    def __xor__(self, other):
        """Implement ^ operator as for builtin set type"""
        return self.symmetric_difference(other)

    def __or__(self, other):
        """Implement | operator as for builtin set type"""
        return self.union(other)

    def __and__(self, other):
        """Implement & operator as for builtin set type"""
        return self.intersection(other)

    def __sub__(self, other):
        """Implement - operator as for builtin set type"""
        return self.difference(other)

GeoSeries._create_indexer('cx', _CoordinateIndexer)
