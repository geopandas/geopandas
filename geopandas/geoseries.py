from __future__ import absolute_import, division, print_function

from functools import partial
import json
from warnings import warn

import numpy as np
import pandas as pd
from pandas import Series, DataFrame, Index
from pandas.core.indexing import _NDFrameIndexer
from pandas.core.internals import SingleBlockManager

import pyproj
from shapely.geometry import box, shape, Polygon, Point
from shapely.geometry.collection import GeometryCollection
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform

from geopandas.plotting import plot_series
from .vectorized import from_shapely, GeometryArray
from .base import GeoPandasBase, _series_unary_op
from ._block import GeometryBlock


def _is_empty(x):
    try:
        return x.is_empty
    except:
        return False


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
        arr = Series.__new__(cls)
        if type(arr) is GeoSeries:
            return arr
        else:
            return arr.view(GeoSeries)

    def __init__(self, *args, **kwargs):
        # fix problem for scalar geometries passed
        crs = kwargs.pop('crs', None)

        assert len(args) == 1  # for now while prototyping

        arg = args[0]

        if isinstance(arg, SingleBlockManager):
            if isinstance(arg.blocks[0], GeometryBlock):
                super(GeoSeries, self).__init__(args[0], **kwargs)
                self.crs = crs
                return
            else:
                values = np.asarray(args[0].blocks[0].external_values())

        if isinstance(arg, BaseGeometry):
            arg = [arg]

        if isinstance(arg, GeoSeries):
            block = arg._data._block
            index = arg.index
            name = arg.name
        else:
            if isinstance(arg, GeometryArray):
                index = kwargs.pop('index', pd.Index(np.arange(len(arg))))
                name = kwargs.get('name', None)
            else:
                s = pd.Series(arg, **kwargs)
                arg = from_shapely(s.values)
                index = s.index
                name = s.name
            block = GeometryBlock(arg, placement=slice(0, len(arg), 1),
                                  ndim=1)

        super(GeoSeries, self).__init__(block, index=index, name=name,
                                        fastpath=True)
        self.crs = crs
        self._invalidate_sindex()

    def append(self, *args, **kwargs):
        return self._wrapped_pandas_method('append', *args, **kwargs)

    @property
    def geometry(self):
        return self

    @property
    def x(self):
        """Return the x location of point geometries in a GeoSeries"""
        if (self.geom_type == "Point").all():
            return _series_unary_op(self, 'x', null_value=np.nan)
        else:
            message = "x attribute access only provided for Point geometries"
            raise ValueError(message)

    @property
    def y(self):
        """Return the y location of point geometries in a GeoSeries"""
        if (self.geom_type == "Point").all():
            return _series_unary_op(self, 'y', null_value=np.nan)
        else:
            message = "y attribute access only provided for Point geometries"
            raise ValueError(message)

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

    @property
    def __geo_interface__(self):
        """Returns a GeoSeries as a python feature collection
        """
        from geopandas import GeoDataFrame
        return GeoDataFrame({'geometry': self}).__geo_interface__

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
    def _constructor(self, *args, **kwargs):
        return GeoSeries

    def _wrapped_pandas_method(self, mtd, *args, **kwargs):
        """Wrap a generic pandas method to ensure it returns a GeoSeries"""
        val = getattr(super(GeoSeries, self), mtd)(*args, **kwargs)
        if type(val) == Series:
            val.__class__ = GeoSeries
            val.crs = self.crs
            val._invalidate_sindex()
        return val

    def __getitem__(self, key):
        if isinstance(key, (slice, list, Series, np.ndarray)):
            block = self._data._block._getitem(key)
            index = self.index[key]
            return GeoSeries(SingleBlockManager(block, axis=index),
                             crs=self.crs, index=index)
        try:
            if key in self.index:
                loc = self.index.get_loc(key)
                return self._geometry_array[loc]
        except TypeError:
            pass
        raise KeyError(key)

    def __iter__(self):
        return iter(self._geometry_array)

    def to_frame(self):
        from .geodataframe import GeoDataFrame
        name = self.name or 'geometry'
        return GeoDataFrame({name: self}, geometry=name, crs=self.crs)

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
            if not hasattr(self, name):
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
        return GeoSeries(self._data.copy(),
                         index=self.index, crs=self.crs,
                         name=self.name)

    def apply(self, func, *args, **kwargs):
        s = Series(self.values, index=self.index, name=self.name)
        s = s.apply(func, *args, **kwargs)
        if len(s) and isinstance(s.iloc[0], BaseGeometry):
            vec = from_shapely(s.values)
            return GeoSeries(vec, index=self.index)
        else:
            return s

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
        return super(GeoSeries, self).fillna(value=value, method=method,
                                             inplace=inplace, **kwargs)

    def align(self, other, join='outer', level=None, copy=True,
              fill_value=0, **kwargs):
        left, right = super(GeoSeries, self).align(other, join=join,
                                                   level=level, copy=copy,
                                                   fill_value=fill_value,
                                                   **kwargs)
        #left = left.astype(np.uintp)  # TODO: maybe avoid this in pandas
        #right = right.astype(np.uintp)
        #left2 = GeoSeries(left)  # TODO: why do we do this?
        left2 = left
        if isinstance(other, GeoSeries):
            right2 = GeoSeries(right)
            return left2, right2
        else: # It is probably a Series, let's keep it that way
            return left2, right


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
        from geopandas.plotting import plot_series
        return plot_series(self, *args, **kwargs)

    # plot.__doc__ = plot_series.__doc__

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

        `to_crs` passes the `crs` argument to the `Proj` function from the
        `pyproj` library (with the option `preserve_units=True`). It can
        therefore accept proj4 projections in any format
        supported by `Proj`, including dictionaries, or proj4 strings.

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
        proj_in = pyproj.Proj(self.crs, preserve_units=True)
        proj_out = pyproj.Proj(crs, preserve_units=True)
        project = partial(pyproj.transform, proj_in, proj_out)
        result = self.apply(lambda geom: transform(project, geom))
        result.__class__ = GeoSeries
        result.crs = crs
        result._invalidate_sindex()
        return result

    def to_json(self, **kwargs):
        """
        Returns a GeoJSON string representation of the GeoSeries.

        Parameters
        ----------
        *kwargs* that will be passed to json.dumps().
        """
        return json.dumps(self.__geo_interface__, **kwargs)

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

    def _reindex_indexer(self, new_index, indexer, copy):
        """ Overwrites the pd.Series method

        This allows us to use the GeometryArray.take method.
        Otherwise the data gets turned into a numpy array.
        """
        if indexer is None:
            if copy:
                return self.copy()
            return self

        new_values = self._geometry_array.take(indexer)
        return self._constructor(new_values, index=new_index)

GeoSeries._create_indexer('cx', _CoordinateIndexer)
