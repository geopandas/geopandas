from __future__ import absolute_import, division, print_function

from functools import partial
import json

import numpy as np

import pandas as pd
from pandas import Series
from pandas.core.internals import SingleBlockManager

import pyproj
from shapely.geometry import shape, Point
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform

from .base import (
    GeoPandasBase, _series_unary_op, _CoordinateIndexer, is_geometry_type)
from .array import from_shapely, GeometryArray, GeometryDtype, _HAS_EXTENSION_ARRAY
from ._block import GeometryBlock


def _is_empty(x):
    try:
        return x.is_empty
    except:
        return False


class GeoSeries(GeoPandasBase, Series):
    """
    A Series object designed to store shapely geometry objects.

    Parameters
    ----------
    data : array-like, dict, scalar value
        The geometries to store in the GeoSeries.
    index : array-like or Index
        The index for the GeoSeries.
    crs : str, dict (optional)
        Coordinate reference system.
    kwargs
        Additional arguments passed to the Series constructor,
         e.g. ``name``.

    Examples
    --------

    >>> from shapely.geometry import Point
    >>> s = GeoSeries([Point(1, 1), Point(2, 2), Point(3, 3)])
    >>> s
    0    POINT (1 1)
    1    POINT (2 2)
    2    POINT (3 3)
    dtype: object

    See Also
    --------
    GeoDataFrame
    pandas.Series

    """
    _metadata = ['name', 'crs']

    def __new__(cls, data=None, index=None, crs=None, **kwargs):
        # we need to use __new__ because we want to return Series instance
        # instead of GeoSeries instance in case of non-geometry data
        if isinstance(data, SingleBlockManager):
            if (isinstance(data.blocks[0], GeometryBlock) 
                    or is_geometry_type(data.blocks[0].dtype)):
                self = super(GeoSeries, cls).__new__(cls)
                super(GeoSeries, self).__init__(data, index=index, **kwargs)
                self.crs = crs
                return self
            # GeometryBlock sometimes gets converted by pandas
            # to ObjectBlock with GeometryArray
            if index is not None:
                data = data.reindex_axis(index, axis=0)
            else:
                index = data.index
            if isinstance(data.blocks[0].values, GeometryArray):
                data = data.blocks[0].values
            else:
                data = np.asarray(data.blocks[0].external_values())

        if isinstance(data, BaseGeometry):
            # fix problem for scalar geometries passed, ensure the list of
            # scalars is of correct length if index is specified
            n = len(index) if index is not None else 1
            data = [data] * n

        if not _HAS_EXTENSION_ARRAY:
            if (isinstance(data, GeoSeries)
                    or (isinstance(data, Series) and isinstance(data._data._block,
                                                                GeometryBlock))):
                block = data._data._block
                index = data.index
                name = data.name
            else:
                if isinstance(data, GeometryArray):
                    index = index if index is not None else pd.Index(np.arange(len(data)))
                    name = kwargs.get('name', None)
                else:
                    s = pd.Series(data, index=index, **kwargs)
                    # prevent trying to convert non-geometry objects
                    if s.dtype != object and not s.empty:
                        return s
                    # try to convert to GeometryArray, if fails return plain Series
                    try:
                        data = from_shapely(s.values)
                    except TypeError:
                        return s
                    index = s.index
                    name = s.name
                block = GeometryBlock(data, placement=slice(0, len(data), 1),
                                    ndim=1)

            self = super(GeoSeries, cls).__new__(cls)
            super(GeoSeries, self).__init__(block, index=index, name=name,
                                            fastpath=True)
            self.crs = crs
            self._invalidate_sindex()
            return self

        name = kwargs.pop('name', None)
        if not is_geometry_type(data):
            s = pd.Series(data, index=index, name=name, **kwargs)
            # prevent trying to convert non-geometry objects
            if s.dtype != object and not s.empty:
                return s
            # try to convert to GeometryArray, if fails return plain Series
            try:
                data = from_shapely(s.values)
            except TypeError:
                return s
            index = s.index
            name = s.name

        self = super(GeoSeries, cls).__new__(cls)
        super(GeoSeries, self).__init__(data, index=index, name=name, **kwargs)

        self.crs = crs
        self._invalidate_sindex()
        return self

    def __init__(self, *args, **kwargs):
        # need to overwrite Series init to prevent converting the
        # manually constructed GeometryBlock back to object block
        # by calling the Series init
        pass

    def append(self, *args):
        from .geodataframe import concat
        return concat((self,) + args)

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
        """Alternate constructor to create a ``GeoSeries`` from a file.

        Can load a ``GeoSeries`` from a file from any format recognized by
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
        """Returns a ``GeoSeries`` as a python feature collection.

        Implements the `geo_interface`. The returned python data structure
        represents the ``GeoSeries`` as a GeoJSON-like ``FeatureCollection``.
        Note that the features will have an empty ``properties`` dict as they
        don't have associated attributes (geometry only).
        """
        from geopandas import GeoDataFrame
        return GeoDataFrame({'geometry': self}).__geo_interface__

    def to_file(self, filename, driver="ESRI Shapefile", **kwargs):
        from geopandas import GeoDataFrame
        data = GeoDataFrame({"geometry": self,
                             "id": self.index.values},
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

        if _HAS_EXTENSION_ARRAY:
            return super(GeoSeries, self).__getitem__(key)
        else:
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

    def isna(self):
        """
        N/A values in a GeoSeries can be represented by empty geometric
        objects, in addition to standard representations such as None and
        np.nan.

        Returns
        -------
        A boolean pandas Series of the same size as the GeoSeries,
        True where a value is N/A.

        See Also
        --------
        GeoSereies.notna : inverse of isna

        """
        return pd.Series(self._geometry_array.data == 0, index=self.index,
                         name=self.name)

    def isnull(self):
        """Alias for `isna` method. See `isna` for more detail."""
        return self.isna()

    def notna(self):
        """
        N/A values in a GeoSeries can be represented by empty geometric
        objects, in addition to standard representations such as None and
        np.nan.

        Returns
        -------
        A boolean pandas Series of the same size as the GeoSeries,
        False where a value is N/A.

        See Also
        --------
        GeoSeries.isna : inverse of notna
        """
        return ~self.isna()

    def notnull(self):
        """Alias for `notna` method. See `notna` for more detail."""
        return self.notna()

    def fillna(self, value=None):
        """ Fill NA/NaN values with a geometry (empty polygon by default) """
        if value is None:
            value = BaseGeometry()
        return GeoSeries(self._geometry_array.fillna(value), index=self.index,
                         crs=self.crs, name=self.name)

    def dropna(self):
        """ Drop NA/NaN values

        Note: the inplace keyword is not currently supported.
        """
        return GeoSeries(self._geometry_array[~self.isna()],
                         index=self.index[~self.isna()],
                         crs=self.crs, name=self.name)

    def align(self, other, join='outer', level=None, copy=True,
              fill_value=None, **kwargs):
        left, right = super(GeoSeries, self).align(other, join=join,
                                                   level=level, copy=copy,
                                                   fill_value=fill_value,
                                                   **kwargs)
        return left, right

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
        """Generate a plot of the geometries in the ``GeoSeries``.

        Wraps the ``plot_series()`` function, and documentation is copied from
        there.
        """
        from geopandas.plotting import plot_series
        return plot_series(self, *args, **kwargs)

    # plot.__doc__ = plot_series.__doc__

    #
    # Additional methods
    #

    def to_crs(self, crs=None, epsg=None):
        """Returns a ``GeoSeries`` with all geometries transformed to a new
        coordinate reference system.

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

    def __eq__(self, other):
        return self.geom_equals(other)

    def __ne__(self, other):
        return ~self.geom_equals(other)

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
