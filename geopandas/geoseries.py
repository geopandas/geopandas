from warnings import warn
from functools import partial

import numpy as np
from pandas import Series, DataFrame

import pyproj
from shapely.geometry import shape, Polygon, Point
from shapely.geometry.collection import GeometryCollection
from shapely.geometry.base import BaseGeometry
from shapely.ops import cascaded_union, unary_union, transform
import fiona
from fiona.crs import from_epsg

from plotting import plot_series

EMPTY_COLLECTION = GeometryCollection()
EMPTY_POLYGON = Polygon()
EMPTY_POINT = Point()




def _is_empty(x):
    try:
        return x.is_empty
    except:
        return False


def _is_geometry(x):
    return isinstance(x, BaseGeometry)


class GeoSeries(Series):
    """A Series object designed to store shapely geometry objects."""

    def __new__(cls, *args, **kwargs):
        kwargs.pop('crs', None)
        arr = Series.__new__(cls, *args, **kwargs)
        return arr.view(GeoSeries)

    def __init__(self, *args, **kwargs):
        crs = kwargs.pop('crs', None)
        super(GeoSeries, self).__init__(*args, **kwargs)
        self.crs = crs

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
    # Internal methods
    #

    def _geo_op(self, other, op):
        """Operation that returns a GeoSeries"""
        if isinstance(other, GeoSeries):
            if self.crs != other.crs:
                warn('GeoSeries crs mismatch: {} and {}'.format(self.crs, other.crs))
            this, other = self.align(other)
            return GeoSeries([getattr(s[0], op)(s[1]) for s in zip(this, other)],
                          index=this.index, crs=self.crs)
        else:
            return GeoSeries([getattr(s, op)(other) for s in self],
                          index=self.index, crs=self.crs)

    # TODO: think about merging with _geo_op
    def _series_op(self, other, op, **kwargs):
        """Geometric operation that returns a pandas Series"""
        if isinstance(other, GeoSeries):
            this, other = self.align(other)
            return Series([getattr(s[0], op)(s[1], **kwargs) for s in zip(this, other)],
                          index=this.index)
        else:
            return Series([getattr(s, op)(other, **kwargs) for s in self],
                          index=self.index)

    def _geo_unary_op(self, op):
        """Unary operation that returns a GeoSeries"""
        return GeoSeries([getattr(geom, op) for geom in self],
                         index=self.index, crs=self.crs)

    def _series_unary_op(self, op):
        """Unary operation that returns a Series"""
        return GeoSeries([getattr(geom, op) for geom in self],
                         index=self.index)

    #
    # Implementation of Shapely methods
    #

    #
    # Unary operations that return a Series
    #

    @property
    def area(self):
        """Return the area of each geometry in the GeoSeries"""
        return self._series_unary_op('area')

    @property
    def geom_type(self):
        """Return the geometry type of each geometry in the GeoSeries"""
        return self._series_unary_op('geom_type')

    @property
    def type(self):
        """Return the geometry type of each geometry in the GeoSeries"""
        return self.geom_type

    @property
    def length(self):
        """Return the length of each geometry in the GeoSeries"""
        return self._series_unary_op('length')

    @property
    def is_valid(self):
        """Return True for each valid geometry, else False"""
        return self._series_unary_op('is_valid')

    @property
    def is_empty(self):
        """Return True for each empty geometry, False for non-empty"""
        return self._series_unary_op('is_empty')

    @property
    def is_simple(self):
        """Return True for each simple geometry, else False"""
        return self._series_unary_op('is_simple')

    @property
    def is_ring(self):
        """Return True for each geometry that is a closed ring, else False"""
        # operates on the exterior, so can't use _series_unary_op()
        return Series([geom.exterior.is_ring for geom in self],
                      index=self.index)

    #
    # Unary operations that return a GeoSeries
    #

    @property
    def boundary(self):
        """Return the bounding geometry for each geometry"""
        return self._geo_unary_op('boundary')

    @property
    def centroid(self):
        """Return the centroid of each geometry in the GeoSeries"""
        return self._geo_unary_op('centroid')

    @property
    def convex_hull(self):
        """Return the convex hull of each geometry"""
        return self._geo_unary_op('convex_hull')

    @property
    def envelope(self):
        """Return a bounding rectangle for each geometry"""
        return self._geo_unary_op('envelope')

    @property
    def exterior(self):
        """Return the outer boundary of each polygon"""
        # TODO: return empty geometry for non-polygons
        return self._geo_unary_op('exterior')

    @property
    def interiors(self):
        """Return the interior rings of each polygon"""
        # TODO: return empty list or None for non-polygons
        return self._geo_unary_op('interiors')

    def representative_point(self):
        """Return a GeoSeries of points guaranteed to be in each geometry"""
        return GeoSeries([geom.representative_point() for geom in self],
                         index=self.index)

    #
    # Reduction operations that return a Shapely geometry
    #

    @property
    def cascaded_union(self):
        """Deprecated: Return the unary_union of all geometries"""
        return cascaded_union(self.values)

    @property
    def unary_union(self):
        """Return the union of all geometries"""
        return unary_union(self.values)

    #
    # Binary operations that return a GeoSeries
    #

    def difference(self, other):
        """Return the set-theoretic difference of each geometry with *other*"""
        return self._geo_op(other, 'difference')

    def symmetric_difference(self, other):
        """Return the symmetric difference of each geometry with *other*"""
        return self._geo_op(other, 'symmetric_difference')

    def union(self, other):
        """Return the set-theoretic union of each geometry with *other*"""
        return self._geo_op(other, 'union')

    def intersection(self, other):
        """Return the set-theoretic intersection of each geometry with *other*"""
        return self._geo_op(other, 'intersection')

    #
    # Binary operations that return a pandas Series
    #

    def contains(self, other):
        """Return True for all geometries that contain *other*, else False"""
        return self._series_op(other, 'contains')

    def equals(self, other):
        """Return True for all geometries that equal *other*, else False"""
        return self._series_op(other, 'equals')

    def almost_equals(self, other, decimal=6):
        """Return True for all geometries that is approximately equal to *other*, else False"""
        # TODO: pass precision argument
        return self._series_op(other, 'almost_equals', decimal=decimal)

    def equals_exact(self, other, tolerance):
        """Return True for all geometries that equal *other* to a given tolerance, else False"""
        # TODO: pass tolerance argument.
        return self._series_op(other, 'equals_exact', tolerance=tolerance)

    def crosses(self, other):
        """Return True for all geometries that cross *other*, else False"""
        return self._series_op(other, 'crosses')

    def disjoint(self, other):
        """Return True for all geometries that are disjoint with *other*, else False"""
        return self._series_op(other, 'disjoint')

    def intersects(self, other):
        """Return True for all geometries that intersect *other*, else False"""
        return self._series_op(other, 'intersects')

    def overlaps(self, other):
        """Return True for all geometries that overlap *other*, else False"""
        return self._series_op(other, 'overlaps')

    def touches(self, other):
        """Return True for all geometries that touch *other*, else False"""
        return self._series_op(other, 'touches')

    def within(self, other):
        """Return True for all geometries that are within *other*, else False"""
        return self._series_op(other, 'within')

    def distance(self, other):
        """Return distance of each geometry to *other*"""
        return self._series_op(other, 'distance')

    #
    # Other operations
    #

    # should this return bounds for entire series, or elementwise?
    @property
    def bounds(self):
        """Return a DataFrame of minx, miny, maxx, maxy values of geometry objects"""
        bounds = np.array([geom.bounds for geom in self])
        return DataFrame(bounds,
                         columns=['minx', 'miny', 'maxx', 'maxy'],
                         index=self.index)

    def buffer(self, distance, resolution=16):
        return GeoSeries([geom.buffer(distance, resolution) for geom in self],
                         index=self.index)

    def simplify(self, *args, **kwargs):
        return Series([geom.simplify(*args, **kwargs) for geom in self],
                      index=self.index)

    def interpolate(self):
        raise NotImplementedError

    def relate(self, other):
        raise NotImplementedError

    def project(self, *args, **kwargs):
        raise NotImplementedError

    #
    # Implement standard operators for GeoSeries
    #

    def __contains__(self, other):
        """Allow tests of the form "geom in s"

        Tests whether a GeoSeries contains a geometry.

        Note: This is not the same as the geometric method "contains".
        """
        if isinstance(other, BaseGeometry):
            return np.any(self.equals(other))
        else:
            return False

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

    #
    # Implement pandas methods
    #

    @property
    def _can_hold_na(self):
        return False

    def copy(self, order='C'):
        """Return new GeoSeries with copy of underlying values

        Returns
        -------
        cp : GeoSeries
        """
        return GeoSeries(self.values.copy(order), index=self.index,
                      name=self.name)

    def isnull(self):
        """Null values in a GeoSeries are represented by empty geometric objects"""
        non_geo_null = super(GeoSeries, self).isnull()
        val = self.apply(_is_empty)
        return np.logical_or(non_geo_null, val)

    def fillna(self, value=EMPTY_POLYGON, method=None, inplace=False,
               limit=None):
        """Fill NA/NaN values with a geometry (empty polygon by default).

        "method" is currently not implemented for GeoSeries.
        """
        if method is not None:
            raise NotImplementedError('Fill method is currently not implemented for GeoSeries')
        if isinstance(value, BaseGeometry):
            result = self.copy() if not inplace else self
            mask = self.isnull()
            np.putmask(result, mask, value)
            if not inplace:
                return GeoSeries(result)
        else:
            raise ValueError('Non-geometric fill values not allowed for GeoSeries')

    def align(self, other, join='outer', level=None, copy=True,
              fill_value=EMPTY_POLYGON, method=None, limit=None):
        left, right = super(GeoSeries, self).align(other, join=join,
                                                   level=level, copy=copy,
                                                   fill_value=fill_value,
                                                   method=method,
                                                   limit=limit)
        return GeoSeries(left), GeoSeries(right)

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
