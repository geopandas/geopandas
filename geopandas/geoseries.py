import numpy as np
from pandas import Series, DataFrame

from shapely.geometry import shape, Polygon, Point
from shapely.geometry.collection import GeometryCollection
from shapely.geometry.base import BaseGeometry
from shapely.ops import cascaded_union, unary_union
import fiona

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
    """
    A Series object designed to store shapely geometry objects.
    """

    def __new__(cls, *args, **kwargs):
        # http://stackoverflow.com/a/11982602/1220158
        arr = Series.__new__(cls, *args, **kwargs)
        return arr.view(GeoSeries)

    def __init__(self, *args, **kwargs):
        super(GeoSeries, self).__init__(*args, **kwargs)
        self.crs = None

    @classmethod
    def from_file(cls, filename):
        """
        Alternate constructor to create a GeoSeries from a file
        """
        geoms = []
        with fiona.open(filename) as f:
            crs = f.crs
            for rec in f:
                geoms.append(shape(rec['geometry']))
        g = GeoSeries(geoms)
        g.crs = crs
        return g

    #
    # Internal methods
    #

    def _geo_op(self, other, op):
        """
        Operation that returns a GeoSeries
        """
        if isinstance(other, GeoSeries):
            this, other = self.align(other)
            return GeoSeries([getattr(s[0], op)(s[1]) for s in zip(this, other)],
                          index=this.index)
        else:
            return GeoSeries([getattr(s, op)(other) for s in self],
                          index=self.index)

    # TODO: think about merging with _geo_op
    def _series_op(self, other, op):
        """
        Geometric operation that returns a pandas Series
        """
        if isinstance(other, GeoSeries):
            this, other = self.align(other)
            return Series([getattr(s[0], op)(s[1]) for s in zip(this, other)],
                          index=this.index)
        else:
            return Series([getattr(s, op)(other) for s in self],
                          index=self.index)

    def _geo_unary_op(self, op):
        """
        Unary operation that returns a GeoSeries
        """
        return GeoSeries([getattr(geom, op) for geom in self],
                         index=self.index)

    def _series_unary_op(self, op):
        """
        Unary operation that returns a Series
        """
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
        """
        Return the area of each member of the GeoSeries
        """
        return self._series_unary_op('area')

    @property
    def geom_type(self):
        return self._series_unary_op('geom_type')

    @property
    def type(self):
        return self.geom_type

    @property
    def length(self):
        return self._series_unary_op('length')

    @property
    def is_valid(self):
        return self._series_unary_op('is_valid')

    @property
    def is_empty(self):
        return self._series_unary_op('is_empty')

    @property
    def is_simple(self):
        return self._series_unary_op('is_simple')

    @property
    def is_ring(self):
        # operates on the exterior, so can't use _series_unary_op()
        return Series([geom.exterior.is_ring for geom in self],
                      index=self.index)

    #
    # Unary operations that return a GeoSeries
    #

    @property
    def boundary(self):
        return self._geo_unary_op('boundary')

    @property
    def centroid(self):
        """
        Return the centroid of each geometry in the GeoSeries
        """
        return self._geo_unary_op('centroid')

    @property
    def convex_hull(self):
        return self._geo_unary_op('convex_hull')

    @property
    def envelope(self):
        return self._geo_unary_op('envelope')

    @property
    def exterior(self):
        return self._geo_unary_op('exterior')

    @property
    def interiors(self):
        return self._geo_unary_op('interiors')

    def representative_point(self):
        return GeoSeries([geom.representative_point() for geom in self],
                         index=self.index)

    #
    # Reduction operations that return a Shapely geometry
    #

    @property
    def cascaded_union(self):
        # Deprecated - use unary_union instead
        return cascaded_union(self.values)

    @property
    def unary_union(self):
        return unary_union(self.values)

    #
    # Binary operations that return a GeoSeries
    #

    def difference(self, other):
        """
        Return a GeoSeries of differences
        Operates on either a GeoSeries or a Shapely geometry
        """
        return self._geo_op(other, 'difference')

    def symmetric_difference(self, other):
        """
        Return a GeoSeries of differences
        Operates on either a GeoSeries or a Shapely geometry
        """
        return self._geo_op(other, 'symmetric_difference')

    def union(self, other):
        """
        Return a GeoSeries of unions
        Operates on either a GeoSeries or a Shapely geometry
        """
        return self._geo_op(other, 'union')

    def intersection(self, other):
        """
        Return a GeoSeries of intersections
        Operates on either a GeoSeries or a Shapely geometry
        """
        return self._geo_op(other, 'intersection')

    #
    # Binary operations that return a pandas Series
    #

    def contains(self, other):
        """
        Return a Series of boolean values.
        Operates on either a GeoSeries or a Shapely geometry
        """
        return self._series_op(other, 'contains')

    def equals(self, other):
        return self._series_op(other, 'equals')

    def almost_equals(self, other):
        return self._series_op(other, 'almost_equals')

    def equals_exact(self, other):
        return self._series_op(other, 'equals_exact')

    def crosses(self, other):
        return self._series_op(other, 'crosses')

    def disjoint(self, other):
        return self._series_op(other, 'disjoint')

    def intersects(self, other):
        return self._series_op(other, 'intersects')

    def overlaps(self, other):
        return self._series_op(other, 'overlaps')

    def touches(self, other):
        return self._series_op(other, 'touches')

    def within(self, other):
        return self._series_op(other, 'within')

    def distance(self, other):
        return self._series_op(other, 'distance')

    #
    # Other operations
    #

    # should this return bounds for entire series, or elementwise?
    @property
    def bounds(self):
        """
        Return a DataFrame of minx, miny, maxx, maxy values of geometry objects
        """
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
        """
        Allow tests of the form "geom in s" to test whether a GeoSeries
        contains a geometry.

        Note: This is not the same as the geometric method "contains".
        """
        if isinstance(other, BaseGeometry):
            return np.any(self.equals(other))
        else:
            return False

    def __xor__(self, other):
        """
        The ^ operator implements symmetric_difference() as it does
        for the builtin set type.
        """
        return self.symmetric_difference(other)

    def __or__(self, other):
        """
        The | operator implements union() as it does
        for the builtin set type.
        """
        return self.union(other)

    def __and__(self, other):
        """
        The & operator implements intersection() as it does
        for the builtin set type.
        """
        return self.intersection(other)

    def __sub__(self, other):
        """
        The - operator implements difference() as it does
        for the builtin set type.
        """
        return self.difference(other)

    #
    # Implement pandas methods
    #

    @property
    def _can_hold_na(self):
        return False

    def copy(self, order='C'):
        """
        Return new GeoSeries with copy of underlying values

        Returns
        -------
        cp : GeoSeries
        """
        return GeoSeries(self.values.copy(order), index=self.index,
                      name=self.name)

    def isnull(self):
        """
        Null values in a GeoSeries are represented by empty geometric objects
        """
        non_geo_null = super(GeoSeries, self).isnull()
        val = self.apply(_is_empty)
        return np.logical_or(non_geo_null, val)

    def fillna(self, value=EMPTY_POLYGON, method=None, inplace=False,
               limit=None):
        """
        Fill NA/NaN values with a geometry (empty polygon by default).

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
