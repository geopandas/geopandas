from warnings import warn

from shapely.geometry import MultiPoint, MultiLineString, MultiPolygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import cascaded_union, unary_union
import shapely.affinity as affinity

import numpy as np
from pandas import Series, DataFrame

import geopandas as gpd


def _geo_op(this, other, op):
    """Operation that returns a GeoSeries"""
    if isinstance(other, GeoPandasBase):
        this = this.geometry
        crs = this.crs
        if crs != other.crs:
            warn('GeoSeries crs mismatch: {0} and {1}'.format(this.crs,
                                                              other.crs))
        this, other = this.align(other.geometry)
        return gpd.GeoSeries([getattr(this_elem, op)(other_elem)
                             for this_elem, other_elem in zip(this, other)],
                             index=this.index, crs=crs)
    else:
        return gpd.GeoSeries([getattr(s, op)(other)
                             for s in this.geometry],
                             index=this.index, crs=this.crs)


# TODO: think about merging with _geo_op
def _series_op(this, other, op, **kwargs):
    """Geometric operation that returns a pandas Series"""
    if isinstance(other, GeoPandasBase):
        this = this.geometry
        this, other = this.align(other.geometry)
        return Series([getattr(this_elem, op)(other_elem, **kwargs)
                      for this_elem, other_elem in zip(this, other)],
                      index=this.index)
    else:
        return Series([getattr(s, op)(other, **kwargs)
                      for s in this.geometry], index=this.index)

def _geo_unary_op(this, op):
    """Unary operation that returns a GeoSeries"""
    return gpd.GeoSeries([getattr(geom, op) for geom in this.geometry],
                     index=this.index, crs=this.crs)

def _series_unary_op(this, op):
    """Unary operation that returns a Series"""
    return Series([getattr(geom, op) for geom in this.geometry],
                     index=this.index)


class GeoPandasBase(object):
    @property
    def area(self):
        """Return the area of each geometry in the GeoSeries"""
        return _series_unary_op(self, 'area')

    @property
    def geom_type(self):
        """Return the geometry type of each geometry in the GeoSeries"""
        return _series_unary_op(self, 'geom_type')

    @property
    def type(self):
        """Return the geometry type of each geometry in the GeoSeries"""
        return self.geom_type

    @property
    def length(self):
        """Return the length of each geometry in the GeoSeries"""
        return _series_unary_op(self, 'length')

    @property
    def is_valid(self):
        """Return True for each valid geometry, else False"""
        return _series_unary_op(self, 'is_valid')

    @property
    def is_empty(self):
        """Return True for each empty geometry, False for non-empty"""
        return _series_unary_op(self, 'is_empty')

    @property
    def is_simple(self):
        """Return True for each simple geometry, else False"""
        return _series_unary_op(self, 'is_simple')

    @property
    def is_ring(self):
        """Return True for each geometry that is a closed ring, else False"""
        # operates on the exterior, so can't use _series_unary_op()
        return Series([geom.exterior.is_ring for geom in self.geometry],
                      index=self.index)

    #
    # Unary operations that return a GeoSeries
    #

    @property
    def boundary(self):
        """Return the bounding geometry for each geometry"""
        return _geo_unary_op(self, 'boundary')

    @property
    def centroid(self):
        """Return the centroid of each geometry in the GeoSeries"""
        return _geo_unary_op(self, 'centroid')

    @property
    def convex_hull(self):
        """Return the convex hull of each geometry"""
        return _geo_unary_op(self, 'convex_hull')

    @property
    def envelope(self):
        """Return a bounding rectangle for each geometry"""
        return _geo_unary_op(self, 'envelope')

    @property
    def exterior(self):
        """Return the outer boundary of each polygon"""
        # TODO: return empty geometry for non-polygons
        return _geo_unary_op(self, 'exterior')

    @property
    def interiors(self):
        """Return the interior rings of each polygon"""
        # TODO: return empty list or None for non-polygons
        return _geo_unary_op(self, 'interiors')

    def representative_point(self):
        """Return a GeoSeries of points guaranteed to be in each geometry"""
        return gpd.GeoSeries([geom.representative_point()
                             for geom in self.geometry],
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
    # Binary operations that return a pandas Series
    #

    def contains(self, other):
        """Return True for all geometries that contain *other*, else False"""
        return _series_op(self, other, 'contains')

    def equals(self, other):
        """Return True for all geometries that equal *other*, else False"""
        return _series_op(self, other, 'equals')

    def almost_equals(self, other, decimal=6):
        """Return True for all geometries that is approximately equal to *other*, else False"""
        # TODO: pass precision argument
        return _series_op(self, other, 'almost_equals', decimal=decimal)

    def equals_exact(self, other, tolerance):
        """Return True for all geometries that equal *other* to a given tolerance, else False"""
        # TODO: pass tolerance argument.
        return _series_op(self, other, 'equals_exact', tolerance=tolerance)

    def crosses(self, other):
        """Return True for all geometries that cross *other*, else False"""
        return _series_op(self, other, 'crosses')

    def disjoint(self, other):
        """Return True for all geometries that are disjoint with *other*, else False"""
        return _series_op(self, other, 'disjoint')

    def intersects(self, other):
        """Return True for all geometries that intersect *other*, else False"""
        return _series_op(self, other, 'intersects')

    def overlaps(self, other):
        """Return True for all geometries that overlap *other*, else False"""
        return _series_op(self, other, 'overlaps')

    def touches(self, other):
        """Return True for all geometries that touch *other*, else False"""
        return _series_op(self, other, 'touches')

    def within(self, other):
        """Return True for all geometries that are within *other*, else False"""
        return _series_op(self, other, 'within')

    def distance(self, other):
        """Return distance of each geometry to *other*"""
        return _series_op(self, other, 'distance')

    #
    # Binary operations that return a GeoSeries
    #

    def difference(self, other):
        """Return the set-theoretic difference of each geometry with *other*"""
        return _geo_op(self, other, 'difference')

    def symmetric_difference(self, other):
        """Return the symmetric difference of each geometry with *other*"""
        return _geo_op(self, other, 'symmetric_difference')

    def union(self, other):
        """Return the set-theoretic union of each geometry with *other*"""
        return _geo_op(self, other, 'union')

    def intersection(self, other):
        """Return the set-theoretic intersection of each geometry with *other*"""
        return _geo_op(self, other, 'intersection')

    #
    # Other operations
    #

    @property
    def bounds(self):
        """Return a DataFrame of minx, miny, maxx, maxy values of geometry objects"""
        bounds = np.array([geom.bounds for geom in self.geometry])
        return DataFrame(bounds,
                         columns=['minx', 'miny', 'maxx', 'maxy'],
                         index=self.index)
                         
    @property
    def total_bounds(self):
        """Return a single bounding box (minx, miny, maxx, maxy) for all geometries

        This is a shortcut for calculating the min/max x and y bounds individually.
        """

        b = self.bounds
        return (b['minx'].min(),
                b['miny'].min(),
                b['maxx'].max(),
                b['maxy'].max())

    def buffer(self, distance, resolution=16):
        return gpd.GeoSeries([geom.buffer(distance, resolution) 
                             for geom in self.geometry],
                         index=self.index, crs=self.crs)

    def simplify(self, *args, **kwargs):
        return gpd.GeoSeries([geom.simplify(*args, **kwargs)
                             for geom in self.geometry],
                      index=self.index, crs=self.crs)

    def relate(self, other):
        raise NotImplementedError

    def project(self, other, normalized=False):
        """
        Return the distance along each geometry nearest to *other*
        
        Parameters
        ----------
        other : BaseGeometry or GeoSeries
            The *other* geometry to computed projected point from.
        normalized : boolean
            If normalized is True, return the distance normalized to
            the length of the object.
        
        The project method is the inverse of interpolate.
        """
        
        return _series_op(self, other, 'project', normalized=normalized)

    def interpolate(self, distance, normalized=False):
        """
        Return a point at the specified distance along each geometry
        
        Parameters
        ----------
        distance : float or Series of floats
            Distance(s) along the geometries at which a point should be returned
        normalized : boolean
            If normalized is True, distance will be interpreted as a fraction 
            of the geometric object's length.
        """
        
        return gpd.GeoSeries([s.interpolate(distance, normalized) 
                             for s in self.geometry],
            index=self.index, crs=self.crs)
        
    def translate(self, xoff=0.0, yoff=0.0, zoff=0.0):
        """
        Shift the coordinates of the GeoSeries.

        Parameters
        ----------
        xoff, yoff, zoff : float, float, float
            Amount of offset along each dimension.
            xoff, yoff, and zoff for translation along the x, y, and z 
            dimensions respectively.

        See shapely manual for more information:
        http://toblerity.org/shapely/manual.html#affine-transformations
        """

        return gpd.GeoSeries([affinity.translate(s, xoff, yoff, zoff) 
                             for s in self.geometry], 
            index=self.index, crs=self.crs)

    # Shift is simply an alias for translate
    shift = translate

    def rotate(self, angle, origin='center', use_radians=False):
        """
        Rotate the coordinates of the GeoSeries.
        
        Parameters
        ----------
        angle : float
            The angle of rotation can be specified in either degrees (default) 
            or radians by setting use_radians=True. Positive angles are 
            counter-clockwise and negative are clockwise rotations.
        origin : string, Point, or tuple (x, y)
            The point of origin can be a keyword 'center' for the bounding box 
            center (default), 'centroid' for the geometry's centroid, a Point 
            object or a coordinate tuple (x, y).
        use_radians : boolean
            Whether to interpret the angle of rotation as degrees or radians
            
        See shapely manual for more information:
        http://toblerity.org/shapely/manual.html#affine-transformations
        """

        return gpd.GeoSeries([affinity.rotate(s, angle, origin=origin, 
            use_radians=use_radians) for s in self.geometry],
            index=self.index, crs=self.crs)

    def scale(self, xfact=1.0, yfact=1.0, zfact=1.0, origin='center'):
        """
        Scale the geometries of the GeoSeries along each (x, y, z) dimension.

        Parameters
        ----------
        xfact, yfact, zfact : float, float, float
            Scaling factors for the x, y, and z dimensions respectively.
        origin : string, Point, or tuple
            The point of origin can be a keyword 'center' for the 2D bounding 
            box center (default), 'centroid' for the geometry's 2D centroid, a 
            Point object or a coordinate tuple (x, y, z).

        Note: Negative scale factors will mirror or reflect coordinates.

        See shapely manual for more information:
        http://toblerity.org/shapely/manual.html#affine-transformations
        """

        return gpd.GeoSeries([affinity.scale(s, xfact, yfact, zfact, 
            origin=origin) for s in self.geometry], index=self.index, 
            crs=self.crs)
                           
    def skew(self, xs=0.0, ys=0.0, origin='center', use_radians=False):
        """
        Shear/Skew the geometries of the GeoSeries by angles along x and y dimensions.
        
        Parameters
        ----------
        xs, ys : float, float
            The shear angle(s) for the x and y axes respectively. These can be 
            specified in either degrees (default) or radians by setting 
            use_radians=True.
        origin : string, Point, or tuple (x, y)
            The point of origin can be a keyword 'center' for the bounding box 
            center (default), 'centroid' for the geometry's centroid, a Point 
            object or a coordinate tuple (x, y).
        use_radians : boolean
            Whether to interpret the shear angle(s) as degrees or radians
            
        See shapely manual for more information:
        http://toblerity.org/shapely/manual.html#affine-transformations
        """
        
        return gpd.GeoSeries([affinity.skew(s, xs, ys, origin=origin, 
            use_radians=use_radians) for s in self.geometry],
            index=self.index, crs=self.crs)


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


def _array_input(arr):
    if isinstance(arr, (MultiPoint, MultiLineString, MultiPolygon)):
        # Prevent against improper length detection when input is a
        # Multi*
        geom = arr
        arr = np.empty(1, dtype=object)
        arr[0] = geom

    return arr


