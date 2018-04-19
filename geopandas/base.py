from __future__ import absolute_import, division, print_function

from warnings import warn

import numpy as np
import pandas as pd
from pandas import Series, DataFrame, MultiIndex
from pandas.core.indexing import _NDFrameIndexer

from shapely.geometry.base import BaseGeometry, CAP_STYLE, JOIN_STYLE
from shapely.geometry import box, MultiPoint, MultiLineString, MultiPolygon
from shapely.ops import cascaded_union
import shapely.affinity as affinity

from . import vectorized
from . import array
from .array import GeometryArray, GeometryDtype, _HAS_EXTENSION_ARRAY
from ._block import GeometryBlock

try:
    from rtree.core import RTreeError
    HAS_SINDEX = True
except ImportError:
    class RTreeError(Exception):
        pass
    HAS_SINDEX = False


def is_geometry_type(data):

    if _HAS_EXTENSION_ARRAY:
        if isinstance(getattr(data, 'dtype', None), GeometryDtype):
            # GeometryArray and Series[GeometryArray]
            return True
        else:
            return False
    else:
        if isinstance(data, GeometryArray):
            return True
        elif (isinstance(data, Series)
                and isinstance(data._data._block, GeometryBlock)):
            return True
        else:
            return False


def binary_geo(op, left, right):
    """ Binary operation on GeoSeries objects that returns a GeoSeries """
    from .geoseries import GeoSeries
    if isinstance(right, GeoPandasBase):
        left = left.geometry
        left, right = left.align(right.geometry)
        t = left._geometry_array
        o = right._geometry_array
        x = vectorized.vector_binary_geo(op, t.data, o.data)
        return GeoSeries(GeometryArray(x), index=left.index, crs=left.crs)
    elif isinstance(right, BaseGeometry):
        x = vectorized.binary_geo(op, left._geometry_array.data, right)
        return GeoSeries(GeometryArray(x), index=left.index, crs=left.crs)
    else:
        raise TypeError(type(left), type(right))


def binary_predicate(op, this, other, *args):
    """ Binary operation on GeoSeries objects that returns a boolean Series """
    if isinstance(other, GeoPandasBase):
        this = this.geometry
        this, other = this.align(other.geometry)
        t = this._geometry_array
        o = other._geometry_array
        if args:
            x = vectorized.vector_binary_predicate_with_arg(op, t.data, o.data, *args)
        else:
            x = vectorized.vector_binary_predicate(op, t.data, o.data)
        return Series(x, index=this.index)
    elif isinstance(other, BaseGeometry):
        if args:
            x = vectorized.binary_predicate_with_arg(op, this._geometry_array.data, other, *args)
        elif op in array.opposite_predicates:
            op2 = array.opposite_predicates[op]
            x = vectorized.prepared_binary_predicate(op2, this._geometry_array.data, other)
        else:
            x = vectorized.binary_predicate(op, this._geometry_array.data, other)
        return Series(x, index=this.index)
    else:
        raise TypeError(type(this), type(other))


def binary_float(op, this, other, *args):
    """ Binary operation on GeoSeries objects that returns a boolean Series """
    if isinstance(other, GeoPandasBase):
        this = this.geometry
        this, other = this.align(other.geometry)
        t = this._geometry_array
        o = other._geometry_array
        x = vectorized.binary_vector_float(op, t.data, o.data)
        return Series(x, index=this.index)
    elif isinstance(other, BaseGeometry):
        x = vectorized.binary_float(op, this._geometry_array.data, other)
        return Series(x, index=this.index)
    else:
        raise TypeError(type(this), type(other))


def prepared_binary_predicate(op, this, other):
    """Geometric operation that returns a pandas Series"""
    if isinstance(other, BaseGeometry):
        x = vectorized.prepared_binary_predicate(this._geometry_array, other)
        return Series(x, index=this.index)
    else:
        raise TypeError(type(this), type(other))


def _geo_unary_op(this, op):
    """Unary operation that returns a GeoSeries"""
    from .geoseries import GeoSeries
    return GeoSeries([getattr(geom, op) for geom in this.geometry],
                     index=this.index, crs=this.crs)


def _series_unary_op(this, op, null_value=False):
    """Unary operation that returns a Series"""
    return Series([getattr(geom, op, null_value) for geom in this.geometry],
                     index=this.index, dtype=np.dtype(type(null_value)))


class GeoPandasBase(object):
    _sindex = None
    _sindex_generated = False

    def _generate_sindex(self):
        if not HAS_SINDEX:
            warn("Cannot generate spatial index: Missing package `rtree`.")
        else:
            from geopandas.sindex import SpatialIndex
            stream = ((i, item.bounds, idx) for i, (idx, item) in
                   enumerate(self.geometry.iteritems()) if
                   pd.notnull(item) and not item.is_empty)
            try:
                self._sindex = SpatialIndex(stream)
            # What we really want here is an empty generator error, or
            # for the bulk loader to log that the generator was empty
            # and move on. See https://github.com/Toblerity/rtree/issues/20.
            except RTreeError:
                pass
        self._sindex_generated = True

    def _invalidate_sindex(self):
        """
        Indicates that the spatial index should be re-built next
        time it's requested.

        """
        self._sindex = None
        self._sindex_generated = False

    @property
    def area(self):
        """Returns a ``Series`` containing the area of each geometry in the
        ``GeoSeries``."""
        x = vectorized.unary_vector_float('area', self._geometry_array.data)
        return Series(x, index=self.index)

    @property
    def geom_type(self):
        """Returns a ``Series`` of strings specifying the `Geometry Type` of each
        object."""
        cat = self._geometry_array.geom_type()
        return Series(cat, index=self.index)

    @property
    def type(self):
        """Return the geometry type of each geometry in the GeoSeries"""
        return self.geom_type

    @property
    def length(self):
        """Returns a ``Series`` containing the length of each geometry."""
        x = vectorized.unary_vector_float('length', self._geometry_array.data)
        return Series(x, index=self.index)

    @property
    def is_valid(self):
        """Returns a ``Series`` of ``dtype('bool')`` with value ``True`` for
        geometries that are valid."""
        x = vectorized.unary_predicate('is_valid', self._geometry_array.data)
        return Series(x, index=self.index)

    @property
    def is_empty(self):
        """Returns a ``Series`` of ``dtype('bool')`` with value ``True`` for
        empty geometries."""
        x = vectorized.unary_predicate('is_empty', self._geometry_array.data)
        return Series(x, index=self.index)

    @property
    def is_simple(self):
        """Returns a ``Series`` of ``dtype('bool')`` with value ``True`` for
        geometries that do not cross themselves.

        This is meaningful only for `LineStrings` and `LinearRings`.
        """
        x = vectorized.unary_predicate('is_simple', self._geometry_array.data)
        return Series(x, index=self.index)

    @property
    def is_ring(self):
        """Returns a ``Series`` of ``dtype('bool')`` with value ``True`` for
        features that are closed."""
        # operates on the exterior, so can't use _series_unary_op()
        x = vectorized.unary_predicate('is_ring', self._geometry_array.data)
        return Series(x, index=self.index)

    @property
    def has_z(self):
        """Returns a ``Series`` of ``dtype('bool')`` with value ``True`` for
        features that have a z-component."""
        # operates on the exterior, so can't use _series_unary_op()
        return _series_unary_op(self, 'has_z', null_value=False)

    #
    # Unary operations that return a GeoSeries
    #

    def unary_geo(self, op):
        from .geoseries import GeoSeries
        x = vectorized.geo_unary_op(op, self._geometry_array.data)
        return GeoSeries(GeometryArray(x), index=self.index, crs=self.crs)

    def coords(self):
        x = vectorized.coords(self._geometry_array.data)
        return Series(x, index=self.index)

    @property
    def boundary(self):
        """Returns a ``GeoSeries`` of lower dimensional objects representing
        each geometries's set-theoretic `boundary`."""
        return self.unary_geo('boundary')

    @property
    def centroid(self):
        """Returns a ``GeoSeries`` of points representing the centroid of each
        geometry."""
        return self.unary_geo('centroid')

    @property
    def convex_hull(self):
        """
        Returns a ``GeoSeries`` of geometries representing the convex hull
        of each geometry.

        The convex hull of a geometry is the smallest convex `Polygon`
        containing all the points in each geometry, unless the number of points
        in the geometric object is less than three. For two points, the convex
        hull collapses to a `LineString`; for 1, a `Point`.
        """
        return self.unary_geo('convex_hull')

    @property
    def envelope(self):
        """
        Returns a ``GeoSeries`` of geometries representing the envelope of
        each geometry.

        The envelope of a geometry is the bounding rectangle. That is, the
        point or smallest rectangular polygon (with sides parallel to the
        coordinate axes) that contains the geometry.
        """
        return self.unary_geo('envelope')

    @property
    def exterior(self):
        """Returns a ``GeoSeries`` of LinearRings representing the outer
        boundary of each polygon in the GeoSeries.

        Applies to GeoSeries containing only Polygons.
        """
        # TODO: return empty geometry for non-polygons
        out = self.unary_geo('exterior')
        out._geometry_array.base = self._geometry_array
        return out  # exterior shares data with self

    @property
    def interiors(self):
        """Returns a ``GeoSeries`` of InteriorRingSequences representing the
        inner rings of each polygon in the GeoSeries.

        Applies to GeoSeries containing only Polygons.
        """
        # TODO: return empty list or None for non-polygons
        return _series_unary_op(self, 'interiors', null_value=False)

    def representative_point(self):
        """Returns a ``GeoSeries`` of (cheaply computed) points that are
        guaranteed to be within each geometry.
        """
        return self.unary_geo('representative_point')

    #
    # Reduction operations that return a Shapely geometry
    #

    @property
    def cascaded_union(self):
        """Deprecated: Return the unary_union of all geometries"""
        return cascaded_union(np.array(self.geometry.values))

    @property
    def unary_union(self):
        """Returns a geometry containing the union of all geometries in the
        ``GeoSeries``."""
        return self._geometry_array.unary_union()

    #
    # Binary operations that return a pandas Series
    #

    @property
    def _geometry_array(self):
        return self.geometry._values

    def contains(self, other):
        """
        Returns a ``Series`` of ``dtype('bool')`` with value ``True`` for
        each geometry that contains `other`.

        An object is said to contain `other` if its `interior` contains the
        `boundary` and `interior` of the other object and their boundaries do
        not touch at all.

        This is the inverse of :meth:`within` in the sense that the expression
        ``a.contains(b) == b.within(a)`` always evaluates to ``True``.

        Parameters
        ----------
        other : GeoSeries or geometric object
            The GeoSeries (elementwise) or geometric object to test if is
            contained.
        """
        return binary_predicate('contains', self, other)

    def geom_equals(self, other):
        """
        Returns a ``Series`` of ``dtype('bool')`` with value ``True`` for
        each geometry equal to `other`.

        An object is said to be equal to `other` if its set-theoretic
        `boundary`, `interior`, and `exterior` coincides with those of the
        other.

        Parameters
        ----------
        other : GeoSeries or geometric object
            The GeoSeries (elementwise) or geometric object to test for
            equality.
        """
        try:
            return binary_predicate('equals', self, other)
        except TypeError:
            raise TypeError("GeoSeries equality checks are only supported "
                            "with shapely geometries or other GeoSeries, "
                            "not type {}".format(type(other)))

    def geom_almost_equals(self, other, decimal=6):
        """
        Returns a ``Series`` of ``dtype('bool')`` with value ``True`` if
        each geometry is approximately equal to `other`.

        Approximate equality is tested at all points to the specified `decimal`
        place precision.  See also :meth:`equals`.

        Parameters
        ----------
        other : GeoSeries or geometric object
            The GeoSeries (elementwise) or geometric object to compare to.
        decimal : int
            Decimal place presion used when testing for approximate equality.
        """
        # TODO: pass precision argument
        tolerance = 0.5 * 10**(-decimal)
        return binary_predicate('equals_exact', self, other, tolerance)

    def geom_equals_exact(self, other, tolerance):
        """Return True for all geometries that equal *other* to a given
        tolerance, else False"""
        # TODO: pass tolerance argument.
        return binary_predicate('equals_exact', self, other, tolerance)

    def crosses(self, other):
        """
        Returns a ``Series`` of ``dtype('bool')`` with value ``True`` for
        each geometry that cross `other`.

        An object is said to cross `other` if its `interior` intersects the
        `interior` of the other but does not contain it, and the dimension of
        the intersection is less than the dimension of the one or the other.

        Parameters
        ----------
        other : GeoSeries or geometric object
            The GeoSeries (elementwise) or geometric object to test if is
            crossed.
        """
        return binary_predicate('crosses', self, other)

    def disjoint(self, other):
        """
        Returns a ``Series`` of ``dtype('bool')`` with value ``True`` for
        each geometry disjoint to `other`.

        An object is said to be disjoint to `other` if its `boundary` and
        `interior` does not intersect at all with those of the other.

        Parameters
        ----------
        other : GeoSeries or geometric object
            The GeoSeries (elementwise) or geometric object to test if is
            disjoint.
        """
        return binary_predicate('disjoint', self, other)

    def intersects(self, other):
        """
        Returns a ``Series`` of ``dtype('bool')`` with value ``True`` for
        each geometry that intersects `other`.

        An object is said to intersect `other` if its `boundary` and `interior`
        intersects in any way with those of the other.

        Parameters
        ----------
        other : GeoSeries or geometric object
            The GeoSeries (elementwise) or geometric object to test if is
            intersected.
        """
        return binary_predicate('intersects', self, other)

    def overlaps(self, other):
        """Return True for all geometries that overlap *other*, else False"""
        return binary_predicate('overlaps', self, other)

    def touches(self, other):
        """
        Returns a ``Series`` of ``dtype('bool')`` with value ``True`` for
        each geometry that touches `other`.

        An object is said to touch `other` if it has at least one point in
        common with `other` and its interior does not intersect with any part
        of the other.

        Parameters
        ----------
        other : GeoSeries or geometric object
            The GeoSeries (elementwise) or geometric object to test if is
            touched.
        """
        return binary_predicate('touches', self, other)

    def within(self, other):
        """
        Returns a ``Series`` of ``dtype('bool')`` with value ``True`` for
        each geometry that is within `other`.

        An object is said to be within `other` if its `boundary` and `interior`
        intersects only with the `interior` of the other (not its `boundary` or
        `exterior`).

        This is the inverse of :meth:`contains` in the sense that the
        expression ``a.within(b) == b.contains(a)`` always evaluates to
        ``True``.

        Parameters
        ----------
        other : GeoSeries or geometric object
            The GeoSeries (elementwise) or geometric object to test if each
            geometry is within.

        """
        return binary_predicate('within', self, other)

    def distance(self, other):
        """
        Returns a ``Series`` containing the distance to `other`.

        Parameters
        ----------
        other : Geoseries or geometric object
            The Geoseries (elementwise) or geometric object to find the
            distance to.
        """
        return binary_float('distance', self, other)

    #
    # Binary operations that return a GeoSeries
    #

    def difference(self, other):
        """
        Returns a ``GeoSeries`` of the points in each geometry that
        are not in `other` (set-theoretic difference).

        Parameters
        ----------
        other : Geoseries or geometric object
            The Geoseries (elementwise) or geometric object to find the
            difference to.
        """
        return binary_geo('difference', self, other)

    def symmetric_difference(self, other):
        """
        Returns a ``GeoSeries`` of the symmetric difference of points in
        each geometry with `other`.

        For each geometry, the symmetric difference consists of points in the
        geometry not in `other`, and points in `other` not in the geometry.

        Parameters
        ----------
        other : Geoseries or geometric object
            The Geoseries (elementwise) or geometric object to find the
            symmetric difference to.
        """
        return binary_geo('symmetric_difference', self, other)

    def union(self, other):
        """
        Returns a ``GeoSeries`` of the union of points in each geometry with
        `other`.

        Parameters
        ----------
        other : Geoseries or geometric object
            The Geoseries (elementwise) or geometric object to find the union
            with.
        """
        return binary_geo('union', self, other)

    def intersection(self, other):
        """
        Returns a ``GeoSeries`` of the intersection of points in each
        geometry with `other`.

        Parameters
        ----------
        other : Geoseries or geometric object
            The Geoseries (elementwise) or geometric object to find the
            intersection with.
        """
        return binary_geo('intersection', self, other)

    #
    # Other operations
    #

    @property
    def bounds(self):
        """Returns a ``DataFrame`` with columns ``minx``, ``miny``, ``maxx``,
        ``maxy`` values containing the bounds for each geometry.

        See ``GeoSeries.total_bounds`` for the limits of the entire series.
        """
        bounds = np.array([geom.bounds for geom in self.geometry])
        return DataFrame(bounds,
                         columns=['minx', 'miny', 'maxx', 'maxy'],
                         index=self.index)

    @property
    def total_bounds(self):
        """Returns a tuple containing ``minx``, ``miny``, ``maxx``, ``maxy``
        values for the bounds of the series as a whole.

        See ``GeoSeries.bounds`` for the bounds of the geometries contained in
        the series.
        """
        b = self.bounds
        return np.array((b['minx'].min(),
                         b['miny'].min(),
                         b['maxx'].max(),
                         b['maxy'].max()))

    @property
    def sindex(self):
        if not self._sindex_generated:
            self._generate_sindex()
        return self._sindex

    def buffer(self, distance, resolution=16, cap_style=CAP_STYLE.round,
               join_style=JOIN_STYLE.round, mitre_limit=5.0):
        """
        Returns a ``GeoSeries`` of geometries representing all points within
        a given `distance` of each geometric object.

        See http://shapely.readthedocs.io/en/latest/manual.html#object.buffer
        for details.

        Parameters
        ----------
        distance : float
            The radius of the buffer.
        resolution: float
            Optional, the resolution of the buffer around each vertex.
        """
        from .geoseries import GeoSeries
        geom_array = self._geometry_array.buffer(
            distance, resolution=resolution, cap_style=cap_style,
            join_style=join_style, mitre_limit=mitre_limit)
        return GeoSeries(geom_array, index=self.index, crs=self.crs)

    def simplify(self, *args, **kwargs):
        """
        Returns a ``GeoSeries`` containing a simplified representation of
        each geometry.

        See http://shapely.readthedocs.io/en/latest/manual.html#object.simplify
        for details.

        Parameters
        ----------
        tolerance : float
            All points in a simplified geometry will be no more than
            `tolerance` distance from the original.
        preserve_topology: bool
            False uses a quicker algorithm, but may produce self-intersecting
            or otherwise invalid geometries.
        """
        from .geoseries import GeoSeries
        return GeoSeries([geom.simplify(*args, **kwargs)
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
        op = 'project' if not normalized else 'project-normalized'
        if isinstance(other, GeoPandasBase):
            self = self.geometry
            self, other = self.align(other.geometry)
            t = self._geometry_array
            o = other._geometry_array
            x = vectorized.binary_vector_float_return(op, t.data, o.data)
            return Series(x, index=self.index)
        else:
            x = vectorized.binary_float_return(op, self._geometry_array.data, other)
            return Series(x, index=self.index)

    def interpolate(self, distance, normalized=False):
        """
        Return a point at the specified distance along each geometry

        Parameters
        ----------
        distance : float or Series of floats
            Distance(s) along the geometries at which a point should be
             returned
        normalized : boolean
            If normalized is True, distance will be interpreted as a fraction
            of the geometric object's length.
        """
        from .geoseries import GeoSeries
        return GeoSeries([s.interpolate(distance, normalized)
                          for s in self.geometry],
                         index=self.index, crs=self.crs)

    def translate(self, xoff=0.0, yoff=0.0, zoff=0.0):
        """Returns a ``GeoSeries`` with translated geometries.

        See http://shapely.readthedocs.io/en/latest/manual.html#shapely.affinity.translate
        for details.

        Parameters
        ----------
        xoff, yoff, zoff : float, float, float
            Amount of offset along each dimension.
            xoff, yoff, and zoff for translation along the x, y, and z
            dimensions respectively.
        """
        from .geoseries import GeoSeries
        return GeoSeries([affinity.translate(s, xoff, yoff, zoff)
                          for s in self.geometry],
                         index=self.index, crs=self.crs)

    def rotate(self, angle, origin='center', use_radians=False):
        """Returns a ``GeoSeries`` with rotated geometries.

        See http://shapely.readthedocs.io/en/latest/manual.html#shapely.affinity.rotate
        for details.

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
        """
        from .geoseries import GeoSeries
        L = [affinity.rotate(s, angle, origin=origin,
             use_radians=use_radians) for s in self.geometry]
        return GeoSeries(L, index=self.index, crs=self.crs)

    def scale(self, xfact=1.0, yfact=1.0, zfact=1.0, origin='center'):
        """Returns a ``GeoSeries`` with scaled geometries.

        The geometries can be scaled by different factors along each
        dimension. Negative scale factors will mirror or reflect coordinates.

        See http://shapely.readthedocs.io/en/latest/manual.html#shapely.affinity.scale
        for details.

        Parameters
        ----------
        xfact, yfact, zfact : float, float, float
            Scaling factors for the x, y, and z dimensions respectively.
        origin : string, Point, or tuple
            The point of origin can be a keyword 'center' for the 2D bounding
            box center (default), 'centroid' for the geometry's 2D centroid, a
            Point object or a coordinate tuple (x, y, z).
        """
        from .geoseries import GeoSeries
        return GeoSeries([affinity.scale(s, xfact, yfact, zfact,
                                         origin=origin)
                          for s in self.geometry],
                         index=self.index, crs=self.crs)

    def skew(self, xs=0.0, ys=0.0, origin='center', use_radians=False):
        """Returns a ``GeoSeries`` with skewed geometries.

        The geometries are sheared by angles along the x and y dimensions.

        See http://shapely.readthedocs.io/en/latest/manual.html#shapely.affinity.skew
        for details.

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
        """
        from .geoseries import GeoSeries
        return GeoSeries([affinity.skew(s, xs, ys, origin=origin,
                                        use_radians=use_radians)
                          for s in self.geometry],
                         index=self.index, crs=self.crs)

    def explode(self):
        """
        Explode multi-part geometries into multiple single geometries.

        Single rows can become multiple rows.
        This is analogous to PostGIS's ST_Dump(). The 'path' index is the
        second level of the returned MultiIndex

        Returns
        ------
        A GeoSeries with a MultiIndex. The levels of the MultiIndex are the
        original index and an integer.

        Example
        -------
        >>> gdf  # gdf is GeoSeries of MultiPoints
        0                 (POINT (0 0), POINT (1 1))
        1    (POINT (2 2), POINT (3 3), POINT (4 4))

        >>> gdf.explode()
        0  0    POINT (0 0)
           1    POINT (1 1)
        1  0    POINT (2 2)
           1    POINT (3 3)
           2    POINT (4 4)
        dtype: object
        """
        from .geoseries import GeoSeries
        index = []
        geometries = []
        for idx, s in self.geometry.iteritems():
            if s.type.startswith('Multi') or s.type == 'GeometryCollection':
                geoms = s.geoms
                idxs = [(idx, i) for i in range(len(geoms))]
            else:
                geoms = [s]
                idxs = [(idx, 0)]
            index.extend(idxs)
            geometries.extend(geoms)
        s = GeoSeries(geometries, index=MultiIndex.from_tuples(index))
        s = s.__finalize__(self)
        return s


class _CoordinateIndexer(_NDFrameIndexer):
    """
    Coordinate based indexer to select by intersection with bounding box.

    Format of input should be ``.cx[xmin:xmax, ymin:ymax]``. Any of ``xmin``,
    ``xmax``, ``ymin``, and ``ymax`` can be provided, but input must
    include a comma separating x and y slices. That is, ``.cx[:, :]`` will
    return the full series/frame, but ``.cx[:]`` is not implemented.
    """

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
        bbox = box(xs.start if xs.start is not None else xmin,
                   ys.start if ys.start is not None else ymin,
                   xs.stop if xs.stop is not None else xmax,
                   ys.stop if ys.stop is not None else ymax)
        idx = obj.intersects(bbox)
        return obj[idx]
