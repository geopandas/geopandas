from warnings import warn

import numpy as np
import pandas as pd
from pandas import Series, DataFrame, MultiIndex
from pandas.core.indexing import _NDFrameIndexer

from shapely.geometry.base import BaseGeometry
from shapely.geometry import box
from shapely.ops import cascaded_union, unary_union
import shapely.affinity as affinity

import geopandas as gpd


try:
    from rtree.core import RTreeError
    HAS_SINDEX = True
except ImportError:
    class RTreeError(Exception):
        pass
    HAS_SINDEX = False


def _binary_geo(op, left, right):
    # type: (str, GeoSeries, GeoSeries) -> GeoSeries
    """Binary operation on GeoSeries objects that returns a GeoSeries"""
    from .geoseries import GeoSeries
    if isinstance(right, GeoPandasBase):
        left = left.geometry
        left, right = left.align(right.geometry)

        if left.crs != right.crs:
            warn('GeoSeries crs mismatch: {0} and {1}'.format(left.crs,
                                                              right.crs))

        # intersection can return empty GeometryCollections, and if the result
        # are only those, numpy will coerce it to empty 2D array
        data = np.empty(len(left), dtype=object)
        data[:] = [getattr(this_elem, op)(other_elem)
                   for this_elem, other_elem in zip(left, right)]

        return GeoSeries(data, index=left.index, crs=left.crs)

    elif isinstance(right, BaseGeometry):
        # ensure 1D output, see note above
        data = np.empty(len(left), dtype=object)
        data[:] = [getattr(s, op)(right) for s in left.geometry]
        return GeoSeries(data, index=left.index, crs=left.crs)
    else:
        raise TypeError(type(left), type(right))


def _binary_op(op, this, other, *args, **kwargs):
    # type: (str, GeoSeries, GeoSeries, args/kwargs) -> Series[bool]
    """Binary operation on GeoSeries objects that returns a Series"""
    if op in ['distance', 'project']:
        null_value = np.nan
    elif op == 'relate':
        null_value = None
    else:
        null_value = False
    if op in ['distance', 'project']:
        dtype = float
    elif op == 'relate':
        dtype = object
    else:
        dtype = bool

    if isinstance(other, GeoPandasBase):

        this = this.geometry
        this, other = this.align(other.geometry)

        data = np.array(
            [getattr(this_elem, op)(other_elem, *args, **kwargs)
             if not this_elem.is_empty | other_elem.is_empty else null_value
             for this_elem, other_elem in zip(this, other)],
            dtype=dtype)

        return Series(data, index=this.index)

    elif isinstance(other, BaseGeometry):
        data = np.array(
            [getattr(s, op)(other, *args, **kwargs) if s else null_value
             for s in this.geometry],
            dtype=dtype)
        return Series(data, index=this.index)
    else:
        raise TypeError(type(this), type(other))


def _unary_geo(op, this):
    # type: (str, GeoSeries) -> GeoSeries
    """Unary operation that returns a GeoSeries"""
    from .geoseries import GeoSeries
    # ensure 1D output, see note above
    data = np.empty(len(this), dtype=object)
    data[:] = [getattr(geom, op) for geom in this.geometry]
    return GeoSeries(data, index=this.index, crs=this.crs)


def _unary_op(op, this, null_value=False):
    # type: (str, GeoSeries, Any) -> Series
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
                      enumerate(self.geometry.iteritems())
                      if pd.notnull(item) and not item.is_empty)
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
        return _unary_op('area', self, null_value=np.nan)

    @property
    def geom_type(self):
        """Returns a ``Series`` of strings specifying the `Geometry Type` of each
        object."""
        return _unary_op('geom_type', self, null_value=None)

    @property
    def type(self):
        """Return the geometry type of each geometry in the GeoSeries"""
        return self.geom_type

    @property
    def length(self):
        """Returns a ``Series`` containing the length of each geometry."""
        return _unary_op('length', self, null_value=np.nan)

    @property
    def is_valid(self):
        """Returns a ``Series`` of ``dtype('bool')`` with value ``True`` for
        geometries that are valid."""
        return _unary_op('is_valid', self, null_value=False)

    @property
    def is_empty(self):
        """Returns a ``Series`` of ``dtype('bool')`` with value ``True`` for
        empty geometries."""
        return _unary_op('is_empty', self, null_value=False)

    @property
    def is_simple(self):
        """Returns a ``Series`` of ``dtype('bool')`` with value ``True`` for
        geometries that do not cross themselves.

        This is meaningful only for `LineStrings` and `LinearRings`.
        """
        return _unary_op('is_simple', self, null_value=False)

    @property
    def is_ring(self):
        """Returns a ``Series`` of ``dtype('bool')`` with value ``True`` for
        features that are closed."""
        # operates on the exterior, so can't use _unary_op()
        return Series([geom.exterior.is_ring for geom in self.geometry],
                      index=self.index)

    @property
    def has_z(self):
        """Returns a ``Series`` of ``dtype('bool')`` with value ``True`` for
        features that have a z-component."""
        return _unary_op('has_z', self, null_value=False)

    #
    # Unary operations that return a GeoSeries
    #

    @property
    def boundary(self):
        """Returns a ``GeoSeries`` of lower dimensional objects representing
        each geometries's set-theoretic `boundary`."""
        return _unary_geo('boundary', self)

    @property
    def centroid(self):
        """Returns a ``GeoSeries`` of points representing the centroid of each
        geometry."""
        return _unary_geo('centroid', self)

    @property
    def convex_hull(self):
        """Returns a ``GeoSeries`` of geometries representing the convex hull
        of each geometry.

        The convex hull of a geometry is the smallest convex `Polygon`
        containing all the points in each geometry, unless the number of points
        in the geometric object is less than three. For two points, the convex
        hull collapses to a `LineString`; for 1, a `Point`."""
        return _unary_geo('convex_hull', self)

    @property
    def envelope(self):
        """Returns a ``GeoSeries`` of geometries representing the envelope of
        each geometry.

        The envelope of a geometry is the bounding rectangle. That is, the
        point or smallest rectangular polygon (with sides parallel to the
        coordinate axes) that contains the geometry."""
        return _unary_geo('envelope', self)

    @property
    def exterior(self):
        """Returns a ``GeoSeries`` of LinearRings representing the outer
        boundary of each polygon in the GeoSeries.

        Applies to GeoSeries containing only Polygons.
        """
        # TODO: return empty geometry for non-polygons
        return _unary_geo('exterior', self)

    @property
    def interiors(self):
        """Returns a ``Series`` of List representing the
        inner rings of each polygon in the GeoSeries.

        Applies to GeoSeries containing only Polygons.

        Returns
        ----------
        inner_rings: Series of List
            Inner rings of each polygon in the GeoSeries.
        """

        has_non_poly = False
        inner_rings = []
        for geom in self.geometry:
            interior_ring_seq = getattr(geom, 'interiors', None)
            # polygon case
            if interior_ring_seq is not None:
                inner_rings.append(list(interior_ring_seq))
            # non-polygon case
            else:
                has_non_poly = True
                inner_rings.append(None)
        if has_non_poly:
            warn("Only Polygon objects have interior rings. For other "
                 "geometry types, None is returned.")

        # _unary_op couldn't be used in order to warning to non-polygon and
        # conversion to list.
        return Series(inner_rings,
                      index=self.index, dtype=object)

    def representative_point(self):
        """Returns a ``GeoSeries`` of (cheaply computed) points that are
        guaranteed to be within each geometry.
        """
        return gpd.GeoSeries([geom.representative_point()
                              for geom in self.geometry],
                             index=self.index)

    #
    # Reduction operations that return a Shapely geometry
    #

    @property
    def cascaded_union(self):
        """Deprecated: Return the unary_union of all geometries"""
        return cascaded_union(self.geometry.values)

    @property
    def unary_union(self):
        """Returns a geometry containing the union of all geometries in the
        ``GeoSeries``."""
        return unary_union(self.geometry.values)

    #
    # Binary operations that return a pandas Series
    #

    def contains(self, other):
        """Returns a ``Series`` of ``dtype('bool')`` with value ``True`` for
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
        return _binary_op('contains', self, other)

    def geom_equals(self, other):
        """Returns a ``Series`` of ``dtype('bool')`` with value ``True`` for
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
        return _binary_op('equals', self, other)

    def geom_almost_equals(self, other, decimal=6):
        """Returns a ``Series`` of ``dtype('bool')`` with value ``True`` if
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
        return _binary_op('almost_equals', self, other, decimal=decimal)

    def geom_equals_exact(self, other, tolerance):
        """Return True for all geometries that equal *other* to a given
        tolerance, else False"""
        return _binary_op('equals_exact', self, other, tolerance=tolerance)

    def crosses(self, other):
        """Returns a ``Series`` of ``dtype('bool')`` with value ``True`` for
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
        return _binary_op('crosses', self, other)

    def disjoint(self, other):
        """Returns a ``Series`` of ``dtype('bool')`` with value ``True`` for
        each geometry disjoint to `other`.

        An object is said to be disjoint to `other` if its `boundary` and
        `interior` does not intersect at all with those of the other.

        Parameters
        ----------
        other : GeoSeries or geometric object
            The GeoSeries (elementwise) or geometric object to test if is
            disjoint.
        """
        return _binary_op('disjoint', self, other)

    def intersects(self, other):
        """Returns a ``Series`` of ``dtype('bool')`` with value ``True`` for
        each geometry that intersects `other`.

        An object is said to intersect `other` if its `boundary` and `interior`
        intersects in any way with those of the other.

        Parameters
        ----------
        other : GeoSeries or geometric object
            The GeoSeries (elementwise) or geometric object to test if is
            intersected.
        """
        return _binary_op('intersects', self, other)

    def overlaps(self, other):
        """Return True for all geometries that overlap *other*, else False"""
        return _binary_op('overlaps', self, other)

    def touches(self, other):
        """Returns a ``Series`` of ``dtype('bool')`` with value ``True`` for
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
        return _binary_op('touches', self, other)

    def within(self, other):
        """Returns a ``Series`` of ``dtype('bool')`` with value ``True`` for
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
        return _binary_op('within', self, other)

    def distance(self, other):
        """Returns a ``Series`` containing the distance to `other`.

        Parameters
        ----------
        other : Geoseries or geometric object
            The Geoseries (elementwise) or geometric object to find the
            distance to.
        """
        return _binary_op('distance', self, other)

    #
    # Binary operations that return a GeoSeries
    #

    def difference(self, other):
        """Returns a ``GeoSeries`` of the points in each geometry that
        are not in `other`.

        Parameters
        ----------
        other : Geoseries or geometric object
            The Geoseries (elementwise) or geometric object to find the
            difference to.
        """
        return _binary_geo('difference', self, other)

    def symmetric_difference(self, other):
        """Returns a ``GeoSeries`` of the symmetric difference of points in
        each geometry with `other`.

        For each geometry, the symmetric difference consists of points in the
        geometry not in `other`, and points in `other` not in the geometry.

        Parameters
        ----------
        other : Geoseries or geometric object
            The Geoseries (elementwise) or geometric object to find the
            symmetric difference to.
        """
        return _binary_geo('symmetric_difference', self, other)

    def union(self, other):
        """Returns a ``GeoSeries`` of the union of points in each geometry with
        `other`.

        Parameters
        ----------
        other : Geoseries or geometric object
            The Geoseries (elementwise) or geometric object to find the union
            with.
        """
        return _binary_geo('union', self, other,)

    def intersection(self, other):
        """Returns a ``GeoSeries`` of the intersection of points in each
        geometry with `other`.

        Parameters
        ----------
        other : Geoseries or geometric object
            The Geoseries (elementwise) or geometric object to find the
            intersection with.
        """
        return _binary_geo('intersection', self, other)

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

    def buffer(self, distance, resolution=16, **kwargs):
        """Returns a ``GeoSeries`` of geometries representing all points within
        a given `distance` of each geometric object.

        See http://shapely.readthedocs.io/en/latest/manual.html#object.buffer
        for details.

        Parameters
        ----------
        distance : float, np.array, pd.Series
            The radius of the buffer. If np.array or pd.Series are used
            then it must have same length as the GeoSeries.
        resolution: int
            Optional, the resolution of the buffer around each vertex.
        """
        if isinstance(distance, (np.ndarray, pd.Series)):
            if len(distance) != len(self.index):
                raise ValueError("Length of distance sequence does not match "
                                 "length of the GeoSeries")
            if isinstance(distance, pd.Series):
                if not self.index.equals(distance.index):
                    raise ValueError("Index values of distance sequence does "
                                     "not match index values of the GeoSeries")
            return gpd.GeoSeries(
                [geom.buffer(dist, resolution, **kwargs)
                 for geom, dist in zip(self.geometry, distance)],
                index=self.index, crs=self.crs)

        return gpd.GeoSeries([geom.buffer(distance, resolution, **kwargs)
                              for geom in self.geometry],
                             index=self.index, crs=self.crs)

    def simplify(self, *args, **kwargs):
        """Returns a ``GeoSeries`` containing a simplified representation of
        each geometry.

        See http://shapely.readthedocs.io/en/latest/manual.html#object.simplify
        for details

        Parameters
        ----------
        tolerance : float
            All points in a simplified geometry will be no more than
            `tolerance` distance from the original.
        preserve_topology: bool
            False uses a quicker algorithm, but may produce self-intersecting
            or otherwise invalid geometries.
        """
        return gpd.GeoSeries(
            [geom.simplify(*args, **kwargs) for geom in self.geometry],
            index=self.index, crs=self.crs)

    def relate(self, other):
        """
        Returns the DE-9IM intersection matrices for the geometries

        Parameters
        ----------
        other : BaseGeometry or GeoSeries
            The other geometry to computed
            the DE-9IM intersection matrices from.

        Returns
        ----------
        spatial_relations: Series of strings
            The DE-9IM intersection matrices which describe
            the spatial relations of the other geometry.
        """
        return _binary_op('relate', self, other)

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

        return _binary_op('project', self, other, normalized=normalized)

    def interpolate(self, distance, normalized=False):
        """
        Return a point at the specified distance along each geometry

        Parameters
        ----------
        distance : float or Series of floats
            Distance(s) along the geometries at which a point should be
            returned. If np.array or pd.Series are used then it must have
            same length as the GeoSeries.
        normalized : boolean
            If normalized is True, distance will be interpreted as a fraction
            of the geometric object's length.
        """
        if isinstance(distance, (np.ndarray, pd.Series)):
            if len(distance) != len(self.index):
                raise ValueError("Length of distance sequence does not match "
                                 "length of the GeoSeries")
            if isinstance(distance, pd.Series):
                if not self.index.equals(distance.index):
                    raise ValueError("Index values of distance sequence does "
                                     "not match index values of the GeoSeries")
            return gpd.GeoSeries(
                [s.interpolate(dist, normalized=normalized)
                 for (s, dist) in zip(self.geometry, distance)],
                index=self.index, crs=self.crs)

        return gpd.GeoSeries([s.interpolate(distance, normalized=normalized)
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
        return gpd.GeoSeries([affinity.translate(s, xoff, yoff, zoff)
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
        return gpd.GeoSeries(
            [affinity.rotate(s, angle, origin=origin,
                             use_radians=use_radians)
             for s in self.geometry],
            index=self.index, crs=self.crs)

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
        return gpd.GeoSeries(
            [affinity.scale(s, xfact, yfact, zfact, origin=origin)
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
        return gpd.GeoSeries(
            [affinity.skew(s, xs, ys, origin=origin, use_radians=use_radians)
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
        original index and a zero-based integer index that counts the
        number of single geometries within a multi-part geometry.

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
        index = MultiIndex.from_tuples(index, names=self.index.names + [None])
        return gpd.GeoSeries(geometries, index=index).__finalize__(self)


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
