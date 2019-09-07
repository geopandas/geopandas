from warnings import warn

import numpy as np
import pandas as pd
from pandas import DataFrame, MultiIndex, Series

from shapely.geometry import box
from shapely.geometry.base import BaseGeometry
from shapely.ops import cascaded_union, unary_union

import geopandas as gpd

from .array import GeometryArray, GeometryDtype

try:
    from rtree.core import RTreeError

    HAS_SINDEX = True
except ImportError:

    class RTreeError(Exception):
        pass

    HAS_SINDEX = False


def is_geometry_type(data):
    """
    Check if the data is of geometry dtype.

    Does not include object array of shapely scalars.
    """
    if isinstance(getattr(data, "dtype", None), GeometryDtype):
        # GeometryArray, GeoSeries and Series[GeometryArray]
        return True
    else:
        return False


def _delegate_binary_method(op, this, other, *args, **kwargs):
    # type: (str, GeoSeries, GeoSeries) -> GeoSeries/Series
    this = this.geometry
    if isinstance(other, GeoPandasBase):
        this, other = this.align(other.geometry)

        # TODO reenable for all operations once we use pyproj > 2
        # if this.crs != other.crs:
        #     warn('GeoSeries crs mismatch: {0} and {1}'.format(this.crs,
        #                                                       other.crs))
        a_this = GeometryArray(this.values)
        other = GeometryArray(other.values)
    elif isinstance(other, BaseGeometry):
        a_this = GeometryArray(this.values)
    else:
        raise TypeError(type(this), type(other))

    data = getattr(a_this, op)(other, *args, **kwargs)
    return data, this.index


def _binary_geo(op, this, other):
    # type: (str, GeoSeries, GeoSeries) -> GeoSeries
    """Binary operation on GeoSeries objects that returns a GeoSeries"""
    from .geoseries import GeoSeries

    geoms, index = _delegate_binary_method(op, this, other)
    return GeoSeries(geoms.data, index=index, crs=this.crs)


def _binary_op(op, this, other, *args, **kwargs):
    # type: (str, GeoSeries, GeoSeries, args/kwargs) -> Series[bool/float]
    """Binary operation on GeoSeries objects that returns a Series"""
    data, index = _delegate_binary_method(op, this, other, *args, **kwargs)
    return Series(data, index=index)


def _delegate_property(op, this):
    # type: (str, GeoSeries) -> GeoSeries/Series
    a_this = GeometryArray(this.geometry.values)
    data = getattr(a_this, op)
    if isinstance(data, GeometryArray):
        from .geoseries import GeoSeries

        return GeoSeries(data.data, index=this.index, crs=this.crs)
    else:
        return Series(data, index=this.index)


def _delegate_geo_method(op, this, *args, **kwargs):
    # type: (str, GeoSeries) -> GeoSeries
    """Unary operation that returns a GeoSeries"""
    from .geoseries import GeoSeries

    a_this = GeometryArray(this.geometry.values)
    data = getattr(a_this, op)(*args, **kwargs).data
    return GeoSeries(data, index=this.index, crs=this.crs)


class GeoPandasBase(object):
    _sindex = None
    _sindex_generated = False

    def _generate_sindex(self):
        if not HAS_SINDEX:
            warn("Cannot generate spatial index: Missing package `rtree`.")
        else:
            from geopandas.sindex import SpatialIndex

            stream = (
                (i, item.bounds, idx)
                for i, (idx, item) in enumerate(self.geometry.iteritems())
                if pd.notnull(item) and not item.is_empty
            )
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
        return _delegate_property("area", self)

    @property
    def geom_type(self):
        """Returns a ``Series`` of strings specifying the `Geometry Type` of each
        object."""
        return _delegate_property("geom_type", self)

    @property
    def type(self):
        """Return the geometry type of each geometry in the GeoSeries"""
        return self.geom_type

    @property
    def length(self):
        """Returns a ``Series`` containing the length of each geometry."""
        return _delegate_property("length", self)

    @property
    def is_valid(self):
        """Returns a ``Series`` of ``dtype('bool')`` with value ``True`` for
        geometries that are valid."""
        return _delegate_property("is_valid", self)

    @property
    def is_empty(self):
        """Returns a ``Series`` of ``dtype('bool')`` with value ``True`` for
        empty geometries."""
        return _delegate_property("is_empty", self)

    @property
    def is_simple(self):
        """Returns a ``Series`` of ``dtype('bool')`` with value ``True`` for
        geometries that do not cross themselves.

        This is meaningful only for `LineStrings` and `LinearRings`.
        """
        return _delegate_property("is_simple", self)

    @property
    def is_ring(self):
        """Returns a ``Series`` of ``dtype('bool')`` with value ``True`` for
        features that are closed."""
        return _delegate_property("is_ring", self)

    @property
    def has_z(self):
        """Returns a ``Series`` of ``dtype('bool')`` with value ``True`` for
        features that have a z-component."""
        return _delegate_property("has_z", self)

    #
    # Unary operations that return a GeoSeries
    #

    @property
    def boundary(self):
        """Returns a ``GeoSeries`` of lower dimensional objects representing
        each geometries's set-theoretic `boundary`."""
        return _delegate_property("boundary", self)

    @property
    def centroid(self):
        """Returns a ``GeoSeries`` of points representing the centroid of each
        geometry."""
        return _delegate_property("centroid", self)

    @property
    def convex_hull(self):
        """Returns a ``GeoSeries`` of geometries representing the convex hull
        of each geometry.

        The convex hull of a geometry is the smallest convex `Polygon`
        containing all the points in each geometry, unless the number of points
        in the geometric object is less than three. For two points, the convex
        hull collapses to a `LineString`; for 1, a `Point`."""
        return _delegate_property("convex_hull", self)

    @property
    def envelope(self):
        """Returns a ``GeoSeries`` of geometries representing the envelope of
        each geometry.

        The envelope of a geometry is the bounding rectangle. That is, the
        point or smallest rectangular polygon (with sides parallel to the
        coordinate axes) that contains the geometry."""
        return _delegate_property("envelope", self)

    @property
    def exterior(self):
        """Returns a ``GeoSeries`` of LinearRings representing the outer
        boundary of each polygon in the GeoSeries.

        Applies to GeoSeries containing only Polygons.
        """
        # TODO: return empty geometry for non-polygons
        return _delegate_property("exterior", self)

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
        return _delegate_property("interiors", self)

    def representative_point(self):
        """Returns a ``GeoSeries`` of (cheaply computed) points that are
        guaranteed to be within each geometry.
        """
        return _delegate_geo_method("representative_point", self)

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
        return _binary_op("contains", self, other)

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
        return _binary_op("equals", self, other)

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
        return _binary_op("almost_equals", self, other, decimal=decimal)

    def geom_equals_exact(self, other, tolerance):
        """Return True for all geometries that equal *other* to a given
        tolerance, else False"""
        return _binary_op("equals_exact", self, other, tolerance=tolerance)

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
        return _binary_op("crosses", self, other)

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
        return _binary_op("disjoint", self, other)

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
        return _binary_op("intersects", self, other)

    def overlaps(self, other):
        """Return True for all geometries that overlap *other*, else False"""
        return _binary_op("overlaps", self, other)

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
        return _binary_op("touches", self, other)

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
        return _binary_op("within", self, other)

    def distance(self, other):
        """Returns a ``Series`` containing the distance to `other`.

        Parameters
        ----------
        other : Geoseries or geometric object
            The Geoseries (elementwise) or geometric object to find the
            distance to.
        """
        return _binary_op("distance", self, other)

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
        return _binary_geo("difference", self, other)

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
        return _binary_geo("symmetric_difference", self, other)

    def union(self, other):
        """Returns a ``GeoSeries`` of the union of points in each geometry with
        `other`.

        Parameters
        ----------
        other : Geoseries or geometric object
            The Geoseries (elementwise) or geometric object to find the union
            with.
        """
        return _binary_geo("union", self, other)

    def intersection(self, other):
        """Returns a ``GeoSeries`` of the intersection of points in each
        geometry with `other`.

        Parameters
        ----------
        other : Geoseries or geometric object
            The Geoseries (elementwise) or geometric object to find the
            intersection with.
        """
        return _binary_geo("intersection", self, other)

    #
    # Other operations
    #

    @property
    def bounds(self):
        """Returns a ``DataFrame`` with columns ``minx``, ``miny``, ``maxx``,
        ``maxy`` values containing the bounds for each geometry.

        See ``GeoSeries.total_bounds`` for the limits of the entire series.
        """
        bounds = GeometryArray(self.geometry.values).bounds
        return DataFrame(
            bounds, columns=["minx", "miny", "maxx", "maxy"], index=self.index
        )

    @property
    def total_bounds(self):
        """Returns a tuple containing ``minx``, ``miny``, ``maxx``, ``maxy``
        values for the bounds of the series as a whole.

        See ``GeoSeries.bounds`` for the bounds of the geometries contained in
        the series.
        """
        return GeometryArray(self.geometry.values).total_bounds

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
        if isinstance(distance, pd.Series):
            if not self.index.equals(distance.index):
                raise ValueError(
                    "Index values of distance sequence does "
                    "not match index values of the GeoSeries"
                )
            distance = np.asarray(distance)

        return _delegate_geo_method(
            "buffer", self, distance, resolution=resolution, **kwargs
        )

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
        return _delegate_geo_method("simplify", self, *args, **kwargs)

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
        return _binary_op("relate", self, other)

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
        return _binary_op("project", self, other, normalized=normalized)

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
        if isinstance(distance, pd.Series):
            if not self.index.equals(distance.index):
                raise ValueError(
                    "Index values of distance sequence does "
                    "not match index values of the GeoSeries"
                )
            distance = np.asarray(distance)
        return _delegate_geo_method(
            "interpolate", self, distance, normalized=normalized
        )

    def affine_transform(self, matrix):
        """Return a ``GeoSeries`` with translated geometries.

        See http://shapely.readthedocs.io/en/stable/manual.html#shapely.affinity.affine_transform
        for details.

        Parameters
        ----------
        matrix: List or tuple
            6 or 12 items for 2D or 3D transformations respectively.
            For 2D affine transformations,
            the 6 parameter matrix is [a, b, d, e, xoff, yoff]
            For 3D affine transformations,
            the 12 parameter matrix is [a, b, c, d, e, f, g, h, i, xoff, yoff, zoff]
        """  # noqa (E501 link is longer than max line length)
        return _delegate_geo_method("affine_transform", self, matrix)

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
        """  # noqa (E501 link is longer than max line length)
        return _delegate_geo_method("translate", self, xoff, yoff, zoff)

    def rotate(self, angle, origin="center", use_radians=False):
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
        return _delegate_geo_method(
            "rotate", self, angle, origin=origin, use_radians=use_radians
        )

    def scale(self, xfact=1.0, yfact=1.0, zfact=1.0, origin="center"):
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
        return _delegate_geo_method("scale", self, xfact, yfact, zfact, origin=origin)

    def skew(self, xs=0.0, ys=0.0, origin="center", use_radians=False):
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
        return _delegate_geo_method(
            "skew", self, xs, ys, origin=origin, use_radians=use_radians
        )

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
        0         MULTIPOINT (0 0, 1 1)
        1    MULTIPOINT (2 2, 3 3, 4 4)
        dtype: geometry

        >>> gdf.explode()
        0  0    POINT (0 0)
           1    POINT (1 1)
        1  0    POINT (2 2)
           1    POINT (3 3)
           2    POINT (4 4)
        dtype: geometry

        """
        index = []
        geometries = []
        for idx, s in self.geometry.iteritems():
            if s.type.startswith("Multi") or s.type == "GeometryCollection":
                geoms = s.geoms
                idxs = [(idx, i) for i in range(len(geoms))]
            else:
                geoms = [s]
                idxs = [(idx, 0)]
            index.extend(idxs)
            geometries.extend(geoms)
        index = MultiIndex.from_tuples(index, names=self.index.names + [None])
        return gpd.GeoSeries(geometries, index=index).__finalize__(self)

    @property
    def cx(self):
        """
        Coordinate based indexer to select by intersection with bounding box.

        Format of input should be ``.cx[xmin:xmax, ymin:ymax]``. Any of
        ``xmin``, ``xmax``, ``ymin``, and ``ymax`` can be provided, but input
        must include a comma separating x and y slices. That is, ``.cx[:, :]``
        will return the full series/frame, but ``.cx[:]`` is not implemented.
        """
        return _CoordinateIndexer(self)


class _CoordinateIndexer(object):
    # see docstring GeoPandasBase.cx property above

    def __init__(self, obj):
        self.obj = obj

    def __getitem__(self, key):
        obj = self.obj
        xs, ys = key
        # handle numeric values as x and/or y coordinate index
        if type(xs) is not slice:
            xs = slice(xs, xs)
        if type(ys) is not slice:
            ys = slice(ys, ys)
        # don't know how to handle step; should this raise?
        if xs.step is not None or ys.step is not None:
            warn("Ignoring step - full interval is used.")
        xmin, ymin, xmax, ymax = obj.total_bounds
        bbox = box(
            xs.start if xs.start is not None else xmin,
            ys.start if ys.start is not None else ymin,
            xs.stop if xs.stop is not None else xmax,
            ys.stop if ys.stop is not None else ymax,
        )
        idx = obj.intersects(bbox)
        return obj[idx]
