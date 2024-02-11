from warnings import warn
import warnings

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from shapely.geometry import box, MultiPoint
from shapely.geometry.base import BaseGeometry

from . import _compat as compat
from .array import GeometryArray, GeometryDtype, points_from_xy


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


def _delegate_binary_method(op, this, other, align, *args, **kwargs):
    # type: (str, GeoSeries, GeoSeries) -> GeoSeries/Series
    this = this.geometry
    if isinstance(other, GeoPandasBase):
        if align and not this.index.equals(other.index):
            warn(
                "The indices of the two GeoSeries are different.",
                stacklevel=4,
            )
            this, other = this.align(other.geometry)
        else:
            other = other.geometry

        a_this = GeometryArray(this.values)
        other = GeometryArray(other.values)
    elif isinstance(other, BaseGeometry):
        a_this = GeometryArray(this.values)
    else:
        raise TypeError(type(this), type(other))

    data = getattr(a_this, op)(other, *args, **kwargs)
    return data, this.index


def _binary_geo(op, this, other, align):
    # type: (str, GeoSeries, GeoSeries) -> GeoSeries
    """Binary operation on GeoSeries objects that returns a GeoSeries"""
    from .geoseries import GeoSeries

    geoms, index = _delegate_binary_method(op, this, other, align)
    return GeoSeries(geoms, index=index, crs=this.crs)


def _binary_op(op, this, other, align, *args, **kwargs):
    # type: (str, GeoSeries, GeoSeries, args/kwargs) -> Series[bool/float]
    """Binary operation on GeoSeries objects that returns a Series"""
    data, index = _delegate_binary_method(op, this, other, align, *args, **kwargs)
    return Series(data, index=index)


def _delegate_property(op, this):
    # type: (str, GeoSeries) -> GeoSeries/Series
    a_this = GeometryArray(this.geometry.values)
    data = getattr(a_this, op)
    if isinstance(data, GeometryArray):
        from .geoseries import GeoSeries

        return GeoSeries(data, index=this.index, crs=this.crs)
    else:
        return Series(data, index=this.index)


def _delegate_geo_method(op, this, *args, **kwargs):
    # type: (str, GeoSeries) -> GeoSeries
    """Unary operation that returns a GeoSeries"""
    from .geoseries import GeoSeries

    a_this = GeometryArray(this.geometry.values)
    data = getattr(a_this, op)(*args, **kwargs)
    return GeoSeries(data, index=this.index, crs=this.crs)


class GeoPandasBase(object):
    @property
    def area(self):
        """Returns a ``Series`` containing the area of each geometry in the
        ``GeoSeries`` expressed in the units of the CRS.

        Examples
        --------

        >>> from shapely.geometry import Polygon, LineString, Point
        >>> s = geopandas.GeoSeries(
        ...     [
        ...         Polygon([(0, 0), (1, 1), (0, 1)]),
        ...         Polygon([(10, 0), (10, 5), (0, 0)]),
        ...         Polygon([(0, 0), (2, 2), (2, 0)]),
        ...         LineString([(0, 0), (1, 1), (0, 1)]),
        ...         Point(0, 1)
        ...     ]
        ... )
        >>> s
        0    POLYGON ((0.00000 0.00000, 1.00000 1.00000, 0....
        1    POLYGON ((10.00000 0.00000, 10.00000 5.00000, ...
        2    POLYGON ((0.00000 0.00000, 2.00000 2.00000, 2....
        3    LINESTRING (0.00000 0.00000, 1.00000 1.00000, ...
        4                              POINT (0.00000 1.00000)
        dtype: geometry

        >>> s.area
        0     0.5
        1    25.0
        2     2.0
        3     0.0
        4     0.0
        dtype: float64

        See also
        --------
        GeoSeries.length : measure length

        Notes
        -----
        Area may be invalid for a geographic CRS using degrees as units;
        use :meth:`GeoSeries.to_crs` to project geometries to a planar
        CRS before using this function.

        Every operation in GeoPandas is planar, i.e. the potential third
        dimension is not taken into account.
        """
        return _delegate_property("area", self)

    @property
    def crs(self):
        """
        The Coordinate Reference System (CRS) represented as a ``pyproj.CRS``
        object.

        Returns None if the CRS is not set, and to set the value it
        :getter: Returns a ``pyproj.CRS`` or None. When setting, the value
        can be anything accepted by
        :meth:`pyproj.CRS.from_user_input() <pyproj.crs.CRS.from_user_input>`,
        such as an authority string (eg "EPSG:4326") or a WKT string.

        Examples
        --------

        >>> s.crs  # doctest: +SKIP
        <Geographic 2D CRS: EPSG:4326>
        Name: WGS 84
        Axis Info [ellipsoidal]:
        - Lat[north]: Geodetic latitude (degree)
        - Lon[east]: Geodetic longitude (degree)
        Area of Use:
        - name: World
        - bounds: (-180.0, -90.0, 180.0, 90.0)
        Datum: World Geodetic System 1984
        - Ellipsoid: WGS 84
        - Prime Meridian: Greenwich

        See also
        --------
        GeoSeries.set_crs : assign CRS
        GeoSeries.to_crs : re-project to another CRS
        """
        return self.geometry.values.crs

    @crs.setter
    def crs(self, value):
        """Sets the value of the crs"""
        self.geometry.values.crs = value

    @property
    def geom_type(self):
        """
        Returns a ``Series`` of strings specifying the `Geometry Type` of each
        object.

        Examples
        --------
        >>> from shapely.geometry import Point, Polygon, LineString
        >>> d = {'geometry': [Point(2, 1), Polygon([(0, 0), (1, 1), (1, 0)]),
        ... LineString([(0, 0), (1, 1)])]}
        >>> gdf = geopandas.GeoDataFrame(d, crs="EPSG:4326")
        >>> gdf.geom_type
        0         Point
        1       Polygon
        2    LineString
        dtype: object
        """
        return _delegate_property("geom_type", self)

    @property
    def type(self):
        """Return the geometry type of each geometry in the GeoSeries"""
        return self.geom_type

    @property
    def length(self):
        """Returns a ``Series`` containing the length of each geometry
        expressed in the units of the CRS.

        In the case of a (Multi)Polygon it measures the length
        of its exterior (i.e. perimeter).

        Examples
        --------

        >>> from shapely.geometry import Polygon, LineString, MultiLineString, Point, \
GeometryCollection
        >>> s = geopandas.GeoSeries(
        ...     [
        ...         LineString([(0, 0), (1, 1), (0, 1)]),
        ...         LineString([(10, 0), (10, 5), (0, 0)]),
        ...         MultiLineString([((0, 0), (1, 0)), ((-1, 0), (1, 0))]),
        ...         Polygon([(0, 0), (1, 1), (0, 1)]),
        ...         Point(0, 1),
        ...         GeometryCollection([Point(1, 0), LineString([(10, 0), (10, 5), (0,\
 0)])])
        ...     ]
        ... )
        >>> s
        0    LINESTRING (0.00000 0.00000, 1.00000 1.00000, ...
        1    LINESTRING (10.00000 0.00000, 10.00000 5.00000...
        2    MULTILINESTRING ((0.00000 0.00000, 1.00000 0.0...
        3    POLYGON ((0.00000 0.00000, 1.00000 1.00000, 0....
        4                              POINT (0.00000 1.00000)
        5    GEOMETRYCOLLECTION (POINT (1.00000 0.00000), L...
        dtype: geometry

        >>> s.length
        0     2.414214
        1    16.180340
        2     3.000000
        3     3.414214
        4     0.000000
        5    16.180340
        dtype: float64

        See also
        --------
        GeoSeries.area : measure area of a polygon

        Notes
        -----
        Length may be invalid for a geographic CRS using degrees as units;
        use :meth:`GeoSeries.to_crs` to project geometries to a planar
        CRS before using this function.

        Every operation in GeoPandas is planar, i.e. the potential third
        dimension is not taken into account.

        """
        return _delegate_property("length", self)

    @property
    def is_valid(self):
        """Returns a ``Series`` of ``dtype('bool')`` with value ``True`` for
        geometries that are valid.

        Examples
        --------

        An example with one invalid polygon (a bowtie geometry crossing itself)
        and one missing geometry:

        >>> from shapely.geometry import Polygon
        >>> s = geopandas.GeoSeries(
        ...     [
        ...         Polygon([(0, 0), (1, 1), (0, 1)]),
        ...         Polygon([(0,0), (1, 1), (1, 0), (0, 1)]),  # bowtie geometry
        ...         Polygon([(0, 0), (2, 2), (2, 0)]),
        ...         None
        ...     ]
        ... )
        >>> s
        0    POLYGON ((0.00000 0.00000, 1.00000 1.00000, 0....
        1    POLYGON ((0.00000 0.00000, 1.00000 1.00000, 1....
        2    POLYGON ((0.00000 0.00000, 2.00000 2.00000, 2....
        3                                                 None
        dtype: geometry

        >>> s.is_valid
        0     True
        1    False
        2     True
        3    False
        dtype: bool

        """
        return _delegate_property("is_valid", self)

    @property
    def is_empty(self):
        """
        Returns a ``Series`` of ``dtype('bool')`` with value ``True`` for
        empty geometries.

        Examples
        --------
        An example of a GeoDataFrame with one empty point, one point and one missing
        value:

        >>> from shapely.geometry import Point
        >>> d = {'geometry': [Point(), Point(2, 1), None]}
        >>> gdf = geopandas.GeoDataFrame(d, crs="EPSG:4326")
        >>> gdf
                           geometry
        0               POINT EMPTY
        1   POINT (2.00000 1.00000)
        2                      None
        >>> gdf.is_empty
        0     True
        1    False
        2    False
        dtype: bool

        See Also
        --------
        GeoSeries.isna : detect missing values
        """
        return _delegate_property("is_empty", self)

    @property
    def is_simple(self):
        """Returns a ``Series`` of ``dtype('bool')`` with value ``True`` for
        geometries that do not cross themselves.

        This is meaningful only for `LineStrings` and `LinearRings`.

        Examples
        --------
        >>> from shapely.geometry import LineString
        >>> s = geopandas.GeoSeries(
        ...     [
        ...         LineString([(0, 0), (1, 1), (1, -1), (0, 1)]),
        ...         LineString([(0, 0), (1, 1), (1, -1)]),
        ...     ]
        ... )
        >>> s
        0    LINESTRING (0.00000 0.00000, 1.00000 1.00000, ...
        1    LINESTRING (0.00000 0.00000, 1.00000 1.00000, ...
        dtype: geometry

        >>> s.is_simple
        0    False
        1     True
        dtype: bool
        """
        return _delegate_property("is_simple", self)

    @property
    def is_ring(self):
        """Returns a ``Series`` of ``dtype('bool')`` with value ``True`` for
        features that are closed.

        When constructing a LinearRing, the sequence of coordinates may be
        explicitly closed by passing identical values in the first and last indices.
        Otherwise, the sequence will be implicitly closed by copying the first tuple
        to the last index.

        Examples
        --------
        >>> from shapely.geometry import LineString, LinearRing
        >>> s = geopandas.GeoSeries(
        ...     [
        ...         LineString([(0, 0), (1, 1), (1, -1)]),
        ...         LineString([(0, 0), (1, 1), (1, -1), (0, 0)]),
        ...         LinearRing([(0, 0), (1, 1), (1, -1)]),
        ...     ]
        ... )
        >>> s
        0    LINESTRING (0.00000 0.00000, 1.00000 1.00000, ...
        1    LINESTRING (0.00000 0.00000, 1.00000 1.00000, ...
        2    LINEARRING (0.00000 0.00000, 1.00000 1.00000, ...
        dtype: geometry

        >>> s.is_ring
        0    False
        1     True
        2     True
        dtype: bool

        """
        return _delegate_property("is_ring", self)

    @property
    def has_z(self):
        """Returns a ``Series`` of ``dtype('bool')`` with value ``True`` for
        features that have a z-component.

        Notes
        -----
        Every operation in GeoPandas is planar, i.e. the potential third
        dimension is not taken into account.

        Examples
        --------
        >>> from shapely.geometry import Point
        >>> s = geopandas.GeoSeries(
        ...     [
        ...         Point(0, 1),
        ...         Point(0, 1, 2),
        ...     ]
        ... )
        >>> s
        0              POINT (0.00000 1.00000)
        1    POINT Z (0.00000 1.00000 2.00000)
        dtype: geometry

        >>> s.has_z
        0    False
        1     True
        dtype: bool
        """
        return _delegate_property("has_z", self)

    #
    # Unary operations that return a GeoSeries
    #

    @property
    def boundary(self):
        """Returns a ``GeoSeries`` of lower dimensional objects representing
        each geometry's set-theoretic `boundary`.

        Examples
        --------

        >>> from shapely.geometry import Polygon, LineString, Point
        >>> s = geopandas.GeoSeries(
        ...     [
        ...         Polygon([(0, 0), (1, 1), (0, 1)]),
        ...         LineString([(0, 0), (1, 1), (1, 0)]),
        ...         Point(0, 0),
        ...     ]
        ... )
        >>> s
        0    POLYGON ((0.00000 0.00000, 1.00000 1.00000, 0....
        1    LINESTRING (0.00000 0.00000, 1.00000 1.00000, ...
        2                              POINT (0.00000 0.00000)
        dtype: geometry

        >>> s.boundary
        0    LINESTRING (0.00000 0.00000, 1.00000 1.00000, ...
        1        MULTIPOINT (0.00000 0.00000, 1.00000 0.00000)
        2                             GEOMETRYCOLLECTION EMPTY
        dtype: geometry

        See also
        --------
        GeoSeries.exterior : outer boundary (without interior rings)

        """
        return _delegate_property("boundary", self)

    @property
    def centroid(self):
        """Returns a ``GeoSeries`` of points representing the centroid of each
        geometry.

        Note that centroid does not have to be on or within original geometry.

        Examples
        --------

        >>> from shapely.geometry import Polygon, LineString, Point
        >>> s = geopandas.GeoSeries(
        ...     [
        ...         Polygon([(0, 0), (1, 1), (0, 1)]),
        ...         LineString([(0, 0), (1, 1), (1, 0)]),
        ...         Point(0, 0),
        ...     ]
        ... )
        >>> s
        0    POLYGON ((0.00000 0.00000, 1.00000 1.00000, 0....
        1    LINESTRING (0.00000 0.00000, 1.00000 1.00000, ...
        2                              POINT (0.00000 0.00000)
        dtype: geometry

        >>> s.centroid
        0    POINT (0.33333 0.66667)
        1    POINT (0.70711 0.50000)
        2    POINT (0.00000 0.00000)
        dtype: geometry

        See also
        --------
        GeoSeries.representative_point : point guaranteed to be within each geometry
        """
        return _delegate_property("centroid", self)

    def concave_hull(self, ratio=0.0, allow_holes=False):
        """Returns a ``GeoSeries`` of geometries representing the concave hull
        of each geometry.

        The concave hull of a geometry is the smallest concave `Polygon`
        containing all the points in each geometry, unless the number of points
        in the geometric object is less than three. For two points, the concave
        hull collapses to a `LineString`; for 1, a `Point`.

        The hull is constructed by removing border triangles of the Delaunay
        Triangulation of the points as long as their "size" is larger than the
        maximum edge length ratio and optionally allowing holes. The edge length factor
        is a fraction of the length difference between the longest and shortest edges
        in the Delaunay Triangulation of the input points. For further information
        on the algorithm used, see
        https://libgeos.org/doxygen/classgeos_1_1algorithm_1_1hull_1_1ConcaveHull.html

        Parameters
        ----------
        ratio : float, (optional, default 0.0)
            Number in the range [0, 1]. Higher numbers will include fewer vertices
            in the hull.
        allow_holes : bool, (optional, default False)
            If set to True, the concave hull may have holes.

        Examples
        --------

        >>> from shapely.geometry import Polygon, LineString, Point, MultiPoint
        >>> s = geopandas.GeoSeries(
        ...     [
        ...         Polygon([(0, 0), (1, 1), (0, 1)]),
        ...         LineString([(0, 0), (1, 1), (1, 0)]),
        ...         MultiPoint([(0, 0), (1, 1), (0, 1), (1, 0), (0.5, 0.5)]),
        ...         MultiPoint([(0, 0), (1, 1)]),
        ...         Point(0, 0),
        ...     ],
        ...     crs=3857
        ... )
        >>> s
        0    POLYGON ((0.000 0.000, 1.000 1.000, 0.000 1.00...
        1    LINESTRING (0.000 0.000, 1.000 1.000, 1.000 0....
        2    MULTIPOINT (0.000 0.000, 1.000 1.000, 0.000 1....
        3                MULTIPOINT (0.000 0.000, 1.000 1.000)
        4                                  POINT (0.000 0.000)
        dtype: geometry

        >>> s.concave_hull()
        0    POLYGON ((0.000 1.000, 1.000 1.000, 0.000 0.00...
        1    POLYGON ((0.000 0.000, 1.000 1.000, 1.000 0.00...
        2    POLYGON ((0.500 0.500, 0.000 1.000, 1.000 1.00...
        3                LINESTRING (0.000 0.000, 1.000 1.000)
        4                                  POINT (0.000 0.000)
        dtype: geometry

        See also
        --------
        GeoSeries.convex_hull : convex hull geometry

        """
        return _delegate_geo_method(
            "concave_hull", self, ratio=ratio, allow_holes=allow_holes
        )

    @property
    def convex_hull(self):
        """Returns a ``GeoSeries`` of geometries representing the convex hull
        of each geometry.

        The convex hull of a geometry is the smallest convex `Polygon`
        containing all the points in each geometry, unless the number of points
        in the geometric object is less than three. For two points, the convex
        hull collapses to a `LineString`; for 1, a `Point`.

        Examples
        --------

        >>> from shapely.geometry import Polygon, LineString, Point, MultiPoint
        >>> s = geopandas.GeoSeries(
        ...     [
        ...         Polygon([(0, 0), (1, 1), (0, 1)]),
        ...         LineString([(0, 0), (1, 1), (1, 0)]),
        ...         MultiPoint([(0, 0), (1, 1), (0, 1), (1, 0), (0.5, 0.5)]),
        ...         MultiPoint([(0, 0), (1, 1)]),
        ...         Point(0, 0),
        ...     ]
        ... )
        >>> s
        0    POLYGON ((0.00000 0.00000, 1.00000 1.00000, 0....
        1    LINESTRING (0.00000 0.00000, 1.00000 1.00000, ...
        2    MULTIPOINT (0.00000 0.00000, 1.00000 1.00000, ...
        3        MULTIPOINT (0.00000 0.00000, 1.00000 1.00000)
        4                              POINT (0.00000 0.00000)
        dtype: geometry

        >>> s.convex_hull
        0    POLYGON ((0.00000 0.00000, 0.00000 1.00000, 1....
        1    POLYGON ((0.00000 0.00000, 1.00000 1.00000, 1....
        2    POLYGON ((0.00000 0.00000, 0.00000 1.00000, 1....
        3        LINESTRING (0.00000 0.00000, 1.00000 1.00000)
        4                              POINT (0.00000 0.00000)
        dtype: geometry

        See also
        --------
        GeoSeries.concave_hull : concave hull geometry
        GeoSeries.envelope : bounding rectangle geometry

        """
        return _delegate_property("convex_hull", self)

    def delaunay_triangles(self, tolerance=0.0, only_edges=False):
        """Returns a ``GeoSeries`` consisting of objects representing
        the computed Delaunay triangulation around the vertices of
        an input geometry.

        The output is a ``GeometryCollection`` containing polygons
        (default) or linestrings (see only_edges).

        Returns an empty GeometryCollection if an input geometry
        contains less than 3 vertices.

        Parameters
        ----------
        tolerance : float | array-like, default 0.0
            Snap input vertices together if their distance is less than this value.
        only_edges : bool | array_like, (optional, default False)
            If set to True, the triangulation will return a collection of
            linestrings instead of polygons.

        Examples
        --------

        >>> from shapely import LineString, MultiPoint, Polygon
        >>> s = geopandas.GeoSeries(
        ...     [
        ...         MultiPoint([(50, 30), (60, 30), (100, 100)]),
        ...         Polygon([(50, 30), (60, 30), (100, 100), (50, 30)]),
        ...         LineString([(50, 30), (60, 30), (100, 100)]),
        ...     ]
        ... )
        >>> s
        0   MULTIPOINT (50.000 30.000, 60.000 30.000, 100....
        1   POLYGON ((50.000 30.000, 60.000 30.000, 100.00...
        2   LINESTRING (50.000 30.000, 60.000 30.000, 100....
        dtype: geometry

        >>> s.delaunay_triangles()
        0    GEOMETRYCOLLECTION (POLYGON ((50.000 30.000, 6...
        1    GEOMETRYCOLLECTION (POLYGON ((50.000 30.000, 6...
        2    GEOMETRYCOLLECTION (POLYGON ((50.000 30.000, 6...
        dtype: geometry

        >>> s.delaunay_triangles(only_edges=True)
        0    MULTILINESTRING ((50.000 30.000, 100.000 100.0...
        1    MULTILINESTRING ((50.000 30.000, 100.000 100.0...
        2    MULTILINESTRING ((50.000 30.000, 100.000 100.0...
        dtype: geometry
        """
        return _delegate_geo_method("delaunay_triangles", self, tolerance, only_edges)

    @property
    def envelope(self):
        """Returns a ``GeoSeries`` of geometries representing the envelope of
        each geometry.

        The envelope of a geometry is the bounding rectangle. That is, the
        point or smallest rectangular polygon (with sides parallel to the
        coordinate axes) that contains the geometry.

        Examples
        --------

        >>> from shapely.geometry import Polygon, LineString, Point, MultiPoint
        >>> s = geopandas.GeoSeries(
        ...     [
        ...         Polygon([(0, 0), (1, 1), (0, 1)]),
        ...         LineString([(0, 0), (1, 1), (1, 0)]),
        ...         MultiPoint([(0, 0), (1, 1)]),
        ...         Point(0, 0),
        ...     ]
        ... )
        >>> s
        0    POLYGON ((0.00000 0.00000, 1.00000 1.00000, 0....
        1    LINESTRING (0.00000 0.00000, 1.00000 1.00000, ...
        2        MULTIPOINT (0.00000 0.00000, 1.00000 1.00000)
        3                              POINT (0.00000 0.00000)
        dtype: geometry

        >>> s.envelope
        0    POLYGON ((0.00000 0.00000, 1.00000 0.00000, 1....
        1    POLYGON ((0.00000 0.00000, 1.00000 0.00000, 1....
        2    POLYGON ((0.00000 0.00000, 1.00000 0.00000, 1....
        3                              POINT (0.00000 0.00000)
        dtype: geometry

        See also
        --------
        GeoSeries.convex_hull : convex hull geometry
        """
        return _delegate_property("envelope", self)

    def minimum_rotated_rectangle(self):
        """Returns a ``GeoSeries`` of the general minimum bounding rectangle
        that contains the object.

        Unlike envelope this rectangle is not constrained to be parallel
        to the coordinate axes. If the convex hull of the object is a
        degenerate (line or point) this degenerate is returned.

        Examples
        --------

        >>> from shapely.geometry import Polygon, LineString, Point, MultiPoint
        >>> s = geopandas.GeoSeries(
        ...     [
        ...         Polygon([(0, 0), (1, 1), (0, 1)]),
        ...         LineString([(0, 0), (1, 1), (1, 0)]),
        ...         MultiPoint([(0, 0), (1, 1)]),
        ...         Point(0, 0),
        ...     ]
        ... )
        >>> s
        0    POLYGON ((0.00000 0.00000, 1.00000 1.00000, 0....
        1    LINESTRING (0.00000 0.00000, 1.00000 1.00000, ...
        2        MULTIPOINT (0.00000 0.00000, 1.00000 1.00000)
        3                              POINT (0.00000 0.00000)
        dtype: geometry

        >>> s.minimum_rotated_rectangle()
        0    POLYGON ((1.00000 1.00000, 0.50000 1.50000, -0...
        1    POLYGON ((0.00000 0.00000, 0.50000 -0.50000, 1...
        2        LINESTRING (0.00000 0.00000, 1.00000 1.00000)
        3                              POINT (0.00000 0.00000)
        dtype: geometry

        See also
        --------
        GeoSeries.envelope : bounding rectangle
        """
        return _delegate_geo_method("minimum_rotated_rectangle", self)

    @property
    def exterior(self):
        """Returns a ``GeoSeries`` of LinearRings representing the outer
        boundary of each polygon in the GeoSeries.

        Applies to GeoSeries containing only Polygons. Returns ``None``` for
        other geometry types.

        Examples
        --------

        >>> from shapely.geometry import Polygon, Point
        >>> s = geopandas.GeoSeries(
        ...     [
        ...         Polygon([(0, 0), (1, 1), (0, 1)]),
        ...         Polygon([(1, 0), (2, 1), (0, 0)]),
        ...         Point(0, 1)
        ...     ]
        ... )
        >>> s
        0    POLYGON ((0.00000 0.00000, 1.00000 1.00000, 0....
        1    POLYGON ((1.00000 0.00000, 2.00000 1.00000, 0....
        2                              POINT (0.00000 1.00000)
        dtype: geometry

        >>> s.exterior
        0    LINEARRING (0.00000 0.00000, 1.00000 1.00000, ...
        1    LINEARRING (1.00000 0.00000, 2.00000 1.00000, ...
        2                                                 None
        dtype: geometry

        See also
        --------
        GeoSeries.boundary : complete set-theoretic boundary
        GeoSeries.interiors : list of inner rings of each polygon
        """
        # TODO: return empty geometry for non-polygons
        return _delegate_property("exterior", self)

    def extract_unique_points(self):
        """Returns a ``GeoSeries`` of MultiPoints representing all
        distinct vertices of an input geometry.

        Examples
        --------

        >>> from shapely import LineString, Polygon
        >>> s = geopandas.GeoSeries(
        ...     [
        ...         LineString([(0, 0), (0, 0), (1, 1), (1, 1)]),
        ...         Polygon([(0, 0), (0, 0), (1, 1), (1, 1)])
        ...     ],
        ...     crs=3857
        ... )
        >>> s
        0    LINESTRING (0.000 0.000, 0.000 0.000, 1.000 1....
        1    POLYGON ((0.000 0.000, 0.000 0.000, 1.000 1.00...
        dtype: geometry

        >>> s.extract_unique_points()
        0    MULTIPOINT (0.000 0.000, 1.000 1.000)
        1    MULTIPOINT (0.000 0.000, 1.000 1.000)
        dtype: geometry

        See also
        --------

        GeoSeries.get_coordinates : extract coordinates as a :class:`~pandas.DataFrame`
        """
        return _delegate_geo_method("extract_unique_points", self)

    def offset_curve(self, distance, quad_segs=8, join_style="round", mitre_limit=5.0):
        """Returns a ``LineString`` or ``MultiLineString`` geometry at a
        distance from the object on its right or its left side.
        Parameters
        ----------
        distance : float | array-like
            Specifies the offset distance from the input geometry. Negative
            for right side offset, positive for left side offset.
        quad_segs : int (optional, default 8)
            Specifies the number of linear segments in a quarter circle in the
            approximation of circular arcs.
        join_style : {'round', 'bevel', 'mitre'}, (optional, default 'round')
            Specifies the shape of outside corners. 'round' results in
            rounded shapes. 'bevel' results in a beveled edge that touches the
            original vertex. 'mitre' results in a single vertex that is beveled
            depending on the ``mitre_limit`` parameter.
        mitre_limit : float (optional, default 5.0)
            Crops of 'mitre'-style joins if the point is displaced from the
            buffered vertex by more than this limit.

        See http://shapely.readthedocs.io/en/latest/manual.html#object.offset_curve
        for details.

        Examples
        --------

        >>> from shapely.geometry import LineString
        >>> s = geopandas.GeoSeries(
        ...     [
        ...         LineString([(0, 0), (0, 1), (1, 1)]),
        ...     ],
        ...     crs=3857
        ... )
        >>> s
        0    LINESTRING (0.000 0.000, 0.000 1.000, 1.000 1....
        dtype: geometry

        >>> s.offset_curve(1)
        0    LINESTRING (-1.000 0.000, -1.000 1.000, -0.981...
        dtype: geometry
        """
        return _delegate_geo_method(
            "offset_curve",
            self,
            distance,
            quad_segs=quad_segs,
            join_style=join_style,
            mitre_limit=mitre_limit,
        )

    @property
    def interiors(self):
        """Returns a ``Series`` of List representing the
        inner rings of each polygon in the GeoSeries.

        Applies to GeoSeries containing only Polygons.

        Returns
        -------
        inner_rings: Series of List
            Inner rings of each polygon in the GeoSeries.

        Examples
        --------

        >>> from shapely.geometry import Polygon
        >>> s = geopandas.GeoSeries(
        ...     [
        ...         Polygon(
        ...             [(0, 0), (0, 5), (5, 5), (5, 0)],
        ...             [[(1, 1), (2, 1), (1, 2)], [(1, 4), (2, 4), (2, 3)]],
        ...         ),
        ...         Polygon([(1, 0), (2, 1), (0, 0)]),
        ...     ]
        ... )
        >>> s
        0    POLYGON ((0.00000 0.00000, 0.00000 5.00000, 5....
        1    POLYGON ((1.00000 0.00000, 2.00000 1.00000, 0....
        dtype: geometry

        >>> s.interiors
        0    [LINEARRING (1 1, 2 1, 1 2, 1 1), LINEARRING (...
        1                                                   []
        dtype: object

        See also
        --------
        GeoSeries.exterior : outer boundary
        """
        return _delegate_property("interiors", self)

    def remove_repeated_points(self, tolerance=0.0):
        """Returns a ``GeoSeries`` containing a copy of the input geometry
        with repeated points removed.

        From the start of the coordinate sequence, each next point within the
        tolerance is removed.

        Removing repeated points with a non-zero tolerance may result in an invalid
        geometry being returned.

        Parameters
        ----------
        tolerance : float, default 0.0
            Remove all points within this distance of each other. Use 0.0
            to remove only exactly repeated points (the default).

        Examples
        --------

        >>> from shapely import LineString, Polygon
        >>> s = geopandas.GeoSeries(
        ...     [
        ...        LineString([(0, 0), (0, 0), (1, 0)]),
        ...        Polygon([(0, 0), (0, 0.5), (0, 1), (0.5, 1), (0,0)]),
        ...     ],
        ...     crs=3857
        ... )
        >>> s
        0    LINESTRING (0.000 0.000, 0.000 0.000, 1.000 0....
        1    POLYGON ((0.000 0.000, 0.000 0.500, 0.000 1.00...
        dtype: geometry

        >>> s.remove_repeated_points(tolerance=0.0)
        0                LINESTRING (0.000 0.000, 1.000 0.000)
        1    POLYGON ((0.000 0.000, 0.000 0.500, 0.000 1.00...
        dtype: geometry
        """
        return _delegate_geo_method("remove_repeated_points", self, tolerance=tolerance)

    def representative_point(self):
        """Returns a ``GeoSeries`` of (cheaply computed) points that are
        guaranteed to be within each geometry.

        Examples
        --------

        >>> from shapely.geometry import Polygon, LineString, Point
        >>> s = geopandas.GeoSeries(
        ...     [
        ...         Polygon([(0, 0), (1, 1), (0, 1)]),
        ...         LineString([(0, 0), (1, 1), (1, 0)]),
        ...         Point(0, 0),
        ...     ]
        ... )
        >>> s
        0    POLYGON ((0.00000 0.00000, 1.00000 1.00000, 0....
        1    LINESTRING (0.00000 0.00000, 1.00000 1.00000, ...
        2                              POINT (0.00000 0.00000)
        dtype: geometry

        >>> s.representative_point()
        0    POINT (0.25000 0.50000)
        1    POINT (1.00000 1.00000)
        2    POINT (0.00000 0.00000)
        dtype: geometry

        See also
        --------
        GeoSeries.centroid : geometric centroid
        """
        return _delegate_geo_method("representative_point", self)

    def minimum_bounding_circle(self):
        """Returns a ``GeoSeries`` of geometries representing the minimum bounding
        circle that encloses each geometry.

        Examples
        --------

        >>> from shapely.geometry import Polygon, LineString, Point
        >>> s = geopandas.GeoSeries(
        ...     [
        ...         Polygon([(0, 0), (1, 1), (0, 1), (0, 0)]),
        ...         LineString([(0, 0), (1, 1), (1, 0)]),
        ...         Point(0, 0),
        ...     ]
        ... )
        >>> s
        0    POLYGON ((0.00000 0.00000, 1.00000 1.00000, 0....
        1    LINESTRING (0.00000 0.00000, 1.00000 1.00000, ...
        2                              POINT (0.00000 0.00000)
        dtype: geometry

        >>> s.minimum_bounding_circle()
        0    POLYGON ((1.20711 0.50000, 1.19352 0.36205, 1....
        1    POLYGON ((1.20711 0.50000, 1.19352 0.36205, 1....
        2                              POINT (0.00000 0.00000)
        dtype: geometry

        See also
        --------
        GeoSeries.convex_hull : convex hull geometry
        """
        return _delegate_geo_method("minimum_bounding_circle", self)

    def minimum_bounding_radius(self):
        """Returns a `Series` of the radii of the minimum bounding circles
        that enclose each geometry.

        Examples
        --------
        >>> from shapely.geometry import Point, LineString, Polygon
        >>> s = geopandas.GeoSeries(
        ...     [
        ...         Polygon([(0, 0), (1, 1), (0, 1), (0, 0)]),
        ...         LineString([(0, 0), (1, 1), (1, 0)]),
        ...         Point(0,0),
        ...     ]
        ... )
        >>> s
        0    POLYGON ((0.00000 0.00000, 1.00000 1.00000, 0....
        1    LINESTRING (0.00000 0.00000, 1.00000 1.00000, ...
        2                              POINT (0.00000 0.00000)
        dtype: geometry
        >>> s.minimum_bounding_radius()
        0    0.707107
        1    0.707107
        2    0.000000
        dtype: float64

        See also
        --------
        GeoSeries.minumum_bounding_circle : minimum bounding circle (geometry)

        """
        return Series(self.geometry.values.minimum_bounding_radius(), index=self.index)

    def normalize(self):
        """Returns a ``GeoSeries`` of normalized
        geometries to normal form (or canonical form).

        This method orders the coordinates, rings of a polygon and parts of
        multi geometries consistently. Typically useful for testing purposes
        (for example in combination with `equals_exact`).

        Examples
        --------

        >>> from shapely.geometry import Polygon, LineString, Point
        >>> s = geopandas.GeoSeries(
        ...     [
        ...         Polygon([(0, 0), (1, 1), (0, 1)]),
        ...         LineString([(0, 0), (1, 1), (1, 0)]),
        ...         Point(0, 0),
        ...     ],
        ...     crs='EPSG:3857'
        ... )
        >>> s
        0    POLYGON ((0.000 0.000, 1.000 1.000, 0.000 1.00...
        1    LINESTRING (0.000 0.000, 1.000 1.000, 1.000 0....
        2                                  POINT (0.000 0.000)
        dtype: geometry

        >>> s.normalize()
        0    POLYGON ((0.000 0.000, 0.000 1.000, 1.000 1.00...
        1    LINESTRING (0.000 0.000, 1.000 1.000, 1.000 0....
        2                                  POINT (0.000 0.000)
        dtype: geometry
        """
        return _delegate_geo_method("normalize", self)

    def make_valid(self):
        """
        Repairs invalid geometries.

        Returns a ``GeoSeries`` with valid geometries.
        If the input geometry is already valid, then it will be preserved.
        In many cases, in order to create a valid geometry, the input
        geometry must be split into multiple parts or multiple geometries.
        If the geometry must be split into multiple parts of the same type
        to be made valid, then a multi-part geometry will be returned
        (e.g. a MultiPolygon).
        If the geometry must be split into multiple parts of different types
        to be made valid, then a GeometryCollection will be returned.

        Examples
        --------
        >>> from shapely.geometry import MultiPolygon, Polygon, LineString, Point
        >>> s = geopandas.GeoSeries(
        ...     [
        ...         Polygon([(0, 0), (0, 2), (1, 1), (2, 2), (2, 0), (1, 1), (0, 0)]),
        ...         Polygon([(0, 2), (0, 1), (2, 0), (0, 0), (0, 2)]),
        ...         LineString([(0, 0), (1, 1), (1, 0)]),
        ...     ],
        ...     crs='EPSG:3857',
        ... )
        >>> s
        0    POLYGON ((0.000 0.000, 0.000 2.000, 1.000 1.00...
        1    POLYGON ((0.000 2.000, 0.000 1.000, 2.000 0.00...
        2    LINESTRING (0.000 0.000, 1.000 1.000, 1.000 0....
        dtype: geometry

        >>> s.make_valid()
        0    MULTIPOLYGON (((1.000 1.000, 0.000 0.000, 0.00...
        1    GEOMETRYCOLLECTION (POLYGON ((2.000 0.000, 0.0...
        2    LINESTRING (0.000 0.000, 1.000 1.000, 1.000 0....
        dtype: geometry
        """
        return _delegate_geo_method("make_valid", self)

    def reverse(self):
        """Returns a ``GeoSeries`` with the order of coordinates reversed.

        Examples
        --------

        >>> from shapely.geometry import Polygon, LineString, Point
        >>> s = geopandas.GeoSeries(
        ...     [
        ...         Polygon([(0, 0), (1, 1), (0, 1)]),
        ...         LineString([(0, 0), (1, 1), (1, 0)]),
        ...         Point(0, 0),
        ...     ]
        ... )
        >>> s
        0    POLYGON ((0.00000 0.00000, 1.00000 1.00000, 0....
        1    LINESTRING (0.00000 0.00000, 1.00000 1.00000, ...
        2                              POINT (0.00000 0.00000)
        dtype: geometry

        >>> s.reverse()
        0    POLYGON ((0.00000 0.00000, 0.00000 1.00000, 1....
        1    LINESTRING (1.00000 0.00000, 1.00000 1.00000, ...
        2    POINT (0.00000 0.00000)
        dtype: geometry

        See also
        --------
        GeoSeries.normalize : normalize order of coordinates
        """
        return _delegate_geo_method("reverse", self)

    def segmentize(self, max_segment_length):
        """Returns a ``GeoSeries`` with vertices added to line segments based on
        maximum segment length.

        Additional vertices will be added to every line segment in an input geometry so
        that segments are no longer than the provided maximum segment length. New
        vertices will evenly subdivide each segment. Only linear components of input
        geometries are densified; other geometries are returned unmodified.

        Parameters
        ----------
        max_segment_length : float | array-like
            Additional vertices will be added so that all line segments are no longer
            than this value. Must be greater than 0.

        Returns
        -------
        GeoSeries

        Examples
        --------
        >>> from shapely.geometry import Polygon, LineString
        >>> s = geopandas.GeoSeries(
        ...     [
        ...         LineString([(0, 0), (0, 10)]),
        ...         Polygon([(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)]),
        ...     ],
        ...     crs=3857
        ... )
        >>> s
        0               LINESTRING (0.000 0.000, 0.000 10.000)
        1    POLYGON ((0.000 0.000, 10.000 0.000, 10.000 10...
        dtype: geometry

        >>> s.segmentize(max_segment_length=5)
        0    LINESTRING (0.000 0.000, 0.000 5.000, 0.000 10...
        1    POLYGON ((0.000 0.000, 5.000 0.000, 10.000 0.0...
        dtype: geometry
        """
        return _delegate_geo_method("segmentize", self, max_segment_length)

    #
    # Reduction operations that return a Shapely geometry
    #

    @property
    def cascaded_union(self):
        """Deprecated: use `unary_union` instead"""
        warn(
            "The 'cascaded_union' attribute is deprecated, use 'unary_union' instead",
            FutureWarning,
            stacklevel=2,
        )
        return self.geometry.values.unary_union()

    @property
    def unary_union(self):
        """Returns a geometry containing the union of all geometries in the
        ``GeoSeries``.

        Examples
        --------

        >>> from shapely.geometry import box
        >>> s = geopandas.GeoSeries([box(0,0,1,1), box(0,0,2,2)])
        >>> s
        0    POLYGON ((1.00000 0.00000, 1.00000 1.00000, 0....
        1    POLYGON ((2.00000 0.00000, 2.00000 2.00000, 0....
        dtype: geometry

        >>> union = s.unary_union
        >>> print(union)
        POLYGON ((0 1, 0 2, 2 2, 2 0, 1 0, 0 0, 0 1))
        """
        return self.geometry.values.unary_union()

    #
    # Binary operations that return a pandas Series
    #

    def contains(self, other, align=True):
        """Returns a ``Series`` of ``dtype('bool')`` with value ``True`` for
        each aligned geometry that contains `other`.

        An object is said to contain `other` if at least one point of `other` lies in
        the interior and no points of `other` lie in the exterior of the object.
        (Therefore, any given polygon does not contain its own boundary – there is not
        any point that lies in the interior.)
        If either object is empty, this operation returns ``False``.

        This is the inverse of :meth:`within` in the sense that the expression
        ``a.contains(b) == b.within(a)`` always evaluates to ``True``.

        The operation works on a 1-to-1 row-wise manner:

        .. image:: ../../../_static/binary_op-01.svg
           :align: center

        Parameters
        ----------
        other : GeoSeries or geometric object
            The GeoSeries (elementwise) or geometric object to test if it
            is contained.
        align : bool (default True)
            If True, automatically aligns GeoSeries based on their indices.
            If False, the order of elements is preserved.

        Returns
        -------
        Series (bool)

        Examples
        --------
        >>> from shapely.geometry import Polygon, LineString, Point
        >>> s = geopandas.GeoSeries(
        ...     [
        ...         Polygon([(0, 0), (1, 1), (0, 1)]),
        ...         LineString([(0, 0), (0, 2)]),
        ...         LineString([(0, 0), (0, 1)]),
        ...         Point(0, 1),
        ...     ],
        ...     index=range(0, 4),
        ... )
        >>> s2 = geopandas.GeoSeries(
        ...     [
        ...         Polygon([(0, 0), (2, 2), (0, 2)]),
        ...         Polygon([(0, 0), (1, 2), (0, 2)]),
        ...         LineString([(0, 0), (0, 2)]),
        ...         Point(0, 1),
        ...     ],
        ...     index=range(1, 5),
        ... )

        >>> s
        0    POLYGON ((0.00000 0.00000, 1.00000 1.00000, 0....
        1        LINESTRING (0.00000 0.00000, 0.00000 2.00000)
        2        LINESTRING (0.00000 0.00000, 0.00000 1.00000)
        3                              POINT (0.00000 1.00000)
        dtype: geometry

        >>> s2
        1    POLYGON ((0.00000 0.00000, 2.00000 2.00000, 0....
        2    POLYGON ((0.00000 0.00000, 1.00000 2.00000, 0....
        3        LINESTRING (0.00000 0.00000, 0.00000 2.00000)
        4                              POINT (0.00000 1.00000)
        dtype: geometry

        We can check if each geometry of GeoSeries contains a single
        geometry:

        .. image:: ../../../_static/binary_op-03.svg
           :align: center

        >>> point = Point(0, 1)
        >>> s.contains(point)
        0    False
        1     True
        2    False
        3     True
        dtype: bool

        We can also check two GeoSeries against each other, row by row.
        The GeoSeries above have different indices. We can either align both GeoSeries
        based on index values and compare elements with the same index using
        ``align=True`` or ignore index and compare elements based on their matching
        order using ``align=False``:

        .. image:: ../../../_static/binary_op-02.svg

        >>> s2.contains(s, align=True)
        0    False
        1    False
        2    False
        3     True
        4    False
        dtype: bool

        >>> s2.contains(s, align=False)
        1     True
        2    False
        3     True
        4     True
        dtype: bool

        Notes
        -----
        This method works in a row-wise manner. It does not check if an element
        of one GeoSeries ``contains`` *any* element of the other one.

        See also
        --------
        GeoSeries.within
        """
        return _binary_op("contains", self, other, align)

    def geom_equals(self, other, align=True):
        """Returns a ``Series`` of ``dtype('bool')`` with value ``True`` for
        each aligned geometry equal to `other`.

        An object is said to be equal to `other` if its set-theoretic
        `boundary`, `interior`, and `exterior` coincides with those of the
        other.

        The operation works on a 1-to-1 row-wise manner:

        .. image:: ../../../_static/binary_op-01.svg
           :align: center

        Parameters
        ----------
        other : GeoSeries or geometric object
            The GeoSeries (elementwise) or geometric object to test for
            equality.
        align : bool (default True)
            If True, automatically aligns GeoSeries based on their indices.
            If False, the order of elements is preserved.

        Returns
        -------
        Series (bool)

        Examples
        --------
        >>> from shapely.geometry import Polygon, LineString, Point
        >>> s = geopandas.GeoSeries(
        ...     [
        ...         Polygon([(0, 0), (2, 2), (0, 2)]),
        ...         Polygon([(0, 0), (1, 2), (0, 2)]),
        ...         LineString([(0, 0), (0, 2)]),
        ...         Point(0, 1),
        ...     ],
        ... )
        >>> s2 = geopandas.GeoSeries(
        ...     [
        ...         Polygon([(0, 0), (2, 2), (0, 2)]),
        ...         Polygon([(0, 0), (1, 2), (0, 2)]),
        ...         Point(0, 1),
        ...         LineString([(0, 0), (0, 2)]),
        ...     ],
        ...     index=range(1, 5),
        ... )

        >>> s
        0    POLYGON ((0.00000 0.00000, 2.00000 2.00000, 0....
        1    POLYGON ((0.00000 0.00000, 1.00000 2.00000, 0....
        2        LINESTRING (0.00000 0.00000, 0.00000 2.00000)
        3                              POINT (0.00000 1.00000)
        dtype: geometry

        >>> s2
        1    POLYGON ((0.00000 0.00000, 2.00000 2.00000, 0....
        2    POLYGON ((0.00000 0.00000, 1.00000 2.00000, 0....
        3                              POINT (0.00000 1.00000)
        4        LINESTRING (0.00000 0.00000, 0.00000 2.00000)
        dtype: geometry

        We can check if each geometry of GeoSeries contains a single
        geometry:

        .. image:: ../../../_static/binary_op-03.svg
           :align: center

        >>> polygon = Polygon([(0, 0), (2, 2), (0, 2)])
        >>> s.geom_equals(polygon)
        0     True
        1    False
        2    False
        3    False
        dtype: bool

        We can also check two GeoSeries against each other, row by row.
        The GeoSeries above have different indices. We can either align both GeoSeries
        based on index values and compare elements with the same index using
        ``align=True`` or ignore index and compare elements based on their matching
        order using ``align=False``:

        .. image:: ../../../_static/binary_op-02.svg

        >>> s.geom_equals(s2)
        0    False
        1    False
        2    False
        3     True
        4    False
        dtype: bool

        >>> s.geom_equals(s2, align=False)
        0     True
        1     True
        2    False
        3    False
        dtype: bool

        Notes
        -----
        This method works in a row-wise manner. It does not check if an element
        of one GeoSeries is equal to *any* element of the other one.

        See also
        --------
        GeoSeries.geom_equals_exact

        """
        return _binary_op("geom_equals", self, other, align)

    def geom_almost_equals(self, other, decimal=6, align=True):
        """Returns a ``Series`` of ``dtype('bool')`` with value ``True`` if
        each aligned geometry is approximately equal to `other`.

        Approximate equality is tested at all points to the specified `decimal`
        place precision.

        The operation works on a 1-to-1 row-wise manner:

        .. image:: ../../../_static/binary_op-01.svg
           :align: center

        Parameters
        ----------
        other : GeoSeries or geometric object
            The GeoSeries (elementwise) or geometric object to compare to.
        decimal : int
            Decimal place precision used when testing for approximate equality.
        align : bool (default True)
            If True, automatically aligns GeoSeries based on their indices.
            If False, the order of elements is preserved.

        Returns
        -------
        Series (bool)

        Examples
        --------
        >>> from shapely.geometry import Point
        >>> s = geopandas.GeoSeries(
        ...     [
        ...         Point(0, 1.1),
        ...         Point(0, 1.01),
        ...         Point(0, 1.001),
        ...     ],
        ... )

        >>> s
        0    POINT (0.00000 1.10000)
        1    POINT (0.00000 1.01000)
        2    POINT (0.00000 1.00100)
        dtype: geometry


        >>> s.geom_almost_equals(Point(0, 1), decimal=2)
        0    False
        1    False
        2     True
        dtype: bool

        >>> s.geom_almost_equals(Point(0, 1), decimal=1)
        0    False
        1     True
        2     True
        dtype: bool

        Notes
        -----
        This method works in a row-wise manner. It does not check if an element
        of one GeoSeries is equal to *any* element of the other one.

        See also
        --------
        GeoSeries.geom_equals
        GeoSeries.geom_equals_exact

        """
        warnings.warn(
            "The 'geom_almost_equals()' method is deprecated because the name is "
            "confusing. The 'geom_equals_exact()' method should be used instead.",
            FutureWarning,
            stacklevel=2,
        )
        tolerance = 0.5 * 10 ** (-decimal)
        return _binary_op(
            "geom_equals_exact", self, other, tolerance=tolerance, align=align
        )

    def geom_equals_exact(self, other, tolerance, align=True):
        """Return True for all geometries that equal aligned *other* to a given
        tolerance, else False.

        The operation works on a 1-to-1 row-wise manner:

        .. image:: ../../../_static/binary_op-01.svg
           :align: center

        Parameters
        ----------
        other : GeoSeries or geometric object
            The GeoSeries (elementwise) or geometric object to compare to.
        tolerance : float
            Decimal place precision used when testing for approximate equality.
        align : bool (default True)
            If True, automatically aligns GeoSeries based on their indices.
            If False, the order of elements is preserved.

        Returns
        -------
        Series (bool)

        Examples
        --------
        >>> from shapely.geometry import Point
        >>> s = geopandas.GeoSeries(
        ...     [
        ...         Point(0, 1.1),
        ...         Point(0, 1.0),
        ...         Point(0, 1.2),
        ...     ]
        ... )

        >>> s
        0    POINT (0.00000 1.10000)
        1    POINT (0.00000 1.00000)
        2    POINT (0.00000 1.20000)
        dtype: geometry


        >>> s.geom_equals_exact(Point(0, 1), tolerance=0.1)
        0    False
        1     True
        2    False
        dtype: bool

        >>> s.geom_equals_exact(Point(0, 1), tolerance=0.15)
        0     True
        1     True
        2    False
        dtype: bool

        Notes
        -----
        This method works in a row-wise manner. It does not check if an element
        of one GeoSeries is equal to *any* element of the other one.

        See also
        --------
        GeoSeries.geom_equals
        """
        return _binary_op(
            "geom_equals_exact", self, other, tolerance=tolerance, align=align
        )

    def crosses(self, other, align=True):
        """Returns a ``Series`` of ``dtype('bool')`` with value ``True`` for
        each aligned geometry that cross `other`.

        An object is said to cross `other` if its `interior` intersects the
        `interior` of the other but does not contain it, and the dimension of
        the intersection is less than the dimension of the one or the other.

        The operation works on a 1-to-1 row-wise manner:

        .. image:: ../../../_static/binary_op-01.svg
           :align: center

        Parameters
        ----------
        other : GeoSeries or geometric object
            The GeoSeries (elementwise) or geometric object to test if is
            crossed.
        align : bool (default True)
            If True, automatically aligns GeoSeries based on their indices.
            If False, the order of elements is preserved.

        Returns
        -------
        Series (bool)

        Examples
        --------
        >>> from shapely.geometry import Polygon, LineString, Point
        >>> s = geopandas.GeoSeries(
        ...     [
        ...         Polygon([(0, 0), (2, 2), (0, 2)]),
        ...         LineString([(0, 0), (2, 2)]),
        ...         LineString([(2, 0), (0, 2)]),
        ...         Point(0, 1),
        ...     ],
        ... )
        >>> s2 = geopandas.GeoSeries(
        ...     [
        ...         LineString([(1, 0), (1, 3)]),
        ...         LineString([(2, 0), (0, 2)]),
        ...         Point(1, 1),
        ...         Point(0, 1),
        ...     ],
        ...     index=range(1, 5),
        ... )

        >>> s
        0    POLYGON ((0.00000 0.00000, 2.00000 2.00000, 0....
        1        LINESTRING (0.00000 0.00000, 2.00000 2.00000)
        2        LINESTRING (2.00000 0.00000, 0.00000 2.00000)
        3                              POINT (0.00000 1.00000)
        dtype: geometry

        >>> s2
        1    LINESTRING (1.00000 0.00000, 1.00000 3.00000)
        2    LINESTRING (2.00000 0.00000, 0.00000 2.00000)
        3                          POINT (1.00000 1.00000)
        4                          POINT (0.00000 1.00000)
        dtype: geometry

        We can check if each geometry of GeoSeries crosses a single
        geometry:

        .. image:: ../../../_static/binary_op-03.svg
           :align: center

        >>> line = LineString([(-1, 1), (3, 1)])
        >>> s.crosses(line)
        0     True
        1     True
        2     True
        3    False
        dtype: bool

        We can also check two GeoSeries against each other, row by row.
        The GeoSeries above have different indices. We can either align both GeoSeries
        based on index values and compare elements with the same index using
        ``align=True`` or ignore index and compare elements based on their matching
        order using ``align=False``:

        .. image:: ../../../_static/binary_op-02.svg

        >>> s.crosses(s2, align=True)
        0    False
        1     True
        2    False
        3    False
        4    False
        dtype: bool

        >>> s.crosses(s2, align=False)
        0     True
        1     True
        2    False
        3    False
        dtype: bool

        Notice that a line does not cross a point that it contains.

        Notes
        -----
        This method works in a row-wise manner. It does not check if an element
        of one GeoSeries ``crosses`` *any* element of the other one.

        See also
        --------
        GeoSeries.disjoint
        GeoSeries.intersects

        """
        return _binary_op("crosses", self, other, align)

    def disjoint(self, other, align=True):
        """Returns a ``Series`` of ``dtype('bool')`` with value ``True`` for
        each aligned geometry disjoint to `other`.

        An object is said to be disjoint to `other` if its `boundary` and
        `interior` does not intersect at all with those of the other.

        The operation works on a 1-to-1 row-wise manner:

        .. image:: ../../../_static/binary_op-01.svg
           :align: center

        Parameters
        ----------
        other : GeoSeries or geometric object
            The GeoSeries (elementwise) or geometric object to test if is
            disjoint.
        align : bool (default True)
            If True, automatically aligns GeoSeries based on their indices.
            If False, the order of elements is preserved.

        Returns
        -------
        Series (bool)

        Examples
        --------
        >>> from shapely.geometry import Polygon, LineString, Point
        >>> s = geopandas.GeoSeries(
        ...     [
        ...         Polygon([(0, 0), (2, 2), (0, 2)]),
        ...         LineString([(0, 0), (2, 2)]),
        ...         LineString([(2, 0), (0, 2)]),
        ...         Point(0, 1),
        ...     ],
        ... )
        >>> s2 = geopandas.GeoSeries(
        ...     [
        ...         Polygon([(-1, 0), (-1, 2), (0, -2)]),
        ...         LineString([(0, 0), (0, 1)]),
        ...         Point(1, 1),
        ...         Point(0, 0),
        ...     ],
        ... )

        >>> s
        0    POLYGON ((0.00000 0.00000, 2.00000 2.00000, 0....
        1        LINESTRING (0.00000 0.00000, 2.00000 2.00000)
        2        LINESTRING (2.00000 0.00000, 0.00000 2.00000)
        3                              POINT (0.00000 1.00000)
        dtype: geometry

        >>> s2
        0    POLYGON ((-1.00000 0.00000, -1.00000 2.00000, ...
        1        LINESTRING (0.00000 0.00000, 0.00000 1.00000)
        2                              POINT (1.00000 1.00000)
        3                              POINT (0.00000 0.00000)
        dtype: geometry

        We can check each geometry of GeoSeries to a single
        geometry:

        .. image:: ../../../_static/binary_op-03.svg
           :align: center

        >>> line = LineString([(0, 0), (2, 0)])
        >>> s.disjoint(line)
        0    False
        1    False
        2    False
        3     True
        dtype: bool

        We can also check two GeoSeries against each other, row by row.
        We can either align both GeoSeries
        based on index values and compare elements with the same index using
        ``align=True`` or ignore index and compare elements based on their matching
        order using ``align=False``:

        .. image:: ../../../_static/binary_op-02.svg

        >>> s.disjoint(s2)
        0     True
        1    False
        2    False
        3     True
        dtype: bool

        Notes
        -----
        This method works in a row-wise manner. It does not check if an element
        of one GeoSeries is equal to *any* element of the other one.

        See also
        --------
        GeoSeries.intersects
        GeoSeries.touches

        """
        return _binary_op("disjoint", self, other, align)

    def intersects(self, other, align=True):
        """Returns a ``Series`` of ``dtype('bool')`` with value ``True`` for
        each aligned geometry that intersects `other`.

        An object is said to intersect `other` if its `boundary` and `interior`
        intersects in any way with those of the other.

        The operation works on a 1-to-1 row-wise manner:

        .. image:: ../../../_static/binary_op-01.svg
           :align: center

        Parameters
        ----------
        other : GeoSeries or geometric object
            The GeoSeries (elementwise) or geometric object to test if is
            intersected.
        align : bool (default True)
            If True, automatically aligns GeoSeries based on their indices.
            If False, the order of elements is preserved.

        Returns
        -------
        Series (bool)

        Examples
        --------
        >>> from shapely.geometry import Polygon, LineString, Point
        >>> s = geopandas.GeoSeries(
        ...     [
        ...         Polygon([(0, 0), (2, 2), (0, 2)]),
        ...         LineString([(0, 0), (2, 2)]),
        ...         LineString([(2, 0), (0, 2)]),
        ...         Point(0, 1),
        ...     ],
        ... )
        >>> s2 = geopandas.GeoSeries(
        ...     [
        ...         LineString([(1, 0), (1, 3)]),
        ...         LineString([(2, 0), (0, 2)]),
        ...         Point(1, 1),
        ...         Point(0, 1),
        ...     ],
        ...     index=range(1, 5),
        ... )

        >>> s
        0    POLYGON ((0.00000 0.00000, 2.00000 2.00000, 0....
        1        LINESTRING (0.00000 0.00000, 2.00000 2.00000)
        2        LINESTRING (2.00000 0.00000, 0.00000 2.00000)
        3                              POINT (0.00000 1.00000)
        dtype: geometry

        >>> s2
        1    LINESTRING (1.00000 0.00000, 1.00000 3.00000)
        2    LINESTRING (2.00000 0.00000, 0.00000 2.00000)
        3                          POINT (1.00000 1.00000)
        4                          POINT (0.00000 1.00000)
        dtype: geometry

        We can check if each geometry of GeoSeries crosses a single
        geometry:

        .. image:: ../../../_static/binary_op-03.svg
           :align: center

        >>> line = LineString([(-1, 1), (3, 1)])
        >>> s.intersects(line)
        0    True
        1    True
        2    True
        3    True
        dtype: bool

        We can also check two GeoSeries against each other, row by row.
        The GeoSeries above have different indices. We can either align both GeoSeries
        based on index values and compare elements with the same index using
        ``align=True`` or ignore index and compare elements based on their matching
        order using ``align=False``:

        .. image:: ../../../_static/binary_op-02.svg

        >>> s.intersects(s2, align=True)
        0    False
        1     True
        2     True
        3    False
        4    False
        dtype: bool

        >>> s.intersects(s2, align=False)
        0    True
        1    True
        2    True
        3    True
        dtype: bool

        Notes
        -----
        This method works in a row-wise manner. It does not check if an element
        of one GeoSeries ``crosses`` *any* element of the other one.

        See also
        --------
        GeoSeries.disjoint
        GeoSeries.crosses
        GeoSeries.touches
        GeoSeries.intersection
        """
        return _binary_op("intersects", self, other, align)

    def overlaps(self, other, align=True):
        """Returns True for all aligned geometries that overlap *other*, else False.

        Geometries overlaps if they have more than one but not all
        points in common, have the same dimension, and the intersection of the
        interiors of the geometries has the same dimension as the geometries
        themselves.

        The operation works on a 1-to-1 row-wise manner:

        .. image:: ../../../_static/binary_op-01.svg
           :align: center

        Parameters
        ----------
        other : GeoSeries or geometric object
            The GeoSeries (elementwise) or geometric object to test if
            overlaps.
        align : bool (default True)
            If True, automatically aligns GeoSeries based on their indices.
            If False, the order of elements is preserved.

        Returns
        -------
        Series (bool)

        Examples
        --------
        >>> from shapely.geometry import Polygon, LineString, MultiPoint, Point
        >>> s = geopandas.GeoSeries(
        ...     [
        ...         Polygon([(0, 0), (2, 2), (0, 2)]),
        ...         Polygon([(0, 0), (2, 2), (0, 2)]),
        ...         LineString([(0, 0), (2, 2)]),
        ...         MultiPoint([(0, 0), (0, 1)]),
        ...     ],
        ... )
        >>> s2 = geopandas.GeoSeries(
        ...     [
        ...         Polygon([(0, 0), (2, 0), (0, 2)]),
        ...         LineString([(0, 1), (1, 1)]),
        ...         LineString([(1, 1), (3, 3)]),
        ...         Point(0, 1),
        ...     ],
        ...     index=range(1, 5),
        ... )

        >>> s
        0    POLYGON ((0.00000 0.00000, 2.00000 2.00000, 0....
        1    POLYGON ((0.00000 0.00000, 2.00000 2.00000, 0....
        2        LINESTRING (0.00000 0.00000, 2.00000 2.00000)
        3        MULTIPOINT (0.00000 0.00000, 0.00000 1.00000)
        dtype: geometry

        >>> s2
        1    POLYGON ((0.00000 0.00000, 2.00000 0.00000, 0....
        2        LINESTRING (0.00000 1.00000, 1.00000 1.00000)
        3        LINESTRING (1.00000 1.00000, 3.00000 3.00000)
        4                              POINT (0.00000 1.00000)
        dtype: geometry

        We can check if each geometry of GeoSeries overlaps a single
        geometry:

        .. image:: ../../../_static/binary_op-03.svg
           :align: center

        >>> polygon = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        >>> s.overlaps(polygon)
        0     True
        1     True
        2    False
        3    False
        dtype: bool

        We can also check two GeoSeries against each other, row by row.
        The GeoSeries above have different indices. We can either align both GeoSeries
        based on index values and compare elements with the same index using
        ``align=True`` or ignore index and compare elements based on their matching
        order using ``align=False``:

        .. image:: ../../../_static/binary_op-02.svg

        >>> s.overlaps(s2)
        0    False
        1     True
        2    False
        3    False
        4    False
        dtype: bool

        >>> s.overlaps(s2, align=False)
        0     True
        1    False
        2     True
        3    False
        dtype: bool

        Notes
        -----
        This method works in a row-wise manner. It does not check if an element
        of one GeoSeries ``overlaps`` *any* element of the other one.

        See also
        --------
        GeoSeries.crosses
        GeoSeries.intersects

        """
        return _binary_op("overlaps", self, other, align)

    def touches(self, other, align=True):
        """Returns a ``Series`` of ``dtype('bool')`` with value ``True`` for
        each aligned geometry that touches `other`.

        An object is said to touch `other` if it has at least one point in
        common with `other` and its interior does not intersect with any part
        of the other. Overlapping features therefore do not touch.

        The operation works on a 1-to-1 row-wise manner:

        .. image:: ../../../_static/binary_op-01.svg
           :align: center

        Parameters
        ----------
        other : GeoSeries or geometric object
            The GeoSeries (elementwise) or geometric object to test if is
            touched.
        align : bool (default True)
            If True, automatically aligns GeoSeries based on their indices.
            If False, the order of elements is preserved.

        Returns
        -------
        Series (bool)

        Examples
        --------
        >>> from shapely.geometry import Polygon, LineString, MultiPoint, Point
        >>> s = geopandas.GeoSeries(
        ...     [
        ...         Polygon([(0, 0), (2, 2), (0, 2)]),
        ...         Polygon([(0, 0), (2, 2), (0, 2)]),
        ...         LineString([(0, 0), (2, 2)]),
        ...         MultiPoint([(0, 0), (0, 1)]),
        ...     ],
        ... )
        >>> s2 = geopandas.GeoSeries(
        ...     [
        ...         Polygon([(0, 0), (-2, 0), (0, -2)]),
        ...         LineString([(0, 1), (1, 1)]),
        ...         LineString([(1, 1), (3, 0)]),
        ...         Point(0, 1),
        ...     ],
        ...     index=range(1, 5),
        ... )

        >>> s
        0    POLYGON ((0.00000 0.00000, 2.00000 2.00000, 0....
        1    POLYGON ((0.00000 0.00000, 2.00000 2.00000, 0....
        2        LINESTRING (0.00000 0.00000, 2.00000 2.00000)
        3        MULTIPOINT (0.00000 0.00000, 0.00000 1.00000)
        dtype: geometry

        >>> s2
        1    POLYGON ((0.00000 0.00000, -2.00000 0.00000, 0...
        2        LINESTRING (0.00000 1.00000, 1.00000 1.00000)
        3        LINESTRING (1.00000 1.00000, 3.00000 0.00000)
        4                              POINT (0.00000 1.00000)
        dtype: geometry

        We can check if each geometry of GeoSeries touches a single
        geometry:

        .. image:: ../../../_static/binary_op-03.svg
           :align: center


        >>> line = LineString([(0, 0), (-1, -2)])
        >>> s.touches(line)
        0    True
        1    True
        2    True
        3    True
        dtype: bool

        We can also check two GeoSeries against each other, row by row.
        The GeoSeries above have different indices. We can either align both GeoSeries
        based on index values and compare elements with the same index using
        ``align=True`` or ignore index and compare elements based on their matching
        order using ``align=False``:

        .. image:: ../../../_static/binary_op-02.svg

        >>> s.touches(s2, align=True)
        0    False
        1     True
        2     True
        3    False
        4    False
        dtype: bool

        >>> s.touches(s2, align=False)
        0     True
        1    False
        2     True
        3    False
        dtype: bool

        Notes
        -----
        This method works in a row-wise manner. It does not check if an element
        of one GeoSeries ``touches`` *any* element of the other one.

        See also
        --------
        GeoSeries.overlaps
        GeoSeries.intersects

        """
        return _binary_op("touches", self, other, align)

    def within(self, other, align=True):
        """Returns a ``Series`` of ``dtype('bool')`` with value ``True`` for
        each aligned geometry that is within `other`.

        An object is said to be within `other` if at least one of its points is located
        in the `interior` and no points are located in the `exterior` of the other.
        If either object is empty, this operation returns ``False``.

        This is the inverse of :meth:`contains` in the sense that the
        expression ``a.within(b) == b.contains(a)`` always evaluates to
        ``True``.

        The operation works on a 1-to-1 row-wise manner:

        .. image:: ../../../_static/binary_op-01.svg
           :align: center

        Parameters
        ----------
        other : GeoSeries or geometric object
            The GeoSeries (elementwise) or geometric object to test if each
            geometry is within.
        align : bool (default True)
            If True, automatically aligns GeoSeries based on their indices.
            If False, the order of elements is preserved.

        Returns
        -------
        Series (bool)


        Examples
        --------
        >>> from shapely.geometry import Polygon, LineString, Point
        >>> s = geopandas.GeoSeries(
        ...     [
        ...         Polygon([(0, 0), (2, 2), (0, 2)]),
        ...         Polygon([(0, 0), (1, 2), (0, 2)]),
        ...         LineString([(0, 0), (0, 2)]),
        ...         Point(0, 1),
        ...     ],
        ... )
        >>> s2 = geopandas.GeoSeries(
        ...     [
        ...         Polygon([(0, 0), (1, 1), (0, 1)]),
        ...         LineString([(0, 0), (0, 2)]),
        ...         LineString([(0, 0), (0, 1)]),
        ...         Point(0, 1),
        ...     ],
        ...     index=range(1, 5),
        ... )

        >>> s
        0    POLYGON ((0.00000 0.00000, 2.00000 2.00000, 0....
        1    POLYGON ((0.00000 0.00000, 1.00000 2.00000, 0....
        2        LINESTRING (0.00000 0.00000, 0.00000 2.00000)
        3                              POINT (0.00000 1.00000)
        dtype: geometry

        >>> s2
        1    POLYGON ((0.00000 0.00000, 1.00000 1.00000, 0....
        2        LINESTRING (0.00000 0.00000, 0.00000 2.00000)
        3        LINESTRING (0.00000 0.00000, 0.00000 1.00000)
        4                              POINT (0.00000 1.00000)
        dtype: geometry

        We can check if each geometry of GeoSeries is within a single
        geometry:

        .. image:: ../../../_static/binary_op-03.svg
           :align: center

        >>> polygon = Polygon([(0, 0), (2, 2), (0, 2)])
        >>> s.within(polygon)
        0     True
        1     True
        2    False
        3    False
        dtype: bool

        We can also check two GeoSeries against each other, row by row.
        The GeoSeries above have different indices. We can either align both GeoSeries
        based on index values and compare elements with the same index using
        ``align=True`` or ignore index and compare elements based on their matching
        order using ``align=False``:

        .. image:: ../../../_static/binary_op-02.svg

        >>> s2.within(s)
        0    False
        1    False
        2     True
        3    False
        4    False
        dtype: bool

        >>> s2.within(s, align=False)
        1     True
        2    False
        3     True
        4     True
        dtype: bool

        Notes
        -----
        This method works in a row-wise manner. It does not check if an element
        of one GeoSeries is ``within`` *any* element of the other one.

        See also
        --------
        GeoSeries.contains
        """
        return _binary_op("within", self, other, align)

    def covers(self, other, align=True):
        """
        Returns a ``Series`` of ``dtype('bool')`` with value ``True`` for
        each aligned geometry that is entirely covering `other`.

        An object A is said to cover another object B if no points of B lie
        in the exterior of A.
        If either object is empty, this operation returns ``False``.

        The operation works on a 1-to-1 row-wise manner:

        .. image:: ../../../_static/binary_op-01.svg
           :align: center

        See
        https://lin-ear-th-inking.blogspot.com/2007/06/subtleties-of-ogc-covers-spatial.html
        for reference.

        Parameters
        ----------
        other : Geoseries or geometric object
            The Geoseries (elementwise) or geometric object to check is being covered.
        align : bool (default True)
            If True, automatically aligns GeoSeries based on their indices.
            If False, the order of elements is preserved.

        Returns
        -------
        Series (bool)

        Examples
        --------
        >>> from shapely.geometry import Polygon, LineString, Point
        >>> s = geopandas.GeoSeries(
        ...     [
        ...         Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]),
        ...         Polygon([(0, 0), (2, 2), (0, 2)]),
        ...         LineString([(0, 0), (2, 2)]),
        ...         Point(0, 0),
        ...     ],
        ... )
        >>> s2 = geopandas.GeoSeries(
        ...     [
        ...         Polygon([(0.5, 0.5), (1.5, 0.5), (1.5, 1.5), (0.5, 1.5)]),
        ...         Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]),
        ...         LineString([(1, 1), (1.5, 1.5)]),
        ...         Point(0, 0),
        ...     ],
        ...     index=range(1, 5),
        ... )

        >>> s
        0    POLYGON ((0.00000 0.00000, 2.00000 0.00000, 2....
        1    POLYGON ((0.00000 0.00000, 2.00000 2.00000, 0....
        2        LINESTRING (0.00000 0.00000, 2.00000 2.00000)
        3                              POINT (0.00000 0.00000)
        dtype: geometry

        >>> s2
        1    POLYGON ((0.50000 0.50000, 1.50000 0.50000, 1....
        2    POLYGON ((0.00000 0.00000, 2.00000 0.00000, 2....
        3        LINESTRING (1.00000 1.00000, 1.50000 1.50000)
        4                              POINT (0.00000 0.00000)
        dtype: geometry

        We can check if each geometry of GeoSeries covers a single
        geometry:

        .. image:: ../../../_static/binary_op-03.svg
           :align: center

        >>> poly = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
        >>> s.covers(poly)
        0     True
        1    False
        2    False
        3    False
        dtype: bool

        We can also check two GeoSeries against each other, row by row.
        The GeoSeries above have different indices. We can either align both GeoSeries
        based on index values and compare elements with the same index using
        ``align=True`` or ignore index and compare elements based on their matching
        order using ``align=False``:

        .. image:: ../../../_static/binary_op-02.svg

        >>> s.covers(s2, align=True)
        0    False
        1    False
        2    False
        3    False
        4    False
        dtype: bool

        >>> s.covers(s2, align=False)
        0     True
        1    False
        2     True
        3     True
        dtype: bool

        Notes
        -----
        This method works in a row-wise manner. It does not check if an element
        of one GeoSeries ``covers`` *any* element of the other one.

        See also
        --------
        GeoSeries.covered_by
        GeoSeries.overlaps
        """
        return _binary_op("covers", self, other, align)

    def covered_by(self, other, align=True):
        """
        Returns a ``Series`` of ``dtype('bool')`` with value ``True`` for
        each aligned geometry that is entirely covered by `other`.

        An object A is said to cover another object B if no points of B lie
        in the exterior of A.

        The operation works on a 1-to-1 row-wise manner:

        .. image:: ../../../_static/binary_op-01.svg
           :align: center

        See
        https://lin-ear-th-inking.blogspot.com/2007/06/subtleties-of-ogc-covers-spatial.html
        for reference.

        Parameters
        ----------
        other : Geoseries or geometric object
            The Geoseries (elementwise) or geometric object to check is being covered.
        align : bool (default True)
            If True, automatically aligns GeoSeries based on their indices.
            If False, the order of elements is preserved.

        Returns
        -------
        Series (bool)

        Examples
        --------
        >>> from shapely.geometry import Polygon, LineString, Point
        >>> s = geopandas.GeoSeries(
        ...     [
        ...         Polygon([(0.5, 0.5), (1.5, 0.5), (1.5, 1.5), (0.5, 1.5)]),
        ...         Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]),
        ...         LineString([(1, 1), (1.5, 1.5)]),
        ...         Point(0, 0),
        ...     ],
        ... )
        >>> s2 = geopandas.GeoSeries(
        ...     [
        ...         Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]),
        ...         Polygon([(0, 0), (2, 2), (0, 2)]),
        ...         LineString([(0, 0), (2, 2)]),
        ...         Point(0, 0),
        ...     ],
        ...     index=range(1, 5),
        ... )

        >>> s
        0    POLYGON ((0.50000 0.50000, 1.50000 0.50000, 1....
        1    POLYGON ((0.00000 0.00000, 2.00000 0.00000, 2....
        2        LINESTRING (1.00000 1.00000, 1.50000 1.50000)
        3                              POINT (0.00000 0.00000)
        dtype: geometry

        >>> s2
        1    POLYGON ((0.00000 0.00000, 2.00000 0.00000, 2....
        2    POLYGON ((0.00000 0.00000, 2.00000 2.00000, 0....
        3        LINESTRING (0.00000 0.00000, 2.00000 2.00000)
        4                              POINT (0.00000 0.00000)
        dtype: geometry

        We can check if each geometry of GeoSeries is covered by a single
        geometry:

        .. image:: ../../../_static/binary_op-03.svg
           :align: center

        >>> poly = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
        >>> s.covered_by(poly)
        0    True
        1    True
        2    True
        3    True
        dtype: bool

        We can also check two GeoSeries against each other, row by row.
        The GeoSeries above have different indices. We can either align both GeoSeries
        based on index values and compare elements with the same index using
        ``align=True`` or ignore index and compare elements based on their matching
        order using ``align=False``:

        .. image:: ../../../_static/binary_op-02.svg

        >>> s.covered_by(s2, align=True)
        0    False
        1     True
        2     True
        3     True
        4    False
        dtype: bool

        >>> s.covered_by(s2, align=False)
        0     True
        1    False
        2     True
        3     True
        dtype: bool

        Notes
        -----
        This method works in a row-wise manner. It does not check if an element
        of one GeoSeries is ``covered_by`` *any* element of the other one.

        See also
        --------
        GeoSeries.covers
        GeoSeries.overlaps
        """
        return _binary_op("covered_by", self, other, align)

    def distance(self, other, align=True):
        """Returns a ``Series`` containing the distance to aligned `other`.

        The operation works on a 1-to-1 row-wise manner:

        .. image:: ../../../_static/binary_op-01.svg
           :align: center

        Parameters
        ----------
        other : Geoseries or geometric object
            The Geoseries (elementwise) or geometric object to find the
            distance to.
        align : bool (default True)
            If True, automatically aligns GeoSeries based on their indices.
            If False, the order of elements is preserved.


        Returns
        -------
        Series (float)

        Examples
        --------
        >>> from shapely.geometry import Polygon, LineString, Point
        >>> s = geopandas.GeoSeries(
        ...     [
        ...         Polygon([(0, 0), (1, 0), (1, 1)]),
        ...         Polygon([(0, 0), (-1, 0), (-1, 1)]),
        ...         LineString([(1, 1), (0, 0)]),
        ...         Point(0, 0),
        ...     ],
        ... )
        >>> s2 = geopandas.GeoSeries(
        ...     [
        ...         Polygon([(0.5, 0.5), (1.5, 0.5), (1.5, 1.5), (0.5, 1.5)]),
        ...         Point(3, 1),
        ...         LineString([(1, 0), (2, 0)]),
        ...         Point(0, 1),
        ...     ],
        ...     index=range(1, 5),
        ... )

        >>> s
        0    POLYGON ((0.00000 0.00000, 1.00000 0.00000, 1....
        1    POLYGON ((0.00000 0.00000, -1.00000 0.00000, -...
        2        LINESTRING (1.00000 1.00000, 0.00000 0.00000)
        3                              POINT (0.00000 0.00000)
        dtype: geometry

        >>> s2
        1    POLYGON ((0.50000 0.50000, 1.50000 0.50000, 1....
        2                              POINT (3.00000 1.00000)
        3        LINESTRING (1.00000 0.00000, 2.00000 0.00000)
        4                              POINT (0.00000 1.00000)
        dtype: geometry

        We can check the distance of each geometry of GeoSeries to a single
        geometry:

        .. image:: ../../../_static/binary_op-03.svg
           :align: center

        >>> point = Point(-1, 0)
        >>> s.distance(point)
        0    1.0
        1    0.0
        2    1.0
        3    1.0
        dtype: float64

        We can also check two GeoSeries against each other, row by row.
        The GeoSeries above have different indices. We can either align both GeoSeries
        based on index values and use elements with the same index using
        ``align=True`` or ignore index and use elements based on their matching
        order using ``align=False``:

        .. image:: ../../../_static/binary_op-02.svg

        >>> s.distance(s2, align=True)
        0         NaN
        1    0.707107
        2    2.000000
        3    1.000000
        4         NaN
        dtype: float64

        >>> s.distance(s2, align=False)
        0    0.000000
        1    3.162278
        2    0.707107
        3    1.000000
        dtype: float64
        """
        return _binary_op("distance", self, other, align)

    def hausdorff_distance(self, other, align=True, densify=None):
        """Returns a ``Series`` containing the Hausdorff distance to aligned `other`.

        The Hausdorff distance is the largest distance consisting of any point in `self`
        with the nearest point in `other`.

        The operation works on a 1-to-1 row-wise manner:

        .. image:: ../../../_static/binary_op-01.svg
           :align: center

        Parameters
        ----------
        other : GeoSeries or geometric object
            The Geoseries (elementwise) or geometric object to find the
            distance to.
        align : bool (default True)
            If True, automatically aligns GeoSeries based on their indices.
            If False, the order of elements is preserved.
        densify : float (default None)
            A value between 0 and 1, that splits each subsegment of a line string
            into equal length segments, making the approximation less coarse.
            A densify value of 0.5 will add a point halfway between each pair of
            points. A densify value of 0.25 will add a point a quarter of the way
            between each pair of points.


        Returns
        -------
        Series (float)

        Examples
        --------
        >>> from shapely.geometry import Polygon, LineString, Point
        >>> s = geopandas.GeoSeries(
        ...     [
        ...         Polygon([(0, 0), (1, 0), (1, 1)]),
        ...         Polygon([(0, 0), (-1, 0), (-1, 1)]),
        ...         LineString([(1, 1), (0, 0)]),
        ...         Point(0, 0),
        ...     ],
        ... )
        >>> s2 = geopandas.GeoSeries(
        ...     [
        ...         Polygon([(0.5, 0.5), (1.5, 0.5), (1.5, 1.5), (0.5, 1.5)]),
        ...         Point(3, 1),
        ...         LineString([(1, 0), (2, 0)]),
        ...         Point(0, 1),
        ...     ],
        ...     index=range(1, 5),
        ... )

        >>> s
        0    POLYGON ((0.00000 0.00000, 1.00000 0.00000, 1....
        1    POLYGON ((0.00000 0.00000, -1.00000 0.00000, -...
        2        LINESTRING (1.00000 1.00000, 0.00000 0.00000)
        3                              POINT (0.00000 0.00000)
        dtype: geometry

        >>> s2
        1    POLYGON ((0.50000 0.50000, 1.50000 0.50000, 1....
        2                              POINT (3.00000 1.00000)
        3        LINESTRING (1.00000 0.00000, 2.00000 0.00000)
        4                              POINT (0.00000 1.00000)
        dtype: geometry

        We can check the hausdorff distance of each geometry of GeoSeries
        to a single geometry:

        >>> point = Point(-1, 0)
        >>> s.hausdorff_distance(point)
        0    2.236068
        1    1.000000
        2    2.236068
        3    1.000000
        dtype: float64

        We can also check two GeoSeries against each other, row by row.
        The GeoSeries above have different indices. We can either align both GeoSeries
        based on index values and use elements with the same index using
        ``align=True`` or ignore index and use elements based on their matching
        order using ``align=False``:

        .. image:: ../../../_static/binary_op-02.svg

        >>> s.hausdorff_distance(s2, align=True)
        0         NaN
        1    2.121320
        2    3.162278
        3    2.000000
        4         NaN
        dtype: float64

        >>> s.hausdorff_distance(s2, align=False)
        0    0.707107
        1    4.123106
        2    1.414214
        3    1.000000
        dtype: float64

        We can also set a densify value, which is a float between 0 and 1 and
        signifies the fraction of the distance between each pair of points that will
        be used as the distance between the points when densifying.

        >>> l1 = geopandas.GeoSeries([LineString([(130, 0), (0, 0), (0, 150)])])
        >>> l2 = geopandas.GeoSeries([LineString([(10, 10), (10, 150), (130, 10)])])
        >>> l1.hausdorff_distance(l2)
        0    14.142136
        dtype: float64
        >>> l1.hausdorff_distance(l2, densify=0.25)
        0    70.0
        dtype: float64
        """
        return _binary_op("hausdorff_distance", self, other, align, densify=densify)

    def frechet_distance(self, other, align=True, densify=None):
        """Returns a ``Series`` containing the Frechet distance to aligned `other`.

        The Fréchet distance is a measure of similarity: it is the greatest distance
        between any point in A and the closest point in B. The discrete distance is an
        approximation of this metric: only vertices are considered. The parameter
        ``densify`` makes this approximation less coarse by splitting the line segments
        between vertices before computing the distance.

        Fréchet distance sweep continuously along their respective curves and the
        direction of curves is significant. This makes it a better measure of similarity
        than Hausdorff distance for curve or surface matching.

        The operation works on a 1-to-1 row-wise manner:

        .. image:: ../../../_static/binary_op-01.svg
           :align: center

        Parameters
        ----------
        other : GeoSeries or geometric object
            The Geoseries (elementwise) or geometric object to find the
            distance to.
        align : bool (default True)
            If True, automatically aligns GeoSeries based on their indices.
            If False, the order of elements is preserved.
        densify : float (default None)
            A value between 0 and 1, that splits each subsegment of a line string
            into equal length segments, making the approximation less coarse.
            A densify value of 0.5 will add a point halfway between each pair of
            points. A densify value of 0.25 will add a point every quarter of the way
            between each pair of points.

        Returns
        -------
        Series (float)

        Examples
        --------
        >>> from shapely.geometry import Polygon, LineString, Point
        >>> s = geopandas.GeoSeries(
        ...     [
        ...         Polygon([(0, 0), (1, 0), (1, 1)]),
        ...         Polygon([(0, 0), (-1, 0), (-1, 1)]),
        ...         LineString([(1, 1), (0, 0)]),
        ...         Point(0, 0),
        ...     ],
        ... )
        >>> s2 = geopandas.GeoSeries(
        ...     [
        ...         Polygon([(0.5, 0.5), (1.5, 0.5), (1.5, 1.5), (0.5, 1.5)]),
        ...         Point(3, 1),
        ...         LineString([(1, 0), (2, 0)]),
        ...         Point(0, 1),
        ...     ],
        ...     index=range(1, 5),
        ... )
        >>> s
        0    POLYGON ((0.00000 0.00000, 1.00000 0.00000, 1....
        1    POLYGON ((0.00000 0.00000, -1.00000 0.00000, -...
        2        LINESTRING (1.00000 1.00000, 0.00000 0.00000)
        3                              POINT (0.00000 0.00000)
        dtype: geometry
        >>> s2
        1    POLYGON ((0.50000 0.50000, 1.50000 0.50000, 1....
        2                              POINT (3.00000 1.00000)
        3        LINESTRING (1.00000 0.00000, 2.00000 0.00000)
        4                              POINT (0.00000 1.00000)
        dtype: geometry

        We can check the frechet distance of each geometry of GeoSeries
        to a single geometry:

        >>> point = Point(-1, 0)
        >>> s.frechet_distance(point)
        0    2.236068
        1    1.000000
        2    2.236068
        3    1.000000
        dtype: float64

        We can also check two GeoSeries against each other, row by row.
        The GeoSeries above have different indices. We can either align both GeoSeries
        based on index values and use elements with the same index using
        ``align=True`` or ignore index and use elements based on their matching
        order using ``align=False``:

        .. image:: ../../../_static/binary_op-02.svg

        >>> s.frechet_distance(s2, align=True)
        0         NaN
        1    2.121320
        2    3.162278
        3    2.000000
        4         NaN
        dtype: float64
        >>> s.frechet_distance(s2, align=False)
        0    0.707107
        1    4.123106
        2    2.000000
        3    1.000000
        dtype: float64

        We can also set a ``densify`` value, which is a float between 0 and 1 and
        signifies the fraction of the distance between each pair of points that will
        be used as the distance between the points when densifying.

        >>> l1 = geopandas.GeoSeries([LineString([(0, 0), (10, 0), (0, 15)])])
        >>> l2 = geopandas.GeoSeries([LineString([(0, 0), (20, 15), (9, 11)])])
        >>> l1.frechet_distance(l2)
        0    18.027756
        dtype: float64
        >>> l1.frechet_distance(l2, densify=0.25)
        0    16.77051
        dtype: float64
        """
        return _binary_op("frechet_distance", self, other, align, densify=densify)

    #
    # Binary operations that return a GeoSeries
    #

    def difference(self, other, align=True):
        """Returns a ``GeoSeries`` of the points in each aligned geometry that
        are not in `other`.

        .. image:: ../../../_static/binary_geo-difference.svg
           :align: center

        The operation works on a 1-to-1 row-wise manner:

        .. image:: ../../../_static/binary_op-01.svg
           :align: center

        Parameters
        ----------
        other : Geoseries or geometric object
            The Geoseries (elementwise) or geometric object to find the
            difference to.
        align : bool (default True)
            If True, automatically aligns GeoSeries based on their indices.
            If False, the order of elements is preserved.

        Returns
        -------
        GeoSeries

        Examples
        --------
        >>> from shapely.geometry import Polygon, LineString, Point
        >>> s = geopandas.GeoSeries(
        ...     [
        ...         Polygon([(0, 0), (2, 2), (0, 2)]),
        ...         Polygon([(0, 0), (2, 2), (0, 2)]),
        ...         LineString([(0, 0), (2, 2)]),
        ...         LineString([(2, 0), (0, 2)]),
        ...         Point(0, 1),
        ...     ],
        ... )
        >>> s2 = geopandas.GeoSeries(
        ...     [
        ...         Polygon([(0, 0), (1, 1), (0, 1)]),
        ...         LineString([(1, 0), (1, 3)]),
        ...         LineString([(2, 0), (0, 2)]),
        ...         Point(1, 1),
        ...         Point(0, 1),
        ...     ],
        ...     index=range(1, 6),
        ... )

        >>> s
        0    POLYGON ((0.00000 0.00000, 2.00000 2.00000, 0....
        1    POLYGON ((0.00000 0.00000, 2.00000 2.00000, 0....
        2        LINESTRING (0.00000 0.00000, 2.00000 2.00000)
        3        LINESTRING (2.00000 0.00000, 0.00000 2.00000)
        4                              POINT (0.00000 1.00000)
        dtype: geometry

        >>> s2
        1    POLYGON ((0.00000 0.00000, 1.00000 1.00000, 0....
        2        LINESTRING (1.00000 0.00000, 1.00000 3.00000)
        3        LINESTRING (2.00000 0.00000, 0.00000 2.00000)
        4                              POINT (1.00000 1.00000)
        5                              POINT (0.00000 1.00000)
        dtype: geometry

        We can do difference of each geometry and a single
        shapely geometry:

        .. image:: ../../../_static/binary_op-03.svg
           :align: center

        >>> s.difference(Polygon([(0, 0), (1, 1), (0, 1)]))
        0    POLYGON ((0.00000 2.00000, 2.00000 2.00000, 1....
        1    POLYGON ((0.00000 2.00000, 2.00000 2.00000, 1....
        2        LINESTRING (1.00000 1.00000, 2.00000 2.00000)
        3    MULTILINESTRING ((2.00000 0.00000, 1.00000 1.0...
        4                                          POINT EMPTY
        dtype: geometry

        We can also check two GeoSeries against each other, row by row.
        The GeoSeries above have different indices. We can either align both GeoSeries
        based on index values and compare elements with the same index using
        ``align=True`` or ignore index and compare elements based on their matching
        order using ``align=False``:

        .. image:: ../../../_static/binary_op-02.svg

        >>> s.difference(s2, align=True)
        0                                                 None
        1    POLYGON ((0.00000 2.00000, 2.00000 2.00000, 1....
        2    MULTILINESTRING ((0.00000 0.00000, 1.00000 1.0...
        3                                   LINESTRING Z EMPTY
        4                              POINT (0.00000 1.00000)
        5                                                 None
        dtype: geometry

        >>> s.difference(s2, align=False)
        0    POLYGON ((0.00000 2.00000, 2.00000 2.00000, 1....
        1    POLYGON ((0.00000 0.00000, 0.00000 2.00000, 1....
        2    MULTILINESTRING ((0.00000 0.00000, 1.00000 1.0...
        3        LINESTRING (2.00000 0.00000, 0.00000 2.00000)
        4                                          POINT EMPTY
        dtype: geometry

        See Also
        --------
        GeoSeries.symmetric_difference
        GeoSeries.union
        GeoSeries.intersection
        """
        return _binary_geo("difference", self, other, align)

    def symmetric_difference(self, other, align=True):
        """Returns a ``GeoSeries`` of the symmetric difference of points in
        each aligned geometry with `other`.

        For each geometry, the symmetric difference consists of points in the
        geometry not in `other`, and points in `other` not in the geometry.

        .. image:: ../../../_static/binary_geo-symm_diff.svg
           :align: center

        The operation works on a 1-to-1 row-wise manner:

        .. image:: ../../../_static/binary_op-01.svg
           :align: center


        Parameters
        ----------
        other : Geoseries or geometric object
            The Geoseries (elementwise) or geometric object to find the
            symmetric difference to.
        align : bool (default True)
            If True, automatically aligns GeoSeries based on their indices.
            If False, the order of elements is preserved.

        Returns
        -------
        GeoSeries

        Examples
        --------
        >>> from shapely.geometry import Polygon, LineString, Point
        >>> s = geopandas.GeoSeries(
        ...     [
        ...         Polygon([(0, 0), (2, 2), (0, 2)]),
        ...         Polygon([(0, 0), (2, 2), (0, 2)]),
        ...         LineString([(0, 0), (2, 2)]),
        ...         LineString([(2, 0), (0, 2)]),
        ...         Point(0, 1),
        ...     ],
        ... )
        >>> s2 = geopandas.GeoSeries(
        ...     [
        ...         Polygon([(0, 0), (1, 1), (0, 1)]),
        ...         LineString([(1, 0), (1, 3)]),
        ...         LineString([(2, 0), (0, 2)]),
        ...         Point(1, 1),
        ...         Point(0, 1),
        ...     ],
        ...     index=range(1, 6),
        ... )

        >>> s
        0    POLYGON ((0.00000 0.00000, 2.00000 2.00000, 0....
        1    POLYGON ((0.00000 0.00000, 2.00000 2.00000, 0....
        2        LINESTRING (0.00000 0.00000, 2.00000 2.00000)
        3        LINESTRING (2.00000 0.00000, 0.00000 2.00000)
        4                              POINT (0.00000 1.00000)
        dtype: geometry

        >>> s2
        1    POLYGON ((0.00000 0.00000, 1.00000 1.00000, 0....
        2        LINESTRING (1.00000 0.00000, 1.00000 3.00000)
        3        LINESTRING (2.00000 0.00000, 0.00000 2.00000)
        4                              POINT (1.00000 1.00000)
        5                              POINT (0.00000 1.00000)
        dtype: geometry

        We can do symmetric difference of each geometry and a single
        shapely geometry:

        .. image:: ../../../_static/binary_op-03.svg
           :align: center

        >>> s.symmetric_difference(Polygon([(0, 0), (1, 1), (0, 1)]))
        0    POLYGON ((0.00000 2.00000, 2.00000 2.00000, 1....
        1    POLYGON ((0.00000 2.00000, 2.00000 2.00000, 1....
        2    GEOMETRYCOLLECTION (POLYGON ((0.00000 0.00000,...
        3    GEOMETRYCOLLECTION (POLYGON ((0.00000 0.00000,...
        4    POLYGON ((0.00000 1.00000, 1.00000 1.00000, 0....
        dtype: geometry

        We can also check two GeoSeries against each other, row by row.
        The GeoSeries above have different indices. We can either align both GeoSeries
        based on index values and compare elements with the same index using
        ``align=True`` or ignore index and compare elements based on their matching
        order using ``align=False``:

        .. image:: ../../../_static/binary_op-02.svg

        >>> s.symmetric_difference(s2, align=True)
        0                                                 None
        1    POLYGON ((0.00000 2.00000, 2.00000 2.00000, 1....
        2    MULTILINESTRING ((0.00000 0.00000, 1.00000 1.0...
        3                                   LINESTRING Z EMPTY
        4        MULTIPOINT (0.00000 1.00000, 1.00000 1.00000)
        5                                                 None
        dtype: geometry

        >>> s.symmetric_difference(s2, align=False)
        0    POLYGON ((0.00000 2.00000, 2.00000 2.00000, 1....
        1    GEOMETRYCOLLECTION (POLYGON ((0.00000 0.00000,...
        2    MULTILINESTRING ((0.00000 0.00000, 1.00000 1.0...
        3        LINESTRING (2.00000 0.00000, 0.00000 2.00000)
        4                                          POINT EMPTY
        dtype: geometry

        See Also
        --------
        GeoSeries.difference
        GeoSeries.union
        GeoSeries.intersection
        """
        return _binary_geo("symmetric_difference", self, other, align)

    def union(self, other, align=True):
        """Returns a ``GeoSeries`` of the union of points in each aligned geometry with
        `other`.

        .. image:: ../../../_static/binary_geo-union.svg
           :align: center

        The operation works on a 1-to-1 row-wise manner:

        .. image:: ../../../_static/binary_op-01.svg
           :align: center


        Parameters
        ----------
        other : Geoseries or geometric object
            The Geoseries (elementwise) or geometric object to find the union
            with.
        align : bool (default True)
            If True, automatically aligns GeoSeries based on their indices.
            If False, the order of elements is preserved.

        Returns
        -------
        GeoSeries

        Examples
        --------
        >>> from shapely.geometry import Polygon, LineString, Point
        >>> s = geopandas.GeoSeries(
        ...     [
        ...         Polygon([(0, 0), (2, 2), (0, 2)]),
        ...         Polygon([(0, 0), (2, 2), (0, 2)]),
        ...         LineString([(0, 0), (2, 2)]),
        ...         LineString([(2, 0), (0, 2)]),
        ...         Point(0, 1),
        ...     ],
        ... )
        >>> s2 = geopandas.GeoSeries(
        ...     [
        ...         Polygon([(0, 0), (1, 1), (0, 1)]),
        ...         LineString([(1, 0), (1, 3)]),
        ...         LineString([(2, 0), (0, 2)]),
        ...         Point(1, 1),
        ...         Point(0, 1),
        ...     ],
        ...     index=range(1, 6),
        ... )

        >>> s
        0    POLYGON ((0.00000 0.00000, 2.00000 2.00000, 0....
        1    POLYGON ((0.00000 0.00000, 2.00000 2.00000, 0....
        2        LINESTRING (0.00000 0.00000, 2.00000 2.00000)
        3        LINESTRING (2.00000 0.00000, 0.00000 2.00000)
        4                              POINT (0.00000 1.00000)
        dtype: geometry

        >>> s2
        1    POLYGON ((0.00000 0.00000, 1.00000 1.00000, 0....
        2        LINESTRING (1.00000 0.00000, 1.00000 3.00000)
        3        LINESTRING (2.00000 0.00000, 0.00000 2.00000)
        4                              POINT (1.00000 1.00000)
        5                              POINT (0.00000 1.00000)
        dtype: geometry

        We can do union of each geometry and a single
        shapely geometry:

        .. image:: ../../../_static/binary_op-03.svg
           :align: center

        >>> s.union(Polygon([(0, 0), (1, 1), (0, 1)]))
        0    POLYGON ((0.00000 0.00000, 0.00000 1.00000, 0....
        1    POLYGON ((0.00000 0.00000, 0.00000 1.00000, 0....
        2    GEOMETRYCOLLECTION (POLYGON ((0.00000 0.00000,...
        3    GEOMETRYCOLLECTION (POLYGON ((0.00000 0.00000,...
        4    POLYGON ((0.00000 1.00000, 1.00000 1.00000, 0....
        dtype: geometry

        We can also check two GeoSeries against each other, row by row.
        The GeoSeries above have different indices. We can either align both GeoSeries
        based on index values and compare elements with the same index using
        ``align=True`` or ignore index and compare elements based on their matching
        order using ``align=False``:

        .. image:: ../../../_static/binary_op-02.svg

        >>> s.union(s2, align=True)
        0                                                 None
        1    POLYGON ((0.00000 0.00000, 0.00000 1.00000, 0....
        2    MULTILINESTRING ((0.00000 0.00000, 1.00000 1.0...
        3        LINESTRING (2.00000 0.00000, 0.00000 2.00000)
        4        MULTIPOINT (0.00000 1.00000, 1.00000 1.00000)
        5                                                 None
        dtype: geometry

        >>> s.union(s2, align=False)
        0    POLYGON ((0.00000 0.00000, 0.00000 1.00000, 0....
        1    GEOMETRYCOLLECTION (POLYGON ((0.00000 0.00000,...
        2    MULTILINESTRING ((0.00000 0.00000, 1.00000 1.0...
        3        LINESTRING (2.00000 0.00000, 0.00000 2.00000)
        4                              POINT (0.00000 1.00000)
        dtype: geometry


        See Also
        --------
        GeoSeries.symmetric_difference
        GeoSeries.difference
        GeoSeries.intersection
        """
        return _binary_geo("union", self, other, align)

    def intersection(self, other, align=True):
        """Returns a ``GeoSeries`` of the intersection of points in each
        aligned geometry with `other`.

        .. image:: ../../../_static/binary_geo-intersection.svg
           :align: center

        The operation works on a 1-to-1 row-wise manner:

        .. image:: ../../../_static/binary_op-01.svg
           :align: center


        Parameters
        ----------
        other : Geoseries or geometric object
            The Geoseries (elementwise) or geometric object to find the
            intersection with.
        align : bool (default True)
            If True, automatically aligns GeoSeries based on their indices.
            If False, the order of elements is preserved.

        Returns
        -------
        GeoSeries

        Examples
        --------
        >>> from shapely.geometry import Polygon, LineString, Point
        >>> s = geopandas.GeoSeries(
        ...     [
        ...         Polygon([(0, 0), (2, 2), (0, 2)]),
        ...         Polygon([(0, 0), (2, 2), (0, 2)]),
        ...         LineString([(0, 0), (2, 2)]),
        ...         LineString([(2, 0), (0, 2)]),
        ...         Point(0, 1),
        ...     ],
        ... )
        >>> s2 = geopandas.GeoSeries(
        ...     [
        ...         Polygon([(0, 0), (1, 1), (0, 1)]),
        ...         LineString([(1, 0), (1, 3)]),
        ...         LineString([(2, 0), (0, 2)]),
        ...         Point(1, 1),
        ...         Point(0, 1),
        ...     ],
        ...     index=range(1, 6),
        ... )

        >>> s
        0    POLYGON ((0.00000 0.00000, 2.00000 2.00000, 0....
        1    POLYGON ((0.00000 0.00000, 2.00000 2.00000, 0....
        2        LINESTRING (0.00000 0.00000, 2.00000 2.00000)
        3        LINESTRING (2.00000 0.00000, 0.00000 2.00000)
        4                              POINT (0.00000 1.00000)
        dtype: geometry

        >>> s2
        1    POLYGON ((0.00000 0.00000, 1.00000 1.00000, 0....
        2        LINESTRING (1.00000 0.00000, 1.00000 3.00000)
        3        LINESTRING (2.00000 0.00000, 0.00000 2.00000)
        4                              POINT (1.00000 1.00000)
        5                              POINT (0.00000 1.00000)
        dtype: geometry

        We can also do intersection of each geometry and a single
        shapely geometry:

        .. image:: ../../../_static/binary_op-03.svg
           :align: center

        >>> s.intersection(Polygon([(0, 0), (1, 1), (0, 1)]))
        0    POLYGON ((0.00000 0.00000, 0.00000 1.00000, 1....
        1    POLYGON ((0.00000 0.00000, 0.00000 1.00000, 1....
        2        LINESTRING (0.00000 0.00000, 1.00000 1.00000)
        3                              POINT (1.00000 1.00000)
        4                              POINT (0.00000 1.00000)
        dtype: geometry

        We can also check two GeoSeries against each other, row by row.
        The GeoSeries above have different indices. We can either align both GeoSeries
        based on index values and compare elements with the same index using
        ``align=True`` or ignore index and compare elements based on their matching
        order using ``align=False``:

        .. image:: ../../../_static/binary_op-02.svg

        >>> s.intersection(s2, align=True)
        0                                                 None
        1    POLYGON ((0.00000 0.00000, 0.00000 1.00000, 1....
        2                              POINT (1.00000 1.00000)
        3        LINESTRING (2.00000 0.00000, 0.00000 2.00000)
        4                                          POINT EMPTY
        5                                                 None
        dtype: geometry

        >>> s.intersection(s2, align=False)
        0    POLYGON ((0.00000 0.00000, 0.00000 1.00000, 1....
        1        LINESTRING (1.00000 1.00000, 1.00000 2.00000)
        2                              POINT (1.00000 1.00000)
        3                              POINT (1.00000 1.00000)
        4                              POINT (0.00000 1.00000)
        dtype: geometry


        See Also
        --------
        GeoSeries.difference
        GeoSeries.symmetric_difference
        GeoSeries.union
        """
        return _binary_geo("intersection", self, other, align)

    def clip_by_rect(self, xmin, ymin, xmax, ymax):
        """Returns a ``GeoSeries`` of the portions of geometry within the given
        rectangle.

        Note that the results are not exactly equal to
        :meth:`~GeoSeries.intersection()`. E.g. in edge cases,
        :meth:`~GeoSeries.clip_by_rect()` will not return a point just touching the
        rectangle. Check the examples section below for some of these exceptions.

        The geometry is clipped in a fast but possibly dirty way. The output is not
        guaranteed to be valid. No exceptions will be raised for topological errors.

        Note: empty geometries or geometries that do not overlap with the specified
        bounds will result in ``GEOMETRYCOLLECTION EMPTY``.

        Parameters
        ----------
        xmin: float
            Minimum x value of the rectangle
        ymin: float
            Minimum y value of the rectangle
        xmax: float
            Maximum x value of the rectangle
        ymax: float
            Maximum y value of the rectangle

        Returns
        -------
        GeoSeries

        Examples
        --------
        >>> from shapely.geometry import Polygon, LineString, Point
        >>> s = geopandas.GeoSeries(
        ...     [
        ...         Polygon([(0, 0), (2, 2), (0, 2)]),
        ...         Polygon([(0, 0), (2, 2), (0, 2)]),
        ...         LineString([(0, 0), (2, 2)]),
        ...         LineString([(2, 0), (0, 2)]),
        ...         Point(0, 1),
        ...     ],
        ...     crs=3857,
        ... )
        >>> bounds = (0, 0, 1, 1)
        >>> s
        0    POLYGON ((0.000 0.000, 2.000 2.000, 0.000 2.00...
        1    POLYGON ((0.000 0.000, 2.000 2.000, 0.000 2.00...
        2                LINESTRING (0.000 0.000, 2.000 2.000)
        3                LINESTRING (2.000 0.000, 0.000 2.000)
        4                                  POINT (0.000 1.000)
        dtype: geometry
        >>> s.clip_by_rect(*bounds)
        0    POLYGON ((0.000 0.000, 0.000 1.000, 1.000 1.00...
        1    POLYGON ((0.000 0.000, 0.000 1.000, 1.000 1.00...
        2                LINESTRING (0.000 0.000, 1.000 1.000)
        3                             GEOMETRYCOLLECTION EMPTY
        4                             GEOMETRYCOLLECTION EMPTY
        dtype: geometry

        See also
        --------
        GeoSeries.intersection
        """
        from .geoseries import GeoSeries

        geometry_array = GeometryArray(self.geometry.values)
        clipped_geometry = geometry_array.clip_by_rect(xmin, ymin, xmax, ymax)
        return GeoSeries(clipped_geometry, index=self.index, crs=self.crs)

    def shortest_line(self, other, align=True):
        """
        Returns the shortest two-point line between two geometries.

        The resulting line consists of two points, representing the nearest points
        between the geometry pair. The line always starts in the first geometry a
        and ends in he second geometry b. The endpoints of the line will not
        necessarily be existing vertices of the input geometries a and b, but
        can also be a point along a line segment.


        The operation works on a 1-to-1 row-wise manner:

        .. image:: ../../../_static/binary_op-01.svg
        :align: center

        Parameters
        ----------
        other : Geoseries or geometric object
            The Geoseries (elementwise) or geometric object to find the
            shortest line with.
        align : bool (default True)
            If True, automatically aligns GeoSeries based on their indices.
            If False, the order of elements is preserved.

        Returns
        -------
        GeoSeries

        Examples
        --------
        >>> from shapely.geometry import Polygon, LineString, Point
        >>> s = geopandas.GeoSeries(
        ...     [
        ...         Polygon([(0, 0), (2, 2), (0, 2)]),
        ...         Polygon([(0, 0), (2, 2), (0, 2)]),
        ...         LineString([(0, 0), (2, 2)]),
        ...         LineString([(2, 0), (0, 2)]),
        ...         Point(0, 1),
        ...     ],
        ...     crs=5514
        ... )
        >>> s
        0    POLYGON ((0.000 0.000, 2.000 2.000, 0.000 2.00...
        1    POLYGON ((0.000 0.000, 2.000 2.000, 0.000 2.00...
        2                LINESTRING (0.000 0.000, 2.000 2.000)
        3                LINESTRING (2.000 0.000, 0.000 2.000)
        4                                  POINT (0.000 1.000)
        dtype: geometry

        We can also do intersection of each geometry and a single
        shapely geometry:

        .. image:: ../../../_static/binary_op-03.svg
           :align: center

        >>> p = Point(3, 3)
        >>> s.shortest_line(p)
        0    LINESTRING (2.000 2.000, 3.000 3.000)
        1    LINESTRING (2.000 2.000, 3.000 3.000)
        2    LINESTRING (2.000 2.000, 3.000 3.000)
        3    LINESTRING (1.000 1.000, 3.000 3.000)
        4    LINESTRING (0.000 1.000, 3.000 3.000)
        dtype: geometry

        We can also check two GeoSeries against each other, row by row.
        The GeoSeries above have different indices than the one below. We can either
        align both GeoSeries based on index values and compare elements with the same
        index using ``align=True`` or ignore index and compare elements based on their
        matching order using ``align=False``:

        .. image:: ../../../_static/binary_op-02.svg

        >>> s2 = geopandas.GeoSeries(
        ...     [
        ...         Polygon([(0.5, 0.5), (1.5, 0.5), (1.5, 1.5), (0.5, 1.5)]),
        ...         Point(3, 1),
        ...         LineString([(1, 0), (2, 0)]),
        ...         Point(10, 15),
        ...         Point(0, 1),
        ...     ],
        ...     index=range(1, 6),
        ...     crs=5514,
        ... )
        >>> s.shortest_line(s2, align=True)
        0                                       None
        1      LINESTRING (0.500 0.500, 0.500 0.500)
        2      LINESTRING (2.000 2.000, 3.000 1.000)
        3      LINESTRING (2.000 0.000, 2.000 0.000)
        4    LINESTRING (0.000 1.000, 10.000 15.000)
        5                                       None
        dtype: geometry

        >>> s.shortest_line(s2, align=False)
        0      LINESTRING (0.500 0.500, 0.500 0.500)
        1      LINESTRING (2.000 2.000, 3.000 1.000)
        2      LINESTRING (0.500 0.500, 1.000 0.000)
        3    LINESTRING (0.000 2.000, 10.000 15.000)
        4      LINESTRING (0.000 1.000, 0.000 1.000)
        dtype: geometry
        """
        return _binary_geo("shortest_line", self, other, align)

    #
    # Other operations
    #

    @property
    def bounds(self):
        """Returns a ``DataFrame`` with columns ``minx``, ``miny``, ``maxx``,
        ``maxy`` values containing the bounds for each geometry.

        See ``GeoSeries.total_bounds`` for the limits of the entire series.

        Examples
        --------
        >>> from shapely.geometry import Point, Polygon, LineString
        >>> d = {'geometry': [Point(2, 1), Polygon([(0, 0), (1, 1), (1, 0)]),
        ... LineString([(0, 1), (1, 2)])]}
        >>> gdf = geopandas.GeoDataFrame(d, crs="EPSG:4326")
        >>> gdf.bounds
           minx  miny  maxx  maxy
        0   2.0   1.0   2.0   1.0
        1   0.0   0.0   1.0   1.0
        2   0.0   1.0   1.0   2.0

        You can assign the bounds to the ``GeoDataFrame`` as:

        >>> import pandas as pd
        >>> gdf = pd.concat([gdf, gdf.bounds], axis=1)
        >>> gdf
                                                    geometry  minx  miny  maxx  maxy
        0                            POINT (2.00000 1.00000)   2.0   1.0   2.0   1.0
        1  POLYGON ((0.00000 0.00000, 1.00000 1.00000, 1....   0.0   0.0   1.0   1.0
        2      LINESTRING (0.00000 1.00000, 1.00000 2.00000)   0.0   1.0   1.0   2.0
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

        Examples
        --------
        >>> from shapely.geometry import Point, Polygon, LineString
        >>> d = {'geometry': [Point(3, -1), Polygon([(0, 0), (1, 1), (1, 0)]),
        ... LineString([(0, 1), (1, 2)])]}
        >>> gdf = geopandas.GeoDataFrame(d, crs="EPSG:4326")
        >>> gdf.total_bounds
        array([ 0., -1.,  3.,  2.])
        """
        return GeometryArray(self.geometry.values).total_bounds

    @property
    def sindex(self):
        """Generate the spatial index

        Creates R-tree spatial index based on ``pygeos.STRtree`` or
        ``rtree.index.Index``.

        Note that the  spatial index may not be fully
        initialized until the first use.

        Examples
        --------
        >>> from shapely.geometry import box
        >>> s = geopandas.GeoSeries(geopandas.points_from_xy(range(5), range(5)))
        >>> s
        0    POINT (0.00000 0.00000)
        1    POINT (1.00000 1.00000)
        2    POINT (2.00000 2.00000)
        3    POINT (3.00000 3.00000)
        4    POINT (4.00000 4.00000)
        dtype: geometry

        Query the spatial index with a single geometry based on the bounding box:

        >>> s.sindex.query(box(1, 1, 3, 3))
        array([1, 2, 3])

        Query the spatial index with a single geometry based on the predicate:

        >>> s.sindex.query(box(1, 1, 3, 3), predicate="contains")
        array([2])

        Query the spatial index with an array of geometries based on the bounding
        box:

        >>> s2 = geopandas.GeoSeries([box(1, 1, 3, 3), box(4, 4, 5, 5)])
        >>> s2
        0    POLYGON ((3.00000 1.00000, 3.00000 3.00000, 1....
        1    POLYGON ((5.00000 4.00000, 5.00000 5.00000, 4....
        dtype: geometry

        >>> s.sindex.query(s2)
        array([[0, 0, 0, 1],
               [1, 2, 3, 4]])

        Query the spatial index with an array of geometries based on the predicate:

        >>> s.sindex.query(s2, predicate="contains")
        array([[0],
               [2]])
        """
        return self.geometry.values.sindex

    @property
    def has_sindex(self):
        """Check the existence of the spatial index without generating it.

        Use the `.sindex` attribute on a GeoDataFrame or GeoSeries
        to generate a spatial index if it does not yet exist,
        which may take considerable time based on the underlying index
        implementation.

        Note that the underlying spatial index may not be fully
        initialized until the first use.

        Examples
        --------

        >>> from shapely.geometry import Point
        >>> d = {'geometry': [Point(1, 2), Point(2, 1)]}
        >>> gdf = geopandas.GeoDataFrame(d)
        >>> gdf.has_sindex
        False
        >>> index = gdf.sindex
        >>> gdf.has_sindex
        True

        Returns
        -------
        bool
            `True` if the spatial index has been generated or
            `False` if not.
        """
        return self.geometry.values.has_sindex

    def buffer(self, distance, resolution=16, **kwargs):
        """Returns a ``GeoSeries`` of geometries representing all points within
        a given ``distance`` of each geometric object.

        See http://shapely.readthedocs.io/en/latest/manual.html#object.buffer
        for details.

        Parameters
        ----------
        distance : float, np.array, pd.Series
            The radius of the buffer. If np.array or pd.Series are used
            then it must have same length as the GeoSeries.
        resolution : int (optional, default 16)
            The resolution of the buffer around each vertex.

        Examples
        --------
        >>> from shapely.geometry import Point, LineString, Polygon
        >>> s = geopandas.GeoSeries(
        ...     [
        ...         Point(0, 0),
        ...         LineString([(1, -1), (1, 0), (2, 0), (2, 1)]),
        ...         Polygon([(3, -1), (4, 0), (3, 1)]),
        ...     ]
        ... )
        >>> s
        0                              POINT (0.00000 0.00000)
        1    LINESTRING (1.00000 -1.00000, 1.00000 0.00000,...
        2    POLYGON ((3.00000 -1.00000, 4.00000 0.00000, 3...
        dtype: geometry

        >>> s.buffer(0.2)
        0    POLYGON ((0.20000 0.00000, 0.19904 -0.01960, 0...
        1    POLYGON ((0.80000 0.00000, 0.80096 0.01960, 0....
        2    POLYGON ((2.80000 -1.00000, 2.80000 1.00000, 2...
        dtype: geometry

        ``**kwargs`` accept further specification as ``join_style`` and ``cap_style``.
        See the following illustration of different options.

        .. plot:: _static/code/buffer.py

        """
        # TODO: update docstring based on pygeos after shapely 2.0
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

        The algorithm (Douglas-Peucker) recursively splits the original line
        into smaller parts and connects these parts’ endpoints
        by a straight line. Then, it removes all points whose distance
        to the straight line is smaller than `tolerance`. It does not
        move any points and it always preserves endpoints of
        the original line or polygon.
        See http://shapely.readthedocs.io/en/latest/manual.html#object.simplify
        for details

        Parameters
        ----------
        tolerance : float
            All parts of a simplified geometry will be no more than
            `tolerance` distance from the original. It has the same units
            as the coordinate reference system of the GeoSeries.
            For example, using `tolerance=100` in a projected CRS with meters
            as units means a distance of 100 meters in reality.
        preserve_topology: bool (default True)
            False uses a quicker algorithm, but may produce self-intersecting
            or otherwise invalid geometries.

        Notes
        -----
        Invalid geometric objects may result from simplification that does not
        preserve topology and simplification may be sensitive to the order of
        coordinates: two geometries differing only in order of coordinates may be
        simplified differently.

        Examples
        --------
        >>> from shapely.geometry import Point, LineString
        >>> s = geopandas.GeoSeries(
        ...     [Point(0, 0).buffer(1), LineString([(0, 0), (1, 10), (0, 20)])]
        ... )
        >>> s
        0    POLYGON ((1.00000 0.00000, 0.99518 -0.09802, 0...
        1    LINESTRING (0.00000 0.00000, 1.00000 10.00000,...
        dtype: geometry

        >>> s.simplify(1)
        0    POLYGON ((1.00000 0.00000, 0.00000 -1.00000, -...
        1       LINESTRING (0.00000 0.00000, 0.00000 20.00000)
        dtype: geometry
        """
        return _delegate_geo_method("simplify", self, *args, **kwargs)

    def relate(self, other, align=True):
        """
        Returns the DE-9IM intersection matrices for the geometries

        The operation works on a 1-to-1 row-wise manner:

        .. image:: ../../../_static/binary_op-01.svg
           :align: center

        Parameters
        ----------
        other : BaseGeometry or GeoSeries
            The other geometry to computed
            the DE-9IM intersection matrices from.
        align : bool (default True)
            If True, automatically aligns GeoSeries based on their indices.
            If False, the order of elements is preserved.

        Returns
        -------
        spatial_relations: Series of strings
            The DE-9IM intersection matrices which describe
            the spatial relations of the other geometry.

        Examples
        --------
        >>> from shapely.geometry import Polygon, LineString, Point
        >>> s = geopandas.GeoSeries(
        ...     [
        ...         Polygon([(0, 0), (2, 2), (0, 2)]),
        ...         Polygon([(0, 0), (2, 2), (0, 2)]),
        ...         LineString([(0, 0), (2, 2)]),
        ...         LineString([(2, 0), (0, 2)]),
        ...         Point(0, 1),
        ...     ],
        ... )
        >>> s2 = geopandas.GeoSeries(
        ...     [
        ...         Polygon([(0, 0), (1, 1), (0, 1)]),
        ...         LineString([(1, 0), (1, 3)]),
        ...         LineString([(2, 0), (0, 2)]),
        ...         Point(1, 1),
        ...         Point(0, 1),
        ...     ],
        ...     index=range(1, 6),
        ... )

        >>> s
        0    POLYGON ((0.00000 0.00000, 2.00000 2.00000, 0....
        1    POLYGON ((0.00000 0.00000, 2.00000 2.00000, 0....
        2        LINESTRING (0.00000 0.00000, 2.00000 2.00000)
        3        LINESTRING (2.00000 0.00000, 0.00000 2.00000)
        4                              POINT (0.00000 1.00000)
        dtype: geometry

        >>> s2
        1    POLYGON ((0.00000 0.00000, 1.00000 1.00000, 0....
        2        LINESTRING (1.00000 0.00000, 1.00000 3.00000)
        3        LINESTRING (2.00000 0.00000, 0.00000 2.00000)
        4                              POINT (1.00000 1.00000)
        5                              POINT (0.00000 1.00000)
        dtype: geometry

        We can relate each geometry and a single
        shapely geometry:

        .. image:: ../../../_static/binary_op-03.svg
           :align: center

        >>> s.relate(Polygon([(0, 0), (1, 1), (0, 1)]))
        0    212F11FF2
        1    212F11FF2
        2    F11F00212
        3    F01FF0212
        4    F0FFFF212
        dtype: object

        We can also check two GeoSeries against each other, row by row.
        The GeoSeries above have different indices. We can either align both GeoSeries
        based on index values and compare elements with the same index using
        ``align=True`` or ignore index and compare elements based on their matching
        order using ``align=False``:

        .. image:: ../../../_static/binary_op-02.svg

        >>> s.relate(s2, align=True)
        0         None
        1    212F11FF2
        2    0F1FF0102
        3    1FFF0FFF2
        4    FF0FFF0F2
        5         None
        dtype: object

        >>> s.relate(s2, align=False)
        0    212F11FF2
        1    1F20F1102
        2    0F1FF0102
        3    0F1FF0FF2
        4    0FFFFFFF2
        dtype: object

        """
        return _binary_op("relate", self, other, align)

    def project(self, other, normalized=False, align=True):
        """
        Return the distance along each geometry nearest to *other*

        The operation works on a 1-to-1 row-wise manner:

        .. image:: ../../../_static/binary_op-01.svg
           :align: center

        The project method is the inverse of interpolate.


        Parameters
        ----------
        other : BaseGeometry or GeoSeries
            The *other* geometry to computed projected point from.
        normalized : boolean
            If normalized is True, return the distance normalized to
            the length of the object.
        align : bool (default True)
            If True, automatically aligns GeoSeries based on their indices.
            If False, the order of elements is preserved.

        Returns
        -------
        Series

        Examples
        --------
        >>> from shapely.geometry import LineString, Point
        >>> s = geopandas.GeoSeries(
        ...     [
        ...         LineString([(0, 0), (2, 0), (0, 2)]),
        ...         LineString([(0, 0), (2, 2)]),
        ...         LineString([(2, 0), (0, 2)]),
        ...     ],
        ... )
        >>> s2 = geopandas.GeoSeries(
        ...     [
        ...         Point(1, 0),
        ...         Point(1, 0),
        ...         Point(2, 1),
        ...     ],
        ...     index=range(1, 4),
        ... )

        >>> s
        0    LINESTRING (0.00000 0.00000, 2.00000 0.00000, ...
        1        LINESTRING (0.00000 0.00000, 2.00000 2.00000)
        2        LINESTRING (2.00000 0.00000, 0.00000 2.00000)
        dtype: geometry

        >>> s2
        1    POINT (1.00000 0.00000)
        2    POINT (1.00000 0.00000)
        3    POINT (2.00000 1.00000)
        dtype: geometry

        We can project each geometry on a single
        shapely geometry:

        .. image:: ../../../_static/binary_op-03.svg
           :align: center

        >>> s.project(Point(1, 0))
        0    1.000000
        1    0.707107
        2    0.707107
        dtype: float64

        We can also check two GeoSeries against each other, row by row.
        The GeoSeries above have different indices. We can either align both GeoSeries
        based on index values and project elements with the same index using
        ``align=True`` or ignore index and project elements based on their matching
        order using ``align=False``:

        .. image:: ../../../_static/binary_op-02.svg

        >>> s.project(s2, align=True)
        0         NaN
        1    0.707107
        2    0.707107
        3         NaN
        dtype: float64

        >>> s.project(s2, align=False)
        0    1.000000
        1    0.707107
        2    0.707107
        dtype: float64

        See also
        --------
        GeoSeries.interpolate
        """
        return _binary_op("project", self, other, normalized=normalized, align=align)

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
            the 6 parameter matrix is ``[a, b, d, e, xoff, yoff]``

            For 3D affine transformations,
            the 12 parameter matrix is ``[a, b, c, d, e, f, g, h, i, xoff, yoff, zoff]``

        Examples
        --------
        >>> from shapely.geometry import Point, LineString, Polygon
        >>> s = geopandas.GeoSeries(
        ...     [
        ...         Point(1, 1),
        ...         LineString([(1, -1), (1, 0)]),
        ...         Polygon([(3, -1), (4, 0), (3, 1)]),
        ...     ]
        ... )
        >>> s
        0                              POINT (1.00000 1.00000)
        1       LINESTRING (1.00000 -1.00000, 1.00000 0.00000)
        2    POLYGON ((3.00000 -1.00000, 4.00000 0.00000, 3...
        dtype: geometry

        >>> s.affine_transform([2, 3, 2, 4, 5, 2])
        0                             POINT (10.00000 8.00000)
        1        LINESTRING (4.00000 0.00000, 7.00000 4.00000)
        2    POLYGON ((8.00000 4.00000, 13.00000 10.00000, ...
        dtype: geometry

        """  # (E501 link is longer than max line length)
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

        Examples
        --------
        >>> from shapely.geometry import Point, LineString, Polygon
        >>> s = geopandas.GeoSeries(
        ...     [
        ...         Point(1, 1),
        ...         LineString([(1, -1), (1, 0)]),
        ...         Polygon([(3, -1), (4, 0), (3, 1)]),
        ...     ]
        ... )
        >>> s
        0                              POINT (1.00000 1.00000)
        1       LINESTRING (1.00000 -1.00000, 1.00000 0.00000)
        2    POLYGON ((3.00000 -1.00000, 4.00000 0.00000, 3...
        dtype: geometry

        >>> s.translate(2, 3)
        0                              POINT (3.00000 4.00000)
        1        LINESTRING (3.00000 2.00000, 3.00000 3.00000)
        2    POLYGON ((5.00000 2.00000, 6.00000 3.00000, 5....
        dtype: geometry

        """  # (E501 link is longer than max line length)
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

        Examples
        --------
        >>> from shapely.geometry import Point, LineString, Polygon
        >>> s = geopandas.GeoSeries(
        ...     [
        ...         Point(1, 1),
        ...         LineString([(1, -1), (1, 0)]),
        ...         Polygon([(3, -1), (4, 0), (3, 1)]),
        ...     ]
        ... )
        >>> s
        0                              POINT (1.00000 1.00000)
        1       LINESTRING (1.00000 -1.00000, 1.00000 0.00000)
        2    POLYGON ((3.00000 -1.00000, 4.00000 0.00000, 3...
        dtype: geometry

        >>> s.rotate(90)
        0                              POINT (1.00000 1.00000)
        1      LINESTRING (1.50000 -0.50000, 0.50000 -0.50000)
        2    POLYGON ((4.50000 -0.50000, 3.50000 0.50000, 2...
        dtype: geometry

        >>> s.rotate(90, origin=(0, 0))
        0                             POINT (-1.00000 1.00000)
        1        LINESTRING (1.00000 1.00000, 0.00000 1.00000)
        2    POLYGON ((1.00000 3.00000, 0.00000 4.00000, -1...
        dtype: geometry

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

        Examples
        --------
        >>> from shapely.geometry import Point, LineString, Polygon
        >>> s = geopandas.GeoSeries(
        ...     [
        ...         Point(1, 1),
        ...         LineString([(1, -1), (1, 0)]),
        ...         Polygon([(3, -1), (4, 0), (3, 1)]),
        ...     ]
        ... )
        >>> s
        0                              POINT (1.00000 1.00000)
        1       LINESTRING (1.00000 -1.00000, 1.00000 0.00000)
        2    POLYGON ((3.00000 -1.00000, 4.00000 0.00000, 3...
        dtype: geometry

        >>> s.scale(2, 3)
        0                              POINT (1.00000 1.00000)
        1       LINESTRING (1.00000 -2.00000, 1.00000 1.00000)
        2    POLYGON ((2.50000 -3.00000, 4.50000 0.00000, 2...
        dtype: geometry

        >>> s.scale(2, 3, origin=(0, 0))
        0                              POINT (2.00000 3.00000)
        1       LINESTRING (2.00000 -3.00000, 2.00000 0.00000)
        2    POLYGON ((6.00000 -3.00000, 8.00000 0.00000, 6...
        dtype: geometry
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

        Examples
        --------
        >>> from shapely.geometry import Point, LineString, Polygon
        >>> s = geopandas.GeoSeries(
        ...     [
        ...         Point(1, 1),
        ...         LineString([(1, -1), (1, 0)]),
        ...         Polygon([(3, -1), (4, 0), (3, 1)]),
        ...     ]
        ... )
        >>> s
        0                              POINT (1.00000 1.00000)
        1       LINESTRING (1.00000 -1.00000, 1.00000 0.00000)
        2    POLYGON ((3.00000 -1.00000, 4.00000 0.00000, 3...
        dtype: geometry

        >>> s.skew(45, 30)
        0                              POINT (1.00000 1.00000)
        1       LINESTRING (0.50000 -1.00000, 1.50000 0.00000)
        2    POLYGON ((2.00000 -1.28868, 4.00000 0.28868, 4...
        dtype: geometry

        >>> s.skew(45, 30, origin=(0, 0))
        0                              POINT (2.00000 1.57735)
        1       LINESTRING (0.00000 -0.42265, 1.00000 0.57735)
        2    POLYGON ((2.00000 0.73205, 4.00000 2.30940, 4....
        dtype: geometry
        """
        return _delegate_geo_method(
            "skew", self, xs, ys, origin=origin, use_radians=use_radians
        )

    @property
    def cx(self):
        """
        Coordinate based indexer to select by intersection with bounding box.

        Format of input should be ``.cx[xmin:xmax, ymin:ymax]``. Any of
        ``xmin``, ``xmax``, ``ymin``, and ``ymax`` can be provided, but input
        must include a comma separating x and y slices. That is, ``.cx[:, :]``
        will return the full series/frame, but ``.cx[:]`` is not implemented.

        Examples
        --------
        >>> from shapely.geometry import LineString, Point
        >>> s = geopandas.GeoSeries(
        ...     [Point(0, 0), Point(1, 2), Point(3, 3), LineString([(0, 0), (3, 3)])]
        ... )
        >>> s
        0                          POINT (0.00000 0.00000)
        1                          POINT (1.00000 2.00000)
        2                          POINT (3.00000 3.00000)
        3    LINESTRING (0.00000 0.00000, 3.00000 3.00000)
        dtype: geometry

        >>> s.cx[0:1, 0:1]
        0                          POINT (0.00000 0.00000)
        3    LINESTRING (0.00000 0.00000, 3.00000 3.00000)
        dtype: geometry

        >>> s.cx[:, 1:]
        1                          POINT (1.00000 2.00000)
        2                          POINT (3.00000 3.00000)
        3    LINESTRING (0.00000 0.00000, 3.00000 3.00000)
        dtype: geometry

        """
        return _CoordinateIndexer(self)

    def get_coordinates(self, include_z=False, ignore_index=False, index_parts=False):
        """Gets coordinates from a :class:`GeoSeries` as a :class:`~pandas.DataFrame` of
        floats.

        The shape of the returned :class:`~pandas.DataFrame` is (N, 2), with N being the
        number of coordinate pairs. With the default of ``include_z=False``,
        three-dimensional data is ignored. When specifying ``include_z=True``, the shape
        of the returned :class:`~pandas.DataFrame` is (N, 3).

        Parameters
        ----------
        include_z : bool, default False
            Include Z coordinates
        ignore_index : bool, default False
            If True, the resulting index will be labelled 0, 1, …, n - 1, ignoring
            ``index_parts``.
        index_parts : bool, default False
           If True, the resulting index will be a :class:`~pandas.MultiIndex` (original
           index with an additional level indicating the ordering of the coordinate
           pairs: a new zero-based index for each geometry in the original GeoSeries).

        Returns
        -------
        pandas.DataFrame

        Examples
        --------
        >>> from shapely.geometry import Point, LineString, Polygon
        >>> s = geopandas.GeoSeries(
        ...     [
        ...         Point(1, 1),
        ...         LineString([(1, -1), (1, 0)]),
        ...         Polygon([(3, -1), (4, 0), (3, 1)]),
        ...     ]
        ... )
        >>> s
        0                              POINT (1.00000 1.00000)
        1       LINESTRING (1.00000 -1.00000, 1.00000 0.00000)
        2    POLYGON ((3.00000 -1.00000, 4.00000 0.00000, 3...
        dtype: geometry

        >>> s.get_coordinates()
             x    y
        0  1.0  1.0
        1  1.0 -1.0
        1  1.0  0.0
        2  3.0 -1.0
        2  4.0  0.0
        2  3.0  1.0
        2  3.0 -1.0

        >>> s.get_coordinates(ignore_index=True)
             x    y
        0  1.0  1.0
        1  1.0 -1.0
        2  1.0  0.0
        3  3.0 -1.0
        4  4.0  0.0
        5  3.0  1.0
        6  3.0 -1.0

        >>> s.get_coordinates(index_parts=True)
               x    y
        0 0  1.0  1.0
        1 0  1.0 -1.0
          1  1.0  0.0
        2 0  3.0 -1.0
          1  4.0  0.0
          2  3.0  1.0
          3  3.0 -1.0
        """
        if compat.USE_SHAPELY_20:
            import shapely

            coords, outer_idx = shapely.get_coordinates(
                self.geometry.values._data, include_z=include_z, return_index=True
            )
        elif compat.USE_PYGEOS:
            import pygeos

            coords, outer_idx = pygeos.get_coordinates(
                self.geometry.values._data, include_z=include_z, return_index=True
            )

        else:
            import shapely

            raise NotImplementedError(
                f"shapely >= 2.0 or PyGEOS are required, "
                f"version {shapely.__version__} is installed."
            )

        column_names = ["x", "y"]
        if include_z:
            column_names.append("z")

        index = _get_index_for_parts(
            self.index,
            outer_idx,
            ignore_index=ignore_index,
            index_parts=index_parts,
        )

        return pd.DataFrame(coords, index=index, columns=column_names)

    def hilbert_distance(self, total_bounds=None, level=16):
        """
        Calculate the distance along a Hilbert curve.

        The distances are calculated for the midpoints of the geometries in the
        GeoDataFrame, and using the total bounds of the GeoDataFrame.

        The Hilbert distance can be used to spatially sort GeoPandas
        objects, by mapping two dimensional geometries along the Hilbert curve.

        Parameters
        ----------
        total_bounds : 4-element array, optional
            The spatial extent in which the curve is constructed (used to
            rescale the geometry midpoints). By default, the total bounds
            of the full GeoDataFrame or GeoSeries will be computed. If known,
            you can pass the total bounds to avoid this extra computation.
        level : int (1 - 16), default 16
            Determines the precision of the curve (points on the curve will
            have coordinates in the range [0, 2^level - 1]).

        Returns
        -------
        Series
            Series containing distance along the curve for geometry
        """
        from geopandas.tools.hilbert_curve import _hilbert_distance

        distances = _hilbert_distance(
            self.geometry.values, total_bounds=total_bounds, level=level
        )

        return pd.Series(distances, index=self.index, name="hilbert_distance")

    def sample_points(self, size, method="uniform", seed=None, rng=None, **kwargs):
        """
        Sample points from each geometry.

        Generate a MultiPoint per each geometry containing points sampled from the
        geometry. You can either sample randomly from a uniform distribution or use an
        advanced sampling algorithm from the ``pointpats`` package.

        For polygons, this samples within the area of the polygon. For lines,
        this samples along the length of the linestring. For multi-part
        geometries, the weights of each part are selected according to their relevant
        attribute (area for Polygons, length for LineStrings), and then points are
        sampled from each part.

        Any other geometry type (e.g. Point, GeometryCollection) is ignored, and an
        empty MultiPoint geometry is returned.

        Parameters
        ----------
        size : int | array-like
            The size of the sample requested. Indicates the number of samples to draw
            from each geometry.  If an array of the same length as a GeoSeries is
            passed, it denotes the size of a sample per geometry.
        method : str, default "uniform"
            The sampling method. ``uniform`` samples uniformly at random from a
            geometry using ``numpy.random.uniform``. Other allowed strings
            (e.g. ``"cluster_poisson"``) denote sampling function name from the
            ``pointpats.random`` module (see
            http://pysal.org/pointpats/api.html#random-distributions). Pointpats methods
            are implemented for (Multi)Polygons only and will return an empty MultiPoint
            for other geometry types.
        rng : {None, int, array_like[ints], SeedSequence, BitGenerator, Generator}, optional
            A random generator or seed to initialize the numpy BitGenerator. If None, then fresh,
            unpredictable entropy will be pulled from the OS.
        **kwargs : dict
            Options for the pointpats sampling algorithms.

        Returns
        -------
        GeoSeries
            Points sampled within (or along) each geometry.

        Examples
        --------
        >>> from shapely.geometry import Polygon
        >>> s = geopandas.GeoSeries(
        ...     [
        ...         Polygon([(1, -1), (1, 0), (0, 0)]),
        ...         Polygon([(3, -1), (4, 0), (3, 1)]),
        ...     ]
        ... )

        >>> s.sample_points(size=10)  # doctest: +SKIP
        0    MULTIPOINT (0.04783 -0.04244, 0.24196 -0.09052...
        1    MULTIPOINT (3.00672 -0.52390, 3.01776 0.30065,...
        Name: sampled_points, dtype: geometry
        """  # noqa: E501
        from .geoseries import GeoSeries
        from .tools._random import uniform

        if seed is not None:
            warn(
                "The 'seed' keyword is deprecated. Use 'rng' instead.",
                FutureWarning,
                stacklevel=2,
            )
            rng = seed

        if method == "uniform":
            if pd.api.types.is_list_like(size):
                result = [uniform(geom, s, rng) for geom, s in zip(self.geometry, size)]
            else:
                result = self.geometry.apply(uniform, size=size, rng=rng)

        else:
            pointpats = compat.import_optional_dependency(
                "pointpats",
                f"For complex sampling methods, the pointpats module is required. "
                f"Your requested method, '{method}' was not a supported option "
                f"and the pointpats package was not able to be imported.",
            )

            if not hasattr(pointpats.random, method):
                raise AttributeError(
                    f"pointpats.random module has no sampling method {method}."
                    f"Consult the pointpats.random module documentation for"
                    f" available random sampling methods."
                )
            sample_function = getattr(pointpats.random, method)
            result = self.geometry.apply(
                lambda x: points_from_xy(
                    *sample_function(x, size=size, **kwargs).T
                ).unary_union()
                if not (x.is_empty or x is None or "Polygon" not in x.geom_type)
                else MultiPoint(),
            )

        return GeoSeries(result, name="sampled_points", crs=self.crs, index=self.index)


def _get_index_for_parts(orig_idx, outer_idx, ignore_index, index_parts):
    """Helper to handle index when geometries get exploded to parts.

    Used in get_coordinates and explode.

    Parameters
    ----------
    orig_idx : pandas.Index
        original index
    outer_idx : array
        the index of each returned geometry as a separate ndarray of integers
    ignore_index : bool
    index_parts : bool

    Returns
    -------
    pandas.Index
        index or multiindex
    """

    if ignore_index:
        return None
    else:
        if len(outer_idx):
            # Generate inner index as a range per value of outer_idx
            # 1. identify the start of each run of values in outer_idx
            # 2. count number of values per run
            # 3. use cumulative sums to create an incremental range
            #    starting at 0 in each run
            run_start = np.r_[True, outer_idx[:-1] != outer_idx[1:]]
            counts = np.diff(np.r_[np.nonzero(run_start)[0], len(outer_idx)])
            inner_index = (~run_start).cumsum(dtype=outer_idx.dtype)
            inner_index -= np.repeat(inner_index[run_start], counts)

        else:
            inner_index = []

        # extract original index values based on integer index
        outer_index = orig_idx.take(outer_idx)

        if index_parts:
            nlevels = outer_index.nlevels
            index_arrays = [outer_index.get_level_values(lvl) for lvl in range(nlevels)]
            index_arrays.append(inner_index)

            index = pd.MultiIndex.from_arrays(
                index_arrays, names=list(orig_idx.names) + [None]
            )

        else:
            index = outer_index

    return index


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
            warn(
                "Ignoring step - full interval is used.",
                stacklevel=2,
            )
        if xs.start is None or xs.stop is None or ys.start is None or ys.stop is None:
            xmin, ymin, xmax, ymax = obj.total_bounds
        bbox = box(
            xs.start if xs.start is not None else xmin,
            ys.start if ys.start is not None else ymin,
            xs.stop if xs.stop is not None else xmax,
            ys.stop if ys.stop is not None else ymax,
        )
        idx = obj.intersects(bbox)
        return obj[idx]
