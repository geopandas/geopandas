from warnings import warn

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

import shapely
from shapely.geometry import MultiPoint, box
from shapely.geometry.base import BaseGeometry

from . import _compat as compat
from .array import GeometryArray, GeometryDtype, points_from_xy


def is_geometry_type(data):
    """Check if the data is of geometry dtype.

    Does not include object array of shapely scalars.
    """
    if isinstance(getattr(data, "dtype", None), GeometryDtype):
        # GeometryArray, GeoSeries and Series[GeometryArray]
        return True
    else:
        return False


def _delegate_binary_method(op, this, other, align, *args, **kwargs):
    # type: (str, GeoSeries, GeoSeries) -> GeoSeries/Series
    if align is None:
        align = True
        maybe_warn = True
    else:
        maybe_warn = False
    this = this.geometry

    # Use same alignment logic, regardless of if `other` is Series or GeoSeries.
    if (
        not isinstance(other, GeoPandasBase)
        and isinstance(other, pd.Series)
        and isinstance(other.dtype, GeometryDtype)
    ):
        # Avoid circular imports by importing here.
        import geopandas.geoseries

        other = geopandas.geoseries.GeoSeries(other)

    if isinstance(other, GeoPandasBase):
        if align and not this.index.equals(other.index):
            if maybe_warn:
                warn(
                    "The indices of the left and right GeoSeries' are not equal, and "
                    "therefore they will be aligned (reordering and/or introducing "
                    "missing values) before executing the operation. If this alignment "
                    "is the desired behaviour, you can silence this warning by passing "
                    "'align=True'. If you don't want alignment and protect yourself of "
                    "accidentally aligning, you can pass 'align=False'.",
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


def _binary_geo(op, this, other, align, *args, **kwargs):
    # type: (str, GeoSeries, GeoSeries) -> GeoSeries
    """Binary operation on GeoSeries objects that returns a GeoSeries."""
    from .geoseries import GeoSeries

    geoms, index = _delegate_binary_method(op, this, other, align, *args, **kwargs)
    return GeoSeries(geoms, index=index, crs=this.crs)


def _binary_op(op, this, other, align, *args, **kwargs):
    # type: (str, GeoSeries, GeoSeries, args/kwargs) -> Series[bool/float]
    """Binary operation on GeoSeries objects that returns a Series."""
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


def _delegate_geo_method(op, this, **kwargs):
    # type: (str, GeoSeries) -> GeoSeries
    """Unary operation that returns a GeoSeries."""
    from .geodataframe import GeoDataFrame
    from .geoseries import GeoSeries

    if isinstance(this, GeoSeries):
        klass, var_name = "GeoSeries", "gs"
    elif isinstance(this, GeoDataFrame):
        klass, var_name = "GeoDataFrame", "gdf"
    else:
        klass, var_name = this.__class__.__name__, "this"

    for key, val in kwargs.items():
        if isinstance(val, pd.Series):
            if not val.index.equals(this.index):
                raise ValueError(
                    f"Index of the Series passed as '{key}' does not match index of "
                    f"the {klass}. If you want both Series to be aligned, align them "
                    f"before passing them to this method as "
                    f"`{var_name}, {key} = {var_name}.align({key})`. If "
                    f"you want to ignore the index, pass the underlying array as "
                    f"'{key}' using `{key}.values`."
                )
            kwargs[key] = np.asarray(val)

    a_this = GeometryArray(this.geometry.values)
    data = getattr(a_this, op)(**kwargs)
    return GeoSeries(data, index=this.index, crs=this.crs)


class GeoPandasBase:
    @property
    def area(self):
        """Return a ``Series`` containing the area of each geometry in the
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
        0       POLYGON ((0 0, 1 1, 0 1, 0 0))
        1    POLYGON ((10 0, 10 5, 0 0, 10 0))
        2       POLYGON ((0 0, 2 2, 2 0, 0 0))
        3           LINESTRING (0 0, 1 1, 0 1)
        4                          POINT (0 1)
        dtype: geometry

        >>> s.area
        0     0.5
        1    25.0
        2     2.0
        3     0.0
        4     0.0
        dtype: float64

        See Also
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
        """The Coordinate Reference System (CRS) as a ``pyproj.CRS`` object.

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

        See Also
        --------
        GeoSeries.set_crs : assign CRS
        GeoSeries.to_crs : re-project to another CRS
        """
        return self.geometry.values.crs

    @crs.setter
    def crs(self, value):
        """Set the value of the crs."""
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
        """Return the geometry type of each geometry in the GeoSeries."""
        return self.geom_type

    @property
    def length(self):
        """Return a ``Series`` containing the length of each geometry
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
        0                           LINESTRING (0 0, 1 1, 0 1)
        1                         LINESTRING (10 0, 10 5, 0 0)
        2            MULTILINESTRING ((0 0, 1 0), (-1 0, 1 0))
        3                       POLYGON ((0 0, 1 1, 0 1, 0 0))
        4                                          POINT (0 1)
        5    GEOMETRYCOLLECTION (POINT (1 0), LINESTRING (1...
        dtype: geometry

        >>> s.length
        0     2.414214
        1    16.180340
        2     3.000000
        3     3.414214
        4     0.000000
        5    16.180340
        dtype: float64

        See Also
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
        """Return a ``Series`` of ``dtype('bool')`` with value ``True`` for
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
        0         POLYGON ((0 0, 1 1, 0 1, 0 0))
        1    POLYGON ((0 0, 1 1, 1 0, 0 1, 0 0))
        2         POLYGON ((0 0, 2 2, 2 0, 0 0))
        3                                   None
        dtype: geometry

        >>> s.is_valid
        0     True
        1    False
        2     True
        3    False
        dtype: bool

        See Also
        --------
        GeoSeries.is_valid_reason : reason for invalidity
        """
        return _delegate_property("is_valid", self)

    def is_valid_reason(self):
        """Return a ``Series`` of strings with the reason for invalidity of
        each geometry.

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
        0         POLYGON ((0 0, 1 1, 0 1, 0 0))
        1    POLYGON ((0 0, 1 1, 1 0, 0 1, 0 0))
        2         POLYGON ((0 0, 2 2, 2 0, 0 0))
        3                                   None
        dtype: geometry

        >>> s.is_valid_reason()
        0    Valid Geometry
        1    Self-intersection[0.5 0.5]
        2    Valid Geometry
        3    None
        dtype: object

        See Also
        --------
        GeoSeries.is_valid : detect invalid geometries
        GeoSeries.make_valid : fix invalid geometries
        """
        return Series(self.geometry.values.is_valid_reason(), index=self.index)

    def is_valid_coverage(self, *, gap_width=0.0):
        """Return a ``bool`` indicating whether a ``GeoSeries`` forms a valid coverage.

        A ``GeoSeries`` of valid polygons is considered a coverage if the polygons are:

        * **Non-overlapping** - polygons do not overlap (their interiors do not
          intersect)
        * **Edge-Matched** - vertices along shared edges are identical

        A valid coverage may contain holes (regions of no coverage). However, sometimes
        it might be desirable to detect narrow gaps as invalidities in the coverage. The
        ``gap_width`` parameter allows to specify the maximum width of gaps to detect.
        When gaps are detected, this method will return ``False`` and the
        :meth:`coverage_invalid_edges` method can be used to find the edges of those
        gaps.

        Geometries that are not Polygon or MultiPolygon are ignored and an empty
        LineString is returned.

        Requires Shapely >= 2.1.

        .. versionadded:: 1.1.0

        Parameters
        ----------
        gap_width : float, optional
            The maximum width of gaps to detect, by default 0.0

        Returns
        -------
        bool

        Examples
        --------
        >>> from shapely import Polygon
        >>> s = geopandas.GeoSeries(
        ...     [
        ...         Polygon([(0, 0), (1, 1), (1, 0), (0, 0)]),
        ...         Polygon([(0, 0), (1, 1), (0, 1), (0, 0)]),
        ...     ]
        ... )
        >>> s
        0    POLYGON ((0 0, 1 1, 1 0, 0 0))
        1    POLYGON ((0 0, 1 1, 0 1, 0 0))
        dtype: geometry

        >>> s.is_valid_coverage()
        True

        Violation of edge-matching:

        >>> s2 = geopandas.GeoSeries(
        ...     [
        ...         Polygon([(0, 0), (1, 1), (1, 0), (0, 0)]),
        ...         Polygon([(0, 0), (0.5, 0.5), (1, 1), (0, 1), (0, 0)])
        ...     ]
        ... )
        >>> s2
        0             POLYGON ((0 0, 1 1, 1 0, 0 0))
        1    POLYGON ((0 0, 0.5 0.5, 1 1, 0 1, 0 0))
        dtype: geometry

        >>> s2.is_valid_coverage()
        False

        See Also
        --------
        GeoSeries.invalid_coverage_edges
        GeoSeries.simplify_coverage
        """
        return self.geometry.values.is_valid_coverage(gap_width=gap_width)

    def invalid_coverage_edges(self, *, gap_width=0.0):
        """Return a ``GeoSeries`` containing edges causing invalid polygonal coverage.

        This method returns (Multi)LineStrings showing the location of edges violating
        polygonal coverage (if any) in each polygon in the input ``GeoSeries``.

        A ``GeoSeries`` of valid polygons is considered a coverage if the polygons are:

        * **Non-overlapping** - polygons do not overlap (their interiors do not
          intersect)
        * **Edge-Matched** - vertices along shared edges are identical

        A valid coverage may contain holes (regions of no coverage). However, sometimes
        it might be desirable to detect narrow gaps as invalidities in the coverage. The
        ``gap_width`` parameter allows to specify the maximum width of gaps to detect.
        When gaps are detected, the :meth:`is_valid_coverage` method will return
        ``False`` and this method can be used to find the edges of those gaps.

        Geometries that are not Polygon or MultiPolygon are ignored.

        Requires Shapely >= 2.1.

        .. versionadded:: 1.1.0

        Parameters
        ----------
        gap_width : float, optional
            The maximum width of gaps to detect, by default 0.0

        Returns
        -------
        GeoSeries

        Examples
        --------
        Violation of edge-matching:

        >>> from shapely import Polygon
        >>> s = geopandas.GeoSeries(
        ...     [
        ...         Polygon([(0, 0), (1, 1), (1, 0), (0, 0)]),
        ...         Polygon([(0, 0), (0.5, 0.5), (1, 1), (0, 1), (0, 0)])
        ...     ]
        ... )
        >>> s
        0             POLYGON ((0 0, 1 1, 1 0, 0 0))
        1    POLYGON ((0 0, 0.5 0.5, 1 1, 0 1, 0 0))
        dtype: geometry

        >>> s.invalid_coverage_edges()
        0             LINESTRING (0 0, 1 1)
        1    LINESTRING (0 0, 0.5 0.5, 1 1)
        dtype: geometry


        See Also
        --------
        GeoSeries.is_valid_coverage
        GeoSeries.simplify_coverage
        """
        return _delegate_geo_method("invalid_coverage_edges", self, gap_width=gap_width)

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
        0  POINT EMPTY
        1  POINT (2 1)
        2         None

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

    def count_coordinates(self):
        """Return a ``Series`` containing the count of the number of coordinate pairs
        in each geometry.

        Examples
        --------
        An example of a GeoDataFrame with two line strings, one point and one None
        value:

        >>> from shapely.geometry import Polygon, LineString, Point
        >>> s = geopandas.GeoSeries(
        ...     [
        ...         LineString([(0, 0), (1, 1), (1, -1), (0, 1)]),
        ...         LineString([(0, 0), (1, 1), (1, -1)]),
        ...         Point(0, 0),
        ...         Polygon([(10, 10), (10, 20), (20, 20), (20, 10), (10, 10)]),
        ...         None
        ...     ]
        ... )
        >>> s
        0                 LINESTRING (0 0, 1 1, 1 -1, 0 1)
        1                      LINESTRING (0 0, 1 1, 1 -1)
        2                                      POINT (0 0)
        3    POLYGON ((10 10, 10 20, 20 20, 20 10, 10 10))
        4                                             None
        dtype: geometry

        >>> s.count_coordinates()
        0    4
        1    3
        2    1
        3    5
        4    0
        dtype: int32

        See Also
        --------
        GeoSeries.get_coordinates : extract coordinates as a :class:`~pandas.DataFrame`
        GoSeries.count_geometries : count the number of geometries in a collection
        """
        return Series(self.geometry.values.count_coordinates(), index=self.index)

    def count_geometries(self):
        """Return a ``Series`` containing the count of geometries in each multi-part
        geometry.

        For single-part geometry objects, this is always 1. For multi-part geometries,
        like ``MultiPoint`` or ``MultiLineString``, it is the number of parts in the
        geometry. For ``GeometryCollection``, it is the number of geometries direct
        parts of the collection (the method does not recurse into collections within
        collections).


        Examples
        --------
        >>> from shapely.geometry import Point, MultiPoint, LineString, MultiLineString
        >>> s = geopandas.GeoSeries(
        ...     [
        ...         MultiPoint([(0, 0), (1, 1), (1, -1), (0, 1)]),
        ...         MultiLineString([((0, 0), (1, 1)), ((-1, 0), (1, 0))]),
        ...         LineString([(0, 0), (1, 1), (1, -1)]),
        ...         Point(0, 0),
        ...     ]
        ... )
        >>> s
        0     MULTIPOINT ((0 0), (1 1), (1 -1), (0 1))
        1    MULTILINESTRING ((0 0, 1 1), (-1 0, 1 0))
        2                  LINESTRING (0 0, 1 1, 1 -1)
        3                                  POINT (0 0)
        dtype: geometry

        >>> s.count_geometries()
        0    4
        1    2
        2    1
        3    1
        dtype: int32

        See Also
        --------
        GeoSeries.count_coordinates : count the number of coordinates in a geometry
        GeoSeries.count_interior_rings : count the number of interior rings
        """
        return Series(self.geometry.values.count_geometries(), index=self.index)

    def count_interior_rings(self):
        """Return a ``Series`` containing the count of the number of interior rings
        in a polygonal geometry.

        For non-polygonal geometries, this is always 0.

        Examples
        --------
        >>> from shapely.geometry import Polygon, Point
        >>> s = geopandas.GeoSeries(
        ...     [
        ...         Polygon(
        ...             [(0, 0), (0, 5), (5, 5), (5, 0)],
        ...             [[(1, 1), (1, 4), (4, 4), (4, 1)]],
        ...         ),
        ...         Polygon(
        ...             [(0, 0), (0, 5), (5, 5), (5, 0)],
        ...             [
        ...                 [(1, 1), (1, 2), (2, 2), (2, 1)],
        ...                 [(3, 2), (3, 3), (4, 3), (4, 2)],
        ...             ],
        ...         ),
        ...         Point(0, 1),
        ...     ]
        ... )
        >>> s
        0    POLYGON ((0 0, 0 5, 5 5, 5 0, 0 0), (1 1, 1 4,...
        1    POLYGON ((0 0, 0 5, 5 5, 5 0, 0 0), (1 1, 1 2,...
        2                                          POINT (0 1)
        dtype: geometry

        >>> s.count_interior_rings()
        0    1
        1    2
        2    0
        dtype: int32

        See Also
        --------
        GeoSeries.count_coordinates : count the number of coordinates in a geometry
        GeoSeries.count_geometries : count the number of geometries in a collection
        """
        return Series(self.geometry.values.count_interior_rings(), index=self.index)

    @property
    def is_simple(self):
        """Return a ``Series`` of ``dtype('bool')`` with value ``True`` for
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
        0    LINESTRING (0 0, 1 1, 1 -1, 0 1)
        1         LINESTRING (0 0, 1 1, 1 -1)
        dtype: geometry

        >>> s.is_simple
        0    False
        1     True
        dtype: bool
        """
        return _delegate_property("is_simple", self)

    @property
    def is_ring(self):
        """Return a ``Series`` of ``dtype('bool')`` with value ``True`` for
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
        0         LINESTRING (0 0, 1 1, 1 -1)
        1    LINESTRING (0 0, 1 1, 1 -1, 0 0)
        2    LINEARRING (0 0, 1 1, 1 -1, 0 0)
        dtype: geometry

        >>> s.is_ring
        0    False
        1     True
        2     True
        dtype: bool

        """
        return _delegate_property("is_ring", self)

    @property
    def is_ccw(self):
        """Return a ``Series`` of ``dtype('bool')`` with value ``True``
        if a LineString or LinearRing is counterclockwise.

        Note that there are no checks on whether lines are actually
        closed and not self-intersecting, while this is a requirement
        for ``is_ccw``. The recommended usage of this property for
        LineStrings is ``GeoSeries.is_ccw & GeoSeries.is_simple`` and for
        LinearRings ``GeoSeries.is_ccw & GeoSeries.is_valid``.

        This property will return False for non-linear geometries and for
        lines with fewer than 4 points (including the closing point).

        Examples
        --------
        >>> from shapely.geometry import LineString, LinearRing, Point
        >>> s = geopandas.GeoSeries(
        ...     [
        ...         LinearRing([(0, 0), (0, 1), (1, 1), (0, 0)]),
        ...         LinearRing([(0, 0), (1, 1), (0, 1), (0, 0)]),
        ...         LineString([(0, 0), (1, 1), (0, 1)]),
        ...         Point(3, 3)
        ...     ]
        ... )
        >>> s
        0    LINEARRING (0 0, 0 1, 1 1, 0 0)
        1    LINEARRING (0 0, 1 1, 0 1, 0 0)
        2         LINESTRING (0 0, 1 1, 0 1)
        3                        POINT (3 3)
        dtype: geometry

        >>> s.is_ccw
        0    False
        1     True
        2    False
        3    False
        dtype: bool
        """
        return _delegate_property("is_ccw", self)

    @property
    def is_closed(self):
        """Return a ``Series`` of ``dtype('bool')`` with value ``True``
        if a LineString's or LinearRing's first and last points are equal.

        Returns False for any other geometry type.

        Examples
        --------
        >>> from shapely.geometry import LineString, Point, Polygon
        >>> s = geopandas.GeoSeries(
        ...     [
        ...         LineString([(0, 0), (1, 1), (0, 1), (0, 0)]),
        ...         LineString([(0, 0), (1, 1), (0, 1)]),
        ...         Polygon([(0, 0), (0, 1), (1, 1), (0, 0)]),
        ...         Point(3, 3)
        ...     ]
        ... )
        >>> s
        0    LINESTRING (0 0, 1 1, 0 1, 0 0)
        1         LINESTRING (0 0, 1 1, 0 1)
        2     POLYGON ((0 0, 0 1, 1 1, 0 0))
        3                        POINT (3 3)
        dtype: geometry

        >>> s.is_closed
        0     True
        1    False
        2    False
        3    False
        dtype: bool
        """
        return _delegate_property("is_closed", self)

    @property
    def has_z(self):
        """Return a ``Series`` of ``dtype('bool')`` with value ``True`` for
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
        0        POINT (0 1)
        1    POINT Z (0 1 2)
        dtype: geometry

        >>> s.has_z
        0    False
        1     True
        dtype: bool
        """
        return _delegate_property("has_z", self)

    @property
    def has_m(self):
        """Return a ``Series`` of ``dtype('bool')`` with value ``True`` for
        features that have a m-component.

        Requires Shapely >= 2.1.

        .. versionadded:: 1.1.0

        Examples
        --------
        >>> from shapely.geometry import Point
        >>> s = geopandas.GeoSeries.from_wkt(
        ...     [
        ...         "POINT M (2 3 5)",
        ...         "POINT Z (1 2 3)",
        ...         "POINT (0 0)",
        ...     ]
        ... )
        >>> s
        0    POINT M (2 3 5)
        1    POINT Z (1 2 3)
        2        POINT (0 0)
        dtype: geometry

        >>> s.has_m
        0     True
        1    False
        2    False
        dtype: bool
        """
        return _delegate_property("has_m", self)

    def get_precision(self):
        """Return a ``Series`` of the precision of each geometry.

        If a precision has not been previously set, it will be 0, indicating regular
        double precision coordinates are in use. Otherwise, it will return the precision
        grid size that was set on a geometry.

        Returns NaN for not-a-geometry values.

        Examples
        --------
        >>> from shapely.geometry import Point
        >>> s = geopandas.GeoSeries(
        ...     [
        ...         Point(0, 1),
        ...         Point(0, 1, 2),
        ...         Point(0, 1.5, 2),
        ...     ]
        ... )
        >>> s
        0          POINT (0 1)
        1      POINT Z (0 1 2)
        2    POINT Z (0 1.5 2)
        dtype: geometry

        >>> s.get_precision()
        0    0.0
        1    0.0
        2    0.0
        dtype: float64

        >>> s1 = s.set_precision(1)
        >>> s1
        0        POINT (0 1)
        1    POINT Z (0 1 2)
        2    POINT Z (0 2 2)
        dtype: geometry

        >>> s1.get_precision()
        0    1.0
        1    1.0
        2    1.0
        dtype: float64

        See Also
        --------
        GeoSeries.set_precision : set precision grid size
        """
        return Series(self.geometry.values.get_precision(), index=self.index)

    def get_geometry(self, index):
        """Return the n-th geometry from a collection of geometries.

        Parameters
        ----------
        index : int or array_like
            Position of a geometry to be retrieved within its collection

        Returns
        -------
        GeoSeries

        Notes
        -----
        Simple geometries act as collections of length 1. Any out-of-range index value
        returns None.

        Examples
        --------
        >>> from shapely.geometry import Point, MultiPoint, GeometryCollection
        >>> s = geopandas.GeoSeries(
        ...     [
        ...         Point(0, 0),
        ...         MultiPoint([(0, 0), (1, 1), (0, 1), (1, 0)]),
        ...         GeometryCollection(
        ...             [MultiPoint([(0, 0), (1, 1), (0, 1), (1, 0)]), Point(0, 1)]
        ...         ),
        ...     ]
        ... )
        >>> s
        0                                          POINT (0 0)
        1              MULTIPOINT ((0 0), (1 1), (0 1), (1 0))
        2    GEOMETRYCOLLECTION (MULTIPOINT ((0 0), (1 1), ...
        dtype: geometry

        >>> s.get_geometry(0)
        0                                POINT (0 0)
        1                                POINT (0 0)
        2    MULTIPOINT ((0 0), (1 1), (0 1), (1 0))
        dtype: geometry

        >>> s.get_geometry(1)
        0           None
        1    POINT (1 1)
        2    POINT (0 1)
        dtype: geometry

        >>> s.get_geometry(-1)
        0    POINT (0 0)
        1    POINT (1 0)
        2    POINT (0 1)
        dtype: geometry

        """
        return _delegate_geo_method("get_geometry", self, index=index)

    #
    # Unary operations that return a GeoSeries
    #

    @property
    def boundary(self):
        """Return a ``GeoSeries`` of lower dimensional objects representing
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
        0    POLYGON ((0 0, 1 1, 0 1, 0 0))
        1        LINESTRING (0 0, 1 1, 1 0)
        2                       POINT (0 0)
        dtype: geometry

        >>> s.boundary
        0    LINESTRING (0 0, 1 1, 0 1, 0 0)
        1          MULTIPOINT ((0 0), (1 0))
        2           GEOMETRYCOLLECTION EMPTY
        dtype: geometry

        See Also
        --------
        GeoSeries.exterior : outer boundary (without interior rings)

        """
        return _delegate_property("boundary", self)

    @property
    def centroid(self):
        """Return a ``GeoSeries`` of points representing the centroid of each
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
        0    POLYGON ((0 0, 1 1, 0 1, 0 0))
        1        LINESTRING (0 0, 1 1, 1 0)
        2                       POINT (0 0)
        dtype: geometry

        >>> s.centroid
        0    POINT (0.33333 0.66667)
        1        POINT (0.70711 0.5)
        2                POINT (0 0)
        dtype: geometry

        See Also
        --------
        GeoSeries.representative_point : point guaranteed to be within each geometry
        """
        return _delegate_property("centroid", self)

    def concave_hull(self, ratio=0.0, allow_holes=False):
        """Return a ``GeoSeries`` of geometries representing the concave hull
        of vertices of each geometry.

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
        0                       POLYGON ((0 0, 1 1, 0 1, 0 0))
        1                           LINESTRING (0 0, 1 1, 1 0)
        2    MULTIPOINT ((0 0), (1 1), (0 1), (1 0), (0.5 0...
        3                            MULTIPOINT ((0 0), (1 1))
        4                                          POINT (0 0)
        dtype: geometry

        >>> s.concave_hull()
        0                      POLYGON ((0 1, 1 1, 0 0, 0 1))
        1                      POLYGON ((0 0, 1 1, 1 0, 0 0))
        2    POLYGON ((0.5 0.5, 0 1, 1 1, 1 0, 0 0, 0.5 0.5))
        3                               LINESTRING (0 0, 1 1)
        4                                         POINT (0 0)
        dtype: geometry

        See Also
        --------
        GeoSeries.convex_hull : convex hull geometry

        Notes
        -----
        The algorithms considers only vertices of each geometry. As a result the
        hull may not fully enclose input geometry. If that happens, increasing ``ratio``
        should resolve the issue.

        """
        return _delegate_geo_method(
            "concave_hull", self, ratio=ratio, allow_holes=allow_holes
        )

    def constrained_delaunay_triangles(self):
        """Return a :class:`~geopandas.GeoSeries` with the constrained
        Delaunay triangulation of polygons.

        A constrained Delaunay triangulation requires the edges of the input
        polygon(s) to be in the set of resulting triangle edges. An
        unconstrained delaunay triangulation only triangulates based on the
        vertices, hence triangle edges could cross polygon boundaries.

        Requires Shapely >= 2.1.

        .. versionadded:: 1.1.0

        Examples
        --------
        >>> from shapely.geometry import Polygon
        >>> s = geopandas.GeoSeries([Polygon([(0, 0), (1, 1), (0, 1)])])
        >>> s
        0                       POLYGON ((0 0, 1 1, 0 1, 0 0))
        dtype: geometry

        >>> s.constrained_delaunay_triangles()
        0         GEOMETRYCOLLECTION (POLYGON ((0 0, 0 1, 1 1, 0...
        dtype: geometry

        See Also
        --------
        GeoSeries.delaunay_triangles : Delaunay triangulation

        """
        return _delegate_geo_method("constrained_delaunay_triangles", self)

    @property
    def convex_hull(self):
        """Return a ``GeoSeries`` of geometries representing the convex hull
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
        0                       POLYGON ((0 0, 1 1, 0 1, 0 0))
        1                           LINESTRING (0 0, 1 1, 1 0)
        2    MULTIPOINT ((0 0), (1 1), (0 1), (1 0), (0.5 0...
        3                            MULTIPOINT ((0 0), (1 1))
        4                                          POINT (0 0)
        dtype: geometry

        >>> s.convex_hull
        0         POLYGON ((0 0, 0 1, 1 1, 0 0))
        1         POLYGON ((0 0, 1 1, 1 0, 0 0))
        2    POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))
        3                  LINESTRING (0 0, 1 1)
        4                            POINT (0 0)
        dtype: geometry

        See Also
        --------
        GeoSeries.concave_hull : concave hull geometry
        GeoSeries.envelope : bounding rectangle geometry

        """
        return _delegate_property("convex_hull", self)

    def delaunay_triangles(self, tolerance=0.0, only_edges=False):
        """Return a ``GeoSeries`` consisting of objects representing
        the computed Delaunay triangulation between the vertices of
        an input geometry.

        All geometries within the GeoSeries are considered together within a single
        Delaunay triangulation. The resulting geometries therefore do not map 1:1
        to input geometries. Note that each vertex of a geometry is considered a site
        for the triangulation, so the triangles will be constructed between the vertices
        of each geometry.

        Notes
        -----
        If you want to generate Delaunay triangles for each geometry separately, use
        :func:`shapely.delaunay_triangles` instead.

        Parameters
        ----------
        tolerance : float, default 0.0
            Snap input vertices together if their distance is less than this value.
        only_edges : bool (optional, default False)
            If set to True, the triangulation will return linestrings instead of
            polygons.

        Examples
        --------
        >>> from shapely import LineString, MultiPoint, Point, Polygon
        >>> s = geopandas.GeoSeries(
        ...     [
        ...         Point(1, 1),
        ...         Point(2, 2),
        ...         Point(1, 3),
        ...         Point(0, 2),
        ...     ]
        ... )
        >>> s
        0    POINT (1 1)
        1    POINT (2 2)
        2    POINT (1 3)
        3    POINT (0 2)
        dtype: geometry

        >>> s.delaunay_triangles()
        0    POLYGON ((0 2, 1 1, 1 3, 0 2))
        1    POLYGON ((1 3, 1 1, 2 2, 1 3))
        dtype: geometry

        >>> s.delaunay_triangles(only_edges=True)
        0    LINESTRING (1 3, 2 2)
        1    LINESTRING (0 2, 1 3)
        2    LINESTRING (0 2, 1 1)
        3    LINESTRING (1 1, 2 2)
        4    LINESTRING (1 1, 1 3)
        dtype: geometry

        The method supports any geometry type but keep in mind that the underlying
        algorithm is based on the vertices of the input geometries only and does not
        consider edge segments between vertices.

        >>> s2 = geopandas.GeoSeries(
        ...     [
        ...         Polygon([(0, 0), (1, 1), (0, 1)]),
        ...         LineString([(1, 0), (2, 1), (1, 2)]),
        ...         MultiPoint([(2, 3), (2, 0), (3, 1)]),
        ...     ]
        ... )
        >>> s2
        0      POLYGON ((0 0, 1 1, 0 1, 0 0))
        1          LINESTRING (1 0, 2 1, 1 2)
        2    MULTIPOINT ((2 3), (2 0), (3 1))
        dtype: geometry

        >>> s2.delaunay_triangles()
        0    POLYGON ((0 1, 0 0, 1 0, 0 1))
        1    POLYGON ((0 1, 1 0, 1 1, 0 1))
        2    POLYGON ((0 1, 1 1, 1 2, 0 1))
        3    POLYGON ((1 2, 1 1, 2 1, 1 2))
        4    POLYGON ((1 2, 2 1, 2 3, 1 2))
        5    POLYGON ((2 3, 2 1, 3 1, 2 3))
        6    POLYGON ((3 1, 2 1, 2 0, 3 1))
        7    POLYGON ((2 0, 2 1, 1 1, 2 0))
        8    POLYGON ((2 0, 1 1, 1 0, 2 0))
        dtype: geometry

        See Also
        --------
        GeoSeries.voronoi_polygons : Voronoi diagram around vertices
        GeoSeries.constrained_delaunay_triangles : constrained Delaunay triangulation
        """
        from .geoseries import GeoSeries

        geometry_input = shapely.geometrycollections(self.geometry.values._data)

        delaunay = shapely.delaunay_triangles(
            geometry_input,
            tolerance=tolerance,
            only_edges=only_edges,
        )
        return GeoSeries(delaunay, crs=self.crs).explode(ignore_index=True)

    def voronoi_polygons(self, tolerance=0.0, extend_to=None, only_edges=False):
        """Return a ``GeoSeries`` consisting of objects representing
        the computed Voronoi diagram around the vertices of an input geometry.

        All geometries within the GeoSeries are considered together within a single
        Voronoi diagram. The resulting geometries therefore do not necessarily map 1:1
        to input geometries. Note that each vertex of a geometry is considered a site
        for the Voronoi diagram, so the diagram will be constructed around the vertices
        of each geometry.

        Notes
        -----
        The order of polygons in the output currently does not correspond to the order
        of input vertices.

        If you want to generate a Voronoi diagram for each geometry separately, use
        :func:`shapely.voronoi_polygons` instead.

        Parameters
        ----------
        tolerance : float, default 0.0
            Snap input vertices together if their distance is less than this value.
        extend_to : shapely.Geometry, default None
            If set, the Voronoi diagram will be extended to cover the
            envelope of this geometry (unless this envelope is smaller than the input
            geometry).
        only_edges : bool (optional, default False)
            If set to True, the diagram will return LineStrings instead
            of Polygons.

        Examples
        --------
        The most common use case is to generate polygons representing the Voronoi
        diagram around a set of points:

        >>> from shapely import LineString, MultiPoint, Point, Polygon
        >>> s = geopandas.GeoSeries(
        ...     [
        ...         Point(1, 1),
        ...         Point(2, 2),
        ...         Point(1, 3),
        ...         Point(0, 2),
        ...     ]
        ... )
        >>> s
        0    POINT (1 1)
        1    POINT (2 2)
        2    POINT (1 3)
        3    POINT (0 2)
        dtype: geometry

        By default, you get back a GeoSeries of polygons:

        >>> s.voronoi_polygons()
        0        POLYGON ((-2 5, 1 2, -2 -1, -2 5))
        1           POLYGON ((4 5, 1 2, -2 5, 4 5))
        2       POLYGON ((-2 -1, 1 2, 4 -1, -2 -1))
        3    POLYGON ((4 -1, 4 -1, 1 2, 4 5, 4 -1))
        dtype: geometry

        If you set only_edges to True, you get back LineStrings representing the
        edges of the Voronoi diagram:

        >>> s.voronoi_polygons(only_edges=True)
        0     LINESTRING (-2 5, 1 2)
        1    LINESTRING (1 2, -2 -1)
        2      LINESTRING (4 5, 1 2)
        3     LINESTRING (1 2, 4 -1)
        dtype: geometry

        You can also extend each diagram to a given geometry:

        >>> limit = Polygon([(-10, -10), (0, 15), (15, 15), (15, 0)])
        >>> s.voronoi_polygons(extend_to=limit)
        0              POLYGON ((-10 13, 1 2, -10 -9, -10 13))
        1    POLYGON ((15 15, 15 -10, 13 -10, 1 2, 14 15, 1...
        2    POLYGON ((-10 -10, -10 -9, 1 2, 13 -10, -10 -10))
        3       POLYGON ((-10 15, 14 15, 1 2, -10 13, -10 15))
        dtype: geometry

        The method supports any geometry type but keep in mind that the underlying
        algorithm is based on the vertices of the input geometries only and does not
        consider edge segments between vertices.

        >>> s2 = geopandas.GeoSeries(
        ...     [
        ...         Polygon([(0, 0), (1, 1), (0, 1)]),
        ...         LineString([(1, 0), (2, 1), (1, 2)]),
        ...         MultiPoint([(2, 3), (2, 0), (3, 1)]),
        ...     ]
        ... )
        >>> s2
        0      POLYGON ((0 0, 1 1, 0 1, 0 0))
        1          LINESTRING (1 0, 2 1, 1 2)
        2    MULTIPOINT ((2 3), (2 0), (3 1))
        dtype: geometry

        >>> s2.voronoi_polygons()
        0    POLYGON ((1.5 1.5, 1.5 0.5, 0.5 0.5, 0.5 1.5, ...
        1    POLYGON ((1.5 0.5, 1.5 1.5, 2 2, 2.5 2, 2.5 0....
        2    POLYGON ((-3 -3, -3 0.5, 0.5 0.5, 0.5 -3, -3 -3))
        3    POLYGON ((0.5 -3, 0.5 0.5, 1.5 0.5, 1.5 -3, 0....
        4     POLYGON ((-3 5, 0.5 1.5, 0.5 0.5, -3 0.5, -3 5))
        5    POLYGON ((-3 6, -2 6, 2 2, 1.5 1.5, 0.5 1.5, -...
        6    POLYGON ((1.5 -3, 1.5 0.5, 2.5 0.5, 6 -3, 1.5 ...
        7       POLYGON ((6 6, 6 3.75, 2.5 2, 2 2, -2 6, 6 6))
        8       POLYGON ((6 -3, 2.5 0.5, 2.5 2, 6 3.75, 6 -3))
        dtype: geometry

        See Also
        --------
        GeoSeries.delaunay_triangles : Delaunay triangulation around vertices
        """
        from .geoseries import GeoSeries

        geometry_input = shapely.geometrycollections(self.geometry.values._data)

        voronoi = shapely.voronoi_polygons(
            geometry_input,
            tolerance=tolerance,
            extend_to=extend_to,
            only_edges=only_edges,
        )

        return GeoSeries(voronoi, crs=self.crs).explode(ignore_index=True)

    @property
    def envelope(self):
        """Return a ``GeoSeries`` of geometries representing the envelope of
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
        0    POLYGON ((0 0, 1 1, 0 1, 0 0))
        1        LINESTRING (0 0, 1 1, 1 0)
        2         MULTIPOINT ((0 0), (1 1))
        3                       POINT (0 0)
        dtype: geometry

        >>> s.envelope
        0    POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))
        1    POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))
        2    POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))
        3                            POINT (0 0)
        dtype: geometry

        See Also
        --------
        GeoSeries.convex_hull : convex hull geometry
        """
        return _delegate_property("envelope", self)

    def minimum_rotated_rectangle(self):
        """Return a ``GeoSeries`` of the general minimum bounding rectangle
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
        0    POLYGON ((0 0, 1 1, 0 1, 0 0))
        1        LINESTRING (0 0, 1 1, 1 0)
        2         MULTIPOINT ((0 0), (1 1))
        3                       POINT (0 0)
        dtype: geometry

        >>> s.minimum_rotated_rectangle()
        0    POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))
        1    POLYGON ((1 1, 1 0, 0 0, 0 1, 1 1))
        2                  LINESTRING (0 0, 1 1)
        3                            POINT (0 0)
        dtype: geometry

        See Also
        --------
        GeoSeries.envelope : bounding rectangle
        """
        return _delegate_geo_method("minimum_rotated_rectangle", self)

    @property
    def exterior(self):
        """Return a ``GeoSeries`` of LinearRings representing the outer
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
        0    POLYGON ((0 0, 1 1, 0 1, 0 0))
        1    POLYGON ((1 0, 2 1, 0 0, 1 0))
        2                       POINT (0 1)
        dtype: geometry

        >>> s.exterior
        0    LINEARRING (0 0, 1 1, 0 1, 0 0)
        1    LINEARRING (1 0, 2 1, 0 0, 1 0)
        2                               None
        dtype: geometry

        See Also
        --------
        GeoSeries.boundary : complete set-theoretic boundary
        GeoSeries.interiors : list of inner rings of each polygon
        """
        # TODO: return empty geometry for non-polygons
        return _delegate_property("exterior", self)

    def extract_unique_points(self):
        """Return a ``GeoSeries`` of MultiPoints representing all
        distinct vertices of an input geometry.

        Examples
        --------
        >>> from shapely import LineString, Polygon
        >>> s = geopandas.GeoSeries(
        ...     [
        ...         LineString([(0, 0), (0, 0), (1, 1), (1, 1)]),
        ...         Polygon([(0, 0), (0, 0), (1, 1), (1, 1)])
        ...     ],
        ... )
        >>> s
        0        LINESTRING (0 0, 0 0, 1 1, 1 1)
        1    POLYGON ((0 0, 0 0, 1 1, 1 1, 0 0))
        dtype: geometry

        >>> s.extract_unique_points()
        0    MULTIPOINT ((0 0), (1 1))
        1    MULTIPOINT ((0 0), (1 1))
        dtype: geometry

        See Also
        --------
        GeoSeries.get_coordinates : extract coordinates as a :class:`~pandas.DataFrame`
        """
        return _delegate_geo_method("extract_unique_points", self)

    def offset_curve(self, distance, quad_segs=8, join_style="round", mitre_limit=5.0):
        """Return a ``LineString`` or ``MultiLineString`` geometry at a
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
        0    LINESTRING (0 0, 0 1, 1 1)
        dtype: geometry

        >>> s.offset_curve(1)
        0    LINESTRING (-1 0, -1 1, -0.981 1.195, -0.924 1...
        dtype: geometry
        """
        return _delegate_geo_method(
            "offset_curve",
            self,
            distance=distance,
            quad_segs=quad_segs,
            join_style=join_style,
            mitre_limit=mitre_limit,
        )

    @property
    def interiors(self):
        """Return a ``Series`` of List representing the
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
        0    POLYGON ((0 0, 0 5, 5 5, 5 0, 0 0), (1 1, 2 1,...
        1                       POLYGON ((1 0, 2 1, 0 0, 1 0))
        dtype: geometry

        >>> s.interiors
        0    [LINEARRING (1 1, 2 1, 1 2, 1 1), LINEARRING (...
        1                                                   []
        dtype: object

        See Also
        --------
        GeoSeries.exterior : outer boundary
        """
        return _delegate_property("interiors", self)

    def remove_repeated_points(self, tolerance=0.0):
        """Return a ``GeoSeries`` containing a copy of the input geometry
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
        ... )
        >>> s
        0                 LINESTRING (0 0, 0 0, 1 0)
        1    POLYGON ((0 0, 0 0.5, 0 1, 0.5 1, 0 0))
        dtype: geometry

        >>> s.remove_repeated_points(tolerance=0.0)
        0                      LINESTRING (0 0, 1 0)
        1    POLYGON ((0 0, 0 0.5, 0 1, 0.5 1, 0 0))
        dtype: geometry
        """
        return _delegate_geo_method("remove_repeated_points", self, tolerance=tolerance)

    def set_precision(self, grid_size, mode="valid_output"):
        """Return a ``GeoSeries`` with the precision set to a precision grid size.

        By default, geometries use double precision coordinates (``grid_size=0``).

        Coordinates will be rounded if a precision grid is less precise than the input
        geometry. Duplicated vertices will be dropped from lines and polygons for grid
        sizes greater than 0. Line and polygon geometries may collapse to empty
        geometries if all vertices are closer together than ``grid_size``. Spikes or
        sections in Polygons narrower than ``grid_size`` after rounding the vertices
        will be removed, which can lead to MultiPolygons or empty geometries. Z values,
        if present, will not be modified.

        Parameters
        ----------
        grid_size : float
            Precision grid size. If 0, will use double precision (will not modify
            geometry if precision grid size was not previously set). If this value is
            more precise than input geometry, the input geometry will not be modified.
        mode : {'valid_output', 'pointwise', 'keep_collapsed'}, default 'valid_output'
            This parameter determines the way a precision reduction is applied on the
            geometry. There are three modes:

            * ``'valid_output'`` (default): The output is always valid. Collapsed
              geometry elements (including both polygons and lines) are removed.
              Duplicate vertices are removed.
            * ``'pointwise'``: Precision reduction is performed pointwise. Output
              geometry may be invalid due to collapse or self-intersection. Duplicate
              vertices are not removed.
            * ``'keep_collapsed'``: Like the default mode, except that collapsed linear
              geometry elements are preserved. Collapsed polygonal input elements are
              removed. Duplicate vertices are removed.

        Examples
        --------
        >>> from shapely import LineString, Point
        >>> s = geopandas.GeoSeries(
        ...     [
        ...        Point(0.9, 0.9),
        ...        Point(0.9, 0.9, 0.9),
        ...        LineString([(0, 0), (0, 0.1), (0, 1), (1, 1)]),
        ...        LineString([(0, 0), (0, 0.1), (0.1, 0.1)])
        ...     ],
        ... )
        >>> s
        0                      POINT (0.9 0.9)
        1                POINT Z (0.9 0.9 0.9)
        2    LINESTRING (0 0, 0 0.1, 0 1, 1 1)
        3     LINESTRING (0 0, 0 0.1, 0.1 0.1)
        dtype: geometry

        >>> s.set_precision(1)
        0                   POINT (1 1)
        1             POINT Z (1 1 0.9)
        2    LINESTRING (0 0, 0 1, 1 1)
        3              LINESTRING EMPTY
        dtype: geometry

        >>> s.set_precision(1, mode="pointwise")
        0                        POINT (1 1)
        1                  POINT Z (1 1 0.9)
        2    LINESTRING (0 0, 0 0, 0 1, 1 1)
        3         LINESTRING (0 0, 0 0, 0 0)
        dtype: geometry

        >>> s.set_precision(1, mode="keep_collapsed")
        0                   POINT (1 1)
        1             POINT Z (1 1 0.9)
        2    LINESTRING (0 0, 0 1, 1 1)
        3         LINESTRING (0 0, 0 0)
        dtype: geometry

        Notes
        -----
        Subsequent operations will always be performed in the precision of the geometry
        with higher precision (smaller ``grid_size``). That same precision will be
        attached to the operation outputs.

        Input geometries should be geometrically valid; unexpected results may occur if
        input geometries are not. You can check the validity with
        :meth:`~GeoSeries.is_valid` and fix invalid geometries with
        :meth:`~GeoSeries.make_valid` methods.

        """
        return _delegate_geo_method(
            "set_precision", self, grid_size=grid_size, mode=mode
        )

    def representative_point(self):
        """Return a ``GeoSeries`` of (cheaply computed) points that are
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
        0    POLYGON ((0 0, 1 1, 0 1, 0 0))
        1        LINESTRING (0 0, 1 1, 1 0)
        2                       POINT (0 0)
        dtype: geometry

        >>> s.representative_point()
        0    POINT (0.25 0.5)
        1         POINT (1 1)
        2         POINT (0 0)
        dtype: geometry

        See Also
        --------
        GeoSeries.centroid : geometric centroid
        """
        return _delegate_geo_method("representative_point", self)

    def minimum_bounding_circle(self):
        """Return a ``GeoSeries`` of geometries representing the minimum bounding
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
        0    POLYGON ((0 0, 1 1, 0 1, 0 0))
        1        LINESTRING (0 0, 1 1, 1 0)
        2                       POINT (0 0)
        dtype: geometry

        >>> s.minimum_bounding_circle()
        0    POLYGON ((1.20711 0.5, 1.19352 0.36205, 1.1532...
        1    POLYGON ((1.20711 0.5, 1.19352 0.36205, 1.1532...
        2                                          POINT (0 0)
        dtype: geometry

        See Also
        --------
        GeoSeries.convex_hull : convex hull geometry
        GeoSeries.maximum_inscribed_circle : the largest circle within the geometry
        """
        return _delegate_geo_method("minimum_bounding_circle", self)

    def maximum_inscribed_circle(self, *, tolerance=None):
        """Return a ``GeoSeries`` of geometries representing the largest circle that
        is fully contained within the input geometry.

        Constructs the “maximum inscribed circle” (MIC) for a polygonal geometry, up to
        a specified tolerance. The MIC is determined by a point in the interior of the
        area which has the farthest distance from the area boundary, along with a
        boundary point at that distance. In the context of geography the center of the
        MIC is known as the “pole of inaccessibility”. A cartographic use case is to
        determine a suitable point to place a map label within a polygon. The radius
        length of the MIC is a measure of how “narrow” a polygon is. It is the distance
        at which the negative buffer becomes empty.

        The method supports polygons with holes and multipolygons but will raise an
        error for any other geometry type.

        Returns a GeoSeries with two-point linestrings rows, with the first point at the
        center of the inscribed circle and the second on the boundary of the inscribed
        circle.

        Requires Shapely >= 2.1.

        .. versionadded:: 1.1.0

        Parameters
        ----------
        tolerance : float, np.array, pd.Series
            Stop the algorithm when the search area is smaller than this tolerance.
            When not specified, uses ``max(width, height) / 1000`` per geometry as the
            default. If np.array or pd.Series are used then it must have same length as
            the GeoSeries.

        Examples
        --------
        >>> from shapely.geometry import Polygon, LineString, Point
        >>> s = geopandas.GeoSeries(
        ...     [
        ...         Polygon([(0, 0), (1, 1), (0, 1), (0, 0)]),
        ...         Polygon([(0, 0), (10, 10), (0, 10), (0, 0)]),
        ...     ]
        ... )
        >>> s
        0       POLYGON ((0 0, 1 1, 0 1, 0 0))
        1    POLYGON ((0 0, 10 10, 0 10, 0 0))
        dtype: geometry

        >>> s.maximum_inscribed_circle()
        0    LINESTRING (0.29297 0.70703, 0.5 0.5)
        1        LINESTRING (2.92969 7.07031, 5 5)
        dtype: geometry

        >>> s.maximum_inscribed_circle(tolerance=2)
        0    LINESTRING (0.25 0.5, 0.375 0.375)
        1          LINESTRING (2.5 7.5, 2.5 10)
        dtype: geometry

        See Also
        --------
        minimum_bounding_circle
        """
        return _delegate_geo_method(
            "maximum_inscribed_circle", self, tolerance=tolerance
        )

    def minimum_bounding_radius(self):
        """Return a `Series` of the radii of the minimum bounding circles
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
        0    POLYGON ((0 0, 1 1, 0 1, 0 0))
        1        LINESTRING (0 0, 1 1, 1 0)
        2                       POINT (0 0)
        dtype: geometry

        >>> s.minimum_bounding_radius()
        0    0.707107
        1    0.707107
        2    0.000000
        dtype: float64

        See Also
        --------
        GeoSeries.minumum_bounding_circle : minimum bounding circle (geometry)

        """
        return Series(self.geometry.values.minimum_bounding_radius(), index=self.index)

    def minimum_clearance(self):
        """Return a ``Series`` containing the minimum clearance distance,
        which is the smallest distance by which a vertex of the geometry
        could be moved to produce an invalid geometry.

        If no minimum clearance exists for a geometry (for example,
        a single point, or an empty geometry), infinity is returned.

        Examples
        --------
        >>> from shapely.geometry import Polygon, LineString, Point
        >>> s = geopandas.GeoSeries(
        ...     [
        ...         Polygon([(0, 0), (1, 1), (0, 1), (0, 0)]),
        ...         LineString([(0, 0), (1, 1), (3, 2)]),
        ...         Point(0, 0),
        ...     ]
        ... )
        >>> s
        0    POLYGON ((0 0, 1 1, 0 1, 0 0))
        1        LINESTRING (0 0, 1 1, 3 2)
        2                       POINT (0 0)
        dtype: geometry

        >>> s.minimum_clearance()
        0    0.707107
        1    1.414214
        2         inf
        dtype: float64

        See Also
        --------
        minimum_clearance_line
        """
        return Series(self.geometry.values.minimum_clearance(), index=self.index)

    def minimum_clearance_line(self):
        """Return a ``GeoSeries`` of linestrings whose endpoints define the
        minimum clearance.

        A geometry's “minimum clearance” is the smallest distance by which a vertex
        of the geometry could be moved to produce an invalid geometry.

        If the geometry has no minimum clearance, an empty LineString will be returned.

        Requires Shapely >= 2.1.

        .. versionadded:: 1.1.0

        Examples
        --------
        >>> from shapely.geometry import Polygon, LineString, Point
        >>> s = geopandas.GeoSeries(
        ...     [
        ...         Polygon([(0, 0), (1, 1), (0, 1), (0, 0)]),
        ...         LineString([(0, 0), (1, 1), (3, 2)]),
        ...         Point(0, 0),
        ...     ]
        ... )
        >>> s
        0    POLYGON ((0 0, 1 1, 0 1, 0 0))
        1        LINESTRING (0 0, 1 1, 3 2)
        2                       POINT (0 0)
        dtype: geometry

        >>> s.minimum_clearance_line()
        0    LINESTRING (0 1, 0.5 0.5)
        1        LINESTRING (0 0, 1 1)
        2             LINESTRING EMPTY
        dtype: geometry

        See Also
        --------
        minimum_clearance
        """
        return _delegate_geo_method("minimum_clearance_line", self)

    def normalize(self):
        """Return a ``GeoSeries`` of normalized
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
        ... )
        >>> s
        0    POLYGON ((0 0, 1 1, 0 1, 0 0))
        1        LINESTRING (0 0, 1 1, 1 0)
        2                       POINT (0 0)
        dtype: geometry

        >>> s.normalize()
        0    POLYGON ((0 0, 0 1, 1 1, 0 0))
        1        LINESTRING (0 0, 1 1, 1 0)
        2                       POINT (0 0)
        dtype: geometry
        """
        return _delegate_geo_method("normalize", self)

    def orient_polygons(self, *, exterior_cw=False):
        """Return a ``GeoSeries`` of geometries with enforced ring orientation.

        Enforce a ring orientation on all polygonal elements in the ``GeoSeries``.

        Forces (Multi)Polygons to use a counter-clockwise orientation for their exterior
        ring, and a clockwise orientation for their interior rings (or the oppposite if
        ``exterior_cw=True``).

        Also processes geometries inside a GeometryCollection in the same way. Other
        geometries are returned unchanged.

        Requires Shapely >= 2.1.

        .. versionadded:: 1.1.0

        Parameters
        ----------
        exterior_cw : bool
            If ``True``, exterior rings will be clockwise and interior rings will be
            counter-clockwise.

        Examples
        --------
        >>> from shapely.geometry import Polygon, LineString, Point
        >>> s = geopandas.GeoSeries(
        ...     [
        ...         Polygon(
        ...             [(0, 0), (0, 10), (10, 10), (10, 0), (0, 0)],
        ...             holes=[[(2, 2), (2, 4), (4, 4), (4, 2), (2, 2)]],
        ...     ),
        ...         LineString([(0, 0), (1, 1), (1, 0)]),
        ...         Point(0, 0),
        ...     ],
        ... )
        >>> s
        0    POLYGON ((0 0, 0 10, 10 10, 10 0, 0 0), (2 2, ...
        1                           LINESTRING (0 0, 1 1, 1 0)
        2                                          POINT (0 0)
        dtype: geometry

        >>> s.orient_polygons()
        0    POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0), (2 2, ...
        1                           LINESTRING (0 0, 1 1, 1 0)
        2                                          POINT (0 0)
        dtype: geometry

        >>> s.orient_polygons(exterior_cw=True)
        0    POLYGON ((0 0, 0 10, 10 10, 10 0, 0 0), (2 2, ...
        1                           LINESTRING (0 0, 1 1, 1 0)
        2                                          POINT (0 0)
        dtype: geometry
        """
        return _delegate_geo_method("orient_polygons", self, exterior_cw=exterior_cw)

    def make_valid(self, *, method="linework", keep_collapsed=True):
        """Repairs invalid geometries.

        Returns a ``GeoSeries`` with valid geometries.

        If the input geometry is already valid, then it will be preserved.
        In many cases, in order to create a valid geometry, the input
        geometry must be split into multiple parts or multiple geometries.
        If the geometry must be split into multiple parts of the same type
        to be made valid, then a multi-part geometry will be returned
        (e.g. a MultiPolygon).
        If the geometry must be split into multiple parts of different types
        to be made valid, then a GeometryCollection will be returned.

        Two ``methods`` are available:

        * the 'linework' algorithm tries to preserve every edge and vertex in
          the input. It combines all rings into a set of noded lines and then
          extracts valid polygons from that linework. An alternating even-odd
          strategy is used to assign areas as interior or exterior. A
          disadvantage is that for some relatively simple invalid geometries
          this produces rather complex results.
        * the 'structure' algorithm tries to reason from the structure of the
          input to find the 'correct' repair: exterior rings bound area,
          interior holes exclude area. It first makes all rings valid, then
          shells are merged and holes are subtracted from the shells to
          generate valid result. It assumes that holes and shells are correctly
          categorized in the input geometry.

        Parameters
        ----------
        method : {'linework', 'structure'}, default 'linework'
            Algorithm to use when repairing geometry. 'structure'
            requires GEOS >= 3.10 and shapely >= 2.1.

            .. versionadded:: 1.1.0
        keep_collapsed : bool, default True
            For the 'structure' method, True will keep components that have
            collapsed into a lower dimensionality. For example, a ring
            collapsing to a line, or a line collapsing to a point. Must be True
            for the 'linework' method.

            .. versionadded:: 1.1.0

        Examples
        --------
        >>> from shapely.geometry import MultiPolygon, Polygon, LineString, Point
        >>> s = geopandas.GeoSeries(
        ...     [
        ...         Polygon([(0, 0), (0, 2), (1, 1), (2, 2), (2, 0), (1, 1), (0, 0)]),
        ...         Polygon([(0, 2), (0, 1), (2, 0), (0, 0), (0, 2)]),
        ...         LineString([(0, 0), (1, 1), (1, 0)]),
        ...     ],
        ... )
        >>> s
        0    POLYGON ((0 0, 0 2, 1 1, 2 2, 2 0, 1 1, 0 0))
        1              POLYGON ((0 2, 0 1, 2 0, 0 0, 0 2))
        2                       LINESTRING (0 0, 1 1, 1 0)
        dtype: geometry

        >>> s.make_valid()
        0    MULTIPOLYGON (((1 1, 0 0, 0 2, 1 1)), ((2 0, 1...
        1    GEOMETRYCOLLECTION (POLYGON ((2 0, 0 0, 0 1, 2...
        2                           LINESTRING (0 0, 1 1, 1 0)
        dtype: geometry
        """
        return _delegate_geo_method(
            "make_valid", self, method=method, keep_collapsed=keep_collapsed
        )

    def reverse(self):
        """Return a ``GeoSeries`` with the order of coordinates reversed.

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
        0    POLYGON ((0 0, 1 1, 0 1, 0 0))
        1        LINESTRING (0 0, 1 1, 1 0)
        2                       POINT (0 0)
        dtype: geometry

        >>> s.reverse()
        0    POLYGON ((0 0, 0 1, 1 1, 0 0))
        1        LINESTRING (1 0, 1 1, 0 0)
        2                       POINT (0 0)
        dtype: geometry

        See Also
        --------
        GeoSeries.normalize : normalize order of coordinates
        """
        return _delegate_geo_method("reverse", self)

    def segmentize(self, max_segment_length):
        """Return a ``GeoSeries`` with vertices added to line segments based on
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
        ... )
        >>> s
        0                     LINESTRING (0 0, 0 10)
        1    POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0))
        dtype: geometry

        >>> s.segmentize(max_segment_length=5)
        0                          LINESTRING (0 0, 0 5, 0 10)
        1    POLYGON ((0 0, 5 0, 10 0, 10 5, 10 10, 5 10, 0...
        dtype: geometry
        """
        return _delegate_geo_method(
            "segmentize", self, max_segment_length=max_segment_length
        )

    def transform(self, transformation, include_z=False):
        """Return a ``GeoSeries`` with the transformation function
        applied to the geometry coordinates.

        Parameters
        ----------
        transformation : Callable
            A function that transforms a (N, 2) or (N, 3) ndarray of float64
            to another (N,2) or (N, 3) ndarray of float64
        include_z : bool, default False
            If True include the third dimension in the coordinates array that
            is passed to the ``transformation`` function. If a geometry has no third
            dimension, the z-coordinates passed to the function will be NaN.

        Returns
        -------
        GeoSeries

        Examples
        --------
        >>> from shapely import Point, Polygon
        >>> s = geopandas.GeoSeries([Point(0, 0)])
        >>> s.transform(lambda x: x + 1)
        0    POINT (1 1)
        dtype: geometry

        >>> s = geopandas.GeoSeries([Polygon([(0, 0), (1, 1), (0, 1)])])
        >>> s.transform(lambda x: x * [2, 3])
        0    POLYGON ((0 0, 2 3, 0 3, 0 0))
        dtype: geometry

        By default the third dimension is ignored and you need explicitly include it:

        >>> s = geopandas.GeoSeries([Point(0, 0, 0)])
        >>> s.transform(lambda x: x + 1, include_z=True)
        0    POINT Z (1 1 1)
        dtype: geometry
        """
        return _delegate_geo_method(
            "transform", self, transformation=transformation, include_z=include_z
        )

    def force_2d(self):
        """Force the dimensionality of a geometry to 2D.

        Removes the additional Z coordinate dimension from all geometries.

        Returns
        -------
        GeoSeries

        Examples
        --------
        >>> from shapely import Polygon, LineString, Point
        >>> s = geopandas.GeoSeries(
        ...     [
        ...         Point(0.5, 2.5, 0),
        ...         LineString([(1, 1, 1), (0, 1, 3), (1, 0, 2)]),
        ...         Polygon([(0, 0, 0), (0, 10, 0), (10, 10, 0)]),
        ...     ],
        ... )
        >>> s
        0                            POINT Z (0.5 2.5 0)
        1             LINESTRING Z (1 1 1, 0 1 3, 1 0 2)
        2    POLYGON Z ((0 0 0, 0 10 0, 10 10 0, 0 0 0))
        dtype: geometry

        >>> s.force_2d()
        0                      POINT (0.5 2.5)
        1           LINESTRING (1 1, 0 1, 1 0)
        2    POLYGON ((0 0, 0 10, 10 10, 0 0))
        dtype: geometry
        """
        return _delegate_geo_method("force_2d", self)

    def force_3d(self, z=0):
        """Force the dimensionality of a geometry to 3D.

        2D geometries will get the provided Z coordinate; 3D geometries
        are unchanged (unless their Z coordinate is ``np.nan``).

        Note that for empty geometries, 3D is only supported since GEOS 3.9 and then
        still only for simple geometries (non-collections).

        Parameters
        ----------
        z : float | array_like (default 0)
            Z coordinate to be assigned

        Returns
        -------
        GeoSeries

        Examples
        --------
        >>> from shapely import Polygon, LineString, Point
        >>> s = geopandas.GeoSeries(
        ...     [
        ...         Point(1, 2),
        ...         Point(0.5, 2.5, 2),
        ...         LineString([(1, 1), (0, 1), (1, 0)]),
        ...         Polygon([(0, 0), (0, 10), (10, 10)]),
        ...     ],
        ... )
        >>> s
        0                          POINT (1 2)
        1                  POINT Z (0.5 2.5 2)
        2           LINESTRING (1 1, 0 1, 1 0)
        3    POLYGON ((0 0, 0 10, 10 10, 0 0))
        dtype: geometry

        >>> s.force_3d()
        0                                POINT Z (1 2 0)
        1                            POINT Z (0.5 2.5 2)
        2             LINESTRING Z (1 1 0, 0 1 0, 1 0 0)
        3    POLYGON Z ((0 0 0, 0 10 0, 10 10 0, 0 0 0))
        dtype: geometry

        Z coordinate can be specified as scalar:

        >>> s.force_3d(4)
        0                                POINT Z (1 2 4)
        1                            POINT Z (0.5 2.5 2)
        2             LINESTRING Z (1 1 4, 0 1 4, 1 0 4)
        3    POLYGON Z ((0 0 4, 0 10 4, 10 10 4, 0 0 4))
        dtype: geometry

        Or as an array-like (one value per geometry):

        >>> s.force_3d(range(4))
        0                                POINT Z (1 2 0)
        1                            POINT Z (0.5 2.5 2)
        2             LINESTRING Z (1 1 2, 0 1 2, 1 0 2)
        3    POLYGON Z ((0 0 3, 0 10 3, 10 10 3, 0 0 3))
        dtype: geometry
        """
        return _delegate_geo_method("force_3d", self, z=z)

    def line_merge(self, directed=False):
        """Return (Multi)LineStrings formed by combining the lines in a
        MultiLineString.

        Lines are joined together at their endpoints in case two lines are intersecting.
        Lines are not joined when 3 or more lines are intersecting at the endpoints.
        Line elements that cannot be joined are kept as is in the resulting
        MultiLineString.

        The direction of each merged LineString will be that of the majority of the
        LineStrings from which it was derived. Except if ``directed=True`` is specified,
        then the operation will not change the order of points within lines and so only
        lines which can be joined with no change in direction are merged.

        Non-linear geometeries result in an empty GeometryCollection.

        Parameters
        ----------
        directed : bool, default False
            Only combine lines if possible without changing point order.
            Requires GEOS >= 3.11.0

        Returns
        -------
        GeoSeries

        Examples
        --------
        >>> from shapely.geometry import MultiLineString, Point
        >>> s = geopandas.GeoSeries(
        ...     [
        ...         MultiLineString([[(0, 2), (0, 10)], [(0, 10), (5, 10)]]),
        ...         MultiLineString([[(0, 2), (0, 10)], [(0, 11), (5, 10)]]),
        ...         MultiLineString(),
        ...         MultiLineString([[(0, 0), (1, 0)], [(0, 0), (3, 0)]]),
        ...         Point(0, 0),
        ...     ]
        ... )
        >>> s
        0    MULTILINESTRING ((0 2, 0 10), (0 10, 5 10))
        1    MULTILINESTRING ((0 2, 0 10), (0 11, 5 10))
        2                          MULTILINESTRING EMPTY
        3       MULTILINESTRING ((0 0, 1 0), (0 0, 3 0))
        4                                    POINT (0 0)
        dtype: geometry

        >>> s.line_merge()
        0                   LINESTRING (0 2, 0 10, 5 10)
        1    MULTILINESTRING ((0 2, 0 10), (0 11, 5 10))
        2                       GEOMETRYCOLLECTION EMPTY
        3                     LINESTRING (1 0, 0 0, 3 0)
        4                       GEOMETRYCOLLECTION EMPTY
        dtype: geometry

        With ``directed=True``, you can avoid changing the order of points within lines
        and merge only lines where no change of direction is required:

        >>> s.line_merge(directed=True)
        0                   LINESTRING (0 2, 0 10, 5 10)
        1    MULTILINESTRING ((0 2, 0 10), (0 11, 5 10))
        2                       GEOMETRYCOLLECTION EMPTY
        3       MULTILINESTRING ((0 0, 1 0), (0 0, 3 0))
        4                       GEOMETRYCOLLECTION EMPTY
        dtype: geometry
        """
        return _delegate_geo_method("line_merge", self, directed=directed)

    #
    # Reduction operations that return a Shapely geometry
    #

    @property
    def unary_union(self):
        """Return a geometry containing the union of all geometries in the
        ``GeoSeries``.

        The ``unary_union`` attribute is deprecated. Use :meth:`union_all`
        instead.

        Examples
        --------
        >>> from shapely.geometry import box
        >>> s = geopandas.GeoSeries([box(0,0,1,1), box(0,0,2,2)])
        >>> s
        0    POLYGON ((1 0, 1 1, 0 1, 0 0, 1 0))
        1    POLYGON ((2 0, 2 2, 0 2, 0 0, 2 0))
        dtype: geometry

        >>> union = s.unary_union
        >>> print(union)
        POLYGON ((0 1, 0 2, 2 2, 2 0, 1 0, 0 0, 0 1))

        See Also
        --------
        GeoSeries.union_all
        """
        warn(
            "The 'unary_union' attribute is deprecated, "
            "use the 'union_all()' method instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        return self.geometry.values.union_all()

    def union_all(self, method="unary", *, grid_size=None):
        """Return a geometry containing the union of all geometries in the
        ``GeoSeries``.

        By default, the unary union algorithm is used. If the geometries are
        non-overlapping (forming a coverage), GeoPandas can use a significantly faster
        algorithm to perform the union using the ``method="coverage"`` option.
        Alternatively, for situations which can be divided into many disjoint subsets,
        ``method="disjoint_subset"`` may be preferable.

        Parameters
        ----------
        method : str (default ``"unary"``)
            The method to use for the union. Options are:

            * ``"unary"``: use the unary union algorithm. This option is the most robust
              but can be slow for large numbers of geometries (default).
            * ``"coverage"``: use the coverage union algorithm. This option is optimized
              for non-overlapping polygons and can be significantly faster than the
              unary union algorithm. However, it can produce invalid geometries if the
              polygons overlap.
            * ``"disjoint_subset:``: use the disjoint subset union algorithm. This
              option is optimized for inputs that can be divided into subsets that do
              not intersect. If there is only one such subset, performance can be
              expected to be worse than ``"unary"``. Requires Shapely >= 2.1.

        grid_size : float, default None
            When grid size is specified, a fixed-precision space is used to perform the
            union operations. This can be useful when unioning geometries that are not
            perfectly snapped or to avoid geometries not being unioned because of
            `robustness issues <https://libgeos.org/usage/faq/#why-doesnt-a-computed-point-lie-exactly-on-a-line>`_.
            The inputs are first snapped to a grid of the given size. When a line
            segment of a geometry is within tolerance off a vertex of another geometry,
            this vertex will be inserted in the line segment. Finally, the result
            vertices are computed on the same grid. Is only supported for ``method``
            ``"unary"``. If None, the highest precision of the inputs will be used.
            Defaults to None.

            .. versionadded:: 1.1.0

        Examples
        --------
        >>> from shapely.geometry import box
        >>> s = geopandas.GeoSeries([box(0, 0, 1, 1), box(0, 0, 2, 2)])
        >>> s
        0    POLYGON ((1 0, 1 1, 0 1, 0 0, 1 0))
        1    POLYGON ((2 0, 2 2, 0 2, 0 0, 2 0))
        dtype: geometry

        >>> s.union_all()
        <POLYGON ((0 1, 0 2, 2 2, 2 0, 1 0, 0 0, 0 1))>
        """
        return self.geometry.values.union_all(method=method, grid_size=grid_size)

    def intersection_all(self):
        """Return a geometry containing the intersection of all geometries in
        the ``GeoSeries``.

        This method ignores None values when other geometries are present.
        If all elements of the GeoSeries are None, an empty GeometryCollection is
        returned.

        Examples
        --------
        >>> from shapely.geometry import box
        >>> s = geopandas.GeoSeries(
        ...     [box(0, 0, 2, 2), box(1, 1, 3, 3), box(0, 0, 1.5, 1.5)]
        ... )
        >>> s
        0              POLYGON ((2 0, 2 2, 0 2, 0 0, 2 0))
        1              POLYGON ((3 1, 3 3, 1 3, 1 1, 3 1))
        2    POLYGON ((1.5 0, 1.5 1.5, 0 1.5, 0 0, 1.5 0))
        dtype: geometry

        >>> s.intersection_all()
        <POLYGON ((1 1, 1 1.5, 1.5 1.5, 1.5 1, 1 1))>
        """
        return self.geometry.values.intersection_all()

    #
    # Binary operations that return a pandas Series
    #

    def contains(self, other, align=None):
        """Return a ``Series`` of ``dtype('bool')`` with value ``True`` for
        each aligned geometry that contains `other`.

        An object is said to contain `other` if at least one point of `other` lies in
        the interior and no points of `other` lie in the exterior of the object.
        (Therefore, any given polygon does not contain its own boundary - there is not
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
        align : bool | None (default None)
            If True, automatically aligns GeoSeries based on their indices.
            If False, the order of elements is preserved. None defaults to True.

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
        0    POLYGON ((0 0, 1 1, 0 1, 0 0))
        1             LINESTRING (0 0, 0 2)
        2             LINESTRING (0 0, 0 1)
        3                       POINT (0 1)
        dtype: geometry

        >>> s2
        1    POLYGON ((0 0, 2 2, 0 2, 0 0))
        2    POLYGON ((0 0, 1 2, 0 2, 0 0))
        3             LINESTRING (0 0, 0 2)
        4                       POINT (0 1)
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

        See Also
        --------
        GeoSeries.contains_properly
        GeoSeries.within
        """
        return _binary_op("contains", self, other, align)

    def contains_properly(self, other, align=None):
        """Return a ``Series`` of ``dtype('bool')`` with value ``True`` for
        each aligned geometry that is completely inside ``other``, with no common
        boundary points.

        Geometry A contains geometry B properly if B intersects the interior of A but
        not the boundary (or exterior). This means that a geometry A does not “contain
        properly” itself, which contrasts with the :meth:`~GeoSeries.contains` method,
        where common points on the boundary are allowed.

        The operation works on a 1-to-1 row-wise manner:

        .. image:: ../../../_static/binary_op-01.svg
           :align: center

        Parameters
        ----------
        other : GeoSeries or geometric object
            The GeoSeries (elementwise) or geometric object to test if it
            is contained.
        align : bool | None (default None)
            If True, automatically aligns GeoSeries based on their indices.
            If False, the order of elements is preserved. None defaults to True.

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
        0    POLYGON ((0 0, 1 1, 0 1, 0 0))
        1             LINESTRING (0 0, 0 2)
        2             LINESTRING (0 0, 0 1)
        3                       POINT (0 1)
        dtype: geometry

        >>> s2
        1    POLYGON ((0 0, 2 2, 0 2, 0 0))
        2    POLYGON ((0 0, 1 2, 0 2, 0 0))
        3             LINESTRING (0 0, 0 2)
        4                       POINT (0 1)
        dtype: geometry

        We can check if each geometry of GeoSeries contains a single
        geometry:

        .. image:: ../../../_static/binary_op-03.svg
           :align: center

        >>> point = Point(0, 1)
        >>> s.contains_properly(point)
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

        >>> s2.contains_properly(s, align=True)
        0    False
        1    False
        2    False
        3     True
        4    False
        dtype: bool

        >>> s2.contains_properly(s, align=False)
        1    False
        2    False
        3    False
        4     True
        dtype: bool

        Compare it to the result of :meth:`~GeoSeries.contains`:

        >>> s2.contains(s, align=False)
        1     True
        2    False
        3     True
        4     True
        dtype: bool

        Notes
        -----
        This method works in a row-wise manner. It does not check if an element
        of one GeoSeries ``contains_properly`` *any* element of the other one.

        See Also
        --------
        GeoSeries.contains
        """
        return _binary_op("contains_properly", self, other, align)

    def dwithin(self, other, distance, align=None):
        """Return a ``Series`` of ``dtype('bool')`` with value ``True`` for
        each aligned geometry that is within a set distance from ``other``.

        The operation works on a 1-to-1 row-wise manner:

        .. image:: ../../../_static/binary_op-01.svg
           :align: center

        Parameters
        ----------
        other : GeoSeries or geometric object
            The GeoSeries (elementwise) or geometric object to test for
            equality.
        distance : float, np.array, pd.Series
            Distance(s) to test if each geometry is within. A scalar distance will be
            applied to all geometries. An array or Series will be applied elementwise.
            If np.array or pd.Series are used then it must have same length as the
            GeoSeries.
        align : bool | None (default None)
            If True, automatically aligns GeoSeries based on their indices.
            If False, the order of elements is preserved. None defaults to True.

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
        ...         Polygon([(1, 0), (4, 2), (2, 2)]),
        ...         Polygon([(2, 0), (3, 2), (2, 2)]),
        ...         LineString([(2, 0), (2, 2)]),
        ...         Point(1, 1),
        ...     ],
        ...     index=range(1, 5),
        ... )

        >>> s
        0    POLYGON ((0 0, 1 1, 0 1, 0 0))
        1             LINESTRING (0 0, 0 2)
        2             LINESTRING (0 0, 0 1)
        3                       POINT (0 1)
        dtype: geometry

        >>> s2
        1    POLYGON ((1 0, 4 2, 2 2, 1 0))
        2    POLYGON ((2 0, 3 2, 2 2, 2 0))
        3             LINESTRING (2 0, 2 2)
        4                       POINT (1 1)
        dtype: geometry

        We can check if each geometry of GeoSeries contains a single
        geometry:

        .. image:: ../../../_static/binary_op-03.svg
           :align: center

        >>> point = Point(0, 1)
        >>> s2.dwithin(point, 1.8)
        1     True
        2    False
        3    False
        4     True
        dtype: bool

        We can also check two GeoSeries against each other, row by row.
        The GeoSeries above have different indices. We can either align both GeoSeries
        based on index values and compare elements with the same index using
        ``align=True`` or ignore index and compare elements based on their matching
        order using ``align=False``:

        .. image:: ../../../_static/binary_op-02.svg

        >>> s.dwithin(s2, distance=1, align=True)
        0    False
        1     True
        2    False
        3    False
        4    False
        dtype: bool

        >>> s.dwithin(s2, distance=1, align=False)
        0     True
        1    False
        2    False
        3     True
        dtype: bool

        Notes
        -----
        This method works in a row-wise manner. It does not check if an element
        of one GeoSeries is within the set distance of *any* element of the other one.

        See Also
        --------
        GeoSeries.within
        """
        return _binary_op("dwithin", self, other, distance=distance, align=align)

    def geom_equals(self, other, align=None):
        """Return a ``Series`` of ``dtype('bool')`` with value ``True`` for
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
        align : bool | None (default None)
            If True, automatically aligns GeoSeries based on their indices.
            If False, the order of elements is preserved. None defaults to True.

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
        0    POLYGON ((0 0, 2 2, 0 2, 0 0))
        1    POLYGON ((0 0, 1 2, 0 2, 0 0))
        2             LINESTRING (0 0, 0 2)
        3                       POINT (0 1)
        dtype: geometry

        >>> s2
        1    POLYGON ((0 0, 2 2, 0 2, 0 0))
        2    POLYGON ((0 0, 1 2, 0 2, 0 0))
        3                       POINT (0 1)
        4             LINESTRING (0 0, 0 2)
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

        See Also
        --------
        GeoSeries.geom_equals_exact
        GeoSeries.geom_equals_identical

        """
        return _binary_op("geom_equals", self, other, align)

    def geom_equals_exact(self, other, tolerance, align=None):
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
        align : bool | None (default None)
            If True, automatically aligns GeoSeries based on their indices.
            If False, the order of elements is preserved. None defaults to True.

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
        0    POINT (0 1.1)
        1      POINT (0 1)
        2    POINT (0 1.2)
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

        See Also
        --------
        GeoSeries.geom_equals
        GeoSeries.geom_equals_identical
        """
        return _binary_op(
            "geom_equals_exact", self, other, tolerance=tolerance, align=align
        )

    def geom_equals_identical(self, other, align=None):
        """Return True for all geometries that are identical aligned *other*, else
        False.

        This function verifies whether geometries are pointwise equivalent by checking
        that the structure, ordering, and values of all vertices are identical in all
        dimensions.

        Similarly to :meth:`geom_equals_exact`, this function uses exact coordinate
        equality and requires coordinates to be in the same order for all components
        (vertices, rings, or parts) of a geometry. However, in contrast to
        :meth:`geom_equals_exact`, this function does not allow specifying specify
        a tolerance, and additionally requires all dimensions to be the same
        (:meth:`geom_equals_exact` ignores the Z and M dimensions), where NaN values
        are considered to be equal to other NaN values.

        This function is the vectorized equivalent of scalar equality of geometry
        objects (``a == b``, i.e. ``__eq__``).

        The operation works on a 1-to-1 row-wise manner:

        .. image:: ../../../_static/binary_op-01.svg
           :align: center

        Requires Shapely >= 2.1.

        .. versionadded:: 1.1.0

        Parameters
        ----------
        other : GeoSeries or geometric object
            The GeoSeries (elementwise) or geometric object to compare to.
        align : bool | None (default None)
            If True, automatically aligns GeoSeries based on their indices.
            If False, the order of elements is preserved. None defaults to True.

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
        0    POINT (0 1.1)
        1      POINT (0 1)
        2    POINT (0 1.2)
        dtype: geometry


        >>> s.geom_equals_identical(Point(0, 1))
        0    False
        1     True
        2    False
        dtype: bool

        Notes
        -----
        This method works in a row-wise manner. It does not check if an element
        of one GeoSeries is equal to *any* element of the other one.

        See Also
        --------
        GeoSeries.geom_equals
        GeoSeries.geom_equals_exact
        """
        return _binary_op("geom_equals_identical", self, other, align=align)

    def crosses(self, other, align=None):
        """Return a ``Series`` of ``dtype('bool')`` with value ``True`` for
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
        align : bool | None (default None)
            If True, automatically aligns GeoSeries based on their indices.
            If False, the order of elements is preserved. None defaults to True.

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
        0    POLYGON ((0 0, 2 2, 0 2, 0 0))
        1             LINESTRING (0 0, 2 2)
        2             LINESTRING (2 0, 0 2)
        3                       POINT (0 1)
        dtype: geometry
        >>> s2
        1    LINESTRING (1 0, 1 3)
        2    LINESTRING (2 0, 0 2)
        3              POINT (1 1)
        4              POINT (0 1)
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

        See Also
        --------
        GeoSeries.disjoint
        GeoSeries.intersects

        """
        return _binary_op("crosses", self, other, align)

    def disjoint(self, other, align=None):
        """Return a ``Series`` of ``dtype('bool')`` with value ``True`` for
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
        align : bool | None (default None)
            If True, automatically aligns GeoSeries based on their indices.
            If False, the order of elements is preserved. None defaults to True.

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
        0    POLYGON ((0 0, 2 2, 0 2, 0 0))
        1             LINESTRING (0 0, 2 2)
        2             LINESTRING (2 0, 0 2)
        3                       POINT (0 1)
        dtype: geometry

        >>> s2
        0    POLYGON ((-1 0, -1 2, 0 -2, -1 0))
        1                 LINESTRING (0 0, 0 1)
        2                           POINT (1 1)
        3                           POINT (0 0)
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

        See Also
        --------
        GeoSeries.intersects
        GeoSeries.touches

        """
        return _binary_op("disjoint", self, other, align)

    def intersects(self, other, align=None):
        """Return a ``Series`` of ``dtype('bool')`` with value ``True`` for
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
        align : bool | None (default None)
            If True, automatically aligns GeoSeries based on their indices.
            If False, the order of elements is preserved. None defaults to True.

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
        0    POLYGON ((0 0, 2 2, 0 2, 0 0))
        1             LINESTRING (0 0, 2 2)
        2             LINESTRING (2 0, 0 2)
        3                       POINT (0 1)
        dtype: geometry

        >>> s2
        1    LINESTRING (1 0, 1 3)
        2    LINESTRING (2 0, 0 2)
        3              POINT (1 1)
        4              POINT (0 1)
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

        See Also
        --------
        GeoSeries.disjoint
        GeoSeries.crosses
        GeoSeries.touches
        GeoSeries.intersection
        """
        return _binary_op("intersects", self, other, align)

    def overlaps(self, other, align=None):
        """Return True for all aligned geometries that overlap *other*, else False.

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
        align : bool | None (default None)
            If True, automatically aligns GeoSeries based on their indices.
            If False, the order of elements is preserved. None defaults to True.

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
        0    POLYGON ((0 0, 2 2, 0 2, 0 0))
        1    POLYGON ((0 0, 2 2, 0 2, 0 0))
        2             LINESTRING (0 0, 2 2)
        3         MULTIPOINT ((0 0), (0 1))
        dtype: geometry

        >>> s2
        1    POLYGON ((0 0, 2 0, 0 2, 0 0))
        2             LINESTRING (0 1, 1 1)
        3             LINESTRING (1 1, 3 3)
        4                       POINT (0 1)
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

        See Also
        --------
        GeoSeries.crosses
        GeoSeries.intersects

        """
        return _binary_op("overlaps", self, other, align)

    def touches(self, other, align=None):
        """Return a ``Series`` of ``dtype('bool')`` with value ``True`` for
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
        align : bool | None (default None)
            If True, automatically aligns GeoSeries based on their indices.
            If False, the order of elements is preserved. None defaults to True.

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
        0    POLYGON ((0 0, 2 2, 0 2, 0 0))
        1    POLYGON ((0 0, 2 2, 0 2, 0 0))
        2             LINESTRING (0 0, 2 2)
        3         MULTIPOINT ((0 0), (0 1))
        dtype: geometry

        >>> s2
        1    POLYGON ((0 0, -2 0, 0 -2, 0 0))
        2               LINESTRING (0 1, 1 1)
        3               LINESTRING (1 1, 3 0)
        4                         POINT (0 1)
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

        See Also
        --------
        GeoSeries.overlaps
        GeoSeries.intersects

        """
        return _binary_op("touches", self, other, align)

    def within(self, other, align=None):
        """Return a ``Series`` of ``dtype('bool')`` with value ``True`` for
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
        align : bool | None (default None)
            If True, automatically aligns GeoSeries based on their indices.
            If False, the order of elements is preserved. None defaults to True.

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
        0    POLYGON ((0 0, 2 2, 0 2, 0 0))
        1    POLYGON ((0 0, 1 2, 0 2, 0 0))
        2             LINESTRING (0 0, 0 2)
        3                       POINT (0 1)
        dtype: geometry

        >>> s2
        1    POLYGON ((0 0, 1 1, 0 1, 0 0))
        2             LINESTRING (0 0, 0 2)
        3             LINESTRING (0 0, 0 1)
        4                       POINT (0 1)
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

        See Also
        --------
        GeoSeries.contains
        """
        return _binary_op("within", self, other, align)

    def covers(self, other, align=None):
        """Return a ``Series`` of ``dtype('bool')`` with value ``True`` for
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
        align : bool | None (default None)
            If True, automatically aligns GeoSeries based on their indices.
            If False, the order of elements is preserved. None defaults to True.

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
        0    POLYGON ((0 0, 2 0, 2 2, 0 2, 0 0))
        1         POLYGON ((0 0, 2 2, 0 2, 0 0))
        2                  LINESTRING (0 0, 2 2)
        3                            POINT (0 0)
        dtype: geometry

        >>> s2
        1    POLYGON ((0.5 0.5, 1.5 0.5, 1.5 1.5, 0.5 1.5, ...
        2                  POLYGON ((0 0, 2 0, 2 2, 0 2, 0 0))
        3                            LINESTRING (1 1, 1.5 1.5)
        4                                          POINT (0 0)
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

        See Also
        --------
        GeoSeries.covered_by
        GeoSeries.overlaps
        """
        return _binary_op("covers", self, other, align)

    def covered_by(self, other, align=None):
        """Return a ``Series`` of ``dtype('bool')`` with value ``True`` for
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
        align : bool | None (default None)
            If True, automatically aligns GeoSeries based on their indices.
            If False, the order of elements is preserved. None defaults to True.

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
        0    POLYGON ((0.5 0.5, 1.5 0.5, 1.5 1.5, 0.5 1.5, ...
        1                  POLYGON ((0 0, 2 0, 2 2, 0 2, 0 0))
        2                            LINESTRING (1 1, 1.5 1.5)
        3                                          POINT (0 0)
        dtype: geometry
        >>>

        >>> s2
        1    POLYGON ((0 0, 2 0, 2 2, 0 2, 0 0))
        2         POLYGON ((0 0, 2 2, 0 2, 0 0))
        3                  LINESTRING (0 0, 2 2)
        4                            POINT (0 0)
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

        See Also
        --------
        GeoSeries.covers
        GeoSeries.overlaps
        """
        return _binary_op("covered_by", self, other, align)

    def distance(self, other, align=None):
        """Return a ``Series`` containing the distance to aligned `other`.

        The operation works on a 1-to-1 row-wise manner:

        .. image:: ../../../_static/binary_op-01.svg
           :align: center

        Parameters
        ----------
        other : Geoseries or geometric object
            The Geoseries (elementwise) or geometric object to find the
            distance to.
        align : bool | None (default None)
            If True, automatically aligns GeoSeries based on their indices.
            If False, the order of elements is preserved. None defaults to True.


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
        0      POLYGON ((0 0, 1 0, 1 1, 0 0))
        1    POLYGON ((0 0, -1 0, -1 1, 0 0))
        2               LINESTRING (1 1, 0 0)
        3                         POINT (0 0)
        dtype: geometry

        >>> s2
        1    POLYGON ((0.5 0.5, 1.5 0.5, 1.5 1.5, 0.5 1.5, ...
        2                                          POINT (3 1)
        3                                LINESTRING (1 0, 2 0)
        4                                          POINT (0 1)
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

    def hausdorff_distance(self, other, align=None, densify=None):
        """Return a ``Series`` containing the Hausdorff distance to aligned `other`.

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
        align : bool | None (default None)
            If True, automatically aligns GeoSeries based on their indices.
            If False, the order of elements is preserved. None defaults to True.
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
        0      POLYGON ((0 0, 1 0, 1 1, 0 0))
        1    POLYGON ((0 0, -1 0, -1 1, 0 0))
        2               LINESTRING (1 1, 0 0)
        3                         POINT (0 0)
        dtype: geometry

        >>> s2
        1    POLYGON ((0.5 0.5, 1.5 0.5, 1.5 1.5, 0.5 1.5, ...
        2                                          POINT (3 1)
        3                                LINESTRING (1 0, 2 0)
        4                                          POINT (0 1)
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

    def frechet_distance(self, other, align=None, densify=None):
        """Return a ``Series`` containing the Frechet distance to aligned `other`.

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
        align : bool | None (default None)
            If True, automatically aligns GeoSeries based on their indices.
            If False, the order of elements is preserved. None defaults to True.
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
        0      POLYGON ((0 0, 1 0, 1 1, 0 0))
        1    POLYGON ((0 0, -1 0, -1 1, 0 0))
        2               LINESTRING (1 1, 0 0)
        3                         POINT (0 0)
        dtype: geometry

        >>> s2
        1    POLYGON ((0.5 0.5, 1.5 0.5, 1.5 1.5, 0.5 1.5, ...
        2                                          POINT (3 1)
        3                                LINESTRING (1 0, 2 0)
        4                                          POINT (0 1)
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

    def difference(self, other, align=None):
        """Return a ``GeoSeries`` of the points in each aligned geometry that
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
        align : bool | None (default None)
            If True, automatically aligns GeoSeries based on their indices.
            If False, the order of elements is preserved. None defaults to True.

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
        0    POLYGON ((0 0, 2 2, 0 2, 0 0))
        1    POLYGON ((0 0, 2 2, 0 2, 0 0))
        2             LINESTRING (0 0, 2 2)
        3             LINESTRING (2 0, 0 2)
        4                       POINT (0 1)
        dtype: geometry

        >>> s2
        1    POLYGON ((0 0, 1 1, 0 1, 0 0))
        2             LINESTRING (1 0, 1 3)
        3             LINESTRING (2 0, 0 2)
        4                       POINT (1 1)
        5                       POINT (0 1)
        dtype: geometry

        We can do difference of each geometry and a single
        shapely geometry:

        .. image:: ../../../_static/binary_op-03.svg
           :align: center

        >>> s.difference(Polygon([(0, 0), (1, 1), (0, 1)]))
        0       POLYGON ((0 2, 2 2, 1 1, 0 1, 0 2))
        1         POLYGON ((0 2, 2 2, 1 1, 0 1, 0 2))
        2                       LINESTRING (1 1, 2 2)
        3    MULTILINESTRING ((2 0, 1 1), (1 1, 0 2))
        4                                 POINT EMPTY
        dtype: geometry

        We can also check two GeoSeries against each other, row by row.
        The GeoSeries above have different indices. We can either align both GeoSeries
        based on index values and compare elements with the same index using
        ``align=True`` or ignore index and compare elements based on their matching
        order using ``align=False``:

        .. image:: ../../../_static/binary_op-02.svg

        >>> s.difference(s2, align=True)
        0                                        None
        1         POLYGON ((0 2, 2 2, 1 1, 0 1, 0 2))
        2    MULTILINESTRING ((0 0, 1 1), (1 1, 2 2))
        3                            LINESTRING EMPTY
        4                                 POINT (0 1)
        5                                        None
        dtype: geometry

        >>> s.difference(s2, align=False)
        0         POLYGON ((0 2, 2 2, 1 1, 0 1, 0 2))
        1    POLYGON ((0 0, 0 2, 1 2, 2 2, 1 1, 0 0))
        2    MULTILINESTRING ((0 0, 1 1), (1 1, 2 2))
        3                       LINESTRING (2 0, 0 2)
        4                                 POINT EMPTY
        dtype: geometry

        See Also
        --------
        GeoSeries.symmetric_difference
        GeoSeries.union
        GeoSeries.intersection
        """
        return _binary_geo("difference", self, other, align)

    def symmetric_difference(self, other, align=None):
        """Return a ``GeoSeries`` of the symmetric difference of points in
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
        align : bool | None (default None)
            If True, automatically aligns GeoSeries based on their indices.
            If False, the order of elements is preserved. None defaults to True.

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
        0    POLYGON ((0 0, 2 2, 0 2, 0 0))
        1    POLYGON ((0 0, 2 2, 0 2, 0 0))
        2             LINESTRING (0 0, 2 2)
        3             LINESTRING (2 0, 0 2)
        4                       POINT (0 1)
        dtype: geometry

        >>> s2
        1    POLYGON ((0 0, 1 1, 0 1, 0 0))
        2             LINESTRING (1 0, 1 3)
        3             LINESTRING (2 0, 0 2)
        4                       POINT (1 1)
        5                       POINT (0 1)
        dtype: geometry

        We can do symmetric difference of each geometry and a single
        shapely geometry:

        .. image:: ../../../_static/binary_op-03.svg
           :align: center

        >>> s.symmetric_difference(Polygon([(0, 0), (1, 1), (0, 1)]))
        0                  POLYGON ((0 2, 2 2, 1 1, 0 1, 0 2))
        1                  POLYGON ((0 2, 2 2, 1 1, 0 1, 0 2))
        2    GEOMETRYCOLLECTION (POLYGON ((0 0, 0 1, 1 1, 0...
        3    GEOMETRYCOLLECTION (POLYGON ((0 0, 0 1, 1 1, 0...
        4                       POLYGON ((0 1, 1 1, 0 0, 0 1))
        dtype: geometry

        We can also check two GeoSeries against each other, row by row.
        The GeoSeries above have different indices. We can either align both GeoSeries
        based on index values and compare elements with the same index using
        ``align=True`` or ignore index and compare elements based on their matching
        order using ``align=False``:

        .. image:: ../../../_static/binary_op-02.svg

        >>> s.symmetric_difference(s2, align=True)
        0                                                 None
        1                  POLYGON ((0 2, 2 2, 1 1, 0 1, 0 2))
        2    MULTILINESTRING ((0 0, 1 1), (1 1, 2 2), (1 0,...
        3                                     LINESTRING EMPTY
        4                            MULTIPOINT ((0 1), (1 1))
        5                                                 None
        dtype: geometry

        >>> s.symmetric_difference(s2, align=False)
        0                  POLYGON ((0 2, 2 2, 1 1, 0 1, 0 2))
        1    GEOMETRYCOLLECTION (POLYGON ((0 0, 0 2, 1 2, 2...
        2    MULTILINESTRING ((0 0, 1 1), (1 1, 2 2), (2 0,...
        3                                LINESTRING (2 0, 0 2)
        4                                          POINT EMPTY
        dtype: geometry

        See Also
        --------
        GeoSeries.difference
        GeoSeries.union
        GeoSeries.intersection
        """
        return _binary_geo("symmetric_difference", self, other, align)

    def union(self, other, align=None):
        """Return a ``GeoSeries`` of the union of points in each aligned geometry with
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
        align : bool | None (default None)
            If True, automatically aligns GeoSeries based on their indices.
            If False, the order of elements is preserved. None defaults to True.

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
        0    POLYGON ((0 0, 2 2, 0 2, 0 0))
        1    POLYGON ((0 0, 2 2, 0 2, 0 0))
        2             LINESTRING (0 0, 2 2)
        3             LINESTRING (2 0, 0 2)
        4                       POINT (0 1)
        dtype: geometry
        >>>

        >>> s2
        1    POLYGON ((0 0, 1 1, 0 1, 0 0))
        2             LINESTRING (1 0, 1 3)
        3             LINESTRING (2 0, 0 2)
        4                       POINT (1 1)
        5                       POINT (0 1)
        dtype: geometry

        We can do union of each geometry and a single
        shapely geometry:

        .. image:: ../../../_static/binary_op-03.svg
           :align: center

        >>> s.union(Polygon([(0, 0), (1, 1), (0, 1)]))
        0             POLYGON ((0 0, 0 1, 0 2, 2 2, 1 1, 0 0))
        1             POLYGON ((0 0, 0 1, 0 2, 2 2, 1 1, 0 0))
        2    GEOMETRYCOLLECTION (POLYGON ((0 0, 0 1, 1 1, 0...
        3    GEOMETRYCOLLECTION (POLYGON ((0 0, 0 1, 1 1, 0...
        4                       POLYGON ((0 1, 1 1, 0 0, 0 1))
        dtype: geometry

        We can also check two GeoSeries against each other, row by row.
        The GeoSeries above have different indices. We can either align both GeoSeries
        based on index values and compare elements with the same index using
        ``align=True`` or ignore index and compare elements based on their matching
        order using ``align=False``:

        .. image:: ../../../_static/binary_op-02.svg

        >>> s.union(s2, align=True)
        0                                                 None
        1             POLYGON ((0 0, 0 1, 0 2, 2 2, 1 1, 0 0))
        2    MULTILINESTRING ((0 0, 1 1), (1 1, 2 2), (1 0,...
        3                                LINESTRING (2 0, 0 2)
        4                            MULTIPOINT ((0 1), (1 1))
        5                                                 None
        dtype: geometry

        >>> s.union(s2, align=False)
        0             POLYGON ((0 0, 0 1, 0 2, 2 2, 1 1, 0 0))
        1    GEOMETRYCOLLECTION (POLYGON ((0 0, 0 2, 1 2, 2...
        2    MULTILINESTRING ((0 0, 1 1), (1 1, 2 2), (2 0,...
        3                                LINESTRING (2 0, 0 2)
        4                                          POINT (0 1)
        dtype: geometry


        See Also
        --------
        GeoSeries.symmetric_difference
        GeoSeries.difference
        GeoSeries.intersection
        """
        return _binary_geo("union", self, other, align)

    def intersection(self, other, align=None):
        """Return a ``GeoSeries`` of the intersection of points in each
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
        align : bool | None (default None)
            If True, automatically aligns GeoSeries based on their indices.
            If False, the order of elements is preserved. None defaults to True.

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
        0    POLYGON ((0 0, 2 2, 0 2, 0 0))
        1    POLYGON ((0 0, 2 2, 0 2, 0 0))
        2             LINESTRING (0 0, 2 2)
        3             LINESTRING (2 0, 0 2)
        4                       POINT (0 1)
        dtype: geometry

        >>> s2
        1    POLYGON ((0 0, 1 1, 0 1, 0 0))
        2             LINESTRING (1 0, 1 3)
        3             LINESTRING (2 0, 0 2)
        4                       POINT (1 1)
        5                       POINT (0 1)
        dtype: geometry

        We can also do intersection of each geometry and a single
        shapely geometry:

        .. image:: ../../../_static/binary_op-03.svg
           :align: center

        >>> s.intersection(Polygon([(0, 0), (1, 1), (0, 1)]))
        0    POLYGON ((0 0, 0 1, 1 1, 0 0))
        1    POLYGON ((0 0, 0 1, 1 1, 0 0))
        2             LINESTRING (0 0, 1 1)
        3                       POINT (1 1)
        4                       POINT (0 1)
        dtype: geometry

        We can also check two GeoSeries against each other, row by row.
        The GeoSeries above have different indices. We can either align both GeoSeries
        based on index values and compare elements with the same index using
        ``align=True`` or ignore index and compare elements based on their matching
        order using ``align=False``:

        .. image:: ../../../_static/binary_op-02.svg

        >>> s.intersection(s2, align=True)
        0                              None
        1    POLYGON ((0 0, 0 1, 1 1, 0 0))
        2                       POINT (1 1)
        3             LINESTRING (2 0, 0 2)
        4                       POINT EMPTY
        5                              None
        dtype: geometry

        >>> s.intersection(s2, align=False)
        0    POLYGON ((0 0, 0 1, 1 1, 0 0))
        1             LINESTRING (1 1, 1 2)
        2                       POINT (1 1)
        3                       POINT (1 1)
        4                       POINT (0 1)
        dtype: geometry


        See Also
        --------
        GeoSeries.difference
        GeoSeries.symmetric_difference
        GeoSeries.union
        """
        return _binary_geo("intersection", self, other, align)

    def clip_by_rect(self, xmin, ymin, xmax, ymax):
        """Return a ``GeoSeries`` of the portions of geometry within the given
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
        ... )
        >>> bounds = (0, 0, 1, 1)
        >>> s
        0    POLYGON ((0 0, 2 2, 0 2, 0 0))
        1    POLYGON ((0 0, 2 2, 0 2, 0 0))
        2             LINESTRING (0 0, 2 2)
        3             LINESTRING (2 0, 0 2)
        4                       POINT (0 1)
        dtype: geometry

        >>> s.clip_by_rect(*bounds)
        0    POLYGON ((0 0, 0 1, 1 1, 0 0))
        1    POLYGON ((0 0, 0 1, 1 1, 0 0))
        2             LINESTRING (0 0, 1 1)
        3          GEOMETRYCOLLECTION EMPTY
        4          GEOMETRYCOLLECTION EMPTY
        dtype: geometry

        See Also
        --------
        GeoSeries.intersection
        """
        from .geoseries import GeoSeries

        geometry_array = GeometryArray(self.geometry.values)
        clipped_geometry = geometry_array.clip_by_rect(xmin, ymin, xmax, ymax)
        return GeoSeries(clipped_geometry, index=self.index, crs=self.crs)

    def shortest_line(self, other, align=None):
        """Return the shortest two-point line between two geometries.

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
        align : bool | None (default None)
            If True, automatically aligns GeoSeries based on their indices.
            If False, the order of elements is preserved. None defaults to True.

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
        >>> s
        0    POLYGON ((0 0, 2 2, 0 2, 0 0))
        1    POLYGON ((0 0, 2 2, 0 2, 0 0))
        2             LINESTRING (0 0, 2 2)
        3             LINESTRING (2 0, 0 2)
        4                       POINT (0 1)
        dtype: geometry

        We can also do intersection of each geometry and a single
        shapely geometry:

        .. image:: ../../../_static/binary_op-03.svg
           :align: center

        >>> p = Point(3, 3)
        >>> s.shortest_line(p)
        0    LINESTRING (2 2, 3 3)
        1    LINESTRING (2 2, 3 3)
        2    LINESTRING (2 2, 3 3)
        3    LINESTRING (1 1, 3 3)
        4    LINESTRING (0 1, 3 3)
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
        ... )

        >>> s.shortest_line(s2, align=True)
        0                             None
        1    LINESTRING (0.5 0.5, 0.5 0.5)
        2            LINESTRING (2 2, 3 1)
        3            LINESTRING (2 0, 2 0)
        4          LINESTRING (0 1, 10 15)
        5                             None
        dtype: geometry
        >>>

        >>> s.shortest_line(s2, align=False)
        0    LINESTRING (0.5 0.5, 0.5 0.5)
        1            LINESTRING (2 2, 3 1)
        2        LINESTRING (0.5 0.5, 1 0)
        3          LINESTRING (0 2, 10 15)
        4            LINESTRING (0 1, 0 1)
        dtype: geometry
        """
        return _binary_geo("shortest_line", self, other, align)

    def snap(self, other, tolerance, align=None):
        """Snap the vertices and segments of the geometry to vertices of the reference.

        Vertices and segments of the input geometry are snapped to vertices of the
        reference geometry, returning a new geometry; the input geometries are not
        modified. The result geometry is the input geometry with the vertices and
        segments snapped. If no snapping occurs then the input geometry is returned
        unchanged. The tolerance is used to control where snapping is performed.

        Where possible, this operation tries to avoid creating invalid geometries;
        however, it does not guarantee that output geometries will be valid. It is
        the responsibility of the caller to check for and handle invalid geometries.

        Because too much snapping can result in invalid geometries being created,
        heuristics are used to determine the number and location of snapped
        vertices that are likely safe to snap. These heuristics may omit
        some potential snaps that are otherwise within the tolerance.

        The operation works in a 1-to-1 row-wise manner:

        .. image:: ../../../_static/binary_op-01.svg
           :align: center

        Parameters
        ----------
        other : GeoSeries or geometric object
            The Geoseries (elementwise) or geometric object to snap to.
        tolerance : float or array like
            Maximum distance between vertices that shall be snapped
        align : bool | None (default None)
            If True, automatically aligns GeoSeries based on their indices.
            If False, the order of elements is preserved. None defaults to True.

        Returns
        -------
        GeoSeries

        Examples
        --------
        >>> from shapely import Polygon, LineString, Point
        >>> s = geopandas.GeoSeries(
        ...     [
        ...         Point(0.5, 2.5),
        ...         LineString([(0.1, 0.1), (0.49, 0.51), (1.01, 0.89)]),
        ...         Polygon([(0, 0), (0, 10), (10, 10), (10, 0), (0, 0)]),
        ...     ],
        ... )
        >>> s
        0                               POINT (0.5 2.5)
        1    LINESTRING (0.1 0.1, 0.49 0.51, 1.01 0.89)
        2       POLYGON ((0 0, 0 10, 10 10, 10 0, 0 0))
        dtype: geometry

        >>> s2 = geopandas.GeoSeries(
        ...     [
        ...         Point(0, 2),
        ...         LineString([(0, 0), (0.5, 0.5), (1.0, 1.0)]),
        ...         Point(8, 10),
        ...     ],
        ...     index=range(1, 4),
        ... )
        >>> s2
        1                       POINT (0 2)
        2    LINESTRING (0 0, 0.5 0.5, 1 1)
        3                      POINT (8 10)
        dtype: geometry

        We can snap each geometry to a single shapely geometry:

        .. image:: ../../../_static/binary_op-03.svg
           :align: center

        >>> s.snap(Point(0, 2), tolerance=1)
        0                                     POINT (0 2)
        1      LINESTRING (0.1 0.1, 0.49 0.51, 1.01 0.89)
        2    POLYGON ((0 0, 0 2, 0 10, 10 10, 10 0, 0 0))
        dtype: geometry

        We can also snap two GeoSeries to each other, row by row.
        The GeoSeries above have different indices. We can either align both GeoSeries
        based on index values and snap elements with the same index using
        ``align=True`` or ignore index and snap elements based on their matching
        order using ``align=False``:

        .. image:: ../../../_static/binary_op-02.svg

        >>> s.snap(s2, tolerance=1, align=True)
        0                                                 None
        1           LINESTRING (0.1 0.1, 0.49 0.51, 1.01 0.89)
        2    POLYGON ((0.5 0.5, 1 1, 0 10, 10 10, 10 0, 0.5...
        3                                                 None
        dtype: geometry

        >>> s.snap(s2, tolerance=1, align=False)
        0                                      POINT (0 2)
        1                   LINESTRING (0 0, 0.5 0.5, 1 1)
        2    POLYGON ((0 0, 0 10, 8 10, 10 10, 10 0, 0 0))
        dtype: geometry
        """
        return _binary_geo("snap", self, other, align, tolerance=tolerance)

    def shared_paths(self, other, align=None):
        """Return the shared paths between two geometries.

        Geometries within the GeoSeries should be only (Multi)LineStrings or
        LinearRings. A GeoSeries of GeometryCollections is returned with two elements
        in each GeometryCollection. The first element is a MultiLineString containing
        shared paths with the same direction for both inputs. The second element is a
        MultiLineString containing shared paths with the opposite direction for the two
        inputs.

        You can extract individual geometries of the resulting GeometryCollection using
        the :meth:`GeoSeries.get_geometry` method.

        The operation works on a 1-to-1 row-wise manner:

        .. image:: ../../../_static/binary_op-01.svg
           :align: center

        Parameters
        ----------
        other : Geoseries or geometric object
            The Geoseries (elementwise) or geometric object to find the shared paths
            with. Has to contain only (Multi)LineString or LinearRing geometry types.
        align : bool | None (default None)
            If True, automatically aligns GeoSeries based on their indices.
            If False, the order of elements is preserved. None defaults to True.

        Returns
        -------
        GeoSeries

        Examples
        --------
        >>> from shapely.geometry import LineString, MultiLineString
        >>> s = geopandas.GeoSeries(
        ...     [
        ...         LineString([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]),
        ...         LineString([(1, 0), (2, 0), (2, 1), (1, 1), (1, 0)]),
        ...         MultiLineString([[(1, 0), (2, 0)], [(2, 1), (1, 1), (1, 0)]]),
        ...     ],
        ... )
        >>> s
        0             LINESTRING (0 0, 1 0, 1 1, 0 1, 0 0)
        1             LINESTRING (1 0, 2 0, 2 1, 1 1, 1 0)
        2    MULTILINESTRING ((1 0, 2 0), (2 1, 1 1, 1 0))
        dtype: geometry

        We can find the shared paths between each geometry and a single shapely
        geometry:

        .. image:: ../../../_static/binary_op-03.svg
           :align: center

        >>> l = LineString([(1, 1), (0, 1)])
        >>> s.shared_paths(l)
        0    GEOMETRYCOLLECTION (MULTILINESTRING ((1 1, 0 1...
        1    GEOMETRYCOLLECTION (MULTILINESTRING EMPTY, MUL...
        2    GEOMETRYCOLLECTION (MULTILINESTRING EMPTY, MUL...
        dtype: geometry

        We can also check two GeoSeries against each other, row by row. The GeoSeries
        above have different indices than the one below. We can either align both
        GeoSeries based on index values and compare elements with the same index using
        ``align=True`` or ignore index and compare elements based on their matching
        order using ``align=False``:

        .. image:: ../../../_static/binary_op-02.svg

        >>> s2 = geopandas.GeoSeries(
        ...     [
        ...         LineString([(1, 1), (0, 1)]),
        ...         LineString([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]),
        ...         LineString([(1, 0), (2, 0), (2, 1), (1, 1), (1, 0)]),
        ...     ],
        ...     index=[1, 2, 3]
        ... )

        >>> s.shared_paths(s2, align=True)
        0                                                 None
        1    GEOMETRYCOLLECTION (MULTILINESTRING EMPTY, MUL...
        2    GEOMETRYCOLLECTION (MULTILINESTRING EMPTY, MUL...
        3                                                 None
        dtype: geometry
        >>>

        >>> s.shared_paths(s2, align=False)
        0    GEOMETRYCOLLECTION (MULTILINESTRING ((1 1, 0 1...
        1    GEOMETRYCOLLECTION (MULTILINESTRING EMPTY, MUL...
        2    GEOMETRYCOLLECTION (MULTILINESTRING ((1 0, 2 0...
        dtype: geometry

        See Also
        --------
        GeoSeries.get_geometry
        """
        return _binary_geo("shared_paths", self, other, align)

    #
    # Other operations
    #

    @property
    def bounds(self):
        """Return a ``DataFrame`` with columns ``minx``, ``miny``, ``maxx``,
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
        0                     POINT (2 1)   2.0   1.0   2.0   1.0
        1  POLYGON ((0 0, 1 1, 1 0, 0 0))   0.0   0.0   1.0   1.0
        2           LINESTRING (0 1, 1 2)   0.0   1.0   1.0   2.0
        """
        bounds = GeometryArray(self.geometry.values).bounds
        return DataFrame(
            bounds, columns=["minx", "miny", "maxx", "maxy"], index=self.index
        )

    @property
    def total_bounds(self):
        """Return a tuple containing ``minx``, ``miny``, ``maxx``, ``maxy``
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
        """Generate the spatial index.

        Creates R-tree spatial index based on ``shapely.STRtree``.

        Note that the spatial index may not be fully
        initialized until the first use.

        Examples
        --------
        >>> from shapely.geometry import box
        >>> s = geopandas.GeoSeries(geopandas.points_from_xy(range(5), range(5)))
        >>> s
        0    POINT (0 0)
        1    POINT (1 1)
        2    POINT (2 2)
        3    POINT (3 3)
        4    POINT (4 4)
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
        0    POLYGON ((3 1, 3 3, 1 3, 1 1, 3 1))
        1    POLYGON ((5 4, 5 5, 4 5, 4 4, 5 4))
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

    def buffer(
        self,
        distance,
        resolution=16,
        cap_style="round",
        join_style="round",
        mitre_limit=5.0,
        single_sided=False,
        **kwargs,
    ):
        """Return a ``GeoSeries`` of geometries representing all points within
        a given ``distance`` of each geometric object.

        Computes the buffer of a geometry for positive and negative buffer distance.

        The buffer of a geometry is defined as the Minkowski sum (or difference, for
        negative distance) of the geometry with a circle with radius equal to the
        absolute value of the buffer distance.

        The buffer operation always returns a polygonal result. The negative or
        zero-distance buffer of lines and points is always empty.

        Parameters
        ----------
        distance : float, np.array, pd.Series
            The radius of the buffer in the Minkowski sum (or difference). If np.array
            or pd.Series are used then it must have same length as the GeoSeries.
        resolution : int (optional, default 16)
            The resolution of the buffer around each vertex. Specifies the number of
            linear segments in a quarter circle in the approximation of circular arcs.
        cap_style : {'round', 'square', 'flat'}, default 'round'
            Specifies the shape of buffered line endings. ``'round'`` results in
            circular line endings (see ``resolution``). Both ``'square'`` and ``'flat'``
            result in rectangular line endings, ``'flat'`` will end at the original
            vertex, while ``'square'`` involves adding the buffer width.
        join_style : {'round', 'mitre', 'bevel'}, default 'round'
            Specifies the shape of buffered line midpoints. ``'round'`` results in
            rounded shapes. ``'bevel'`` results in a beveled edge that touches the
            original vertex. ``'mitre'`` results in a single vertex that is beveled
            depending on the ``mitre_limit`` parameter.
        mitre_limit : float, default 5.0
            Crops of ``'mitre'``-style joins if the point is displaced from the
            buffered vertex by more than this limit.
        single_sided : bool, default False
            Only buffer at one side of the geometry.

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
        0                         POINT (0 0)
        1    LINESTRING (1 -1, 1 0, 2 0, 2 1)
        2    POLYGON ((3 -1, 4 0, 3 1, 3 -1))
        dtype: geometry

        >>> s.buffer(0.2)
        0    POLYGON ((0.2 0, 0.19904 -0.0196, 0.19616 -0.0...
        1    POLYGON ((0.8 0, 0.80096 0.0196, 0.80384 0.039...
        2    POLYGON ((2.8 -1, 2.8 1, 2.80096 1.0196, 2.803...
        dtype: geometry

        ``Further specification as ``join_style`` and ``cap_style`` are shown in the
        following illustration:

        .. plot:: _static/code/buffer.py

        """
        return _delegate_geo_method(
            "buffer",
            self,
            distance=distance,
            resolution=resolution,
            cap_style=cap_style,
            join_style=join_style,
            mitre_limit=mitre_limit,
            single_sided=single_sided,
            **kwargs,
        )

    def simplify(self, tolerance, preserve_topology=True):
        """Return a ``GeoSeries`` containing a simplified representation of
        each geometry.

        The algorithm (Douglas-Peucker) recursively splits the original line
        into smaller parts and connects these parts' endpoints
        by a straight line. Then, it removes all points whose distance
        to the straight line is smaller than `tolerance`. It does not
        move any points and it always preserves endpoints of
        the original line or polygon.
        See https://shapely.readthedocs.io/en/latest/manual.html#object.simplify
        for details

        Simplifies individual geometries independently, without considering
        the topology of a potential polygonal coverage. If you would like to treat
        the ``GeoSeries`` as a coverage and simplify its edges, while preserving the
        coverage topology, see :meth:`simplify_coverage`.

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

        See Also
        --------
        simplify_coverage : simplify geometries using coverage simplification

        Examples
        --------
        >>> from shapely.geometry import Point, LineString
        >>> s = geopandas.GeoSeries(
        ...     [Point(0, 0).buffer(1), LineString([(0, 0), (1, 10), (0, 20)])]
        ... )
        >>> s
        0    POLYGON ((1 0, 0.99518 -0.09802, 0.98079 -0.19...
        1                         LINESTRING (0 0, 1 10, 0 20)
        dtype: geometry

        >>> s.simplify(1)
        0    POLYGON ((0 1, 0 -1, -1 0, 0 1))
        1              LINESTRING (0 0, 0 20)
        dtype: geometry
        """
        return _delegate_geo_method(
            "simplify", self, tolerance=tolerance, preserve_topology=preserve_topology
        )

    def simplify_coverage(self, tolerance, *, simplify_boundary=True):
        """Return a ``GeoSeries`` containing a simplified representation of
        polygonal coverage.

        Assumes that the ``GeoSeries`` forms a polygonal coverage. Under this
        assumption, the method simplifies the edges using the Visvalingam-Whyatt
        algorithm, while preserving a valid coverage. In the most simplified case,
        polygons are reduced to triangles.

        A ``GeoSeries`` of valid polygons is considered a coverage if the polygons are:

        * **Non-overlapping** - polygons do not overlap (their interiors do not
          intersect)
        * **Edge-Matched** - vertices along shared edges are identical

        The method allows simplification of all edges including the outer boundaries of
        the coverage or simplification of only the inner (shared) edges.

        If there are other geometry types than Polygons or MultiPolygons present, the
        method will raise an error.

        If the geometry is polygonal but does not form a valid coverage due to overlaps,
        it will be simplified but it may result in invalid coverage topology.

        Requires Shapely >= 2.1.

        .. versionadded:: 1.1.0

        Parameters
        ----------
        tolerance : float
            The degree of simplification roughly equal to the square root of the area
            of triangles that will be removed. It has the same units
            as the coordinate reference system of the GeoSeries.
            For example, using `tolerance=100` in a projected CRS with meters
            as units means a distance of 100 meters in reality.
        simplify_boundary: bool (default True)
            By default (True), simplifies both internal edges of the coverage as well
            as its boundary. If set to False, only simplifies internal edges.


        See Also
        --------
        simplify : simplification of individual geometries

        Examples
        --------
        >>> from shapely.geometry import Polygon
        >>> s = geopandas.GeoSeries(
        ...     [
        ...         Polygon([(0, 0), (1, 1.1), (2, 0), (1.5, 1), (2, 2), (0, 2)]),
        ...         Polygon([(2, 0), (4, 0), (4, 2), (2, 2), (1.5, 1)]),
        ...     ]
        ... )
        >>> s
        0    POLYGON ((0 0, 1 1.1, 2 0, 1.5 1, 2 2, 0 2, 0 0))
        1           POLYGON ((2 0, 4 0, 4 2, 2 2, 1.5 1, 2 0))
        dtype: geometry

        >>> s.simplify_coverage(1)
        0         POLYGON ((2 0, 2 2, 0 2, 2 0))
        1    POLYGON ((2 0, 4 0, 4 2, 2 2, 2 0))
        dtype: geometry

        >>> s.simplify_coverage(1, simplify_boundary=False)
        0    POLYGON ((2 0, 2 2, 0 2, 0 0, 1 1.1, 2 0))
        1           POLYGON ((2 0, 4 0, 4 2, 2 2, 2 0))
        dtype: geometry
        """
        return _delegate_geo_method(
            "simplify_coverage",
            self,
            tolerance=tolerance,
            simplify_boundary=simplify_boundary,
        )

    def relate(self, other, align=None):
        """Return the DE-9IM intersection matrices for the geometries.

        The operation works on a 1-to-1 row-wise manner:

        .. image:: ../../../_static/binary_op-01.svg
           :align: center

        Parameters
        ----------
        other : BaseGeometry or GeoSeries
            The other geometry to computed
            the DE-9IM intersection matrices from.
        align : bool | None (default None)
            If True, automatically aligns GeoSeries based on their indices.
            If False, the order of elements is preserved. None defaults to True.

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
        0    POLYGON ((0 0, 2 2, 0 2, 0 0))
        1    POLYGON ((0 0, 2 2, 0 2, 0 0))
        2             LINESTRING (0 0, 2 2)
        3             LINESTRING (2 0, 0 2)
        4                       POINT (0 1)
        dtype: geometry

        >>> s2
        1    POLYGON ((0 0, 1 1, 0 1, 0 0))
        2             LINESTRING (1 0, 1 3)
        3             LINESTRING (2 0, 0 2)
        4                       POINT (1 1)
        5                       POINT (0 1)
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

    def relate_pattern(self, other, pattern, align=None):
        """Return True if the DE-9IM string code for the relationship between
        the geometries satisfies the pattern, else False.

        This function compares the DE-9IM code string for two geometries
        against a specified pattern. If the string matches the pattern then
        ``True`` is returned, otherwise ``False``. The pattern specified can
        be an exact match (``0``, ``1`` or ``2``), a boolean match
        (uppercase ``T`` or ``F``), or a wildcard (``*``). For example,
        the pattern for the ``within`` predicate is ``'T*F**F***'``

        The operation works on a 1-to-1 row-wise manner:

        .. image:: ../../../_static/binary_op-01.svg
           :align: center

        Parameters
        ----------
        other : BaseGeometry or GeoSeries
            The other geometry to be tested agains the pattern.
        pattern : str
            The DE-9IM pattern to test against.
        align : bool | None (default None)
            If True, automatically aligns GeoSeries based on their indices.
            If False, the order of elements is preserved. None defaults to True.

        Returns
        -------
        Series

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
        0    POLYGON ((0 0, 2 2, 0 2, 0 0))
        1    POLYGON ((0 0, 2 2, 0 2, 0 0))
        2             LINESTRING (0 0, 2 2)
        3             LINESTRING (2 0, 0 2)
        4                       POINT (0 1)
        dtype: geometry

        >>> s2
        1    POLYGON ((0 0, 1 1, 0 1, 0 0))
        2             LINESTRING (1 0, 1 3)
        3             LINESTRING (2 0, 0 2)
        4                       POINT (1 1)
        5                       POINT (0 1)
        dtype: geometry

        We can check the relate pattern of each geometry and a single
        shapely geometry:

        .. image:: ../../../_static/binary_op-03.svg
           :align: center

        >>> s.relate_pattern(Polygon([(0, 0), (1, 1), (0, 1)]), "2*T***F**")
        0     True
        1     True
        2    False
        3    False
        4    False
        dtype: bool

        We can also check two GeoSeries against each other, row by row.
        The GeoSeries above have different indices. We can either align both GeoSeries
        based on index values and compare elements with the same index using
        ``align=True`` or ignore index and compare elements based on their matching
        order using ``align=False``:

        .. image:: ../../../_static/binary_op-02.svg

        >>> s.relate_pattern(s2, "TF******T", align=True)
        0    False
        1    False
        2     True
        3     True
        4    False
        5    False
        dtype: bool

        >>> s.relate_pattern(s2, "TF******T", align=False)
        0    False
        1     True
        2     True
        3     True
        4     True
        dtype: bool

        """
        return _binary_op("relate_pattern", self, other, pattern=pattern, align=align)

    def project(self, other, normalized=False, align=None):
        """Return the distance along each geometry nearest to *other*.

        The operation works on a 1-to-1 row-wise manner:

        .. image:: ../../../_static/binary_op-01.svg
           :align: center

        The project method is the inverse of interpolate.

        In shapely, this is equal to ``line_locate_point``.


        Parameters
        ----------
        other : BaseGeometry or GeoSeries
            The *other* geometry to computed projected point from.
        normalized : boolean
            If normalized is True, return the distance normalized to
            the length of the object.
        align : bool | None (default None)
            If True, automatically aligns GeoSeries based on their indices.
            If False, the order of elements is preserved. None defaults to True.

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
        0    LINESTRING (0 0, 2 0, 0 2)
        1         LINESTRING (0 0, 2 2)
        2         LINESTRING (2 0, 0 2)
        dtype: geometry

        >>> s2
        1    POINT (1 0)
        2    POINT (1 0)
        3    POINT (2 1)
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

        See Also
        --------
        GeoSeries.interpolate
        """
        return _binary_op("project", self, other, normalized=normalized, align=align)

    def interpolate(self, distance, normalized=False):
        """Return a point at the specified distance along each geometry.

        Parameters
        ----------
        distance : float or Series of floats
            Distance(s) along the geometries at which a point should be
            returned. If np.array or pd.Series are used then it must have
            same length as the GeoSeries.
        normalized : boolean
            If normalized is True, distance will be interpreted as a fraction
            of the geometric object's length.

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
        >>> s
        0    LINESTRING (0 0, 2 0, 0 2)
        1         LINESTRING (0 0, 2 2)
        2         LINESTRING (2 0, 0 2)
        dtype: geometry

        >>> s.interpolate(1)
        0                POINT (1 0)
        1    POINT (0.70711 0.70711)
        2    POINT (1.29289 0.70711)
        dtype: geometry

        >>> s.interpolate([1, 2, 3])
        0                POINT (1 0)
        1    POINT (1.41421 1.41421)
        2                POINT (0 2)
        dtype: geometry
        """
        return _delegate_geo_method(
            "interpolate", self, distance=distance, normalized=normalized
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
        0                         POINT (1 1)
        1              LINESTRING (1 -1, 1 0)
        2    POLYGON ((3 -1, 4 0, 3 1, 3 -1))
        dtype: geometry

        >>> s.affine_transform([2, 3, 2, 4, 5, 2])
        0                          POINT (10 8)
        1                 LINESTRING (4 0, 7 4)
        2    POLYGON ((8 4, 13 10, 14 12, 8 4))
        dtype: geometry

        """  # (E501 link is longer than max line length)
        return _delegate_geo_method("affine_transform", self, matrix=matrix)

    def translate(self, xoff=0.0, yoff=0.0, zoff=0.0):
        """Return a ``GeoSeries`` with translated geometries.

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
        0                         POINT (1 1)
        1              LINESTRING (1 -1, 1 0)
        2    POLYGON ((3 -1, 4 0, 3 1, 3 -1))
        dtype: geometry

        >>> s.translate(2, 3)
        0                       POINT (3 4)
        1             LINESTRING (3 2, 3 3)
        2    POLYGON ((5 2, 6 3, 5 4, 5 2))
        dtype: geometry

        """  # (E501 link is longer than max line length)
        return _delegate_geo_method("translate", self, xoff=xoff, yoff=yoff, zoff=zoff)

    def rotate(self, angle, origin="center", use_radians=False):
        """Return a ``GeoSeries`` with rotated geometries.

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
        0                         POINT (1 1)
        1              LINESTRING (1 -1, 1 0)
        2    POLYGON ((3 -1, 4 0, 3 1, 3 -1))
        dtype: geometry

        >>> s.rotate(90)
        0                                          POINT (1 1)
        1                      LINESTRING (1.5 -0.5, 0.5 -0.5)
        2    POLYGON ((4.5 -0.5, 3.5 0.5, 2.5 -0.5, 4.5 -0.5))
        dtype: geometry

        >>> s.rotate(90, origin=(0, 0))
        0                       POINT (-1 1)
        1              LINESTRING (1 1, 0 1)
        2    POLYGON ((1 3, 0 4, -1 3, 1 3))
        dtype: geometry

        """
        return _delegate_geo_method(
            "rotate", self, angle=angle, origin=origin, use_radians=use_radians
        )

    def scale(self, xfact=1.0, yfact=1.0, zfact=1.0, origin="center"):
        """Return a ``GeoSeries`` with scaled geometries.

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
        0                         POINT (1 1)
        1              LINESTRING (1 -1, 1 0)
        2    POLYGON ((3 -1, 4 0, 3 1, 3 -1))
        dtype: geometry

        >>> s.scale(2, 3)
        0                                 POINT (1 1)
        1                      LINESTRING (1 -2, 1 1)
        2    POLYGON ((2.5 -3, 4.5 0, 2.5 3, 2.5 -3))
        dtype: geometry

        >>> s.scale(2, 3, origin=(0, 0))
        0                         POINT (2 3)
        1              LINESTRING (2 -3, 2 0)
        2    POLYGON ((6 -3, 8 0, 6 3, 6 -3))
        dtype: geometry
        """
        return _delegate_geo_method(
            "scale", self, xfact=xfact, yfact=yfact, zfact=zfact, origin=origin
        )

    def skew(self, xs=0.0, ys=0.0, origin="center", use_radians=False):
        """Return a ``GeoSeries`` with skewed geometries.

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
        0                         POINT (1 1)
        1              LINESTRING (1 -1, 1 0)
        2    POLYGON ((3 -1, 4 0, 3 1, 3 -1))
        dtype: geometry

        >>> s.skew(45, 30)
        0                                          POINT (1 1)
        1                           LINESTRING (0.5 -1, 1.5 0)
        2    POLYGON ((2 -1.28868, 4 0.28868, 4 0.71132, 2 ...
        dtype: geometry

        >>> s.skew(45, 30, origin=(0, 0))
        0                                    POINT (2 1.57735)
        1         LINESTRING (1.11022e-16 -0.42265, 1 0.57735)
        2    POLYGON ((2 0.73205, 4 2.3094, 4 2.73205, 2 0....
        dtype: geometry
        """
        return _delegate_geo_method(
            "skew", self, xs=xs, ys=ys, origin=origin, use_radians=use_radians
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
        0              POINT (0 0)
        1              POINT (1 2)
        2              POINT (3 3)
        3    LINESTRING (0 0, 3 3)
        dtype: geometry

        >>> s.cx[0:1, 0:1]
        0              POINT (0 0)
        3    LINESTRING (0 0, 3 3)
        dtype: geometry

        >>> s.cx[:, 1:]
        1              POINT (1 2)
        2              POINT (3 3)
        3    LINESTRING (0 0, 3 3)
        dtype: geometry

        """
        return _CoordinateIndexer(self)

    def get_coordinates(
        self, include_z=False, ignore_index=False, index_parts=False, *, include_m=False
    ):
        """Get coordinates from a :class:`GeoSeries` as a :class:`~pandas.DataFrame` of
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
        include_m : bool, default False
            Include M coordinates. Requires shapely >= 2.1.

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
        0                         POINT (1 1)
        1              LINESTRING (1 -1, 1 0)
        2    POLYGON ((3 -1, 4 0, 3 1, 3 -1))
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
        if include_m:
            if not compat.SHAPELY_GE_21:
                raise ImportError("Shapely >= 2.1 is required for include_m=True.")

            # can be merged with the one below once min requirement is shapely 2.1
            coords, outer_idx = shapely.get_coordinates(
                self.geometry.values._data,
                include_z=include_z,
                include_m=include_m,
                return_index=True,
            )
        else:
            coords, outer_idx = shapely.get_coordinates(
                self.geometry.values._data, include_z=include_z, return_index=True
            )

        column_names = ["x", "y"]
        if include_z:
            column_names.append("z")
        if include_m:
            column_names.append("m")

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
        0    MULTIPOINT ((0.1045 -0.10294), (0.35249 -0.264...
        1    MULTIPOINT ((3.03261 -0.43069), (3.10068 0.114...
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
                lambda x: (
                    points_from_xy(
                        *sample_function(x, size=size, **kwargs).T
                    ).union_all()
                    if not (x.is_empty or x is None or "Polygon" not in x.geom_type)
                    else MultiPoint()
                ),
            )

        return GeoSeries(result, name="sampled_points", crs=self.crs, index=self.index)

    def build_area(self, node=True):
        """Create an areal geometry formed by the constituent linework.

        Builds areas from the GeoSeries that contain linework which represents the edges
        of a planar graph. Any geometry type may be provided as input; only the
        constituent lines and rings will be used to create the output polygons. All
        geometries within the GeoSeries are considered together and the resulting
        polygons therefore do not map 1:1 to input geometries.

        This function converts inner rings into holes. To turn inner rings into polygons
        as well, use polygonize.

        Unless you know that the input GeoSeries represents a planar graph with a clean
        topology (e.g. there is a node on both lines where they intersect), it is
        recommended to use ``node=True`` which performs noding prior to building areal
        geometry. Using ``node=False`` will provide performance benefits but may result
        in incorrect polygons if the input is not of the proper topology.

        If the input linework crosses, this function may produce invalid polygons. Use
        :meth:`GeoSeries.make_valid` to ensure valid geometries.

        Parameters
        ----------
        node : bool, default True
            Perform noding prior to building the areas, by default True.

        Returns
        -------
        GeoSeries
            GeoSeries with polygons

        Examples
        --------
        >>> from shapely.geometry import LineString, Polygon
        >>> s = geopandas.GeoSeries([
        ...     LineString([(18, 4), (4, 2), (2, 9)]),
        ...     LineString([(18, 4), (16, 16)]),
        ...     LineString([(16, 16), (8, 19), (8, 12), (2, 9)]),
        ...     LineString([(8, 6), (12, 13), (15, 8)]),
        ...     LineString([(8, 6), (15, 8)]),
        ...     LineString([(0, 0), (0, 3), (3, 3), (3, 0), (0, 0)]),
        ...     Polygon([(1, 1), (2, 2), (1, 2), (1, 1)]),
        ... ])
        >>> s.build_area()
        0    POLYGON ((0 3, 3 3, 3 0, 0 0, 0 3), (1 1, 2 2,...
        1    POLYGON ((4 2, 2 9, 8 12, 8 19, 16 16, 18 4, 4...
        Name: polygons, dtype: geometry

        """
        from .geoseries import GeoSeries

        if node:
            geometry_input = self.geometry.union_all()
        else:
            geometry_input = shapely.geometrycollections(self.geometry.values._data)

        polygons = shapely.build_area(geometry_input)
        return GeoSeries(polygons, crs=self.crs, name="polygons").explode(
            ignore_index=True
        )

    def polygonize(self, node=True, full=False):
        """Create polygons formed from the linework of a GeoSeries.

        Polygonizes the GeoSeries that contain linework which represents the
        edges of a planar graph. Any geometry type may be provided as input; only the
        constituent lines and rings will be used to create the output polygons.

        Lines or rings that when combined do not completely close a polygon will be
        ignored. Duplicate segments are ignored.

        Unless you know that the input GeoSeries represents a planar graph with a clean
        topology (e.g. there is a node on both lines where they intersect), it is
        recommended to use ``node=True`` which performs noding prior to polygonization.
        Using ``node=False`` will provide performance benefits but may result in
        incorrect polygons if the input is not of the proper topology.

        When ``full=True``, the return value is a 4-tuple containing output polygons,
        along with lines which could not be converted to polygons. The return value
        consists of 4 elements or varying lenghts:

        - GeoSeries of the valid polygons (same as with ``full=False``)
        - GeoSeries of cut edges: edges connected on both ends but not part of
          polygonal output
        - GeoSeries of dangles: edges connected on one end but not part of polygonal
          output
        - GeoSeries of invalid rings: polygons that are formed but are not valid
          (bowties, etc)

        Parameters
        ----------
        node : bool, default True
            Perform noding prior to polygonization, by default True.
        full : bool, default False
            Return the full output composed of a tuple of GeoSeries, by default False.

        Returns
        -------
        GeoSeries | tuple(GeoSeries, GeoSeries, GeoSeries, GeoSeries)
            GeoSeries with the polygons or a tuple of four GeoSeries as
            ``(polygons, cuts, dangles, invalid)``

        Examples
        --------
        >>> from shapely.geometry import LineString
        >>> s = geopandas.GeoSeries([
        ...     LineString([(0, 0), (1, 1)]),
        ...     LineString([(0, 0), (0, 1), (1, 1), (1, 0), (0, 0)]),
        ...     LineString([(0.5, 0.2), (0.5, 0.8)]),
        ... ])
        >>> s.polygonize()
        0        POLYGON ((0 0, 0.5 0.5, 1 1, 1 0, 0 0))
        1    POLYGON ((0.5 0.5, 0 0, 0 1, 1 1, 0.5 0.5))
        Name: polygons, dtype: geometry

        >>> polygons, cuts, dangles, invalid = s.polygonize(full=True)

        """
        from .geoseries import GeoSeries

        if node:
            geometry_input = [self.geometry.union_all()]
        else:
            geometry_input = self.geometry.values

        if full:
            polygons, cuts, dangles, invalid = shapely.polygonize_full(geometry_input)

            cuts = GeoSeries(cuts, crs=self.crs, name="cut_edges").explode(
                ignore_index=True
            )
            dangles = GeoSeries(dangles, crs=self.crs, name="dangles").explode(
                ignore_index=True
            )
            invalid = GeoSeries(invalid, crs=self.crs, name="invalid_rings").explode(
                ignore_index=True
            )
            polygons = GeoSeries(polygons, crs=self.crs, name="polygons").explode(
                ignore_index=True
            )

            return (polygons, cuts, dangles, invalid)

        polygons = shapely.polygonize(geometry_input)
        return GeoSeries(polygons, crs=self.crs, name="polygons").explode(
            ignore_index=True
        )


def _get_index_for_parts(orig_idx, outer_idx, ignore_index, index_parts):
    """Handle index when geometries get exploded to parts.

    Helper function used in get_coordinates and explode.

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


class _CoordinateIndexer:
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
