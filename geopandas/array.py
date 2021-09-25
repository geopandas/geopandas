from collections.abc import Iterable
import numbers
import operator
import warnings
import inspect

import numpy as np
import pandas as pd
from pandas.api.extensions import (
    ExtensionArray,
    ExtensionDtype,
    register_extension_dtype,
)

import shapely
import shapely.affinity
import shapely.geometry
from shapely.geometry.base import BaseGeometry
import shapely.ops
import shapely.wkt
from pyproj import CRS, Transformer

try:
    import pygeos
except ImportError:
    geos = None

from . import _compat as compat
from . import _vectorized as vectorized
from .sindex import _get_sindex_class


class GeometryDtype(ExtensionDtype):
    type = BaseGeometry
    name = "geometry"
    na_value = np.nan

    @classmethod
    def construct_from_string(cls, string):
        if not isinstance(string, str):
            raise TypeError(
                "'construct_from_string' expects a string, got {}".format(type(string))
            )
        elif string == cls.name:
            return cls()
        else:
            raise TypeError(
                "Cannot construct a '{}' from '{}'".format(cls.__name__, string)
            )

    @classmethod
    def construct_array_type(cls):
        return GeometryArray


register_extension_dtype(GeometryDtype)


def _check_crs(left, right, allow_none=False):
    """
    Check if the projection of both arrays is the same.

    If allow_none is True, empty CRS is treated as the same.
    """
    if allow_none:
        if not left.crs or not right.crs:
            return True
    if not left.crs == right.crs:
        return False
    return True


def _crs_mismatch_warn(left, right, stacklevel=3):
    """
    Raise a CRS mismatch warning with the information on the assigned CRS.
    """
    if left.crs:
        left_srs = left.crs.to_string()
        left_srs = left_srs if len(left_srs) <= 50 else " ".join([left_srs[:50], "..."])
    else:
        left_srs = None

    if right.crs:
        right_srs = right.crs.to_string()
        right_srs = (
            right_srs if len(right_srs) <= 50 else " ".join([right_srs[:50], "..."])
        )
    else:
        right_srs = None

    warnings.warn(
        "CRS mismatch between the CRS of left geometries "
        "and the CRS of right geometries.\n"
        "Use `to_crs()` to reproject one of "
        "the input geometries to match the CRS of the other.\n\n"
        "Left CRS: {0}\n"
        "Right CRS: {1}\n".format(left_srs, right_srs),
        UserWarning,
        stacklevel=stacklevel,
    )


# -----------------------------------------------------------------------------
# Constructors / converters to other formats
# -----------------------------------------------------------------------------


def _geom_to_shapely(geom):
    """
    Convert internal representation (PyGEOS or Shapely) to external Shapely object.
    """
    if not compat.USE_PYGEOS:
        return geom
    else:
        return vectorized._pygeos_to_shapely(geom)


def _shapely_to_geom(geom):
    """
    Convert external Shapely object to internal representation (PyGEOS or Shapely).
    """
    if not compat.USE_PYGEOS:
        return geom
    else:
        return vectorized._shapely_to_pygeos(geom)


def _is_scalar_geometry(geom):
    if compat.USE_PYGEOS:
        return isinstance(geom, (pygeos.Geometry, BaseGeometry))
    else:
        return isinstance(geom, BaseGeometry)


def from_shapely(data, crs=None):
    """
    Convert a list or array of shapely objects to a GeometryArray.

    Validates the elements.

    Parameters
    ----------
    data : array-like
        list or array of shapely objects
    crs : value, optional
        Coordinate Reference System of the geometry objects. Can be anything accepted by
        :meth:`pyproj.CRS.from_user_input() <pyproj.crs.CRS.from_user_input>`,
        such as an authority string (eg "EPSG:4326") or a WKT string.

    """
    return GeometryArray(vectorized.from_shapely(data), crs=crs)


def to_shapely(geoms):
    """
    Convert GeometryArray to numpy object array of shapely objects.
    """
    if not isinstance(geoms, GeometryArray):
        raise ValueError("'geoms' must be a GeometryArray")
    return vectorized.to_shapely(geoms.data)


def from_wkb(data, crs=None):
    """
    Convert a list or array of WKB objects to a GeometryArray.

    Parameters
    ----------
    data : array-like
        list or array of WKB objects
    crs : value, optional
        Coordinate Reference System of the geometry objects. Can be anything accepted by
        :meth:`pyproj.CRS.from_user_input() <pyproj.crs.CRS.from_user_input>`,
        such as an authority string (eg "EPSG:4326") or a WKT string.

    """
    return GeometryArray(vectorized.from_wkb(data), crs=crs)


def to_wkb(geoms, hex=False, **kwargs):
    """
    Convert GeometryArray to a numpy object array of WKB objects.
    """
    if not isinstance(geoms, GeometryArray):
        raise ValueError("'geoms' must be a GeometryArray")
    return vectorized.to_wkb(geoms.data, hex=hex, **kwargs)


def from_wkt(data, crs=None):
    """
    Convert a list or array of WKT objects to a GeometryArray.

    Parameters
    ----------
    data : array-like
        list or array of WKT objects
    crs : value, optional
        Coordinate Reference System of the geometry objects. Can be anything accepted by
        :meth:`pyproj.CRS.from_user_input() <pyproj.crs.CRS.from_user_input>`,
        such as an authority string (eg "EPSG:4326") or a WKT string.

    """
    return GeometryArray(vectorized.from_wkt(data), crs=crs)


def to_wkt(geoms, **kwargs):
    """
    Convert GeometryArray to a numpy object array of WKT objects.
    """
    if not isinstance(geoms, GeometryArray):
        raise ValueError("'geoms' must be a GeometryArray")
    return vectorized.to_wkt(geoms.data, **kwargs)


def points_from_xy(x, y, z=None, crs=None):
    """
    Generate GeometryArray of shapely Point geometries from x, y(, z) coordinates.

    In case of geographic coordinates, it is assumed that longitude is captured by
    ``x`` coordinates and latitude by ``y``.

    Parameters
    ----------
    x, y, z : iterable
    crs : value, optional
        Coordinate Reference System of the geometry objects. Can be anything accepted by
        :meth:`pyproj.CRS.from_user_input() <pyproj.crs.CRS.from_user_input>`,
        such as an authority string (eg "EPSG:4326") or a WKT string.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({'x': [0, 1, 2], 'y': [0, 1, 2], 'z': [0, 1, 2]})
    >>> df
       x  y  z
    0  0  0  0
    1  1  1  1
    2  2  2  2
    >>> geometry = geopandas.points_from_xy(x=[1, 0], y=[0, 1])
    >>> geometry = geopandas.points_from_xy(df['x'], df['y'], df['z'])
    >>> gdf = geopandas.GeoDataFrame(
    ...     df, geometry=geopandas.points_from_xy(df['x'], df['y']))

    Having geographic coordinates:

    >>> df = pd.DataFrame({'longitude': [-140, 0, 123], 'latitude': [-65, 1, 48]})
    >>> df
       longitude  latitude
    0       -140       -65
    1          0         1
    2        123        48
    >>> geometry = geopandas.points_from_xy(df.longitude, df.latitude, crs="EPSG:4326")

    Returns
    -------
    output : GeometryArray
    """
    return GeometryArray(vectorized.points_from_xy(x, y, z), crs=crs)


class GeometryArray(ExtensionArray):
    """
    Class wrapping a numpy array of Shapely objects and
    holding the array-based implementations.
    """

    _dtype = GeometryDtype()

    def __init__(self, data, crs=None):
        if isinstance(data, self.__class__):
            if not crs:
                crs = data.crs
            data = data.data
        elif not isinstance(data, np.ndarray):
            raise TypeError(
                "'data' should be array of geometry objects. Use from_shapely, "
                "from_wkb, from_wkt functions to construct a GeometryArray."
            )
        elif not data.ndim == 1:
            raise ValueError(
                "'data' should be a 1-dimensional array of geometry objects."
            )
        self.data = data

        self._crs = None
        self.crs = crs
        self._sindex = None

    @property
    def sindex(self):
        if self._sindex is None:
            self._sindex = _get_sindex_class()(self.data)
        return self._sindex

    @property
    def has_sindex(self):
        """Check the existence of the spatial index without generating it.

        Use the `.sindex` attribute on a GeoDataFrame or GeoSeries
        to generate a spatial index if it does not yet exist,
        which may take considerable time based on the underlying index
        implementation.

        Note that the underlying spatial index may not be fully
        initialized until the first use.

        See Also
        ---------
        GeoDataFrame.has_sindex

        Returns
        -------
        bool
            `True` if the spatial index has been generated or
            `False` if not.
        """
        return self._sindex is not None

    @property
    def crs(self):
        """
        The Coordinate Reference System (CRS) represented as a ``pyproj.CRS``
        object.

        Returns None if the CRS is not set, and to set the value it
        :getter: Returns a ``pyproj.CRS`` or None. When setting, the value
        Coordinate Reference System of the geometry objects. Can be anything accepted by
        :meth:`pyproj.CRS.from_user_input() <pyproj.crs.CRS.from_user_input>`,
        such as an authority string (eg "EPSG:4326") or a WKT string.
        """
        return self._crs

    @crs.setter
    def crs(self, value):
        """Sets the value of the crs"""
        self._crs = None if not value else CRS.from_user_input(value)

    def check_geographic_crs(self, stacklevel):
        """Check CRS and warn if the planar operation is done in a geographic CRS"""
        if self.crs and self.crs.is_geographic:
            warnings.warn(
                "Geometry is in a geographic CRS. Results from '{}' are likely "
                "incorrect. Use 'GeoSeries.to_crs()' to re-project geometries to a "
                "projected CRS before this operation.\n".format(
                    inspect.stack()[1].function
                ),
                UserWarning,
                stacklevel=stacklevel,
            )

    @property
    def dtype(self):
        return self._dtype

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, idx):
        if isinstance(idx, numbers.Integral):
            return _geom_to_shapely(self.data[idx])
        # array-like, slice
        if compat.PANDAS_GE_10:
            # for pandas >= 1.0, validate and convert IntegerArray/BooleanArray
            # to numpy array, pass-through non-array-like indexers
            idx = pd.api.indexers.check_array_indexer(self, idx)
        if isinstance(idx, (Iterable, slice)):
            return GeometryArray(self.data[idx], crs=self.crs)
        else:
            raise TypeError("Index type not supported", idx)

    def __setitem__(self, key, value):
        if compat.PANDAS_GE_10:
            # for pandas >= 1.0, validate and convert IntegerArray/BooleanArray
            # keys to numpy array, pass-through non-array-like indexers
            key = pd.api.indexers.check_array_indexer(self, key)
        if isinstance(value, pd.Series):
            value = value.values
        if isinstance(value, (list, np.ndarray)):
            value = from_shapely(value)
        if isinstance(value, GeometryArray):
            if isinstance(key, numbers.Integral):
                raise ValueError("cannot set a single element with an array")
            self.data[key] = value.data
        elif isinstance(value, BaseGeometry) or vectorized.isna(value):
            if vectorized.isna(value):
                # internally only use None as missing value indicator
                # but accept others
                value = None
            elif isinstance(value, BaseGeometry):
                value = from_shapely([value]).data[0]
            else:
                raise TypeError("should be valid geometry")
            if isinstance(key, (slice, list, np.ndarray)):
                value_array = np.empty(1, dtype=object)
                with compat.ignore_shapely2_warnings():
                    value_array[:] = [value]
                self.data[key] = value_array
            else:
                self.data[key] = value
        else:
            raise TypeError(
                "Value should be either a BaseGeometry or None, got %s" % str(value)
            )

        # invalidate spatial index
        self._sindex = None

        # TODO: use this once pandas-dev/pandas#33457 is fixed
        # if hasattr(value, "crs"):
        #     if value.crs and (value.crs != self.crs):
        #         raise ValueError(
        #             "CRS mismatch between CRS of the passed geometries "
        #             "and CRS of existing geometries."
        #         )

    def __getstate__(self):
        if compat.USE_PYGEOS:
            return (pygeos.to_wkb(self.data), self._crs)
        else:
            return self.__dict__

    def __setstate__(self, state):
        if compat.USE_PYGEOS:
            geoms = pygeos.from_wkb(state[0])
            self._crs = state[1]
            self._sindex = None  # pygeos.STRtree could not be pickled yet
            self.data = geoms
            self.base = None
        else:
            if "_crs" not in state:
                state["_crs"] = None
            self.__dict__.update(state)

    # -------------------------------------------------------------------------
    # Geometry related methods
    # -------------------------------------------------------------------------

    @property
    def is_valid(self):
        return vectorized.is_valid(self.data)

    @property
    def is_empty(self):
        return vectorized.is_empty(self.data)

    @property
    def is_simple(self):
        return vectorized.is_simple(self.data)

    @property
    def is_ring(self):
        return vectorized.is_ring(self.data)

    @property
    def is_closed(self):
        return vectorized.is_closed(self.data)

    @property
    def has_z(self):
        return vectorized.has_z(self.data)

    @property
    def geom_type(self):
        return vectorized.geom_type(self.data)

    @property
    def area(self):
        self.check_geographic_crs(stacklevel=5)
        return vectorized.area(self.data)

    @property
    def length(self):
        self.check_geographic_crs(stacklevel=5)
        return vectorized.length(self.data)

    #
    # Unary operations that return new geometries
    #

    @property
    def boundary(self):
        return GeometryArray(vectorized.boundary(self.data), crs=self.crs)

    @property
    def centroid(self):
        self.check_geographic_crs(stacklevel=5)
        return GeometryArray(vectorized.centroid(self.data), crs=self.crs)

    @property
    def convex_hull(self):
        return GeometryArray(vectorized.convex_hull(self.data), crs=self.crs)

    @property
    def envelope(self):
        return GeometryArray(vectorized.envelope(self.data), crs=self.crs)

    @property
    def exterior(self):
        return GeometryArray(vectorized.exterior(self.data), crs=self.crs)

    @property
    def interiors(self):
        # no GeometryArray as result
        return vectorized.interiors(self.data)

    def representative_point(self):
        return GeometryArray(vectorized.representative_point(self.data), crs=self.crs)

    #
    # Binary predicates
    #

    @staticmethod
    def _binary_method(op, left, right, **kwargs):
        if isinstance(right, GeometryArray):
            if len(left) != len(right):
                msg = "Lengths of inputs do not match. Left: {0}, Right: {1}".format(
                    len(left), len(right)
                )
                raise ValueError(msg)
            if not _check_crs(left, right):
                _crs_mismatch_warn(left, right, stacklevel=7)
            right = right.data

        return getattr(vectorized, op)(left.data, right, **kwargs)

    def covers(self, other):
        return self._binary_method("covers", self, other)

    def covered_by(self, other):
        return self._binary_method("covered_by", self, other)

    def contains(self, other):
        return self._binary_method("contains", self, other)

    def crosses(self, other):
        return self._binary_method("crosses", self, other)

    def disjoint(self, other):
        return self._binary_method("disjoint", self, other)

    def geom_equals(self, other):
        return self._binary_method("equals", self, other)

    def intersects(self, other):
        return self._binary_method("intersects", self, other)

    def overlaps(self, other):
        return self._binary_method("overlaps", self, other)

    def touches(self, other):
        return self._binary_method("touches", self, other)

    def within(self, other):
        return self._binary_method("within", self, other)

    def geom_equals_exact(self, other, tolerance):
        return self._binary_method("equals_exact", self, other, tolerance=tolerance)

    def geom_almost_equals(self, other, decimal):
        return self.geom_equals_exact(other, 0.5 * 10 ** (-decimal))
        # return _binary_predicate("almost_equals", self, other, decimal=decimal)

    def equals_exact(self, other, tolerance):
        warnings.warn(
            "GeometryArray.equals_exact() is now GeometryArray.geom_equals_exact(). "
            "GeometryArray.equals_exact() will be deprecated in the future.",
            FutureWarning,
            stacklevel=2,
        )
        return self._binary_method("equals_exact", self, other, tolerance=tolerance)

    def almost_equals(self, other, decimal):
        warnings.warn(
            "GeometryArray.almost_equals() is now GeometryArray.geom_almost_equals(). "
            "GeometryArray.almost_equals() will be deprecated in the future.",
            FutureWarning,
            stacklevel=2,
        )
        return self.geom_equals_exact(other, 0.5 * 10 ** (-decimal))

    #
    # Binary operations that return new geometries
    #

    def difference(self, other):
        return GeometryArray(
            self._binary_method("difference", self, other), crs=self.crs
        )

    def intersection(self, other):
        return GeometryArray(
            self._binary_method("intersection", self, other), crs=self.crs
        )

    def symmetric_difference(self, other):
        return GeometryArray(
            self._binary_method("symmetric_difference", self, other), crs=self.crs
        )

    def union(self, other):
        return GeometryArray(self._binary_method("union", self, other), crs=self.crs)

    #
    # Other operations
    #

    def distance(self, other):
        self.check_geographic_crs(stacklevel=6)
        return self._binary_method("distance", self, other)

    def buffer(self, distance, resolution=16, **kwargs):
        if not (isinstance(distance, (int, float)) and distance == 0):
            self.check_geographic_crs(stacklevel=5)
        return GeometryArray(
            vectorized.buffer(self.data, distance, resolution=resolution, **kwargs),
            crs=self.crs,
        )

    def interpolate(self, distance, normalized=False):
        self.check_geographic_crs(stacklevel=5)
        return GeometryArray(
            vectorized.interpolate(self.data, distance, normalized=normalized),
            crs=self.crs,
        )

    def simplify(self, tolerance, preserve_topology=True):
        return GeometryArray(
            vectorized.simplify(
                self.data, tolerance, preserve_topology=preserve_topology
            ),
            crs=self.crs,
        )

    def project(self, other, normalized=False):
        if isinstance(other, BaseGeometry):
            other = _shapely_to_geom(other)
        elif isinstance(other, GeometryArray):
            other = other.data
        return vectorized.project(self.data, other, normalized=normalized)

    def relate(self, other):
        if isinstance(other, GeometryArray):
            other = other.data
        return vectorized.relate(self.data, other)

    #
    # Reduction operations that return a Shapely geometry
    #

    def unary_union(self):
        return vectorized.unary_union(self.data)

    #
    # Affinity operations
    #

    def affine_transform(self, matrix):
        return GeometryArray(
            vectorized._affinity_method("affine_transform", self.data, matrix),
            crs=self.crs,
        )

    def translate(self, xoff=0.0, yoff=0.0, zoff=0.0):
        return GeometryArray(
            vectorized._affinity_method("translate", self.data, xoff, yoff, zoff),
            crs=self.crs,
        )

    def rotate(self, angle, origin="center", use_radians=False):
        return GeometryArray(
            vectorized._affinity_method(
                "rotate", self.data, angle, origin=origin, use_radians=use_radians
            ),
            crs=self.crs,
        )

    def scale(self, xfact=1.0, yfact=1.0, zfact=1.0, origin="center"):
        return GeometryArray(
            vectorized._affinity_method(
                "scale", self.data, xfact, yfact, zfact, origin=origin
            ),
            crs=self.crs,
        )

    def skew(self, xs=0.0, ys=0.0, origin="center", use_radians=False):
        return GeometryArray(
            vectorized._affinity_method(
                "skew", self.data, xs, ys, origin=origin, use_radians=use_radians
            ),
            crs=self.crs,
        )

    def to_crs(self, crs=None, epsg=None):
        """Returns a ``GeometryArray`` with all geometries transformed to a new
        coordinate reference system.

        Transform all geometries in a GeometryArray to a different coordinate
        reference system.  The ``crs`` attribute on the current GeometryArray must
        be set.  Either ``crs`` or ``epsg`` may be specified for output.

        This method will transform all points in all objects.  It has no notion
        or projecting entire geometries.  All segments joining points are
        assumed to be lines in the current projection, not geodesics.  Objects
        crossing the dateline (or other projection boundary) will have
        undesirable behavior.

        Parameters
        ----------
        crs : pyproj.CRS, optional if `epsg` is specified
            The value can be anything accepted
            by :meth:`pyproj.CRS.from_user_input() <pyproj.crs.CRS.from_user_input>`,
            such as an authority string (eg "EPSG:4326") or a WKT string.
        epsg : int, optional if `crs` is specified
            EPSG code specifying output projection.

        Returns
        -------
        GeometryArray

        Examples
        --------
        >>> from shapely.geometry import Point
        >>> from geopandas.array import from_shapely, to_wkt
        >>> a = from_shapely([Point(1, 1), Point(2, 2), Point(3, 3)], crs=4326)
        >>> to_wkt(a)
        array(['POINT (1 1)', 'POINT (2 2)', 'POINT (3 3)'], dtype=object)
        >>> a.crs  # doctest: +SKIP
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

        >>> a = a.to_crs(3857)
        >>> to_wkt(a)
        array(['POINT (111319 111325)', 'POINT (222639 222684)',
               'POINT (333958 334111)'], dtype=object)
        >>> a.crs  # doctest: +SKIP
        <Projected CRS: EPSG:3857>
        Name: WGS 84 / Pseudo-Mercator
        Axis Info [cartesian]:
        - X[east]: Easting (metre)
        - Y[north]: Northing (metre)
        Area of Use:
        - name: World - 85°S to 85°N
        - bounds: (-180.0, -85.06, 180.0, 85.06)
        Coordinate Operation:
        - name: Popular Visualisation Pseudo-Mercator
        - method: Popular Visualisation Pseudo Mercator
        Datum: World Geodetic System 1984
        - Ellipsoid: WGS 84
        - Prime Meridian: Greenwich

        """
        if self.crs is None:
            raise ValueError(
                "Cannot transform naive geometries.  "
                "Please set a crs on the object first."
            )
        if crs is not None:
            crs = CRS.from_user_input(crs)
        elif epsg is not None:
            crs = CRS.from_epsg(epsg)
        else:
            raise ValueError("Must pass either crs or epsg.")

        # skip if the input CRS and output CRS are the exact same
        if self.crs.is_exact_same(crs):
            return self

        transformer = Transformer.from_crs(self.crs, crs, always_xy=True)

        new_data = vectorized.transform(self.data, transformer.transform)
        return GeometryArray(new_data, crs=crs)

    def estimate_utm_crs(self, datum_name="WGS 84"):
        """Returns the estimated UTM CRS based on the bounds of the dataset.

        .. versionadded:: 0.9

        .. note:: Requires pyproj 3+

        Parameters
        ----------
        datum_name : str, optional
            The name of the datum to use in the query. Default is WGS 84.

        Returns
        -------
        pyproj.CRS

        Examples
        --------
        >>> world = geopandas.read_file(
        ...     geopandas.datasets.get_path("naturalearth_lowres")
        ... )
        >>> germany = world.loc[world.name == "Germany"]
        >>> germany.geometry.values.estimate_utm_crs()  # doctest: +SKIP
        <Projected CRS: EPSG:32632>
        Name: WGS 84 / UTM zone 32N
        Axis Info [cartesian]:
        - E[east]: Easting (metre)
        - N[north]: Northing (metre)
        Area of Use:
        - name: World - N hemisphere - 6°E to 12°E - by country
        - bounds: (6.0, 0.0, 12.0, 84.0)
        Coordinate Operation:
        - name: UTM zone 32N
        - method: Transverse Mercator
        Datum: World Geodetic System 1984
        - Ellipsoid: WGS 84
        - Prime Meridian: Greenwich
        """
        try:
            from pyproj.aoi import AreaOfInterest
            from pyproj.database import query_utm_crs_info
        except ImportError:
            raise RuntimeError("pyproj 3+ required for estimate_utm_crs.")

        if not self.crs:
            raise RuntimeError("crs must be set to estimate UTM CRS.")

        minx, miny, maxx, maxy = self.total_bounds
        if self.crs.is_geographic:
            x_center = np.mean([minx, maxx])
            y_center = np.mean([miny, maxy])
        # ensure using geographic coordinates
        else:
            transformer = Transformer.from_crs(self.crs, "EPSG:4326", always_xy=True)
            if compat.PYPROJ_GE_31:
                minx, miny, maxx, maxy = transformer.transform_bounds(
                    minx, miny, maxx, maxy
                )
                y_center = np.mean([miny, maxy])
                # crossed the antimeridian
                if minx > maxx:
                    # shift maxx from [-180,180] to [0,360]
                    # so both numbers are positive for center calculation
                    # Example: -175 to 185
                    maxx += 360
                    x_center = np.mean([minx, maxx])
                    # shift back to [-180,180]
                    x_center = ((x_center + 180) % 360) - 180
                else:
                    x_center = np.mean([minx, maxx])
            else:
                lon, lat = transformer.transform(
                    (minx, maxx, minx, maxx), (miny, miny, maxy, maxy)
                )
                x_center = np.mean(lon)
                y_center = np.mean(lat)

        utm_crs_list = query_utm_crs_info(
            datum_name=datum_name,
            area_of_interest=AreaOfInterest(
                west_lon_degree=x_center,
                south_lat_degree=y_center,
                east_lon_degree=x_center,
                north_lat_degree=y_center,
            ),
        )
        try:
            return CRS.from_epsg(utm_crs_list[0].code)
        except IndexError:
            raise RuntimeError("Unable to determine UTM CRS")

    #
    # Coordinate related properties
    #

    @property
    def x(self):
        """Return the x location of point geometries in a GeoSeries"""
        if (self.geom_type[~self.isna()] == "Point").all():
            return vectorized.get_x(self.data)
        else:
            message = "x attribute access only provided for Point geometries"
            raise ValueError(message)

    @property
    def y(self):
        """Return the y location of point geometries in a GeoSeries"""
        if (self.geom_type[~self.isna()] == "Point").all():
            return vectorized.get_y(self.data)
        else:
            message = "y attribute access only provided for Point geometries"
            raise ValueError(message)

    @property
    def z(self):
        """Return the z location of point geometries in a GeoSeries"""
        if (self.geom_type[~self.isna()] == "Point").all():
            return vectorized.get_z(self.data)
        else:
            message = "z attribute access only provided for Point geometries"
            raise ValueError(message)

    @property
    def bounds(self):
        return vectorized.bounds(self.data)

    @property
    def total_bounds(self):
        if len(self) == 0:
            # numpy 'min' cannot handle empty arrays
            # TODO with numpy >= 1.15, the 'initial' argument can be used
            return np.array([np.nan, np.nan, np.nan, np.nan])
        b = self.bounds
        return np.array(
            (
                np.nanmin(b[:, 0]),  # minx
                np.nanmin(b[:, 1]),  # miny
                np.nanmax(b[:, 2]),  # maxx
                np.nanmax(b[:, 3]),  # maxy
            )
        )

    # -------------------------------------------------------------------------
    # general array like compat
    # -------------------------------------------------------------------------

    @property
    def size(self):
        return self.data.size

    @property
    def shape(self):
        return (self.size,)

    @property
    def ndim(self):
        return len(self.shape)

    def copy(self, *args, **kwargs):
        # still taking args/kwargs for compat with pandas 0.24
        return GeometryArray(self.data.copy(), crs=self._crs)

    def take(self, indices, allow_fill=False, fill_value=None):
        from pandas.api.extensions import take

        if allow_fill:
            if fill_value is None or pd.isna(fill_value):
                fill_value = None
            elif isinstance(fill_value, BaseGeometry):
                fill_value = _shapely_to_geom(fill_value)
            elif not _is_scalar_geometry(fill_value):
                raise TypeError("provide geometry or None as fill value")

        result = take(self.data, indices, allow_fill=allow_fill, fill_value=fill_value)
        if allow_fill and fill_value is None:
            result[pd.isna(result)] = None
        return GeometryArray(result, crs=self.crs)

    def _fill(self, idx, value):
        """Fill index locations with value

        Value should be a BaseGeometry
        """
        if not (_is_scalar_geometry(value) or value is None):
            raise TypeError(
                "Value should be either a BaseGeometry or None, got %s" % str(value)
            )
        # self.data[idx] = value
        value_arr = np.empty(1, dtype=object)
        with compat.ignore_shapely2_warnings():
            value_arr[:] = [value]
        self.data[idx] = value_arr
        return self

    def fillna(self, value=None, method=None, limit=None):
        """Fill NA/NaN values using the specified method.

        Parameters
        ----------
        value : scalar, array-like
            If a scalar value is passed it is used to fill all missing values.
            Alternatively, an array-like 'value' can be given. It's expected
            that the array-like have the same length as 'self'.
        method : {'backfill', 'bfill', 'pad', 'ffill', None}, default None
            Method to use for filling holes in reindexed Series
            pad / ffill: propagate last valid observation forward to next valid
            backfill / bfill: use NEXT valid observation to fill gap
        limit : int, default None
            If method is specified, this is the maximum number of consecutive
            NaN values to forward/backward fill. In other words, if there is
            a gap with more than this number of consecutive NaNs, it will only
            be partially filled. If method is not specified, this is the
            maximum number of entries along the entire axis where NaNs will be
            filled.

        Returns
        -------
        filled : ExtensionArray with NA/NaN filled
        """
        if method is not None:
            raise NotImplementedError("fillna with a method is not yet supported")

        mask = self.isna()
        new_values = self.copy()

        if mask.any():
            # fill with value
            if vectorized.isna(value):
                value = None
            elif not isinstance(value, BaseGeometry):
                raise NotImplementedError(
                    "fillna currently only supports filling with a scalar geometry"
                )
            value = _shapely_to_geom(value)
            new_values = new_values._fill(mask, value)

        return new_values

    def astype(self, dtype, copy=True):
        """
        Cast to a NumPy array with 'dtype'.

        Parameters
        ----------
        dtype : str or dtype
            Typecode or data-type to which the array is cast.
        copy : bool, default True
            Whether to copy the data, even if not necessary. If False,
            a copy is made only if the old dtype does not match the
            new dtype.

        Returns
        -------
        array : ndarray
            NumPy ndarray with 'dtype' for its dtype.
        """
        if isinstance(dtype, GeometryDtype):
            if copy:
                return self.copy()
            else:
                return self
        elif pd.api.types.is_string_dtype(dtype) and not pd.api.types.is_object_dtype(
            dtype
        ):
            string_values = to_wkt(self)
            if compat.PANDAS_GE_10:
                pd_dtype = pd.api.types.pandas_dtype(dtype)
                if isinstance(pd_dtype, pd.StringDtype):
                    # ensure to return a pandas string array instead of numpy array
                    return pd.array(string_values, dtype=pd_dtype)
            return string_values.astype(dtype, copy=False)
        else:
            return np.array(self, dtype=dtype, copy=copy)

    def isna(self):
        """
        Boolean NumPy array indicating if each value is missing
        """
        if compat.USE_PYGEOS:
            return pygeos.is_missing(self.data)
        else:
            return np.array([g is None for g in self.data], dtype="bool")

    def value_counts(
        self,
        dropna: bool = True,
    ):
        """
        Compute a histogram of the counts of non-null values.

        Parameters
        ----------
        dropna : bool, default True
            Don't include counts of NaN

        Returns
        -------
        pd.Series
        """

        # note ExtensionArray usage of value_counts only specifies dropna,
        # so sort, normalize and bins are not arguments
        values = to_wkb(self)
        from pandas import Series, Index

        result = Series(values).value_counts(dropna=dropna)
        # value_counts converts None to nan, need to convert back for from_wkb to work
        # note result.index already has object dtype, not geometry
        # Can't use fillna(None) or Index.putmask, as this gets converted back to nan
        # for object dtypes
        result.index = Index(
            from_wkb(np.where(result.index.isna(), None, result.index))
        )
        return result

    def unique(self):
        """Compute the ExtensionArray of unique values.

        Returns
        -------
        uniques : ExtensionArray
        """
        from pandas import factorize

        _, uniques = factorize(self)
        return uniques

    @property
    def nbytes(self):
        return self.data.nbytes

    def shift(self, periods=1, fill_value=None):
        """
        Shift values by desired number.

        Newly introduced missing values are filled with
        ``self.dtype.na_value``.

        Parameters
        ----------
        periods : int, default 1
            The number of periods to shift. Negative values are allowed
            for shifting backwards.

        fill_value : object, optional (default None)
            The scalar value to use for newly introduced missing values.
            The default is ``self.dtype.na_value``.

        Returns
        -------
        GeometryArray
            Shifted.

        Notes
        -----
        If ``self`` is empty or ``periods`` is 0, a copy of ``self`` is
        returned.

        If ``periods > len(self)``, then an array of size
        len(self) is returned, with all values filled with
        ``self.dtype.na_value``.
        """
        shifted = super().shift(periods, fill_value)
        shifted.crs = self.crs
        return shifted

    # -------------------------------------------------------------------------
    # ExtensionArray specific
    # -------------------------------------------------------------------------

    @classmethod
    def _from_sequence(cls, scalars, dtype=None, copy=False):
        """
        Construct a new ExtensionArray from a sequence of scalars.

        Parameters
        ----------
        scalars : Sequence
            Each element will be an instance of the scalar type for this
            array, ``cls.dtype.type``.
        dtype : dtype, optional
            Construct for this particular dtype. This should be a Dtype
            compatible with the ExtensionArray.
        copy : boolean, default False
            If True, copy the underlying data.

        Returns
        -------
        ExtensionArray
        """
        # GH 1413
        if isinstance(scalars, BaseGeometry):
            scalars = [scalars]
        return from_shapely(scalars)

    def _values_for_factorize(self):
        # type: () -> Tuple[np.ndarray, Any]
        """Return an array and missing value suitable for factorization.

        Returns
        -------
        values : ndarray
            An array suitable for factoraization. This should maintain order
            and be a supported dtype (Float64, Int64, UInt64, String, Object).
            By default, the extension array is cast to object dtype.
        na_value : object
            The value in `values` to consider missing. This will be treated
            as NA in the factorization routines, so it will be coded as
            `na_sentinal` and not included in `uniques`. By default,
            ``np.nan`` is used.
        """
        vals = to_wkb(self)
        return vals, None

    @classmethod
    def _from_factorized(cls, values, original):
        """
        Reconstruct an ExtensionArray after factorization.

        Parameters
        ----------
        values : ndarray
            An integer ndarray with the factorized values.
        original : ExtensionArray
            The original ExtensionArray that factorize was called on.

        See Also
        --------
        pandas.factorize
        ExtensionArray.factorize
        """
        return from_wkb(values, crs=original.crs)

    def _values_for_argsort(self):
        # type: () -> np.ndarray
        """Return values for sorting.

        Returns
        -------
        ndarray
            The transformed values should maintain the ordering between values
            within the array.

        See Also
        --------
        ExtensionArray.argsort
        """
        # Note: this is used in `ExtensionArray.argsort`.
        raise TypeError("geometries are not orderable")

    def _formatter(self, boxed=False):
        """Formatting function for scalar values.

        This is used in the default '__repr__'. The returned formatting
        function receives instances of your scalar type.

        Parameters
        ----------
        boxed: bool, default False
            An indicated for whether or not your array is being printed
            within a Series, DataFrame, or Index (True), or just by
            itself (False). This may be useful if you want scalar values
            to appear differently within a Series versus on its own (e.g.
            quoted or not).

        Returns
        -------
        Callable[[Any], str]
            A callable that gets instances of the scalar type and
            returns a string. By default, :func:`repr` is used
            when ``boxed=False`` and :func:`str` is used when
            ``boxed=True``.
        """
        if boxed:
            import geopandas

            precision = geopandas.options.display_precision
            if precision is None:
                if self.crs:
                    if self.crs.is_projected:
                        precision = 3
                    else:
                        precision = 5
                else:
                    # fallback
                    # dummy heuristic based on 10 first geometries that should
                    # work in most cases
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=RuntimeWarning)
                        xmin, ymin, xmax, ymax = self[~self.isna()][:10].total_bounds
                    if (
                        (-180 <= xmin <= 180)
                        and (-180 <= xmax <= 180)
                        and (-90 <= ymin <= 90)
                        and (-90 <= ymax <= 90)
                    ):
                        # geographic coordinates
                        precision = 5
                    else:
                        # typically projected coordinates
                        # (in case of unit meter: mm precision)
                        precision = 3
            return lambda geom: shapely.wkt.dumps(geom, rounding_precision=precision)
        return repr

    @classmethod
    def _concat_same_type(cls, to_concat):
        """
        Concatenate multiple array

        Parameters
        ----------
        to_concat : sequence of this type

        Returns
        -------
        ExtensionArray
        """
        data = np.concatenate([ga.data for ga in to_concat])
        return GeometryArray(data, crs=to_concat[0].crs)

    def _reduce(self, name, skipna=True, **kwargs):
        # including the base class version here (that raises by default)
        # because this was not yet defined in pandas 0.23
        if name == "any" or name == "all":
            # TODO(pygeos)
            return getattr(to_shapely(self), name)()
        raise TypeError(
            "cannot perform {name} with type {dtype}".format(
                name=name, dtype=self.dtype
            )
        )

    def __array__(self, dtype=None):
        """
        The numpy array interface.

        Returns
        -------
        values : numpy array
        """
        return to_shapely(self)

    def _binop(self, other, op):
        def convert_values(param):
            if not _is_scalar_geometry(param) and (
                isinstance(param, ExtensionArray) or pd.api.types.is_list_like(param)
            ):
                ovalues = param
            else:  # Assume its an object
                ovalues = [param] * len(self)
            return ovalues

        if isinstance(other, (pd.Series, pd.Index, pd.DataFrame)):
            # rely on pandas to unbox and dispatch to us
            return NotImplemented

        lvalues = self
        rvalues = convert_values(other)

        if len(lvalues) != len(rvalues):
            raise ValueError("Lengths must match to compare")

        # If the operator is not defined for the underlying objects,
        # a TypeError should be raised
        res = [op(a, b) for (a, b) in zip(lvalues, rvalues)]

        res = np.asarray(res, dtype=bool)
        return res

    def __eq__(self, other):
        return self._binop(other, operator.eq)

    def __ne__(self, other):
        return self._binop(other, operator.ne)

    def __contains__(self, item):
        """
        Return for `item in self`.
        """
        if vectorized.isna(item):
            if (
                item is self.dtype.na_value
                or isinstance(item, self.dtype.type)
                or item is None
            ):
                return self.isna().any()
            else:
                return False
        return (self == item).any()
