import numbers
import operator
import warnings
from functools import lru_cache

import numpy as np
import pandas as pd
import spherely
from pandas.api.extensions import ExtensionArray

import shapely
import shapely.affinity
import shapely.geometry

from ._compat import HAS_PYPROJ, PANDAS_GE_21, PANDAS_GE_22, requires_pyproj
from .array import BaseGeometryArray, GeometryDtype

if HAS_PYPROJ:
    from pyproj import Transformer

    TransformerFromCRS = lru_cache(Transformer.from_crs)


_names = {
    "NONE": None,
    "POINT": "Point",
    "LINESTRING": "LineString",
    "POLYGON": "Polygon",
    "MULTIPOINT": "MultiPoint",
    "MULTILINESTRING": "MultiLineString",
    "MULTIPOLYGON": "MultiPolygon",
    "GEOMETRYCOLLECTION": "GeometryCollection",
}

POLYGON_GEOM_TYPES = {"Polygon", "MultiPolygon"}
LINE_GEOM_TYPES = {"LineString", "MultiLineString", "LinearRing"}
POINT_GEOM_TYPES = {"Point", "MultiPoint"}

type_mapping = {
    p.value: _names[p.name] for p in [spherely.GeographyType(i) for i in range(-1, 7)]
}
geometry_type_ids = list(type_mapping.keys())
geometry_type_values = np.array(list(type_mapping.values()), dtype=object)


def isna(value):
    """
    Check if scalar value is NA-like (None, np.nan or pd.NA).

    Custom version that only works for scalars (returning True or False),
    as `pd.isna` also works for array-like input returning a boolean array.
    """
    if value is None:
        return True
    elif isinstance(value, float) and np.isnan(value):
        return True
    elif value is pd.NA:
        return True
    else:
        return False


# -----------------------------------------------------------------------------
# Constructors / converters to other formats
# -----------------------------------------------------------------------------


def _is_scalar_geometry(geom):
    # TODO should we also accept shapely geometries in certain cases?
    return isinstance(geom, spherely.Geography)


def from_spherely(data, crs=None):
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
    if not isinstance(data, np.ndarray):
        arr = np.empty(len(data), dtype=object)
        arr[:] = data
    elif len(data) == 0 and data.dtype == "float64":
        arr = data.astype(object)
    else:
        arr = data

    if not shapely.is_geography(arr).all():
        # TODO do we want to support creating from the geo interface?
        # out = []

        # for geom in data:
        #     if isinstance(geom, BaseGeometry):
        #         out.append(geom)
        #     elif hasattr(geom, "__geo_interface__"):
        #         geom = shapely.geometry.shape(geom)
        #         out.append(geom)
        #     elif isna(geom):
        #         out.append(None)
        #     else:
        #         raise TypeError(f"Input must be valid geometry objects: {geom}")
        # arr = np.array(out, dtype=object)
        raise TypeError("Input must be valid spherely geography objects")

    return GeographyArray(arr, crs=crs)


def from_wkb(data, crs=None, on_invalid="raise"):
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
    on_invalid: {"raise", "warn", "ignore"}, default "raise"
        - raise: an exception will be raised if a WKB input geometry is invalid.
        - warn: a warning will be raised and invalid WKB geometries will be returned as
          None.
        - ignore: invalid WKB geometries will be returned as None without a warning.

    """
    # TODO raise for unsupported keyword (and pass through spherely-speficic kwargs)
    return GeographyArray(shapely.from_wkb(data), crs=crs)


def from_wkt(data, crs=None, on_invalid="raise"):
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
    on_invalid : {"raise", "warn", "ignore"}, default "raise"
        - raise: an exception will be raised if a WKT input geometry is invalid.
        - warn: a warning will be raised and invalid WKT geometries will be
          returned as ``None``.
        - ignore: invalid WKT geometries will be returned as ``None`` without a warning.

    """
    return GeographyArray(spherely.from_wkt(data), crs=crs)


class GeographyArray(BaseGeometryArray):
    """
    Class wrapping a numpy array of Spherely objects
    """

    _dtype = GeometryDtype(engine="spherical")

    def __init__(self, data, crs=None):
        if isinstance(data, self.__class__):
            if not crs:
                crs = data.crs
            data = data._data
        elif not isinstance(data, np.ndarray):
            raise TypeError(
                "'data' should be array of geometry objects. Use from_spherely, "
                "from_wkb, from_wkt functions to construct a GeometryArray."
            )
        elif not data.ndim == 1:
            raise ValueError(
                "'data' should be a 1-dimensional array of geometry objects."
            )
        self._data = data

        # TODO do we want to still store an actual CRS object?
        # Or hardcode it to EPSG:4326?
        # (there are still various geographic CRS options)
        self._crs = None
        if crs is None:
            crs = "EPSG:4326"
        self.crs = crs
        self._sindex = None

    @property
    def sindex(self):
        raise NotImplementedError

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
        if HAS_PYPROJ:
            from pyproj import CRS

            self._crs = None if not value else CRS.from_user_input(value)
        else:
            if value is not None:
                warnings.warn(
                    "Cannot set the CRS, falling back to None. The CRS support requires"
                    " the 'pyproj' package, but it is not installed or does not import"
                    " correctly. The functions depending on CRS will raise an error or"
                    " may produce unexpected results.",
                    UserWarning,
                    stacklevel=2,
                )
            self._crs = None

    @property
    def dtype(self):
        return self._dtype

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, idx):
        if isinstance(idx, numbers.Integral):
            return self._data[idx]
        # array-like, slice
        # validate and convert IntegerArray/BooleanArray
        # to numpy array, pass-through non-array-like indexers
        idx = pd.api.indexers.check_array_indexer(self, idx)
        return GeographyArray(self._data[idx], crs=self.crs)

    def __setitem__(self, key, value):
        # validate and convert IntegerArray/BooleanArray
        # keys to numpy array, pass-through non-array-like indexers
        key = pd.api.indexers.check_array_indexer(self, key)
        if isinstance(value, pd.Series):
            value = value.values
        if isinstance(value, pd.DataFrame):
            value = value.values.flatten()
        if isinstance(value, list | np.ndarray):
            value = from_spherely(value)
        if isinstance(value, GeographyArray):
            if isinstance(key, numbers.Integral):
                raise ValueError("cannot set a single element with an array")
            self._data[key] = value._data
        elif isinstance(value, spherely.Geography) or isna(value):
            if isna(value):
                # internally only use None as missing value indicator
                # but accept others
                value = None
            elif isinstance(value, spherely.Geography):
                value = from_spherely([value])._data[0]
            else:
                raise TypeError("should be valid geometry")
            if isinstance(key, slice | list | np.ndarray):
                value_array = np.empty(1, dtype=object)
                value_array[:] = [value]
                self._data[key] = value_array
            else:
                self._data[key] = value
        else:
            raise TypeError(
                f"Value should be either a BaseGeometry or None, got {value!s}"
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
        return (self.to_wkb(), self._crs)

    def __setstate__(self, state):
        if not isinstance(state, dict):
            # pickle file saved with pygeos
            geoms = spherely.from_wkb(state[0])
            self._crs = state[1]
            self._sindex = None  # pygeos.STRtree could not be pickled yet
            self._data = geoms
            self.base = None
        else:
            if "data" in state:
                state["_data"] = state.pop("data")
            if "_crs" not in state:
                state["_crs"] = None
            self.__dict__.update(state)

    def to_wkb(self, hex=False, **kwargs):
        """
        Convert GeometryArray to a numpy object array of WKB objects.
        """
        if hex:
            raise NotImplementedError
        return spherely.to_wkb(self._data, **kwargs)

    @classmethod
    def from_shapely(cls, data, planar=True):
        data_wkb = shapely.to_wkb(
            shapely.set_precision(shapely.remove_repeated_points(data), 0.000001)
        )
        data_geog = spherely.from_wkb(data_wkb, planar=planar)
        return cls(data_geog)

    # -------------------------------------------------------------------------
    # Geometry related methods
    # -------------------------------------------------------------------------

    @property
    def is_valid(self):
        raise NotImplementedError

    def is_valid_reason(self):
        raise NotImplementedError

    @property
    def is_empty(self):
        return spherely.is_empty(self._data)

    @property
    def is_simple(self):
        raise NotImplementedError

    @property
    def is_ring(self):
        raise NotImplementedError

    @property
    def is_closed(self):
        raise NotImplementedError

    @property
    def is_ccw(self):
        # return all True?
        raise NotImplementedError

    @property
    def has_z(self):
        # return all False?
        raise NotImplementedError

    @property
    def geom_type(self):
        res = spherely.get_type_id(self._data)
        return geometry_type_values[np.searchsorted(geometry_type_ids, res)]

    @property
    def area(self):
        return spherely.area(self._data)

    @property
    def length(self):
        return spherely.perimeter(self._data)

    def count_coordinates(self):
        # return shapely.get_num_coordinates(self._data)
        raise NotImplementedError

    def count_geometries(self):
        # return shapely.get_num_geometries(self._data)
        raise NotImplementedError

    def count_interior_rings(self):
        # return shapely.get_num_interior_rings(self._data)
        raise NotImplementedError

    def get_precision(self):
        raise NotImplementedError

    def get_geometry(self, index):
        # return shapely.get_geometry(self._data, index=index)
        raise NotImplementedError

    #
    # Unary operations that return new geometries
    #

    @property
    def boundary(self):
        return GeographyArray(spherely.boundary(self._data), crs=self.crs)

    @property
    def centroid(self):
        return GeographyArray(spherely.centroid(self._data), crs=self.crs)

    def concave_hull(self, ratio, allow_holes):
        # return shapely.concave_hull(self._data, ratio=ratio, allow_holes=allow_holes)
        raise NotImplementedError

    @property
    def convex_hull(self):
        return GeographyArray(spherely.convex_hull(self._data), crs=self.crs)

    @property
    def envelope(self):
        # return GeometryArray(shapely.envelope(self._data), crs=self.crs)
        raise NotImplementedError

    def minimum_rotated_rectangle(self):
        # return GeometryArray(shapely.oriented_envelope(self._data), crs=self.crs)
        raise NotImplementedError

    @property
    def exterior(self):
        # return GeometryArray(shapely.get_exterior_ring(self._data), crs=self.crs)
        raise NotImplementedError

    def extract_unique_points(self):
        # TODO can be implemented by going through geoarrow or shapely
        # return GeometryArray(shapely.extract_unique_points(self._data), crs=self.crs)
        raise NotImplementedError

    def offset_curve(self, distance, quad_segs=8, join_style="round", mitre_limit=5.0):
        raise NotImplementedError

    @property
    def interiors(self):
        raise NotImplementedError

    def remove_repeated_points(self, tolerance=0.0):
        # return as is because repeated points are not allowed in S2 ?
        raise NotImplementedError

    def representative_point(self):
        raise NotImplementedError

    def minimum_bounding_circle(self):
        raise NotImplementedError

    def minimum_bounding_radius(self):
        raise NotImplementedError

    def minimum_clearance(self):
        raise NotImplementedError

    def normalize(self):
        # TODO return as is?
        # return GeometryArray(shapely.normalize(self._data), crs=self.crs)
        raise NotImplementedError

    def make_valid(self):
        raise NotImplementedError

    def reverse(self):
        # TODO raise error about this not being possible
        raise NotImplementedError

    def segmentize(self, max_segment_length):
        raise NotImplementedError

    def force_2d(self):
        # TODO return as is?
        raise NotImplementedError

    def force_3d(self, z=0):
        raise NotImplementedError

    def transform(self, transformation, include_z=False):
        # return GeometryArray(
        #     shapely.transform(self._data, transformation, include_z), crs=self.crs
        # )
        raise NotImplementedError

    def line_merge(self, directed=False):
        raise NotImplementedError

    def set_precision(self, grid_size, mode="valid_output"):
        raise NotImplementedError

    #
    # Binary predicates
    #

    @staticmethod
    def _binary_method(op, left, right, **kwargs):
        if isinstance(right, GeographyArray):
            if len(left) != len(right):
                msg = (
                    "Lengths of inputs do not match. "
                    f"Left: {len(left)}, Right: {len(right)}"
                )
                raise ValueError(msg)
            right = right._data

        return getattr(spherely, op)(left._data, right, **kwargs)

    def covers(self, other):
        return self._binary_method("covers", self, other)

    def covered_by(self, other):
        return self._binary_method("covered_by", self, other)

    def contains(self, other):
        return self._binary_method("contains", self, other)

    def contains_properly(self, other):
        # return self._binary_method("contains_properly", self, other)
        raise NotImplementedError

    def crosses(self, other):
        # return self._binary_method("crosses", self, other)
        raise NotImplementedError

    def disjoint(self, other):
        return self._binary_method("disjoint", self, other)

    def geom_equals(self, other):
        return self._binary_method("equals", self, other)

    def intersects(self, other):
        return self._binary_method("intersects", self, other)

    def overlaps(self, other):
        # return self._binary_method("overlaps", self, other)
        raise NotImplementedError

    def touches(self, other):
        return self._binary_method("touches", self, other)

    def within(self, other):
        return self._binary_method("within", self, other)

    def dwithin(self, other, distance):
        # return self._binary_method("dwithin", self, other, distance=distance)
        raise NotImplementedError

    def geom_equals_exact(self, other, tolerance):
        # return self._binary_method("equals_exact", self, other, tolerance=tolerance)
        raise NotImplementedError

    #
    # Binary operations that return new geometries
    #

    def clip_by_rect(self, xmin, ymin, xmax, ymax):
        raise NotImplementedError

    def difference(self, other):
        return GeographyArray(
            self._binary_method("difference", self, other), crs=self.crs
        )

    def intersection(self, other):
        return GeographyArray(
            self._binary_method("intersection", self, other), crs=self.crs
        )

    def symmetric_difference(self, other):
        return GeographyArray(
            self._binary_method("symmetric_difference", self, other), crs=self.crs
        )

    def union(self, other):
        return GeographyArray(self._binary_method("union", self, other), crs=self.crs)

    def shortest_line(self, other):
        raise NotImplementedError

    def snap(self, other, tolerance):
        raise NotImplementedError

    def shared_paths(self, other):
        raise NotImplementedError

    #
    # Other operations
    #

    def distance(self, other):
        return self._binary_method("distance", self, other)

    def hausdorff_distance(self, other, **kwargs):
        raise NotImplementedError

    def frechet_distance(self, other, **kwargs):
        raise NotImplementedError

    def buffer(self, distance, resolution=16, **kwargs):
        raise NotImplementedError

    def interpolate(self, distance, normalized=False):
        raise NotImplementedError

    def simplify(self, tolerance, preserve_topology=True):
        raise NotImplementedError

    def project(self, other, normalized=False):
        raise NotImplementedError

    def relate(self, other):
        raise NotImplementedError

    def relate_pattern(self, other, pattern):
        NotImplementedError

    #
    # Reduction operations that return a Shapely geometry
    #

    def union_all(self, method="unary", grid_size=None):
        raise NotImplementedError

    def intersection_all(self):
        raise NotImplementedError

    #
    # Affinity operations
    #

    def affine_transform(self, matrix):
        raise NotImplementedError

    def translate(self, xoff=0.0, yoff=0.0, zoff=0.0):
        raise NotImplementedError

    def rotate(self, angle, origin="center", use_radians=False):
        raise NotImplementedError

    def scale(self, xfact=1.0, yfact=1.0, zfact=1.0, origin="center"):
        raise NotImplementedError

    def skew(self, xs=0.0, ys=0.0, origin="center", use_radians=False):
        raise NotImplementedError

    @requires_pyproj
    def to_crs(self, crs=None, epsg=None):
        """Returns a ``GeometryArray`` with all geometries transformed to a new
        coordinate reference system.

        Transform all geometries in a GeometryArray to a different coordinate
        reference system.  The ``crs`` attribute on the current GeometryArray must
        be set.  Either ``crs`` or ``epsg`` may be specified for output.

        This method will transform all points in all objects.  It has no notion
        of projecting entire geometries.  All segments joining points are
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
        array(['POINT (111319.490793 111325.142866)',
               'POINT (222638.981587 222684.208506)',
               'POINT (333958.47238 334111.171402)'], dtype=object)
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
        # TODO add custom implementation of to_crs
        # - when converting to projected CRS, convert to GeometryArray
        # - when converting to geographic CRS, preserve the GeographyArray type?
        #   (this would require some transform method in spherely, or convert to
        #   geoarrow flat coordinates, transform, and convert back to shapely)

        raise NotImplementedError
        # from pyproj import CRS

        # if self.crs is None:
        #     raise ValueError(
        #         "Cannot transform naive geometries.  "
        #         "Please set a crs on the object first."
        #     )
        # if crs is not None:
        #     crs = CRS.from_user_input(crs)
        # elif epsg is not None:
        #     crs = CRS.from_epsg(epsg)
        # else:
        #     raise ValueError("Must pass either crs or epsg.")

        # # skip if the input CRS and output CRS are the exact same
        # if self.crs.is_exact_same(crs):
        #     return self

        # transformer = TransformerFromCRS(self.crs, crs, always_xy=True)

        # new_data = transform(self._data, transformer.transform)
        # return GeometryArray(new_data, crs=crs)

    def estimate_utm_crs(self, datum_name="WGS 84"):
        raise NotImplementedError

    #
    # Coordinate related properties
    #

    @property
    def x(self):
        """Return the x location of point geometries in a GeoSeries"""
        if (self.geom_type[~self.isna()] == "Point").all():
            return spherely.get_x(self._data)
        else:
            message = "x attribute access only provided for Point geometries"
            raise ValueError(message)

    @property
    def y(self):
        """Return the y location of point geometries in a GeoSeries"""
        if (self.geom_type[~self.isna()] == "Point").all():
            return spherely.get_y(self._data)
        else:
            message = "y attribute access only provided for Point geometries"
            raise ValueError(message)

    @property
    def z(self):
        """Return the z location of point geometries in a GeoSeries"""
        raise ValueError("no z coords")

    @property
    def bounds(self):
        # return shapely.bounds(self._data)
        raise NotImplementedError

    @property
    def total_bounds(self):
        # if len(self) == 0:
        #     # numpy 'min' cannot handle empty arrays
        #     # TODO with numpy >= 1.15, the 'initial' argument can be used
        #     return np.array([np.nan, np.nan, np.nan, np.nan])
        # b = self.bounds
        # with warnings.catch_warnings():
        #     # if all rows are empty geometry / none, nan is expected
        #     warnings.filterwarnings(
        #         "ignore", r"All-NaN slice encountered", RuntimeWarning
        #     )
        #     return np.array(
        #         (
        #             np.nanmin(b[:, 0]),  # minx
        #             np.nanmin(b[:, 1]),  # miny
        #             np.nanmax(b[:, 2]),  # maxx
        #             np.nanmax(b[:, 3]),  # maxy
        #         )
        #     )
        raise NotImplementedError

    # -------------------------------------------------------------------------
    # general array like compat
    # -------------------------------------------------------------------------

    @property
    def size(self):
        return self._data.size

    @property
    def shape(self):
        return (self.size,)

    @property
    def ndim(self):
        return len(self.shape)

    def copy(self, *args, **kwargs):
        # still taking args/kwargs for compat with pandas 0.24
        return GeographyArray(self._data.copy(), crs=self._crs)

    def take(self, indices, allow_fill=False, fill_value=None):
        from pandas.api.extensions import take

        if allow_fill:
            if fill_value is None or pd.isna(fill_value):
                fill_value = None
            elif not _is_scalar_geometry(fill_value):
                raise TypeError("provide geometry or None as fill value")

        result = take(self._data, indices, allow_fill=allow_fill, fill_value=fill_value)
        # TODO check why this is needed
        # if allow_fill and fill_value is None:
        #     result[~shapely.is_valid_input(result)] = None
        return GeographyArray(result, crs=self.crs)

    # compat for pandas < 3.0
    def _pad_or_backfill(
        self, method, limit=None, limit_area=None, copy=True, **kwargs
    ):
        if PANDAS_GE_21 and not PANDAS_GE_22:
            if limit_area is not None:
                # limit area not supported, but, but we feed through
                # so the caller gets the pandas exception
                kwargs["limit_area"] = limit_area
        else:
            kwargs["limit_area"] = limit_area
        return super()._pad_or_backfill(method=method, limit=limit, copy=copy, **kwargs)

    def fillna(self, value=None, method=None, limit=None, copy=True):
        """
        Fill NA values with geometry (or geometries) or using the specified method.

        Parameters
        ----------
        value : shapely geometry object or GeometryArray
            If a geometry value is passed it is used to fill all missing values.
            Alternatively, an GeometryArray 'value' can be given. It's expected
            that the GeometryArray has the same length as 'self'.

        method : {'backfill', 'bfill', 'pad', 'ffill', None}, default None
            Method to use for filling holes in reindexed Series
            pad / ffill: propagate last valid observation forward to next valid
            backfill / bfill: use NEXT valid observation to fill gap

        limit : int, default None
            The maximum number of entries where NA values will be filled.

        copy : bool, default True
            Whether to make a copy of the data before filling. If False, then
            the original should be modified and no new memory should be allocated.

        Returns
        -------
        GeometryArray
        """
        if method is not None:
            raise NotImplementedError("fillna with a method is not yet supported")

        if copy:
            new_values = self.copy()
        else:
            new_values = self

        # spherely does not yet support missing values
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
            string_values = spherely.to_wkt(self._data)
            pd_dtype = pd.api.types.pandas_dtype(dtype)
            if isinstance(pd_dtype, pd.StringDtype):
                # ensure to return a pandas string array instead of numpy array
                return pd.array(string_values, dtype=pd_dtype)
            return string_values.astype(dtype, copy=False)
        else:
            # numpy 2.0 makes copy=False case strict (errors if cannot avoid the copy)
            # -> in that case use `np.asarray` as backwards compatible alternative
            # for `copy=None` (when requiring numpy 2+, this can be cleaned up)
            if not copy:
                return np.asarray(self, dtype=dtype)
            else:
                return np.array(self, dtype=dtype, copy=copy)

    def isna(self):
        """
        Boolean NumPy array indicating if each value is missing
        """
        # spherely does not yet support missing values
        return np.full(self.shape, dtype="bool", fill_value=False)

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
        values = self.to_wkb()
        from pandas import Index, Series

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
        return self._data.nbytes

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
        if isinstance(scalars, spherely.Geography):
            scalars = [scalars]
        return from_spherely(scalars)

    @classmethod
    def _from_sequence_of_strings(cls, strings, *, dtype=None, copy=False):
        """
        Construct a new ExtensionArray from a sequence of strings.

        Parameters
        ----------
        strings : Sequence
            Each element will be an instance of the scalar type for this
            array, ``cls.dtype.type``.
        dtype : dtype, optional
            Construct for this particular dtype. This should be a Dtype
            compatible with the ExtensionArray.
        copy : bool, default False
            If True, copy the underlying data.

        Returns
        -------
        ExtensionArray
        """
        # GH 3099
        return from_wkt(strings)

    def _values_for_factorize(self):
        # type: () -> Tuple[np.ndarray, Any]
        """Return an array and missing value suitable for factorization.

        Returns
        -------
        values : ndarray
            An array suitable for factorization. This should maintain order
            and be a supported dtype (Float64, Int64, UInt64, String, Object).
            By default, the extension array is cast to object dtype.
        na_value : object
            The value in `values` to consider missing. This will be treated
            as NA in the factorization routines, so it will be coded as
            `na_sentinal` and not included in `uniques`. By default,
            ``np.nan`` is used.
        """
        vals = self.to_wkb()
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
        from geopandas.tools.hilbert_curve import _hilbert_distance

        if self.size == 0:
            # TODO _hilbert_distance fails for empty array
            return np.array([], dtype="uint32")

        mask_empty = self.is_empty
        has_empty = mask_empty.any()
        mask = self.isna() | mask_empty
        if mask.any():
            # if there are missing or empty geometries, we fill those with
            # a dummy geometry so that the _hilbert_distance function can
            # process those. The missing values are handled separately by
            # pandas regardless of the values we return here (to sort
            # first/last depending on 'na_position'), the distances for the
            # empty geometries are substitued below with an appropriate value
            geoms = self.copy()
            indices = np.nonzero(~mask)[0]
            if indices.size:
                geom = self[indices[0]]
            else:
                # for all-empty/NA, just take random geometry
                geom = shapely.geometry.Point(0, 0)

            geoms[mask] = geom
        else:
            geoms = self
        if has_empty:
            # in case we have empty geometries, we need to expand the total
            # bounds with a small percentage, so the empties can be
            # deterministically sorted first
            total_bounds = geoms.total_bounds
            xoff = (total_bounds[2] - total_bounds[0]) * 0.01
            yoff = (total_bounds[3] - total_bounds[1]) * 0.01
            total_bounds += np.array([-xoff, -yoff, xoff, yoff])
        else:
            total_bounds = None
        distances = _hilbert_distance(geoms, total_bounds=total_bounds)
        if has_empty:
            # empty geometries are sorted first ("smallest"), so fill in
            # smallest possible value for uints
            distances[mask_empty] = 0
        return distances

    def argmin(self, skipna: bool = True) -> int:
        raise TypeError("geometries have no minimum or maximum")

    def argmax(self, skipna: bool = True) -> int:
        raise TypeError("geometries have no minimum or maximum")

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
                precision = 5
            return lambda geom: spherely.to_wkt(geom, precision=precision)
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
        data = np.concatenate([ga._data for ga in to_concat])
        return GeographyArray(data, crs=_get_common_crs(to_concat))

    def _reduce(self, name, skipna=True, keepdims=False, **kwargs):
        # including the base class version here (that raises by default)
        # because this was not yet defined in pandas 0.23
        if name in ("any", "all"):
            return getattr(self._data, name)(keepdims=keepdims)
        raise TypeError(
            f"'{type(self).__name__}' with dtype {self.dtype} "
            f"does not support reduction '{name}'"
        )

    def __array__(self, dtype=None, copy=None):
        """
        The numpy array interface.

        Returns
        -------
        values : numpy array
        """
        if copy and (dtype is None or dtype == np.dtype("object")):
            return self._data.copy()
        return self._data

    def _binop(self, other, op):
        def convert_values(param):
            if not _is_scalar_geometry(param) and (
                isinstance(param, ExtensionArray) or pd.api.types.is_list_like(param)
            ):
                ovalues = param
            else:  # Assume its an object
                ovalues = [param] * len(self)
            return ovalues

        if isinstance(other, pd.Series | pd.Index | pd.DataFrame):
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
        if isna(item):
            if (
                item is self.dtype.na_value
                or isinstance(item, self.dtype.type)
                or item is None
            ):
                return self.isna().any()
            else:
                return False
        return (self == item).any()


def _get_common_crs(arr_seq):
    # mask out all None arrays with no crs (most likely auto generated by pandas
    # from concat with missing column)
    arr_seq = [ga for ga in arr_seq if not (ga.isna().all() and ga.crs is None)]
    # determine unique crs without using a set, because CRS hash can be different
    # for objects with the same CRS
    unique_crs = []
    for arr in arr_seq:
        if arr.crs not in unique_crs:
            unique_crs.append(arr.crs)

    crs_not_none = [crs for crs in unique_crs if crs is not None]
    names = [crs.name for crs in crs_not_none]

    if len(crs_not_none) == 0:
        return None
    if len(crs_not_none) == 1:
        if len(unique_crs) != 1:
            warnings.warn(
                "CRS not set for some of the concatenation inputs. "
                f"Setting output's CRS as {names[0]} "
                "(the single non-null crs provided).",
                stacklevel=2,
            )
        return crs_not_none[0]

    raise ValueError(
        f"Cannot determine common CRS for concatenation inputs, got {names}. "
        "Use `to_crs()` to transform geometries to the same CRS before merging."
    )


def transform(data, func):
    has_z = shapely.has_z(data)

    result = np.empty_like(data)

    coords = shapely.get_coordinates(data[~has_z], include_z=False)
    new_coords_z = func(coords[:, 0], coords[:, 1])
    result[~has_z] = shapely.set_coordinates(
        data[~has_z].copy(), np.array(new_coords_z).T
    )

    coords_z = shapely.get_coordinates(data[has_z], include_z=True)
    new_coords_z = func(coords_z[:, 0], coords_z[:, 1], coords_z[:, 2])
    result[has_z] = shapely.set_coordinates(
        data[has_z].copy(), np.array(new_coords_z).T
    )

    return result
