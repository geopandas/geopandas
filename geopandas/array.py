from collections.abc import Iterable
import numbers
import operator

import numpy as np
import pandas as pd
from pandas.api.extensions import ExtensionArray, ExtensionDtype

import shapely
import shapely.affinity
import shapely.geometry
from shapely.geometry.base import BaseGeometry
import shapely.ops
import shapely.wkt

try:
    import pygeos
except ImportError:
    geos = None

from . import _compat as compat
from . import _vectorized as vectorized


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


if compat.PANDAS_GE_024:
    from pandas.api.extensions import register_extension_dtype

    register_extension_dtype(GeometryDtype)


def _isna(value):
    """
    Check if scalar value is NA-like (None or np.nan).

    Custom version that only works for scalars (returning True or False),
    as `pd.isna` also works for array-like input returning a boolean array.
    """
    if value is None:
        return True
    elif isinstance(value, float) and np.isnan(value):
        return True
    else:
        return False


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
        return isinstance(geom, pygeos.Geometry)
    else:
        return isinstance(geom, BaseGeometry)


def from_shapely(data):
    """
    Convert a list or array of shapely objects to a GeometryArray.

    Validates the elements.
    """
    return GeometryArray(vectorized.from_shapely(data))


def to_shapely(geoms):
    """
    Convert GeometryArray to numpy object array of shapely objects.
    """
    if not isinstance(geoms, GeometryArray):
        raise ValueError("'geoms' must be a GeometryArray")
    return vectorized.to_shapely(geoms.data)


def from_wkb(data):
    """
    Convert a list or array of WKB objects to a GeometryArray.
    """
    return GeometryArray(vectorized.from_wkb(data))


def to_wkb(geoms):
    """
    Convert GeometryArray to a numpy object array of WKB objects.
    """
    if not isinstance(geoms, GeometryArray):
        raise ValueError("'geoms' must be a GeometryArray")
    return vectorized.to_wkb(geoms.data)


def from_wkt(data):
    """
    Convert a list or array of WKT objects to a GeometryArray.
    """
    return GeometryArray(vectorized.from_wkt(data))


def to_wkt(geoms, **kwargs):
    """
    Convert GeometryArray to a numpy object array of WKT objects.
    """
    if not isinstance(geoms, GeometryArray):
        raise ValueError("'geoms' must be a GeometryArray")
    return vectorized.to_wkt(geoms.data, **kwargs)


def points_from_xy(x, y, z=None):
    """
    Generate GeometryArray of shapely Point geometries from x, y(, z) coordinates.

    Parameters
    ----------
    x, y, z : iterable

    Examples
    --------
    >>> geometry = geopandas.points_from_xy(x=[1, 0], y=[0, 1])
    >>> geometry = geopandas.points_from_xy(df['x'], df['y'], df['z'])
    >>> gdf = geopandas.GeoDataFrame(
            df, geometry=geopandas.points_from_xy(df['x'], df['y']))

    Returns
    -------
    output : GeometryArray
    """
    return GeometryArray(vectorized.points_from_xy(x, y, z))


class GeometryArray(ExtensionArray):
    """
    Class wrapping a numpy array of Shapely objects and
    holding the array-based implementations.
    """

    _dtype = GeometryDtype()

    def __init__(self, data):
        if isinstance(data, self.__class__):
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
            return GeometryArray(self.data[idx])
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
        elif isinstance(value, BaseGeometry) or _isna(value):
            if _isna(value):
                # internally only use None as missing value indicator
                # but accept others
                value = None
            elif isinstance(value, BaseGeometry):
                value = from_shapely([value]).data[0]
            else:
                raise TypeError("should be valid geometry")
            if isinstance(key, (slice, list, np.ndarray)):
                value_array = np.empty(1, dtype=object)
                value_array[:] = [value]
                self.data[key] = value_array
            else:
                self.data[key] = value
        else:
            raise TypeError(
                "Value should be either a BaseGeometry or None, got %s" % str(value)
            )

    if compat.USE_PYGEOS:

        def __getstate__(self):
            return pygeos.to_wkb(self.data)

        def __setstate__(self, state):
            geoms = pygeos.from_wkb(state)
            self.data = geoms
            self.base = None

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
        return vectorized.area(self.data)

    @property
    def length(self):
        return vectorized.length(self.data)

    #
    # Unary operations that return new geometries
    #

    @property
    def boundary(self):
        return GeometryArray(vectorized.boundary(self.data))

    @property
    def centroid(self):
        return GeometryArray(vectorized.centroid(self.data))

    @property
    def convex_hull(self):
        return GeometryArray(vectorized.convex_hull(self.data))

    @property
    def envelope(self):
        return GeometryArray(vectorized.envelope(self.data))

    @property
    def exterior(self):
        return GeometryArray(vectorized.exterior(self.data))

    @property
    def interiors(self):
        # no GeometryArray as result
        return vectorized.interiors(self.data)

    def representative_point(self):
        return GeometryArray(vectorized.representative_point(self.data))

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
            right = right.data

        return getattr(vectorized, op)(left.data, right, **kwargs)

    def covers(self, other):
        return self._binary_method("covers", self, other)

    def contains(self, other):
        return self._binary_method("contains", self, other)

    def crosses(self, other):
        return self._binary_method("crosses", self, other)

    def disjoint(self, other):
        return self._binary_method("disjoint", self, other)

    def equals(self, other):
        return self._binary_method("equals", self, other)

    def intersects(self, other):
        return self._binary_method("intersects", self, other)

    def overlaps(self, other):
        return self._binary_method("overlaps", self, other)

    def touches(self, other):
        return self._binary_method("touches", self, other)

    def within(self, other):
        return self._binary_method("within", self, other)

    def equals_exact(self, other, tolerance):
        return self._binary_method("equals_exact", self, other, tolerance=tolerance)

    def almost_equals(self, other, decimal):
        return self.equals_exact(other, 0.5 * 10 ** (-decimal))
        # return _binary_predicate("almost_equals", self, other, decimal=decimal)

    #
    # Binary operations that return new geometries
    #

    def difference(self, other):
        return GeometryArray(self._binary_method("difference", self, other))

    def intersection(self, other):
        return GeometryArray(self._binary_method("intersection", self, other))

    def symmetric_difference(self, other):
        return GeometryArray(self._binary_method("symmetric_difference", self, other))

    def union(self, other):
        return GeometryArray(self._binary_method("union", self, other))

    #
    # Other operations
    #

    def distance(self, other):
        return self._binary_method("distance", self, other)

    def buffer(self, distance, resolution=16, **kwargs):
        return GeometryArray(
            vectorized.buffer(self.data, distance, resolution=resolution, **kwargs)
        )

    def interpolate(self, distance, normalized=False):
        return GeometryArray(
            vectorized.interpolate(self.data, distance, normalized=normalized)
        )

    def simplify(self, tolerance, preserve_topology=True):
        return GeometryArray(
            vectorized.simplify(
                self.data, tolerance, preserve_topology=preserve_topology
            )
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
            vectorized._affinity_method("affine_transform", self.data, matrix)
        )

    def translate(self, xoff=0.0, yoff=0.0, zoff=0.0):
        return GeometryArray(
            vectorized._affinity_method("translate", self.data, xoff, yoff, zoff)
        )

    def rotate(self, angle, origin="center", use_radians=False):
        return GeometryArray(
            vectorized._affinity_method(
                "rotate", self.data, angle, origin=origin, use_radians=use_radians
            )
        )

    def scale(self, xfact=1.0, yfact=1.0, zfact=1.0, origin="center"):
        return GeometryArray(
            vectorized._affinity_method(
                "scale", self.data, xfact, yfact, zfact, origin=origin
            )
        )

    def skew(self, xs=0.0, ys=0.0, origin="center", use_radians=False):
        return GeometryArray(
            vectorized._affinity_method(
                "skew", self.data, xs, ys, origin=origin, use_radians=use_radians
            )
        )

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
        return GeometryArray(self.data.copy())

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
        return GeometryArray(result)

    def _fill(self, idx, value):
        """ Fill index locations with value

        Value should be a BaseGeometry
        """
        if not (_is_scalar_geometry(value) or value is None):
            raise TypeError(
                "Value should be either a BaseGeometry or None, got %s" % str(value)
            )
        # self.data[idx] = value
        self.data[idx] = np.array([value], dtype=object)
        return self

    def fillna(self, value=None, method=None, limit=None):
        """ Fill NA/NaN values using the specified method.

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
            if _isna(value):
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
            return to_wkt(self).astype(dtype, copy=False)
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
        return from_wkb(values)

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
                # dummy heuristic based on 10 first geometries that should
                # work in most cases
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
        return GeometryArray(data)

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
            if isinstance(param, ExtensionArray) or pd.api.types.is_list_like(param):
                ovalues = param
            else:  # Assume its an object
                ovalues = [param] * len(self)
            return ovalues

        if isinstance(other, (pd.Series, pd.Index)):
            # rely on pandas to unbox and dispatch to us
            return NotImplemented

        lvalues = self
        rvalues = convert_values(other)

        if len(lvalues) != len(rvalues):
            raise ValueError("Lengths must match to compare")

        # If the operator is not defined for the underlying objects,
        # a TypeError should be raised
        res = [op(a, b) for (a, b) in zip(lvalues, rvalues)]

        res = np.asarray(res)
        return res

    def __eq__(self, other):
        return self._binop(other, operator.eq)

    def __ne__(self, other):
        return self._binop(other, operator.ne)
