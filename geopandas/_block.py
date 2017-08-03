from __future__ import absolute_import, division, print_function

import numpy as np

from pandas.core.internals import Block, NonConsolidatableMixIn
from pandas.core.common import is_null_slice
from shapely.geometry.base import geom_factory, BaseGeometry

from .vectorized import VectorizedGeometry, to_shapely


class GeometryBlock(NonConsolidatableMixIn, Block):
    """ implement a datetime64 block with a tz attribute """
    __slots__ = ()

    @property
    def _holder(self):
        return VectorizedGeometry

    def __init__(self, values, placement, ndim=2, **kwargs):

        if not isinstance(values, self._holder):
            raise TypeError("values must be a VectorizedGeometry object")

        super(GeometryBlock, self).__init__(values, placement=placement,
                                            ndim=ndim, **kwargs)

    @property
    def _box_func(self):
        # TODO does not seems to be used at the moment (from the examples) ?
        print("I am boxed")
        return geom_factory

    # @property
    # def _na_value(self):
    #     return None
    #
    # @property
    # def fill_value(self):
    #     return tslib.iNaT

    # TODO
    # def copy(self, deep=True, mgr=None):
    #     """ copy constructor """
    #     values = self.values
    #     if deep:
    #         values = values.copy(deep=True)
    #     return self.make_block_same_class(values)

    def external_values(self):
        """ we internally represent the data as a DatetimeIndex, but for
        external compat with ndarray, export as a ndarray of Timestamps
        """
        #return np.asarray(self.values)
        print("I am densified (external_values, {} elements)".format(len(self)))
        return self.values.to_dense()

    def formatting_values(self, dtype=None):
        """ return an internal format, currently just the ndarray
        this should be the pure internal API format
        """
        return self.to_dense()

    def to_dense(self):
        print("I am densified ({} elements)".format(len(self)))
        return self.values.to_dense().view()

    def _getitem(self, key):
        values = self.values[key]
        return GeometryBlock(values, placement=slice(0, len(values), 1),
                             ndim=1)

    # TODO is this needed?
    # def get_values(self, dtype=None):
    #     """
    #     return object dtype as boxed values, as shapely objects
    #     """
    #     if is_object_dtype(dtype):
    #         return lib.map_infer(self.values.ravel(),
    #                              self._box_func).reshape(self.values.shape)
    #     return self.values

    def to_native_types(self, slicer=None, na_rep=None, date_format=None,
                        quoting=None, **kwargs):
        """ convert to our native types format, slicing if desired """

        values = self.values
        if slicer is not None:
            values = values[slicer]

        values = to_shapely(values.data)

        return np.atleast_2d(values)

    # TODO needed for what?
    def _can_hold_element(self, element):
        # if is_list_like(element):
        #     element = np.array(element)
        #     return element.dtype == _NS_DTYPE or element.dtype == np.int64
        return isinstance(element, BaseGeometry)

    def _slice(self, slicer):
        """ return a slice of my values """
        print("I am sliced")
        if isinstance(slicer, tuple):
            col, loc = slicer
            if not is_null_slice(col) and col != 0:
                raise IndexError("{0} only contains one item".format(self))
            return self.values[loc]
        return self.values[slicer]

    def take_nd(self, indexer, axis=0, new_mgr_locs=None, fill_tuple=None):
        """
        Take values according to indexer and return them as a block.bb
        """
        print("I am in take_nd")
        if fill_tuple is None:
            fill_value = None
        else:
            fill_value = fill_tuple[0]

        # axis doesn't matter; we are really a single-dim object
        # but are passed the axis depending on the calling routing
        # if its REALLY axis 0, then this will be a reindex and not a take

        # TODO implement take_nd on VectorizedGeometry
        # new_values = self.values.take_nd(indexer, fill_value=fill_value)
        new_values = self.values[indexer]

        # if we are a 1-dim object, then always place at 0
        if self.ndim == 1:
            new_mgr_locs = [0]
        else:
            if new_mgr_locs is None:
                new_mgr_locs = self.mgr_locs

        return self.make_block_same_class(new_values, new_mgr_locs)

    def eval(self, func, other, raise_on_error=True, try_cast=False,
             mgr=None):
        if func.__name__ == 'eq':
            super(GeometryBlock, self).eval(
                func, other, raise_on_error=raise_on_error, try_cast=try_cast,
                mgr=mgr)
        raise TypeError("{} not supported on geometry blocks".format(func.__name__))


    def _astype(self, dtype, copy=False, errors='raise', values=None,
                klass=None, mgr=None):
        """
        Coerce to the new type (if copy=True, return a new copy)
        raise on an except if raise == True
        """

        if dtype != np.object_:
            if errors == 'raise':
                raise TypeError('cannot astype geometries')
        values = self.values

        if copy:
            values = values.copy()

        return self.make_block(values)

    # def should_store(self, value):
    #     return (issubclass(value.dtype.type, np.uint64)
    #             and value.dtype == self.dtype)

    def set(self, locs, values, check=False):
        """
        Modify Block in-place with new item value

        Returns
        -------
        None
        """
        if values.dtype != self.dtype:
            # Workaround for numpy 1.6 bug
            if isinstance(values, BaseGeometry):
                values = values.__geom__
            else:
                raise ValueError()

            self.values[locs] = values

    # def _astype(self, dtype, mgr=None, **kwargs):
    #     """
    #     these automatically copy, so copy=True has no effect
    #     raise on an except if raise == True
    #     """
    #
    #     # if we are passed a datetime64[ns, tz]
    #     if is_datetime64tz_dtype(dtype):
    #         dtype = DatetimeTZDtype(dtype)
    #
    #         values = self.values
    #         if getattr(values, 'tz', None) is None:
    #             values = DatetimeIndex(values).tz_localize('UTC')
    #         values = values.tz_convert(dtype.tz)
    #         return self.make_block(values)
    #
    #     # delegate
    #     return super(DatetimeBlock, self)._astype(dtype=dtype, **kwargs)


    # def _try_coerce_args(self, values, other):
    #     """
    #     Coerce values and other to dtype 'i8'. NaN and NaT convert to
    #     the smallest i8, and will correctly round-trip to NaT if converted
    #     back in _try_coerce_result. values is always ndarray-like, other
    #     may not be
    #
    #     Parameters
    #     ----------
    #     values : ndarray-like
    #     other : ndarray-like or scalar
    #
    #     Returns
    #     -------
    #     base-type values, values mask, base-type other, other mask
    #     """
    #
    #     values_mask = isna(values)
    #     values = values.view('i8')
    #     other_mask = False
    #
    #     if isinstance(other, bool):
    #         raise TypeError
    #     elif is_null_datelike_scalar(other):
    #         other = tslib.iNaT
    #         other_mask = True
    #     elif isinstance(other, (datetime, np.datetime64, date)):
    #         other = self._box_func(other)
    #         if getattr(other, 'tz') is not None:
    #             raise TypeError("cannot coerce a Timestamp with a tz on a "
    #                             "naive Block")
    #         other_mask = isna(other)
    #         other = other.asm8.view('i8')
    #     elif hasattr(other, 'dtype') and is_datetime64_dtype(other):
    #         other_mask = isna(other)
    #         other = other.astype('i8', copy=False).view('i8')
    #     else:
    #         # coercion issues
    #         # let higher levels handle
    #         raise TypeError
    #
    #     return values, values_mask, other, other_mask
    #
    # def _try_coerce_result(self, result):
    #     """ reverse of try_coerce_args """
    #     if isinstance(result, np.ndarray):
    #         if result.dtype.kind in ['i', 'f', 'O']:
    #             try:
    #                 result = result.astype('M8[ns]')
    #             except ValueError:
    #                 pass
    #     elif isinstance(result, (np.integer, np.float, np.datetime64)):
    #         result = self._box_func(result)
    #     return result
    #
    # def _try_coerce_args(self, values, other):
    #     """
    #     localize and return i8 for the values
    #
    #     Parameters
    #     ----------
    #     values : ndarray-like
    #     other : ndarray-like or scalar
    #
    #     Returns
    #     -------
    #     base-type values, values mask, base-type other, other mask
    #     """
    #     values_mask = _block_shape(isna(values), ndim=self.ndim)
    #     # asi8 is a view, needs copy
    #     values = _block_shape(values.asi8, ndim=self.ndim)
    #     other_mask = False
    #
    #     if isinstance(other, ABCSeries):
    #         other = self._holder(other)
    #         other_mask = isna(other)
    #
    #     if isinstance(other, bool):
    #         raise TypeError
    #     elif (is_null_datelike_scalar(other) or
    #           (is_scalar(other) and isna(other))):
    #         other = tslib.iNaT
    #         other_mask = True
    #     elif isinstance(other, self._holder):
    #         if other.tz != self.values.tz:
    #             raise ValueError("incompatible or non tz-aware value")
    #         other = other.asi8
    #         other_mask = isna(other)
    #     elif isinstance(other, (np.datetime64, datetime, date)):
    #         other = lib.Timestamp(other)
    #         tz = getattr(other, 'tz', None)
    #
    #         # test we can have an equal time zone
    #         if tz is None or str(tz) != str(self.values.tz):
    #             raise ValueError("incompatible or non tz-aware value")
    #         other_mask = isna(other)
    #         other = other.value
    #     else:
    #         raise TypeError
    #
    #     return values, values_mask, other, other_mask
    #
    # def _try_coerce_result(self, result):
    #     """ reverse of try_coerce_args """
    #     if isinstance(result, np.ndarray):
    #         if result.dtype.kind in ['i', 'f', 'O']:
    #             result = result.astype('M8[ns]')
    #     elif isinstance(result, (np.integer, np.float, np.datetime64)):
    #         result = lib.Timestamp(result, tz=self.values.tz)
    #     if isinstance(result, np.ndarray):
    #         # allow passing of > 1dim if its trivial
    #         if result.ndim > 1:
    #             result = result.reshape(np.prod(result.shape))
    #         result = self.values._shallow_copy(result)
    #
    #     return result
    #

    # def shift(self, periods, axis=0, mgr=None):
    #     """ shift the block by periods """
    #
    #     # think about moving this to the DatetimeIndex. This is a non-freq
    #     # (number of periods) shift ###
    #
    #     N = len(self)
    #     indexer = np.zeros(N, dtype=int)
    #     if periods > 0:
    #         indexer[periods:] = np.arange(N - periods)
    #     else:
    #         indexer[:periods] = np.arange(-periods, N)
    #
    #     new_values = self.values.asi8.take(indexer)
    #
    #     if periods > 0:
    #         new_values[:periods] = tslib.iNaT
    #     else:
    #         new_values[periods:] = tslib.iNaT
    #
    #     new_values = self.values._shallow_copy(new_values)
    #     return [self.make_block_same_class(new_values,
    #                                        placement=self.mgr_locs)]
