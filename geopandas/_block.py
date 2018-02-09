from __future__ import absolute_import, division, print_function

import numpy as np

import pandas as pd
from pandas.core.internals import Block, NonConsolidatableMixIn, BlockManager
from pandas.core.common import is_null_slice
from shapely.geometry.base import geom_factory, BaseGeometry

from .array import GeometryArray, to_shapely


class GeometryBlock(NonConsolidatableMixIn, Block):
    """ Pandas Geometry block with pointers to C GEOS objects """
    __slots__ = ()

    @property
    def _holder(self):
        return GeometryArray

    def __init__(self, values, placement, ndim=2, **kwargs):
        print("Ai, I am constructing a GeometryBlock")

        if not isinstance(values, self._holder):
            raise TypeError("values must be a GeometryArray object")

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
        if fill_tuple is None:
            fill_value = None
        else:
            fill_value = fill_tuple[0]

        # axis doesn't matter; we are really a single-dim object
        # but are passed the axis depending on the calling routing
        # if its REALLY axis 0, then this will be a reindex and not a take

        # TODO implement take_nd on GeometryArray
        # new_values = self.values.take_nd(indexer, fill_value=fill_value)
        new_values = self.values.take(indexer)

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

        if dtype == np.object_:
            values = self.to_dense()
        elif dtype == str:
            values = np.array(list(map(str, self.to_dense())))
        else:
            if errors == 'raise':
                raise TypeError('cannot astype geometries')
            else:
                values = self.to_dense()

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

    def concat_same_type(self, to_concat, placement=None):
        """
        Concatenate list of single blocks of the same type.
        """
        from .geodataframe import _concat_arrays
        values = _concat_arrays([blk.values for blk in to_concat])
        return self.make_block_same_class(
            values, placement=placement or slice(0, len(values), 1))


def add_geometries(blk_mgr, geo_items):
    # blk_mgr: BlockManager, geo_items: Dict[name: GeoSeries] -> BlockManager
    columns, index = blk_mgr.axes
    blocks = blk_mgr.blocks
    for name, geom in geo_items.items():
        geom_block = geom._data._block
        geom_block = GeometryBlock(geom_block.values,
                                   slice(len(columns), len(columns) + 1))
        columns = columns.append(pd.Index([name]))
        blocks = blocks + (geom_block,)

    block_manager = BlockManager(blocks, [columns, index])
    return block_manager


def frame_set_geometry(frame, key, geoms):
    # frame: DataFrame, key: str, geoms: GeometryArray -> None (edit in place)

    # if column does not exist, add dummy non-consolidatable
    # (therefore categorical) column to create the correct final
    # BlockManager structure
    # do the same if column exists but is not yet a geometry
    if (key not in frame.columns
            or (key in frame.columns
                and not isinstance(frame[key]._data._block, GeometryBlock))):
        frame[key] = pd.Categorical.from_codes([-1]*len(frame), [])

    # determine location of the target GeometryBlock
    key_loc = frame.columns.get_loc(key)
    block_loc = frame._data._blknos[key_loc]

    columns, index = frame._data.axes
    blocks = frame._data.blocks
    blocks = list(blocks)

    # create new GeometryBlock and update existing blocks
    orig_geom_block = blocks[block_loc]
    new_geom_block = GeometryBlock(geoms, orig_geom_block.mgr_locs.as_slice)
    blocks[block_loc] = new_geom_block

    block_manager = BlockManager(blocks, [columns, index])
    frame._data = block_manager
    frame._clear_item_cache(key)
