from __future__ import absolute_import, division, print_function

import numpy as np
import pandas as pd

from ..geodataframe import GeoDataFrame
from ..geodataframe import GeoSeries

from ..vectorized import cysjoin


def sjoin(left_df, right_df, op='intersects', how='inner',
          lsuffix='left', rsuffix='right'):
    """Spatial join of two GeoDataFrames.

    Parameters
    ----------
    left_df, right_df : GeoDataFrames
    how : string, default 'inner'
        The type of join:

        * 'left': use keys from left_df; retain only left_df geometry column
        * 'right': use keys from right_df; retain only right_df geometry column
        * 'inner': use intersection of keys from both dfs; retain only
          left_df geometry column
    op : string, default 'intersection'
        Binary predicate, one of {'intersects', 'contains', 'within'}.
        See http://toblerity.org/shapely/manual.html#binary-predicates.
    lsuffix : string, default 'left'
        Suffix to apply to overlapping column names (left GeoDataFrame).
    rsuffix : string, default 'right'
        Suffix to apply to overlapping column names (right GeoDataFrame).
    """
    allowed_hows = ('left', 'right', 'inner')
    if how not in allowed_hows:
        raise ValueError("How keyword should be one of %s, got %s"
                         % (allowed_hows, how))

    if left_df.crs != right_df.crs:
        print("Warning: CRS does not match")

    left_indices, right_indices = cysjoin(left_df.geometry._geometry_array.data,
                                          right_df.geometry._geometry_array.data,
                                          op)
    n = len(left_indices)

    if how == 'left':
        missing = pd.Index(np.arange(len(left_df))).difference(pd.Index(left_indices))
        if len(missing):
            left_indices = np.concatenate([left_indices, missing])

    if how == 'right':
        missing = pd.Index(np.arange(len(right_df))).difference(pd.Index(right_indices))
        if len(missing):
            right_indices = np.concatenate([right_indices, missing])

    n_left = len(left_indices)
    n_right = len(right_indices)

    left = left_df.take(left_indices)
    right = right_df.take(right_indices)

    if how in ('inner', 'left'):
        del right[right._geometry_column_name]
        index = left.index
    else:
        del left[left._geometry_column_name]
        index = right.index

    columns = {}
    names = []
    for name, series in left.iteritems():
        if name in right.columns:
            name = name + '_' + lsuffix
        series.index = index[:n_left]
        if how == 'right':
            new = series.iloc[:0].reindex(index[n:])
            series = pd.concat([series, new], axis=0)
        columns[name] = series
        names.append(name)
    for name, series in right.iteritems():
        if name in left.columns:
            name = name + '_' + rsuffix
        series.index = index[:n_right]
        if how == 'left':
            new = series.iloc[:0].reindex(index[n:])
            series = pd.concat([series, new], axis=0)
        columns[name] = series
        names.append(name)

    if how in ('inner', 'left'):
        series = pd.Series(right.index.values, index=index[:n_right])
        new = series.iloc[:0].reindex(index[n:])
        series = pd.concat([series, new], axis=0)
        series.index = index
        columns['right_index'] = series
        names.append('right_index')
        geo_name = left_df._geometry_column_name
    else:
        series = pd.Series(left.index.values, index=index[:n_left])
        new = series.iloc[:0].reindex(index[n:])
        series = pd.concat([series, new], axis=0)
        series.index = index
        columns['left_index'] = series
        names.append('left_index')
        geo_name = right_df._geometry_column_name

    geometry = columns[geo_name]
    geometry = GeoSeries(geometry._values, index=index, name=geometry.name)
    columns[geo_name] = geometry

    return GeoDataFrame(columns, columns=names, index=index, geometry=geo_name)
