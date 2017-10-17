from __future__ import absolute_import, division, print_function
from warnings import warn

from warnings import warn

import numpy as np
import pandas as pd

from shapely import prepared

from ..geodataframe import GeoDataFrame
from ..geodataframe import GeoSeries
from ..vectorized import cysjoin


def sjoin_old(
    left_df, right_df, op="intersects", how="inner", lsuffix="left", rsuffix="right"
):
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
        Binary predicate, such as 'intersects', 'contains', 'within'.
        See http://shapely.readthedocs.io/en/latest/manual.html#binary-predicates.
    lsuffix : string, default 'left'
        Suffix to apply to overlapping column names (left GeoDataFrame).
    rsuffix : string, default 'right'
        Suffix to apply to overlapping column names (right GeoDataFrame).
    """
    if not isinstance(left_df, GeoDataFrame):
        raise ValueError(
            "'left_df' should be GeoDataFrame, got {}".format(type(left_df))
        )

    if not isinstance(right_df, GeoDataFrame):
        raise ValueError(
            "'right_df' should be GeoDataFrame, got {}".format(type(right_df))
        )

    allowed_hows = ["left", "right", "inner"]
    if how not in allowed_hows:
        raise ValueError(
            '`how` was "%s" but is expected to be in %s' % (how, allowed_hows)
        )

    allowed_ops = ["contains", "within", "intersects"]
    if op not in allowed_ops:
        raise ValueError(
            '`op` was "%s" but is expected to be in %s' % (op, allowed_ops)
        )

    if left_df.crs != right_df.crs:
        warn(
            (
                "CRS of frames being joined does not match!"
                "(%s != %s)" % (left_df.crs, right_df.crs)
            )
        )

    index_left = "index_%s" % lsuffix
    index_right = "index_%s" % rsuffix

    # due to GH 352
    if any(left_df.columns.isin([index_left, index_right])) or any(
        right_df.columns.isin([index_left, index_right])
    ):
        raise ValueError(
            "'{0}' and '{1}' cannot be names in the frames being"
            " joined".format(index_left, index_right)
        )

    # Attempt to re-use spatial indexes, otherwise generate the spatial index
    # for the longer dataframe
    if right_df._sindex_generated or (
        not left_df._sindex_generated and right_df.shape[0] > left_df.shape[0]
    ):
        tree_idx = right_df.sindex
        tree_idx_right = True
    else:
        tree_idx = left_df.sindex
        tree_idx_right = False

    # the rtree spatial index only allows limited (numeric) index types, but an
    # index in geopandas may be any arbitrary dtype. so reset both indices now
    # and store references to the original indices, to be reaffixed later.
    # GH 352
    left_df = left_df.copy(deep=True)
    left_df.index = left_df.index.rename(index_left)
    left_df = left_df.reset_index()
    right_df = right_df.copy(deep=True)
    right_df.index = right_df.index.rename(index_right)
    right_df = right_df.reset_index()

    if op == "within":
        # within implemented as the inverse of contains; swap names
        left_df, right_df = right_df, left_df
        tree_idx_right = not tree_idx_right

    r_idx = np.empty((0, 0))
    l_idx = np.empty((0, 0))
    # get rtree spatial index
    if tree_idx_right:
        idxmatch = left_df.geometry.apply(lambda x: x.bounds).apply(
            lambda x: list(tree_idx.intersection(x)) if not x == () else []
        )
        idxmatch = idxmatch[idxmatch.apply(len) > 0]
        # indexes of overlapping boundaries
        if idxmatch.shape[0] > 0:
            r_idx = np.concatenate(idxmatch.values)
            l_idx = np.concatenate([[i] * len(v) for i, v in idxmatch.iteritems()])
    else:
        # tree_idx_df == 'left'
        idxmatch = right_df.geometry.apply(lambda x: x.bounds).apply(
            lambda x: list(tree_idx.intersection(x)) if not x == () else []
        )
        idxmatch = idxmatch[idxmatch.apply(len) > 0]
        if idxmatch.shape[0] > 0:
            # indexes of overlapping boundaries
            l_idx = np.concatenate(idxmatch.values)
            r_idx = np.concatenate([[i] * len(v) for i, v in idxmatch.iteritems()])

    if len(r_idx) > 0 and len(l_idx) > 0:
        # Vectorize predicate operations
        def find_intersects(a1, a2):
            return a1.intersects(a2)

        def find_contains(a1, a2):
            return a1.contains(a2)

        predicate_d = {
            "intersects": find_intersects,
            "contains": find_contains,
            "within": find_contains,
        }

        check_predicates = np.vectorize(predicate_d[op])

        result = pd.DataFrame(
            np.column_stack(
                [
                    l_idx,
                    r_idx,
                    check_predicates(
                        left_df.geometry.apply(lambda x: prepared.prep(x))[l_idx],
                        right_df[right_df.geometry.name][r_idx],
                    ),
                ]
            )
        )

        result.columns = ["_key_left", "_key_right", "match_bool"]
        result = pd.DataFrame(result[result["match_bool"] == 1]).drop(
            "match_bool", axis=1
        )

    else:
        # when output from the join has no overlapping geometries
        result = pd.DataFrame(columns=["_key_left", "_key_right"], dtype=float)

    if op == "within":
        # within implemented as the inverse of contains; swap names
        left_df, right_df = right_df, left_df
        result = result.rename(
            columns={"_key_left": "_key_right", "_key_right": "_key_left"}
        )

    if how == "inner":
        result = result.set_index("_key_left")
        joined = left_df.merge(result, left_index=True, right_index=True).merge(
            right_df.drop(right_df.geometry.name, axis=1),
            left_on="_key_right",
            right_index=True,
            suffixes=("_%s" % lsuffix, "_%s" % rsuffix),
        )
        joined = joined.set_index(index_left).drop(["_key_right"], axis=1)
        joined.index.name = None
    elif how == "left":
        result = result.set_index("_key_left")
        joined = left_df.merge(
            result, left_index=True, right_index=True, how="left"
        ).merge(
            right_df.drop(right_df.geometry.name, axis=1),
            how="left",
            left_on="_key_right",
            right_index=True,
            suffixes=("_%s" % lsuffix, "_%s" % rsuffix),
        )
        joined = joined.set_index(index_left).drop(["_key_right"], axis=1)
        joined.index.name = None
    else:  # how == 'right':
        joined = (
            left_df.drop(left_df.geometry.name, axis=1)
            .merge(
                result.merge(
                    right_df, left_on="_key_right", right_index=True, how="right"
                ),
                left_index=True,
                right_on="_key_left",
                how="right",
            )
            .set_index(index_right)
        )
        joined = joined.drop(["_key_left", "_key_right"], axis=1)

    return joined


def sjoin(
    left_df, right_df, op="intersects", how="inner", lsuffix="left", rsuffix="right"
):
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
        Binary predicate, such as 'intersects', 'contains', 'within'.
        See http://shapely.readthedocs.io/en/latest/manual.html#binary-predicates.
    lsuffix : string, default 'left'
        Suffix to apply to overlapping column names (left GeoDataFrame).
    rsuffix : string, default 'right'
        Suffix to apply to overlapping column names (right GeoDataFrame).
    """
    if not isinstance(left_df, GeoDataFrame):
        raise ValueError(
            "'left_df' should be GeoDataFrame, got {}".format(type(left_df))
        )

    if not isinstance(right_df, GeoDataFrame):
        raise ValueError(
            "'right_df' should be GeoDataFrame, got {}".format(type(right_df))
        )

    allowed_hows = ("left", "right", "inner")
    if how not in allowed_hows:
        raise ValueError(
            '`how` was "%s" but is expected to be in %s' % (how, allowed_hows)
        )

    original_op = op
    original_how = how
    if op == "within":
        # within implemented as the inverse of contains; swap names
        # This is done for efficiency reasons
        op = "contains"
        left_df, right_df = right_df, left_df
        if how == "left":
            how = "right"
        elif how == "right":
            how = "left"

    if left_df.crs != right_df.crs:
        warn(
            (
                "CRS of frames being joined does not match!"
                "(%s != %s)" % (left_df.crs, right_df.crs)
            )
        )

    left_indices, right_indices = cysjoin(
        left_df.geometry.array.data, right_df.geometry.array.data, op
    )
    n = len(left_indices)

    if how == "left":
        missing = pd.Index(np.arange(len(left_df))).difference(pd.Index(left_indices))
        if len(missing):
            left_indices = np.concatenate([left_indices, missing])

    if how == "right":
        missing = pd.Index(np.arange(len(right_df))).difference(pd.Index(right_indices))
        if len(missing):
            right_indices = np.concatenate([right_indices, missing])

    n_left = len(left_indices)
    n_right = len(right_indices)

    left = left_df.take(left_indices)
    right = right_df.take(right_indices)

    if original_op == "within":  # switch back
        left, right = right, left
        n_left, n_right = n_right, n_left
        how = original_how

    if how in ("inner", "left"):
        del right[right._geometry_column_name]
        index = left.index
    else:
        del left[left._geometry_column_name]
        index = right.index

    columns = {}
    names = []
    for name, series in left.iteritems():
        if name in right.columns:
            name = name + "_" + lsuffix
        series.index = index[:n_left]
        if how == "right":
            new = series.iloc[:0].reindex(index[n:])
            series = pd.concat([series, new], axis=0)
        columns[name] = series
        names.append(name)
    for name, series in right.iteritems():
        if name in left.columns:
            name = name + "_" + rsuffix
        series.index = index[:n_right]
        if how == "left":
            new = series.iloc[:0].reindex(index[n:])
            series = pd.concat([series, new], axis=0)
        columns[name] = series
        names.append(name)

    if how in ("inner", "left"):
        series = pd.Series(right.index.values, index=index[:n_right])
        new = series.iloc[:0].reindex(index[n:])
        series = pd.concat([series, new], axis=0)
        series.index = index
        columns["index_right"] = series
        names.append("index_right")
        geo_name = left_df._geometry_column_name
    else:
        series = pd.Series(left.index.values, index=index[:n_left])
        new = series.iloc[:0].reindex(index[n:])
        series = pd.concat([series, new], axis=0)
        series.index = index
        columns["index_left"] = series
        names.append("index_left")
        geo_name = right_df._geometry_column_name

    geometry = columns[geo_name]
    geometry = GeoSeries(geometry._values, index=index, name=geometry.name)
    columns[geo_name] = geometry

    return GeoDataFrame(columns, columns=names, index=index, geometry=geo_name)
