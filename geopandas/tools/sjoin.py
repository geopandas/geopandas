import numpy as np
import pandas as pd

from geopandas import GeoDataFrame
from geopandas._compat import HAS_RTREE


def sjoin(
    left_df, right_df, how="inner", op="intersects", lsuffix="left", rsuffix="right"
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
        Binary predicate, one of {'intersects', 'contains', 'within'}.
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

    if not _check_crs(left_df, right_df):
        _crs_mismatch_warn(left_df, right_df, stacklevel=3)

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
    # for the longer dataframe. If we are joining to an empty dataframe,
    # don't bother generating the index.
    if right_df._sindex_generated or (
        not left_df._sindex_generated and right_df.shape[0] > left_df.shape[0]
    ):
        tree_idx = right_df.sindex if len(left_df) > 0 else None
        tree_idx_right = True
    else:
        tree_idx = left_df.sindex if len(right_df) > 0 else None
        tree_idx_right = False

    # the rtree spatial index only allows limited (numeric) index types, but an
    # index in geopandas may be any arbitrary dtype. so reset both indices now
    # and store references to the original indices, to be reaffixed later.
    # GH 352
    left_df = left_df.copy(deep=True)
    try:
        left_index_name = left_df.index.name
        left_df.index = left_df.index.rename(index_left)
    except TypeError:
        index_left = [
            "index_%s" % lsuffix + str(l) for l, ix in enumerate(left_df.index.names)
        ]
        left_index_name = left_df.index.names
        left_df.index = left_df.index.rename(index_left)
    left_df = left_df.reset_index()

    right_df = right_df.copy(deep=True)
    try:
        right_index_name = right_df.index.name
        right_df.index = right_df.index.rename(index_right)
    except TypeError:
        index_right = [
            "index_%s" % rsuffix + str(l) for l, ix in enumerate(right_df.index.names)
        ]
        right_index_name = right_df.index.names
        right_df.index = right_df.index.rename(index_right)
    right_df = right_df.reset_index()

    # for historical reasons, this logic is flipped in sjoin vs. pygeos query_bulk
    if op == "contains":
        op = "within"
    elif op == "within":
        op = "contains"

    r_idx = np.empty((0, 0))
    l_idx = np.empty((0, 0))
    # get rtree spatial index. If tree_idx does not exist, it is due to either a
    # failure to generate the index (e.g., if the column is empty), or the
    # other dataframe is empty so it wasn't necessary to generate it.
    if tree_idx_right and tree_idx:
        l_idx, r_idx = tree_idx.query_bulk(left_df.geometry, predicate=op, sort=False)
    elif not tree_idx_right and tree_idx:
        # tree_idx_df == 'left'
        r_idx, l_idx = tree_idx.query_bulk(right_df.geometry, predicate=op, sort=False)

    if r_idx.size > 0 and l_idx.size > 0:
        result = pd.DataFrame({"_key_left": l_idx, "_key_right": r_idx})
    else:
        # when output from the join has no overlapping geometries
        result = pd.DataFrame(columns=["_key_left", "_key_right"], dtype=float)

    if how == "inner":
        result = result.set_index("_key_left")
        joined = (
            left_df.merge(result, left_index=True, right_index=True)
            .merge(
                right_df.drop(right_df.geometry.name, axis=1),
                left_on="_key_right",
                right_index=True,
                suffixes=("_%s" % lsuffix, "_%s" % rsuffix),
            )
            .set_index(index_left)
            .drop(["_key_right"], axis=1)
        )
        if isinstance(index_left, list):
            joined.index.names = left_index_name
        else:
            joined.index.name = left_index_name

    elif how == "left":
        result = result.set_index("_key_left")
        joined = (
            left_df.merge(result, left_index=True, right_index=True, how="left")
            .merge(
                right_df.drop(right_df.geometry.name, axis=1),
                how="left",
                left_on="_key_right",
                right_index=True,
                suffixes=("_%s" % lsuffix, "_%s" % rsuffix),
            )
            .set_index(index_left)
            .drop(["_key_right"], axis=1)
        )
        if isinstance(index_left, list):
            joined.index.names = left_index_name
        else:
            joined.index.name = left_index_name

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
            .drop(["_key_left", "_key_right"], axis=1)
        )
        if isinstance(index_right, list):
            joined.index.names = right_index_name
        else:
            joined.index.name = right_index_name

    return joined
