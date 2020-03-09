""""Spatial joins for GeoPandas

The general algorithm for these spatial join functions is:
 1. Use a spatial index (rtree in the current implementation) to roughly compute
    the results. Because rtree operates on bounding boxes, this result is not final
    and must be refined, but refines the search to a much smaller subset of geometries
    in most cases.
 2. Use basic geometric operations on each of the matches from the spatial index
    query to get an exact result. This operation can be slow since it is iterative.
    In the future, this may be sped up by vectorizing operations, see PR#1154.

In order to avoid duplication, functionality required by all (current) spatial
join implementations has been moved to helper methods, resulting in the following
overall flow:
 1. Input checks: delegated to _basic_checks()
 2. Rename left_df and right_df indexes for compatibility with rtree (which only
   works with numeric indexes): delegated to _rename_indexes()
 3. Compute raw result of spatial join (i.e. which indexes match) using the
    algorithm mentioned above (and handle the "op" parameter in the case of
    sjoin): this happens within the public user-facing sjoin_... functions.
 4. Take the raw result from (3) and join left_df and right_df according to the
    matches and the "how" parameter: delegated to _join_results.

Currently, 2 types of spatial join are implemented:
 * sjoin: spatial join with basic binary predicates (intersection, contains and
   within)
 * sjoin_nearest: matches the nearest geometries.
"""

from warnings import warn

from distutils.version import LooseVersion

import numpy as np
import pandas as pd

from shapely import prepared

from geopandas import GeoDataFrame, base


RTREE_VERSION = ""  # string to match expected type from rtree.__version__
if base.HAS_SINDEX:
    import rtree

    RTREE_VERSION = rtree.__version__


def sjoin(
    left_df, right_df, how="inner", op="intersects", lsuffix="left", rsuffix="right"
):
    """Spatial join of two GeoDataFrames based on binary predicates.

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
    # ------------------------------ CHECK INPUTS ------------------------------
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

    _basic_checks(left_df, right_df)

    # ----------------- RENAME INDEXES FOR RTREE COMPATIBILITY -----------------
    (
        left_df,
        right_df,
        left_index_name,
        right_index_name,
        index_left,
        index_right,
    ) = _rename_indexes(left_df, right_df, lsuffix, rsuffix)

    # ------------------------ COMPUTE SPATIAL JOIN ----------------------------

    # Attempt to re-use spatial indexes, otherwise generate the spatial index
    # for the longer dataframe. If we are joining to an empty dataframe,
    # don't bother generating the index
    if right_df._sindex_generated or (
        not left_df._sindex_generated and right_df.shape[0] > left_df.shape[0]
    ):
        tree_idx = right_df.sindex if len(left_df) > 0 else None
        tree_idx_right = True
    else:
        tree_idx = left_df.sindex if len(right_df) > 0 else None
        tree_idx_right = False

    if op == "within":
        # within implemented as the inverse of contains; swap names
        left_df, right_df = right_df, left_df
        tree_idx_right = not tree_idx_right

    r_idx = np.empty((0, 0))
    l_idx = np.empty((0, 0))
    # get rtree spatial index. If tree_idx does not exist, it is due to either a
    # failure to generate the index (e.g., if the column is empty), or the
    # other dataframe is empty so it wasn't necessary to generate it.
    if tree_idx_right and tree_idx:
        idxmatch = left_df.geometry.apply(lambda x: x.bounds).apply(
            lambda x: list(tree_idx.intersection(x)) if not x == () else []
        )
        idxmatch = idxmatch[idxmatch.apply(len) > 0]
        # indexes of overlapping boundaries
        if idxmatch.shape[0] > 0:
            r_idx = np.concatenate(idxmatch.values)
            l_idx = np.concatenate([[i] * len(v) for i, v in idxmatch.iteritems()])
    elif not tree_idx_right and tree_idx:
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
        # Vectorize predicate operations (no added speed, just convenience)
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

    # ------------------- HANDLE HOW PARAM, CREATE FINAL DF --------------------
    return _join_results(
        result,
        left_df,
        right_df,
        how,
        left_index_name,
        right_index_name,
        index_left,
        index_right,
        lsuffix,
        rsuffix,
    )


def sjoin_nearest(
    left_df,
    right_df,
    how="inner",
    lsuffix="left",
    rsuffix="right",
    search_radius=None,
    max_search_neighbors=50,
    nearest_distances=False,
):
    """Spatial join of two GeoDataFrames, matching by nearest neighbor.
    Results can be restricted to a radius using the search_radius parameter.
    If the search is not bounded by a radius, execution can be sped up by
      using the max_search_neighbors parameter.
    Mixing both options usually does not improve performance over using only
     search_radius and is not recommended.
    See below for details on choosing a value for max_search_neighbors.

    Parameters
    ----------
    left_df, right_df : GeoDataFrames
    how : string, default 'inner'
        The type of join:

        * 'left': use keys from left_df; retain only left_df geometry column
        * 'right': use keys from right_df; retain only right_df geometry column
        * 'inner': use intersection of keys from both dfs; retain only
          left_df geometry column
    lsuffix : string, default 'left'
        Suffix to apply to overlapping column names (left GeoDataFrame).
    rsuffix : string, default 'right'
        Suffix to apply to overlapping column names (right GeoDataFrame).
    max_search_neighbors : int, default 50
        Number of nearest neighbors to check for proximity.
        Useful if you do not want to use search_radius.
        note:: using a very small number (~< 10) can cause unexpected results
          due to the implementation of the underlaying spatial index.
          Geometries are converted to bounding boxes, thus results for the
            nearest geometry may unexpected. Using a larger number mitigates
            this, but using too large of a number will provide little performance
            improvement.
          If it is not too slow, you may want to run at least once with
            max_search_neighbors=None (or at least a larger number) to validate
            results for your data.
    search_radius : int or float, default None
        Restricts search to a certain radius.
        This can significantly speed up execution.
        If using a tight search radius, it is recommended that you do not use the
          max_search_neighbors option as it will not improve performance.
        If you are not using search_radius, consider using max_search_neighbors
          to speed up execution.
    nearest_distances: bool, default False
        If True, report the distance for each match in new column
          named "nearest_distances".
    """
    # ------------------------------ CHECK INPUTS ------------------------------
    allowed_hows = ["left", "right", "inner"]
    if how not in allowed_hows:
        raise ValueError(
            '`how` was "%s" but is expected to be in %s' % (how, allowed_hows)
        )

    _basic_checks(left_df, right_df)

    # ----------------- RENAME INDEXES FOR RTREE COMPATIBILITY -----------------
    (
        left_df,
        right_df,
        left_index_name,
        right_index_name,
        index_left,
        index_right,
    ) = _rename_indexes(left_df, right_df, lsuffix, rsuffix)

    # ------------------------ COMPUTE SPATIAL JOIN ----------------------------

    # get spatial index
    tree_idx = right_df.sindex if len(right_df) > 0 else None

    # validate max_search_neighbors and search_radius params
    if max_search_neighbors is not None:
        if not isinstance(max_search_neighbors, int) or max_search_neighbors < 1:
            # warn about using rtree < 0.9.4 and max_search_neighbors option
            # see https://github.com/Toblerity/rtree/pull/141
            raise ValueError("max_search_neighbors must be an integer and >= 1")

        if str(RTREE_VERSION) < LooseVersion("0.9.4"):
            warn(
                "Using an rtree version < 0.9.4 may cause inconsistent "
                "results when using max_search_neighbors. Consider using a "
                "large number or max_search_neighbors=None."
            )

    if search_radius is not None and search_radius < 0:
        raise ValueError("search_radius must be >= 0")

    def _query_index(geo_in_l, buff_geo_in_l):
        """
        Queries the spatial index to filter results.
        """
        # restric by radius
        if search_radius is not None:
            in_radius = list(tree_idx.intersection(buff_geo_in_l))
            check_radius = True
        else:
            in_radius = right_df.index
            check_radius = False
        # find neighbors
        if max_search_neighbors is not None:
            neighbors = list(
                tree_idx.nearest(geo_in_l.bounds, num_results=max_search_neighbors)
            )
            check_neighbors = True
        else:
            neighbors = right_df.index
            check_neighbors = False

        if check_radius and check_neighbors:
            return np.array(
                [int(idx) for idx in neighbors if idx in in_radius], dtype=int
            )
        elif check_radius:
            return np.array(in_radius, dtype=int)
        elif check_neighbors:
            return np.array(neighbors, dtype=int)
        else:
            return right_df.index  # check all indexes in right_df

    if tree_idx is not None:
        # the accuracy of the spatial index is limited, so we need to manually
        # check each set of matches for actual distance
        l_idx = []
        r_idx = []
        distances = []

        # pre-buffer the bounds of right_df geometries by search radius
        # these will be used by the spatial index
        if search_radius is not None:
            bbox_delta = np.array(
                [-search_radius, -search_radius, search_radius, search_radius]
            )  # minx, miny, maxx, maxy
            buff_geos_in_l = (left_df.geometry.bounds + bbox_delta).values
        else:
            buff_geos_in_l = left_df.geometry.bounds.values

        for ind_in_left, (buff_geo_in_l, geo_in_l) in enumerate(
            zip(buff_geos_in_l, left_df.geometry)
        ):
            if not geo_in_l or geo_in_l.is_empty:
                # see https://github.com/Toblerity/Shapely/issues/799
                continue
            possible_nearest = _query_index(geo_in_l, buff_geo_in_l)
            min_dist = np.inf  # initialize
            min_ind = []  # initialize
            for ind_in_right, geo_in_r in zip(
                possible_nearest, right_df.geometry.iloc[possible_nearest],
            ):
                if not geo_in_r or geo_in_r.is_empty:
                    # see https://github.com/Toblerity/Shapely/issues/799
                    continue
                dist = geo_in_l.distance(geo_in_r)
                if dist < min_dist and (search_radius is None or dist <= search_radius):
                    # new closest
                    min_dist = dist  # reset
                    min_ind = [ind_in_right]  # re-initialize
                elif dist == min_dist:
                    # matching closest, extend list of closest
                    min_ind.append(ind_in_right)
            # extend final results
            if min_ind:  # possible_nearest may have been empty
                r_idx.extend(min_ind)
                l_idx.extend([ind_in_left] * len(min_ind))
                if nearest_distances:  # avoid memory use if unwarrented
                    distances.extend([min_dist] * len(min_ind))

        if len(r_idx) > 0 and len(l_idx) > 0:
            # assemble resultant df
            result = pd.DataFrame(np.column_stack([l_idx, r_idx]))
            result.columns = ["_key_left", "_key_right"]

        if nearest_distances:
            result["nearest_distances"] = distances

    if tree_idx is None or len(l_idx) == 0 or len(r_idx) == 0:
        # when output from the join has no overlapping geometries
        result = pd.DataFrame(columns=["_key_left", "_key_right"], dtype=float)

    # ------------------- HANDLE HOW PARAM, CREATE FINAL DF --------------------
    return _join_results(
        result,
        left_df,
        right_df,
        how,
        left_index_name,
        right_index_name,
        index_left,
        index_right,
        lsuffix,
        rsuffix,
    )


def _basic_checks(left_df, right_df):
    """
    Helper method for other sjoin methods.
    Runs type checks and crs checks.

    Parameters
    ----------
    left_df, right_df : GeoDataFrame
        The geodataframe's being joined.
    """
    if not isinstance(left_df, GeoDataFrame):
        raise ValueError(
            "'left_df' should be GeoDataFrame, got {}".format(type(left_df))
        )

    if not isinstance(right_df, GeoDataFrame):
        raise ValueError(
            "'right_df' should be GeoDataFrame, got {}".format(type(right_df))
        )

    if left_df.crs != right_df.crs:
        warn(
            (
                "CRS of frames being joined does not match!"
                "(%s != %s)" % (left_df.crs, right_df.crs)
            )
        )

    # check that rtree is installed
    if not RTREE_VERSION:
        raise RuntimeError("Spatial joins require `rtree`.")


def _rename_indexes(left_df, right_df, lsuffix, rsuffix):
    """
    Helper method for other sjoin methods.
    Renames indexes to numeric for compatibility with rtree.
    Returns renamed DataFrames as well as old indexes and their names.

    Parameters
    ----------
    left_df, right_df : GeoDataFrame
        The geodataframe's being joined.
    lsuffix : string, default 'left'
        Suffix to apply to overlapping column names (left GeoDataFrame).
    rsuffix : string, default 'right'
        Suffix to apply to overlapping column names (right GeoDataFrame).

    Returns
    -------
    left_df, right_df : GeoDataFrame
        The geodataframe's being joined, with indexes renamed.
    index_left : string
        Original index for left_df, to be restored by _join_results.
    index_right : string
        Original index for right_df, to be restored by _join_results
    left_index_name : string
        Original index name for left_df, to be restored by _join_results
    right_index_name : string
        Original index name for right_df, to be restored by _join_results
    """

    # store index names
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

    return left_df, right_df, left_index_name, right_index_name, index_left, index_right


def _join_results(
    result,
    left_df,
    right_df,  # data
    how,  # how to join the dataframes
    left_index_name,
    right_index_name,  # original names for the ind
    index_left,
    index_right,  # original indexes
    lsuffix,
    rsuffix,  # suffixes for merged column names
):
    """
    Helper method for other sjoin methods.
    Takes result DataFrame and handles the application of the "how" parameter.

    Parameters
    ----------
    result: GeoDataFrame
        The result of applying the spatial join operation, containing columns
        "_key_left" and "_key_right" inidicating which entries to match.
    left_df, right_df : GeoDataFrame
        The geodataframe's being joined, with indexes renamed.
    how : string
        The type of join:

        * 'left': use keys from left_df; retain only left_df geometry column
        * 'right': use keys from right_df; retain only right_df geometry column
        * 'inner': use intersection of keys from both dfs; retain only
          left_df geometry column
    op : string, default 'intersection'
        Binary predicate, one of {'intersects', 'contains', 'within'}.
        See http://shapely.readthedocs.io/en/latest/manual.html#binary-predicates.
    left_index_name : string
        Original index name for left_df to be restored.
    right_index_name : string
        Original index name for right_df to be restored.
    lsuffix : string, default 'left'
        Suffix to apply to overlapping column names (left GeoDataFrame).
    rsuffix : string, default 'right'
        Suffix to apply to overlapping column names (right GeoDataFrame).

    Returns
    -------
    joined: DataFram
        The final result of a spatial join.
    """

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
