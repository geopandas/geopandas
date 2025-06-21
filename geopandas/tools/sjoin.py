import warnings
from functools import partial

import numpy as np
import pandas as pd

from geopandas import GeoDataFrame
from geopandas._compat import PANDAS_GE_30
from geopandas.array import _check_crs, _crs_mismatch_warn


def sjoin(
    left_df,
    right_df,
    how="inner",
    predicate="intersects",
    lsuffix="left",
    rsuffix="right",
    distance=None,
    on_attribute=None,
    **kwargs,
):
    """Spatial join of two GeoDataFrames.

    See the User Guide page :doc:`../../user_guide/mergingdata` for details.


    Parameters
    ----------
    left_df, right_df : GeoDataFrames
    how : string, default 'inner'
        The type of join:

        * 'left': use keys from left_df; retain only left_df geometry column
        * 'right': use keys from right_df; retain only right_df geometry column
        * 'inner': use intersection of keys from both dfs; retain only
          left_df geometry column
    predicate : string, default 'intersects'
        Binary predicate. Valid values are determined by the spatial index used.
        You can check the valid values in left_df or right_df as
        ``left_df.sindex.valid_query_predicates`` or
        ``right_df.sindex.valid_query_predicates``
        Replaces deprecated ``op`` parameter.
    lsuffix : string, default 'left'
        Suffix to apply to overlapping column names (left GeoDataFrame).
    rsuffix : string, default 'right'
        Suffix to apply to overlapping column names (right GeoDataFrame).
    distance : number or array_like, optional
        Distance(s) around each input geometry within which to query the tree
        for the 'dwithin' predicate. If array_like, must be
        one-dimesional with length equal to length of left GeoDataFrame.
        Required if ``predicate='dwithin'``.
    on_attribute : string, list or tuple
        Column name(s) to join on as an additional join restriction on top
        of the spatial predicate. These must be found in both DataFrames.
        If set, observations are joined only if the predicate applies
        and values in specified columns match.

    Examples
    --------
    >>> import geodatasets
    >>> chicago = geopandas.read_file(
    ...     geodatasets.get_path("geoda.chicago_health")
    ... )
    >>> groceries = geopandas.read_file(
    ...     geodatasets.get_path("geoda.groceries")
    ... ).to_crs(chicago.crs)

    >>> chicago.head()  # doctest: +SKIP
        ComAreaID  ...                                           geometry
    0         35  ...  POLYGON ((-87.60914 41.84469, -87.60915 41.844...
    1         36  ...  POLYGON ((-87.59215 41.81693, -87.59231 41.816...
    2         37  ...  POLYGON ((-87.62880 41.80189, -87.62879 41.801...
    3         38  ...  POLYGON ((-87.60671 41.81681, -87.60670 41.816...
    4         39  ...  POLYGON ((-87.59215 41.81693, -87.59215 41.816...
    [5 rows x 87 columns]

    >>> groceries.head()  # doctest: +SKIP
        OBJECTID     Ycoord  ...  Category                         geometry
    0        16  41.973266  ...       NaN  MULTIPOINT (-87.65661 41.97321)
    1        18  41.696367  ...       NaN  MULTIPOINT (-87.68136 41.69713)
    2        22  41.868634  ...       NaN  MULTIPOINT (-87.63918 41.86847)
    3        23  41.877590  ...       new  MULTIPOINT (-87.65495 41.87783)
    4        27  41.737696  ...       NaN  MULTIPOINT (-87.62715 41.73623)
    [5 rows x 8 columns]

    >>> groceries_w_communities = geopandas.sjoin(groceries, chicago)
    >>> groceries_w_communities.head()  # doctest: +SKIP
       OBJECTID       community                           geometry
    0        16          UPTOWN  MULTIPOINT ((-87.65661 41.97321))
    1        18     MORGAN PARK  MULTIPOINT ((-87.68136 41.69713))
    2        22  NEAR WEST SIDE  MULTIPOINT ((-87.63918 41.86847))
    3        23  NEAR WEST SIDE  MULTIPOINT ((-87.65495 41.87783))
    4        27         CHATHAM  MULTIPOINT ((-87.62715 41.73623))
    [5 rows x 95 columns]

    See Also
    --------
    overlay : overlay operation resulting in a new geometry
    GeoDataFrame.sjoin : equivalent method

    Notes
    -----
    Every operation in GeoPandas is planar, i.e. the potential third
    dimension is not taken into account.
    """
    if kwargs:
        first = next(iter(kwargs.keys()))
        raise TypeError(f"sjoin() got an unexpected keyword argument '{first}'")

    on_attribute = _maybe_make_list(on_attribute)

    _basic_checks(left_df, right_df, how, lsuffix, rsuffix, on_attribute=on_attribute)

    indices = _geom_predicate_query(
        left_df, right_df, predicate, distance, on_attribute=on_attribute
    )

    joined, _ = _frame_join(
        left_df,
        right_df,
        indices,
        None,
        how,
        lsuffix,
        rsuffix,
        predicate,
        on_attribute=on_attribute,
    )

    return joined


def _maybe_make_list(obj):
    if isinstance(obj, tuple):
        return list(obj)
    if obj is not None and not isinstance(obj, list):
        return [obj]
    return obj


def _basic_checks(left_df, right_df, how, lsuffix, rsuffix, on_attribute=None):
    """Check the validity of join input parameters.

    `how` must be one of the valid options.
    `'index_'` concatenated with `lsuffix` or `rsuffix` must not already
    exist as columns in the left or right data frames.

    Parameters
    ----------
    left_df : GeoDataFrame
    right_df : GeoData Frame
    how : str, one of 'left', 'right', 'inner'
        join type
    lsuffix : str
        left index suffix
    rsuffix : str
        right index suffix
    on_attribute : list, default None
        list of column names to merge on along with geometry
    """
    if not isinstance(left_df, GeoDataFrame):
        raise ValueError(f"'left_df' should be GeoDataFrame, got {type(left_df)}")

    if not isinstance(right_df, GeoDataFrame):
        raise ValueError(f"'right_df' should be GeoDataFrame, got {type(right_df)}")

    allowed_hows = ["left", "right", "inner"]
    if how not in allowed_hows:
        raise ValueError(f'`how` was "{how}" but is expected to be in {allowed_hows}')

    if not _check_crs(left_df, right_df):
        _crs_mismatch_warn(left_df, right_df, stacklevel=4)

    if on_attribute:
        for attr in on_attribute:
            if (attr not in left_df) and (attr not in right_df):
                raise ValueError(
                    f"Expected column {attr} is missing from both of the dataframes."
                )
            if attr not in left_df:
                raise ValueError(
                    f"Expected column {attr} is missing from the left dataframe."
                )
            if attr not in right_df:
                raise ValueError(
                    f"Expected column {attr} is missing from the right dataframe."
                )
            if attr in (left_df.geometry.name, right_df.geometry.name):
                raise ValueError(
                    "Active geometry column cannot be used as an input "
                    "for on_attribute parameter."
                )


def _geom_predicate_query(left_df, right_df, predicate, distance, on_attribute=None):
    """Compute geometric comparisons and get matching indices.

    Parameters
    ----------
    left_df : GeoDataFrame
    right_df : GeoDataFrame
    predicate : string
        Binary predicate to query.
    on_attribute: list, default None
        list of column names to merge on along with geometry


    Returns
    -------
    DataFrame
        DataFrame with matching indices in
        columns named `_key_left` and `_key_right`.
    """
    original_predicate = predicate

    if predicate == "within":
        # within is implemented as the inverse of contains
        # contains is a faster predicate
        # see discussion at https://github.com/geopandas/geopandas/pull/1421
        predicate = "contains"
        sindex = left_df.sindex
        input_geoms = right_df.geometry
    else:
        # all other predicates are symmetric
        # keep them the same
        sindex = right_df.sindex
        input_geoms = left_df.geometry

    if sindex:
        l_idx, r_idx = sindex.query(
            input_geoms, predicate=predicate, sort=False, distance=distance
        )
    else:
        # when sindex is empty / has no valid geometries
        l_idx, r_idx = np.array([], dtype=np.intp), np.array([], dtype=np.intp)

    if original_predicate == "within":
        # within is implemented as the inverse of contains
        # flip back the results
        r_idx, l_idx = l_idx, r_idx
        indexer = np.lexsort((r_idx, l_idx))
        l_idx = l_idx[indexer]
        r_idx = r_idx[indexer]

    if on_attribute:
        for attr in on_attribute:
            (l_idx, r_idx), _ = _filter_shared_attribute(
                left_df, right_df, l_idx, r_idx, attr
            )

    return l_idx, r_idx


def _reset_index_with_suffix(df, suffix, other):
    """
    Equivalent of df.reset_index(), but with adding 'suffix' to auto-generated
    column names.
    """
    index_original = df.index.names
    if PANDAS_GE_30:
        df_reset = df.reset_index()
    else:
        # we already made a copy of the dataframe in _frame_join before getting here
        df_reset = df
        df_reset.reset_index(inplace=True)
    column_names = df_reset.columns.to_numpy(copy=True)
    for i, label in enumerate(index_original):
        # if the original label was None, add suffix to auto-generated name
        if label is None:
            new_label = column_names[i]
            if "level" in new_label:
                # reset_index of MultiIndex gives "level_i" names, preserve the "i"
                lev = new_label.split("_")[1]
                new_label = f"index_{suffix}{lev}"
            else:
                new_label = f"index_{suffix}"
            # check new label will not be in other dataframe
            if new_label in df.columns or new_label in other.columns:
                raise ValueError(
                    f"'{new_label}' cannot be a column name in the frames being joined"
                )
            column_names[i] = new_label
    return df_reset, pd.Index(column_names)


def _process_column_names_with_suffix(
    left: pd.Index, right: pd.Index, suffixes, left_df, right_df
):
    """
    Add suffixes to overlapping labels (ignoring the geometry column).

    This is based on pandas' merge logic at https://github.com/pandas-dev/pandas/blob/
    a0779adb183345a8eb4be58b3ad00c223da58768/pandas/core/reshape/merge.py#L2300-L2370
    """
    to_rename = left.intersection(right)
    if len(to_rename) == 0:
        return left, right

    lsuffix, rsuffix = suffixes

    if not lsuffix and not rsuffix:
        raise ValueError(f"columns overlap but no suffix specified: {to_rename}")

    def renamer(x, suffix, geometry):
        if x in to_rename and x != geometry and suffix is not None:
            return f"{x}_{suffix}"
        return x

    lrenamer = partial(
        renamer,
        suffix=lsuffix,
        geometry=getattr(left_df, "_geometry_column_name", None),
    )
    rrenamer = partial(
        renamer,
        suffix=rsuffix,
        geometry=getattr(right_df, "_geometry_column_name", None),
    )

    # TODO retain index name?
    left_renamed = pd.Index([lrenamer(lab) for lab in left])
    right_renamed = pd.Index([rrenamer(lab) for lab in right])

    dups = []
    if not left_renamed.is_unique:
        # Only warn when duplicates are caused because of suffixes, already duplicated
        # columns in origin should not warn
        dups = left_renamed[(left_renamed.duplicated()) & (~left.duplicated())].tolist()
    if not right_renamed.is_unique:
        dups.extend(
            right_renamed[(right_renamed.duplicated()) & (~right.duplicated())].tolist()
        )
    # TODO turn this into an error (pandas has done so as well)
    if dups:
        warnings.warn(
            f"Passing 'suffixes' which cause duplicate columns {set(dups)} in the "
            f"result is deprecated and will raise a MergeError in a future version.",
            FutureWarning,
            stacklevel=4,
        )

    return left_renamed, right_renamed


def _restore_index(joined, index_names, index_names_original):
    """
    Set back the the original index columns, and restoring their name as `None`
    if they didn't have a name originally.
    """
    if PANDAS_GE_30:
        joined = joined.set_index(list(index_names))
    else:
        joined.set_index(list(index_names), inplace=True)

    # restore the fact that the index didn't have a name
    joined_index_names = list(joined.index.names)
    for i, label in enumerate(index_names_original):
        if label is None:
            joined_index_names[i] = None
    joined.index.names = joined_index_names
    return joined


def _adjust_indexers(indices, distances, original_length, how, predicate):
    """Adjust the indexers for the join based on the `how` parameter.

    The left/right indexers from the query represents an inner join.
    For a left or right join, we need to adjust them to include the rows
    that would not be present in an inner join.
    """
    # the indices represent an inner join, no adjustment needed
    if how == "inner":
        return indices, distances

    l_idx, r_idx = indices

    if how == "right":
        # re-sort so it is sorted by the right indexer
        indexer = np.lexsort((l_idx, r_idx))
        l_idx, r_idx = l_idx[indexer], r_idx[indexer]
        if distances is not None:
            distances = distances[indexer]

        # switch order
        r_idx, l_idx = l_idx, r_idx

    # determine which indices are missing and where they would need to be inserted
    idx = np.arange(original_length)
    l_idx_missing = idx[~np.isin(idx, l_idx)]
    insert_idx = np.searchsorted(l_idx, l_idx_missing)
    # for the left indexer, insert those missing indices
    l_idx = np.insert(l_idx, insert_idx, l_idx_missing)
    # for the right indexer, insert -1 -> to get missing values in pandas' reindexing
    r_idx = np.insert(r_idx, insert_idx, -1)
    # for the indices, already insert those missing values manually
    if distances is not None:
        distances = np.insert(distances, insert_idx, np.nan)

    if how == "right":
        # switch back
        l_idx, r_idx = r_idx, l_idx

    return (l_idx, r_idx), distances


def _frame_join(
    left_df,
    right_df,
    indices,
    distances,
    how,
    lsuffix,
    rsuffix,
    predicate,
    on_attribute=None,
):
    """Join the GeoDataFrames at the DataFrame level.

    Parameters
    ----------
    left_df : GeoDataFrame
    right_df : GeoDataFrame
    indices : tuple of ndarray
        Indices returned by the geometric join. Tuple with with integer
        indices representing the matches from `left_df` and `right_df`
        respectively.
    distances : ndarray, optional
        Passed trough and adapted based on the indices, if needed.
    how : string
        The type of join to use on the DataFrame level.
    lsuffix : string
        Suffix to apply to overlapping column names (left GeoDataFrame).
    rsuffix : string
        Suffix to apply to overlapping column names (right GeoDataFrame).
    on_attribute: list, default None
        list of column names to merge on along with geometry


    Returns
    -------
    GeoDataFrame
        Joined GeoDataFrame.
    """
    if on_attribute:  # avoid renaming or duplicating shared column
        right_df = right_df.drop(on_attribute, axis=1)

    if how in ("inner", "left"):
        right_df = right_df.drop(right_df.geometry.name, axis=1)
    else:  # how == 'right':
        left_df = left_df.drop(left_df.geometry.name, axis=1)

    left_df = left_df.copy(deep=False)
    left_nlevels = left_df.index.nlevels
    left_index_original = left_df.index.names
    left_df, left_column_names = _reset_index_with_suffix(left_df, lsuffix, right_df)

    right_df = right_df.copy(deep=False)
    right_nlevels = right_df.index.nlevels
    right_index_original = right_df.index.names
    right_df, right_column_names = _reset_index_with_suffix(right_df, rsuffix, left_df)

    # if conflicting names in left and right, add suffix
    left_column_names, right_column_names = _process_column_names_with_suffix(
        left_column_names,
        right_column_names,
        (lsuffix, rsuffix),
        left_df,
        right_df,
    )
    left_df.columns = left_column_names
    right_df.columns = right_column_names
    left_index = left_df.columns[:left_nlevels]
    right_index = right_df.columns[:right_nlevels]

    # perform join on the dataframes
    original_length = len(right_df) if how == "right" else len(left_df)
    (l_idx, r_idx), distances = _adjust_indexers(
        indices, distances, original_length, how, predicate
    )
    # the `take` method doesn't allow introducing NaNs with -1 indices
    # left = left_df.take(l_idx)
    # therefore we are using the private _reindex_with_indexers as workaround
    new_index = pd.RangeIndex(len(l_idx))
    left = left_df._reindex_with_indexers({0: (new_index, l_idx)})
    right = right_df._reindex_with_indexers({0: (new_index, r_idx)})
    if PANDAS_GE_30:
        kwargs = {}
    else:
        kwargs = dict(copy=False)
    joined = pd.concat([left, right], axis=1, **kwargs)

    if how in ("inner", "left"):
        joined = _restore_index(joined, left_index, left_index_original)
    else:  # how == 'right':
        joined = joined.set_geometry(right_df.geometry.name)
        joined = _restore_index(joined, right_index, right_index_original)

    return joined, distances


def _nearest_query(
    left_df: GeoDataFrame,
    right_df: GeoDataFrame,
    max_distance: float,
    how: str,
    return_distance: bool,
    exclusive: bool,
    on_attribute: list | None = None,
):
    # use the opposite of the join direction for the index
    use_left_as_sindex = how == "right"
    if use_left_as_sindex:
        sindex = left_df.sindex
        query = right_df.geometry
    else:
        sindex = right_df.sindex
        query = left_df.geometry
    if sindex:
        res = sindex.nearest(
            query,
            return_all=True,
            max_distance=max_distance,
            return_distance=return_distance,
            exclusive=exclusive,
        )
        if return_distance:
            (input_idx, tree_idx), distances = res
        else:
            (input_idx, tree_idx) = res
            distances = None
        if use_left_as_sindex:
            l_idx, r_idx = tree_idx, input_idx
            sort_order = np.argsort(l_idx, kind="stable")
            l_idx, r_idx = l_idx[sort_order], r_idx[sort_order]
            if distances is not None:
                distances = distances[sort_order]
        else:
            l_idx, r_idx = input_idx, tree_idx
    else:
        # when sindex is empty / has no valid geometries
        l_idx, r_idx = np.array([], dtype=np.intp), np.array([], dtype=np.intp)
        if return_distance:
            distances = np.array([], dtype=np.float64)
        else:
            distances = None

    if on_attribute:
        for attr in on_attribute:
            (l_idx, r_idx), shared_attribute_rows = _filter_shared_attribute(
                left_df, right_df, l_idx, r_idx, attr
            )
            distances = distances[shared_attribute_rows]

    return (l_idx, r_idx), distances


def _filter_shared_attribute(left_df, right_df, l_idx, r_idx, attribute):
    """Return the indices for the left and right dataframe that share the same entry
    in the attribute column.

    Also returns a Boolean `shared_attribute_rows` for rows with the same entry.
    """
    shared_attribute_rows = (
        left_df[attribute].iloc[l_idx].values == right_df[attribute].iloc[r_idx].values
    )

    l_idx = l_idx[shared_attribute_rows]
    r_idx = r_idx[shared_attribute_rows]
    return (l_idx, r_idx), shared_attribute_rows


def sjoin_nearest(
    left_df: GeoDataFrame,
    right_df: GeoDataFrame,
    how: str = "inner",
    max_distance: float | None = None,
    lsuffix: str = "left",
    rsuffix: str = "right",
    distance_col: str | None = None,
    exclusive: bool = False,
) -> GeoDataFrame:
    """Spatial join of two GeoDataFrames based on the distance between their geometries.

    Results will include multiple output records for a single input record
    where there are multiple equidistant nearest or intersected neighbors.

    Distance is calculated in CRS units and can be returned using the
    `distance_col` parameter.

    See the User Guide page
    https://geopandas.readthedocs.io/en/latest/docs/user_guide/mergingdata.html
    for more details.


    Parameters
    ----------
    left_df, right_df : GeoDataFrames
    how : string, default 'inner'
        The type of join:

        * 'left': use keys from left_df; retain only left_df geometry column
        * 'right': use keys from right_df; retain only right_df geometry column
        * 'inner': use intersection of keys from both dfs; retain only
          left_df geometry column
    max_distance : float, default None
        Maximum distance within which to query for nearest geometry.
        Must be greater than 0.
        The max_distance used to search for nearest items in the tree may have a
        significant impact on performance by reducing the number of input
        geometries that are evaluated for nearest items in the tree.
    lsuffix : string, default 'left'
        Suffix to apply to overlapping column names (left GeoDataFrame).
    rsuffix : string, default 'right'
        Suffix to apply to overlapping column names (right GeoDataFrame).
    distance_col : string, default None
        If set, save the distances computed between matching geometries under a
        column of this name in the joined GeoDataFrame.
    exclusive : bool, default False
        If True, the nearest geometries that are equal to the input geometry
        will not be returned, default False.

    Examples
    --------
    >>> import geodatasets
    >>> groceries = geopandas.read_file(
    ...     geodatasets.get_path("geoda.groceries")
    ... )
    >>> chicago = geopandas.read_file(
    ...     geodatasets.get_path("geoda.chicago_health")
    ... ).to_crs(groceries.crs)

    >>> chicago.head()  # doctest: +SKIP
       ComAreaID  ...                                           geometry
    0         35  ...  POLYGON ((-87.60914 41.84469, -87.60915 41.844...
    1         36  ...  POLYGON ((-87.59215 41.81693, -87.59231 41.816...
    2         37  ...  POLYGON ((-87.62880 41.80189, -87.62879 41.801...
    3         38  ...  POLYGON ((-87.60671 41.81681, -87.60670 41.816...
    4         39  ...  POLYGON ((-87.59215 41.81693, -87.59215 41.816...
    [5 rows x 87 columns]

    >>> groceries.head()  # doctest: +SKIP
       OBJECTID     Ycoord  ...  Category                           geometry
    0        16  41.973266  ...       NaN  MULTIPOINT ((-87.65661 41.97321))
    1        18  41.696367  ...       NaN  MULTIPOINT ((-87.68136 41.69713))
    2        22  41.868634  ...       NaN  MULTIPOINT ((-87.63918 41.86847))
    3        23  41.877590  ...       new  MULTIPOINT ((-87.65495 41.87783))
    4        27  41.737696  ...       NaN  MULTIPOINT ((-87.62715 41.73623))
    [5 rows x 8 columns]

    >>> groceries_w_communities = geopandas.sjoin_nearest(groceries, chicago)
    >>> groceries_w_communities[["Chain", "community", "geometry"]].head(2)
                   Chain    community                                geometry
    0     VIET HOA PLAZA       UPTOWN   MULTIPOINT ((1168268.672 1933554.35))
    1  COUNTY FAIR FOODS  MORGAN PARK  MULTIPOINT ((1162302.618 1832900.224))


    To include the distances:

    >>> groceries_w_communities = geopandas.sjoin_nearest(groceries, chicago, \
distance_col="distances")
    >>> groceries_w_communities[["Chain", "community", \
"distances"]].head(2)
                   Chain    community  distances
    0     VIET HOA PLAZA       UPTOWN        0.0
    1  COUNTY FAIR FOODS  MORGAN PARK        0.0

    In the following example, we get multiple groceries for Uptown because all
    results are equidistant (in this case zero because they intersect).
    In fact, we get 4 results in total:

    >>> chicago_w_groceries = geopandas.sjoin_nearest(groceries, chicago, \
distance_col="distances", how="right")
    >>> uptown_results = \
chicago_w_groceries[chicago_w_groceries["community"] == "UPTOWN"]
    >>> uptown_results[["Chain", "community"]]
                Chain community
    30  VIET HOA PLAZA    UPTOWN
    30      JEWEL OSCO    UPTOWN
    30          TARGET    UPTOWN
    30       Mariano's    UPTOWN

    See Also
    --------
    sjoin : binary predicate joins
    GeoDataFrame.sjoin_nearest : equivalent method

    Notes
    -----
    Since this join relies on distances, results will be inaccurate
    if your geometries are in a geographic CRS.

    Every operation in GeoPandas is planar, i.e. the potential third
    dimension is not taken into account.
    """
    _basic_checks(left_df, right_df, how, lsuffix, rsuffix)

    left_df.geometry.values.check_geographic_crs(stacklevel=1)
    right_df.geometry.values.check_geographic_crs(stacklevel=1)

    return_distance = distance_col is not None

    indices, distances = _nearest_query(
        left_df,
        right_df,
        max_distance,
        how,
        return_distance,
        exclusive,
    )
    joined, distances = _frame_join(
        left_df,
        right_df,
        indices,
        distances,
        how,
        lsuffix,
        rsuffix,
        None,
    )

    if return_distance:
        joined[distance_col] = distances

    return joined
