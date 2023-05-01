from typing import Optional
import warnings

import numpy as np
import pandas as pd

from geopandas import GeoDataFrame
from geopandas import _compat as compat
from geopandas.array import _check_crs, _crs_mismatch_warn


def sjoin(
    left_df,
    right_df,
    how="inner",
    predicate="intersects",
    lsuffix="left",
    rsuffix="right",
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
            OBJECTID     Ycoord     Xcoord  ... GonorrF GonorrM Tuberc
    0          16  41.973266 -87.657073  ...   170.8   468.7   13.6
    87        365  41.961707 -87.654058  ...   170.8   468.7   13.6
    90        373  41.963131 -87.656352  ...   170.8   468.7   13.6
    140       582  41.969131 -87.674882  ...   170.8   468.7   13.6
    1          18  41.696367 -87.681315  ...   800.5   741.1    2.6
    [5 rows x 95 columns]

    See also
    --------
    overlay : overlay operation resulting in a new geometry
    GeoDataFrame.sjoin : equivalent method

    Notes
    -----
    Every operation in GeoPandas is planar, i.e. the potential third
    dimension is not taken into account.
    """
    if "op" in kwargs:
        op = kwargs.pop("op")
        deprecation_message = (
            "The `op` parameter is deprecated and will be removed"
            " in a future release. Please use the `predicate` parameter"
            " instead."
        )
        if predicate != "intersects" and op != predicate:
            override_message = (
                "A non-default value for `predicate` was passed"
                f' (got `predicate="{predicate}"`'
                f' in combination with `op="{op}"`).'
                " The value of `predicate` will be overridden by the value of `op`,"
                " , which may result in unexpected behavior."
                f"\n{deprecation_message}"
            )
            warnings.warn(override_message, UserWarning, stacklevel=4)
        else:
            warnings.warn(deprecation_message, FutureWarning, stacklevel=4)
        predicate = op
    if kwargs:
        first = next(iter(kwargs.keys()))
        raise TypeError(f"sjoin() got an unexpected keyword argument '{first}'")

    _basic_checks(left_df, right_df, how, lsuffix, rsuffix)

    indices = _geom_predicate_query(left_df, right_df, predicate)

    joined = _frame_join(indices, left_df, right_df, how, lsuffix, rsuffix)

    return joined


def _basic_checks(left_df, right_df, how, lsuffix, rsuffix):
    """Checks the validity of join input parameters.

    `how` must be one of the valid options.
    `'index_'` concatenated with `lsuffix` or `rsuffix` must not already
    exist as columns in the left or right data frames.

    Parameters
    ------------
    left_df : GeoDataFrame
    right_df : GeoData Frame
    how : str, one of 'left', 'right', 'inner'
        join type
    lsuffix : str
        left index suffix
    rsuffix : str
        right index suffix
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
            '`how` was "{}" but is expected to be in {}'.format(how, allowed_hows)
        )

    if not _check_crs(left_df, right_df):
        _crs_mismatch_warn(left_df, right_df, stacklevel=4)

    index_left = "index_{}".format(lsuffix)
    index_right = "index_{}".format(rsuffix)

    # due to GH 352
    if any(left_df.columns.isin([index_left, index_right])) or any(
        right_df.columns.isin([index_left, index_right])
    ):
        raise ValueError(
            "'{0}' and '{1}' cannot be names in the frames being"
            " joined".format(index_left, index_right)
        )


def _geom_predicate_query(left_df, right_df, predicate):
    """Compute geometric comparisons and get matching indices.

    Parameters
    ----------
    left_df : GeoDataFrame
    right_df : GeoDataFrame
    predicate : string
        Binary predicate to query.

    Returns
    -------
    DataFrame
        DataFrame with matching indices in
        columns named `_key_left` and `_key_right`.
    """
    with warnings.catch_warnings():
        # We don't need to show our own warning here
        # TODO remove this once the deprecation has been enforced
        warnings.filterwarnings(
            "ignore", "Generated spatial index is empty", FutureWarning
        )

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
        l_idx, r_idx = sindex.query(input_geoms, predicate=predicate, sort=False)
        indices = pd.DataFrame({"_key_left": l_idx, "_key_right": r_idx})
    else:
        # when sindex is empty / has no valid geometries
        indices = pd.DataFrame(columns=["_key_left", "_key_right"], dtype=float)

    if original_predicate == "within":
        # within is implemented as the inverse of contains
        # flip back the results
        indices = indices.rename(
            columns={"_key_left": "_key_right", "_key_right": "_key_left"}
        )

    return indices


def _frame_join(join_df, left_df, right_df, how, lsuffix, rsuffix):
    """Join the GeoDataFrames at the DataFrame level.

    Parameters
    ----------
    join_df : DataFrame
        Indices and join data returned by the geometric join.
        Must have columns `_key_left` and `_key_right`
        with integer indices representing the matches
        from `left_df` and `right_df` respectively.
        Additional columns may be included and will be copied to
        the resultant GeoDataFrame.
    left_df : GeoDataFrame
    right_df : GeoDataFrame
    lsuffix : string
        Suffix to apply to overlapping column names (left GeoDataFrame).
    rsuffix : string
        Suffix to apply to overlapping column names (right GeoDataFrame).
    how : string
        The type of join to use on the DataFrame level.

    Returns
    -------
    GeoDataFrame
        Joined GeoDataFrame.
    """
    # the spatial index only allows limited (numeric) index types, but an
    # index in geopandas may be any arbitrary dtype. so reset both indices now
    # and store references to the original indices, to be reaffixed later.
    # GH 352
    index_left = "index_{}".format(lsuffix)
    left_df = left_df.copy(deep=True)
    try:
        left_index_name = left_df.index.name
        left_df.index = left_df.index.rename(index_left)
    except TypeError:
        index_left = [
            "index_{}".format(lsuffix + str(pos))
            for pos, ix in enumerate(left_df.index.names)
        ]
        left_index_name = left_df.index.names
        left_df.index = left_df.index.rename(index_left)
    left_df = left_df.reset_index()

    index_right = "index_{}".format(rsuffix)
    right_df = right_df.copy(deep=True)
    try:
        right_index_name = right_df.index.name
        right_df.index = right_df.index.rename(index_right)
    except TypeError:
        index_right = [
            "index_{}".format(rsuffix + str(pos))
            for pos, ix in enumerate(right_df.index.names)
        ]
        right_index_name = right_df.index.names
        right_df.index = right_df.index.rename(index_right)
    right_df = right_df.reset_index()

    # perform join on the dataframes
    if how == "inner":
        join_df = join_df.set_index("_key_left")
        joined = (
            left_df.merge(join_df, left_index=True, right_index=True)
            .merge(
                right_df.drop(right_df.geometry.name, axis=1),
                left_on="_key_right",
                right_index=True,
                suffixes=("_{}".format(lsuffix), "_{}".format(rsuffix)),
            )
            .set_index(index_left)
            .drop(["_key_right"], axis=1)
        )
        if isinstance(index_left, list):
            joined.index.names = left_index_name
        else:
            joined.index.name = left_index_name

    elif how == "left":
        join_df = join_df.set_index("_key_left")
        joined = (
            left_df.merge(join_df, left_index=True, right_index=True, how="left")
            .merge(
                right_df.drop(right_df.geometry.name, axis=1),
                how="left",
                left_on="_key_right",
                right_index=True,
                suffixes=("_{}".format(lsuffix), "_{}".format(rsuffix)),
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
                join_df.merge(
                    right_df, left_on="_key_right", right_index=True, how="right"
                ),
                left_index=True,
                right_on="_key_left",
                how="right",
                suffixes=("_{}".format(lsuffix), "_{}".format(rsuffix)),
            )
            .set_index(index_right)
            .drop(["_key_left", "_key_right"], axis=1)
            .set_geometry(right_df.geometry.name)
        )
        if isinstance(index_right, list):
            joined.index.names = right_index_name
        else:
            joined.index.name = right_index_name

    return joined


def _nearest_query(
    left_df: GeoDataFrame,
    right_df: GeoDataFrame,
    max_distance: float,
    how: str,
    return_distance: bool,
):
    if not (compat.USE_SHAPELY_20 or (compat.USE_PYGEOS and compat.PYGEOS_GE_010)):
        raise NotImplementedError(
            "Currently, only PyGEOS >= 0.10.0 or Shapely >= 2.0 supports "
            "`nearest_all`. " + compat.INSTALL_PYGEOS_ERROR
        )
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
        join_df = pd.DataFrame(
            {"_key_left": l_idx, "_key_right": r_idx, "distances": distances}
        )
    else:
        # when sindex is empty / has no valid geometries
        join_df = pd.DataFrame(
            columns=["_key_left", "_key_right", "distances"], dtype=float
        )
    return join_df


def sjoin_nearest(
    left_df: GeoDataFrame,
    right_df: GeoDataFrame,
    how: str = "inner",
    max_distance: Optional[float] = None,
    lsuffix: str = "left",
    rsuffix: str = "right",
    distance_col: Optional[str] = None,
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
        OBJECTID     Ycoord  ...  Category                         geometry
    0        16  41.973266  ...       NaN  MULTIPOINT (-87.65661 41.97321)
    1        18  41.696367  ...       NaN  MULTIPOINT (-87.68136 41.69713)
    2        22  41.868634  ...       NaN  MULTIPOINT (-87.63918 41.86847)
    3        23  41.877590  ...       new  MULTIPOINT (-87.65495 41.87783)
    4        27  41.737696  ...       NaN  MULTIPOINT (-87.62715 41.73623)
    [5 rows x 8 columns]

    >>> groceries_w_communities = geopandas.sjoin_nearest(groceries, chicago)
    >>> groceries_w_communities[["Chain", "community", "geometry"]].head(2)
                    Chain community                              geometry
    0   VIET HOA PLAZA    UPTOWN  MULTIPOINT (1168268.672 1933554.350)
    87      JEWEL OSCO    UPTOWN  MULTIPOINT (1168837.980 1929246.962)


    To include the distances:

    >>> groceries_w_communities = geopandas.sjoin_nearest(groceries, chicago, \
distance_col="distances")
    >>> groceries_w_communities[["Chain", "community", \
"distances"]].head(2)  # doctest: +SKIP
                    Chain community  distances
    0   VIET HOA PLAZA    UPTOWN        0.0
    87      JEWEL OSCO    UPTOWN        0.0

    In the following example, we get multiple groceries for Uptown because all
    results are equidistant (in this case zero because they intersect).
    In fact, we get 4 results in total:

    >>> chicago_w_groceries = geopandas.sjoin_nearest(groceries, chicago, \
distance_col="distances", how="right")
    >>> uptown_results = \
chicago_w_groceries[chicago_w_groceries["community"] == "UPTOWN"]
    >>> uptown_results[["Chain", "community"]]  # doctest: +SKIP
                Chain community
    30  VIET HOA PLAZA    UPTOWN
    30      JEWEL OSCO    UPTOWN
    30          TARGET    UPTOWN
    30       Mariano's    UPTOWN

    See also
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

    join_df = _nearest_query(left_df, right_df, max_distance, how, return_distance)

    if return_distance:
        join_df = join_df.rename(columns={"distances": distance_col})
    else:
        join_df.pop("distances")

    joined = _frame_join(join_df, left_df, right_df, how, lsuffix, rsuffix)

    if return_distance:
        columns = [c for c in joined.columns if c != distance_col] + [distance_col]
        joined = joined[columns]

    return joined
