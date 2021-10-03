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
    >>> countries = geopandas.read_file(geopandas.datasets.get_\
path("naturalearth_lowres"))
    >>> cities = geopandas.read_file(geopandas.datasets.get_path("naturalearth_cities"))
    >>> countries.head()  # doctest: +SKIP
        pop_est      continent                      name \
iso_a3  gdp_md_est                                           geometry
    0     920938        Oceania                      Fiji    FJI      8374.0  MULTIPOLY\
GON (((180.00000 -16.06713, 180.00000...
    1   53950935         Africa                  Tanzania    TZA    150600.0  POLYGON (\
(33.90371 -0.95000, 34.07262 -1.05982...
    2     603253         Africa                 W. Sahara    ESH       906.5  POLYGON (\
(-8.66559 27.65643, -8.66512 27.58948...
    3   35623680  North America                    Canada    CAN   1674000.0  MULTIPOLY\
GON (((-122.84000 49.00000, -122.9742...
    4  326625791  North America  United States of America    USA  18560000.0  MULTIPOLY\
GON (((-122.84000 49.00000, -120.0000...
    >>> cities.head()
            name                   geometry
    0  Vatican City  POINT (12.45339 41.90328)
    1    San Marino  POINT (12.44177 43.93610)
    2         Vaduz   POINT (9.51667 47.13372)
    3    Luxembourg   POINT (6.13000 49.61166)
    4       Palikir  POINT (158.14997 6.91664)

    >>> cities_w_country_data = geopandas.sjoin(cities, countries)
    >>> cities_w_country_data.head()  # doctest: +SKIP
            name_left                   geometry  index_right   pop_est continent name_\
right iso_a3  gdp_md_est
    0    Vatican City  POINT (12.45339 41.90328)          141  62137802    Europe      \
Italy    ITA   2221000.0
    1      San Marino  POINT (12.44177 43.93610)          141  62137802    Europe      \
Italy    ITA   2221000.0
    192          Rome  POINT (12.48131 41.89790)          141  62137802    Europe      \
Italy    ITA   2221000.0
    2           Vaduz   POINT (9.51667 47.13372)          114   8754413    Europe    Au\
stria    AUT    416600.0
    184        Vienna  POINT (16.36469 48.20196)          114   8754413    Europe    Au\
stria    AUT    416600.0

    See also
    --------
    overlay : overlay operation resulting in a new geometry
    GeoDataFrame.sjoin : equivalent method

    Notes
    ------
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
                " The value of `predicate` will be overriden by the value of `op`,"
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
        l_idx, r_idx = sindex.query_bulk(input_geoms, predicate=predicate, sort=False)
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
    if not (compat.PYGEOS_GE_010 and compat.USE_PYGEOS):
        raise NotImplementedError(
            "Currently, only PyGEOS >= 0.10.0 supports `nearest_all`. "
            + compat.INSTALL_PYGEOS_ERROR
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
    >>> countries = geopandas.read_file(geopandas.datasets.get_\
path("naturalearth_lowres"))
    >>> cities = geopandas.read_file(geopandas.datasets.get_path("naturalearth_cities"))
    >>> countries.head(2).name  # doctest: +SKIP
        pop_est      continent                      name \
iso_a3  gdp_md_est                                           geometry
    0     920938        Oceania                      Fiji    FJI      8374.0  MULTIPOLY\
GON (((180.00000 -16.06713, 180.00000...
    1   53950935         Africa                  Tanzania    TZA    150600.0  POLYGON (\
(33.90371 -0.95000, 34.07262 -1.05982...
    >>> cities.head(2).name  # doctest: +SKIP
            name                   geometry
    0  Vatican City  POINT (12.45339 41.90328)
    1    San Marino  POINT (12.44177 43.93610)

    >>> cities_w_country_data = geopandas.sjoin_nearest(cities, countries)
    >>> cities_w_country_data[['name_left', 'name_right']].head(2)  # doctest: +SKIP
            name_left                   geometry  index_right   pop_est continent name_\
right iso_a3  gdp_md_est
    0    Vatican City  POINT (12.45339 41.90328)          141  62137802    Europe      \
Italy    ITA   2221000.0
    1      San Marino  POINT (12.44177 43.93610)          141  62137802    Europe      \
Italy    ITA   2221000.0

    To include the distances:

    >>> cities_w_country_data = geopandas.sjoin_nearest\
(cities, countries, distance_col="distances")
    >>> cities_w_country_data[["name_left", "name_right", \
"distances"]].head(2)  # doctest: +SKIP
            name_left name_right distances
    0    Vatican City      Italy       0.0
    1      San Marino      Italy       0.0

    In the following example, we get multiple cities for Italy because all results are
    equidistant (in this case zero because they intersect).
    In fact, we get 3 results in total:

    >>> countries_w_city_data = geopandas.sjoin_nearest\
(cities, countries, distance_col="distances", how="right")
    >>> italy_results = \
countries_w_city_data[countries_w_city_data["name_left"] == "Italy"]
    >>> italy_results  # doctest: +SKIP
         name_x        name_y
    141  Vatican City  Italy
    141    San Marino  Italy
    141          Rome  Italy

    See also
    --------
    sjoin : binary predicate joins
    GeoDataFrame.sjoin_nearest : equivalent method

    Notes
    -----
    Since this join relies on distances, results will be innaccurate
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
