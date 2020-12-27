import warnings

import pandas as pd

from geopandas import GeoDataFrame
from geopandas.array import _check_crs, _crs_mismatch_warn


def sjoin(
    left_df, right_df, how="inner", op="intersects", lsuffix="left", rsuffix="right"
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
    op : string, default 'intersects'
        Binary predicate. Valid values are determined by the spatial index used.
        You can check the valid values in left_df or right_df as
        ``left_df.sindex.valid_query_predicates`` or
        ``right_df.sindex.valid_query_predicates``
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

    Notes
    ------
    Every operation in GeoPandas is planar, i.e. the potential third
    dimension is not taken into account.
    """
    _basic_checks(left_df, right_df, how, lsuffix, rsuffix)

    indices = _geom_predicate_query(left_df, right_df, op)

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


def _geom_predicate_query(left_df, right_df, op):
    """Compute geometric comparisons and get matching indices.

    Parameters
    ----------
    left_df : GeoDataFrame
    right_df : GeoDataFrame
    op : string
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
        if op == "within":
            # within is implemented as the inverse of contains
            # contains is a faster predicate
            # see discussion at https://github.com/geopandas/geopandas/pull/1421
            predicate = "contains"
            sindex = left_df.sindex
            input_geoms = right_df.geometry
        else:
            # all other predicates are symmetric
            # keep them the same
            predicate = op
            sindex = right_df.sindex
            input_geoms = left_df.geometry

    if sindex:
        l_idx, r_idx = sindex.query_bulk(input_geoms, predicate=predicate, sort=False)
        indices = pd.DataFrame({"_key_left": l_idx, "_key_right": r_idx})
    else:
        # when sindex is empty / has no valid geometries
        indices = pd.DataFrame(columns=["_key_left", "_key_right"], dtype=float)
    if op == "within":
        # within is implemented as the inverse of contains
        # flip back the results
        indices = indices.rename(
            columns={"_key_left": "_key_right", "_key_right": "_key_left"}
        )

    return indices


def _frame_join(indices, left_df, right_df, how, lsuffix, rsuffix):
    """Join the GeoDataFrames at the DataFrame level.

    Parameters
    ----------
    indices : DataFrame
        Indexes returned by the geometric join.
        Must have columns `_key_left` and `_key_right`
        with integer indices representing the matches
        from `left_df` and `right_df` respectively.
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
        indices = indices.set_index("_key_left")
        joined = (
            left_df.merge(indices, left_index=True, right_index=True)
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
        indices = indices.set_index("_key_left")
        joined = (
            left_df.merge(indices, left_index=True, right_index=True, how="left")
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
                indices.merge(
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
