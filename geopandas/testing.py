"""
Testing functionality for geopandas objects.
"""
import warnings

import pandas as pd

from geopandas import GeoDataFrame, GeoSeries
from geopandas.array import GeometryDtype
from geopandas import _vectorized


def _isna(this):
    """isna version that works for both scalars and (Geo)Series"""
    with warnings.catch_warnings():
        # GeoSeries.isna will raise a warning about no longer returning True
        # for empty geometries. This helper is used below always in combination
        # with an is_empty check to preserve behaviour, and thus we ignore the
        # warning here to avoid it bubbling up to the user
        warnings.filterwarnings(
            "ignore", r"GeoSeries.isna\(\) previously returned", UserWarning
        )
        if hasattr(this, "isna"):
            return this.isna()
        elif hasattr(this, "isnull"):
            return this.isnull()
        else:
            return pd.isnull(this)


def _geom_equals_mask(this, that):
    """
    Test for geometric equality. Empty or missing geometries are considered
    equal.

    Parameters
    ----------
    this, that : arrays of Geo objects (or anything that has an `is_empty`
                 attribute)

    Returns
    -------
    Series
        boolean Series, True if geometries in left equal geometries in right
    """

    return (
        this.geom_equals(that)
        | (this.is_empty & that.is_empty)
        | (_isna(this) & _isna(that))
    )


def geom_equals(this, that):
    """
    Test for geometric equality. Empty or missing geometries are considered
    equal.

    Parameters
    ----------
    this, that : arrays of Geo objects (or anything that has an `is_empty`
                 attribute)

    Returns
    -------
    bool
        True if all geometries in left equal geometries in right
    """

    return _geom_equals_mask(this, that).all()


def _geom_almost_equals_mask(this, that):
    """
    Test for 'almost' geometric equality. Empty or missing geometries
    considered equal.

    This method allows small difference in the coordinates, but this
    requires coordinates be in the same order for all components of a geometry.

    Parameters
    ----------
    this, that : arrays of Geo objects (or anything that has an `is_empty`
                 property)

    Returns
    -------
    Series
        boolean Series, True if geometries in left almost equal geometries in right
    """

    return (
        this.geom_almost_equals(that)
        | (this.is_empty & that.is_empty)
        | (_isna(this) & _isna(that))
    )


def geom_almost_equals(this, that):
    """
    Test for 'almost' geometric equality. Empty or missing geometries
    considered equal.

    This method allows small difference in the coordinates, but this
    requires coordinates be in the same order for all components of a geometry.

    Parameters
    ----------
    this, that : arrays of Geo objects (or anything that has an `is_empty`
                 property)

    Returns
    -------
    bool
        True if all geometries in left almost equal geometries in right
    """

    return _geom_almost_equals_mask(this, that).all()


def assert_geoseries_equal(
    left,
    right,
    check_dtype=True,
    check_index_type=False,
    check_series_type=True,
    check_less_precise=False,
    check_geom_type=False,
    check_crs=True,
    normalize=False,
):
    """
    Test util for checking that two GeoSeries are equal.

    Parameters
    ----------
    left, right : two GeoSeries
    check_dtype : bool, default False
        If True, check geo dtype [only included so it's a drop-in replacement
        for assert_series_equal].
    check_index_type : bool, default False
        Check that index types are equal.
    check_series_type : bool, default True
        Check that both are same type (*and* are GeoSeries). If False,
        will attempt to convert both into GeoSeries.
    check_less_precise : bool, default False
        If True, use geom_almost_equals. if False, use geom_equals.
    check_geom_type : bool, default False
        If True, check that all the geom types are equal.
    check_crs: bool, default True
        If `check_series_type` is True, then also check that the
        crs matches.
    normalize: bool, default False
        If True, normalize the geometries before comparing equality.
        Typically useful with ``check_less_precise=True``, which uses
        ``geom_almost_equals`` and requires exact coordinate order.
    """
    assert len(left) == len(right), "%d != %d" % (len(left), len(right))

    if check_dtype:
        msg = "dtype should be a GeometryDtype, got {0}"
        assert isinstance(left.dtype, GeometryDtype), msg.format(left.dtype)
        assert isinstance(right.dtype, GeometryDtype), msg.format(left.dtype)

    if check_index_type:
        assert isinstance(left.index, type(right.index))

    if check_series_type:
        assert isinstance(left, GeoSeries)
        assert isinstance(left, type(right))

        if check_crs:
            assert left.crs == right.crs
    else:
        if not isinstance(left, GeoSeries):
            left = GeoSeries(left)
        if not isinstance(right, GeoSeries):
            right = GeoSeries(right, index=left.index)

    assert left.index.equals(right.index), "index: %s != %s" % (left.index, right.index)

    if check_geom_type:
        assert (left.geom_type == right.geom_type).all(), "type: %s != %s" % (
            left.geom_type,
            right.geom_type,
        )

    if normalize:
        left = GeoSeries(_vectorized.normalize(left.array._data))
        right = GeoSeries(_vectorized.normalize(right.array._data))

    if not check_crs:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "CRS mismatch", UserWarning)
            _check_equality(left, right, check_less_precise)
    else:
        _check_equality(left, right, check_less_precise)


def _truncated_string(geom):
    """Truncated WKT repr of geom"""
    s = str(geom)
    if len(s) > 100:
        return s[:100] + "..."
    else:
        return s


def _check_equality(left, right, check_less_precise):
    assert_error_message = (
        "{0} out of {1} geometries are not {3}equal.\n"
        "Indices where geometries are not {3}equal: {2} \n"
        "The first not {3}equal geometry:\n"
        "Left: {4}\n"
        "Right: {5}\n"
    )
    if check_less_precise:
        precise = "almost "
        equal = _geom_almost_equals_mask(left, right)
    else:
        precise = ""
        equal = _geom_equals_mask(left, right)

    if not equal.all():
        unequal_left_geoms = left[~equal]
        unequal_right_geoms = right[~equal]
        raise AssertionError(
            assert_error_message.format(
                len(unequal_left_geoms),
                len(left),
                unequal_left_geoms.index.to_list(),
                precise,
                _truncated_string(unequal_left_geoms.iloc[0]),
                _truncated_string(unequal_right_geoms.iloc[0]),
            )
        )


def assert_geodataframe_equal(
    left,
    right,
    check_dtype=True,
    check_index_type="equiv",
    check_column_type="equiv",
    check_frame_type=True,
    check_like=False,
    check_less_precise=False,
    check_geom_type=False,
    check_crs=True,
    normalize=False,
):
    """
    Check that two GeoDataFrames are equal/

    Parameters
    ----------
    left, right : two GeoDataFrames
    check_dtype : bool, default True
        Whether to check the DataFrame dtype is identical.
    check_index_type, check_column_type : bool, default 'equiv'
        Check that index types are equal.
    check_frame_type : bool, default True
        Check that both are same type (*and* are GeoDataFrames). If False,
        will attempt to convert both into GeoDataFrame.
    check_like : bool, default False
        If true, ignore the order of rows & columns
    check_less_precise : bool, default False
        If True, use geom_almost_equals. if False, use geom_equals.
    check_geom_type : bool, default False
        If True, check that all the geom types are equal.
    check_crs: bool, default True
        If `check_frame_type` is True, then also check that the
        crs matches.
    normalize: bool, default False
        If True, normalize the geometries before comparing equality.
        Typically useful with ``check_less_precise=True``, which uses
        ``geom_almost_equals`` and requires exact coordinate order.
    """
    try:
        # added from pandas 0.20
        from pandas.testing import assert_frame_equal, assert_index_equal
    except ImportError:
        from pandas.util.testing import assert_frame_equal, assert_index_equal

    # instance validation
    if check_frame_type:
        assert isinstance(left, GeoDataFrame)
        assert isinstance(left, type(right))

        if check_crs:
            # no crs can be either None or {}
            if not left.crs and not right.crs:
                pass
            else:
                assert left.crs == right.crs
    else:
        if not isinstance(left, GeoDataFrame):
            left = GeoDataFrame(left)
        if not isinstance(right, GeoDataFrame):
            right = GeoDataFrame(right)

    # shape comparison
    assert left.shape == right.shape, (
        "GeoDataFrame shape mismatch, left: {lshape!r}, right: {rshape!r}.\n"
        "Left columns: {lcols!r}, right columns: {rcols!r}"
    ).format(
        lshape=left.shape, rshape=right.shape, lcols=left.columns, rcols=right.columns
    )

    if check_like:
        left, right = left.reindex_like(right), right

    # column comparison
    assert_index_equal(
        left.columns, right.columns, exact=check_column_type, obj="GeoDataFrame.columns"
    )

    # geometry comparison
    for col, dtype in left.dtypes.items():
        if isinstance(dtype, GeometryDtype):
            assert_geoseries_equal(
                left[col],
                right[col],
                normalize=normalize,
                check_dtype=check_dtype,
                check_less_precise=check_less_precise,
                check_geom_type=check_geom_type,
                check_crs=check_crs,
            )

    # drop geometries and check remaining columns
    left2 = left.drop([left._geometry_column_name], axis=1)
    right2 = right.drop([right._geometry_column_name], axis=1)
    assert_frame_equal(
        left2,
        right2,
        check_dtype=check_dtype,
        check_index_type=check_index_type,
        check_column_type=check_column_type,
        obj="GeoDataFrame",
    )
