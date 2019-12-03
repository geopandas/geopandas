"""
Testing functionality for geopandas objects.
"""
import warnings

import pandas as pd

from geopandas import GeoDataFrame, GeoSeries
from geopandas.array import GeometryDtype


def _isna(this):
    """isna version that works for both scalars and (Geo)Series"""
    if hasattr(this, "isna"):
        return this.isna()
    elif hasattr(this, "isnull"):
        return this.isnull()
    else:
        return pd.isnull(this)


def geom_equals(this, that):
    """
    Test for geometric equality. Empty or missing geometries are considered
    equal.

    Parameters
    ----------
    this, that : arrays of Geo objects (or anything that has an `is_empty`
                 attribute)
    """

    return (
        this.geom_equals(that)
        | (this.is_empty & that.is_empty)
        | (_isna(this) & _isna(that))
    ).all()


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
    """

    return (
        this.geom_almost_equals(that)
        | (this.is_empty & that.is_empty)
        | (_isna(this) & _isna(that))
    ).all()


def assert_geoseries_equal(
    left,
    right,
    check_dtype=False,
    check_index_type=False,
    check_series_type=True,
    check_less_precise=False,
    check_geom_type=False,
    check_crs=True,
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
    """
    assert len(left) == len(right), "%d != %d" % (len(left), len(right))

    msg = "dtype should be a GeometryDtype, got {0}"
    assert isinstance(left.dtype, GeometryDtype), msg.format(left.dtype)
    assert isinstance(right.dtype, GeometryDtype), msg.format(left.dtype)

    if check_index_type:
        assert isinstance(left.index, type(right.index))

    if check_dtype:
        assert left.dtype == right.dtype, "dtype: %s != %s" % (left.dtype, right.dtype)

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
        assert (left.type == right.type).all(), "type: %s != %s" % (
            left.type,
            right.type,
        )

    if not check_crs:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "GeoSeries crs mismatch", UserWarning)
            if check_less_precise:
                assert geom_almost_equals(left, right)
            else:
                assert geom_equals(left, right)
    else:
        if check_less_precise:
            assert geom_almost_equals(left, right)
        else:
            assert geom_equals(left, right)


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
        "Left columns: {lcols!r}, right columns: {rcols!r}".format(
            lshape=left.shape,
            rshape=right.shape,
            lcols=left.columns,
            rcols=right.columns,
        )
    )

    if check_like:
        left, right = left.reindex_like(right), right

    # column comparison
    assert_index_equal(
        left.columns, right.columns, exact=check_column_type, obj="GeoDataFrame.columns"
    )

    # geometry comparison
    assert_geoseries_equal(
        left.geometry,
        right.geometry,
        check_dtype=check_dtype,
        check_less_precise=check_less_precise,
        check_geom_type=check_geom_type,
        check_crs=False,
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
