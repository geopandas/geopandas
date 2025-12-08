"""
This file contains a minimal set of tests for compliance with the extension
array interface test suite (by inheriting the pandas test suite), and should
contain no other tests.
Other tests (eg related to the spatial functionality or integration
with GeoSeries/GeoDataFrame) should be added to test_array.py and others.

The tests in this file are inherited from the BaseExtensionTests, and only
minimal tweaks should be applied to get the tests passing (by overwriting a
parent method).

A set of fixtures are defined to provide data for the tests (the fixtures
expected to be available to pytest by the inherited pandas tests).

"""

import itertools
import operator

import numpy as np
import pandas as pd
from pandas.api.types import is_object_dtype
from pandas.tests.extension import base as extension_tests

import shapely.geometry
from shapely.geometry import Point

from geopandas._compat import PANDAS_GE_21, PANDAS_GE_22, PANDAS_GE_30
from geopandas.array import GeometryArray, GeometryDtype, from_shapely

import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

# -----------------------------------------------------------------------------
# Compat with extension tests in older pandas versions
# -----------------------------------------------------------------------------


not_yet_implemented = pytest.mark.skip(reason="Not yet implemented")
no_minmax = pytest.mark.skip(reason="Min/max not supported")


# -----------------------------------------------------------------------------
# Required fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def dtype():
    """A fixture providing the ExtensionDtype to validate."""
    return GeometryDtype()


def make_data():
    if PANDAS_GE_30:
        size = 10
    else:
        size = 100
    a = np.empty(size, dtype=object)
    a[:] = [shapely.geometry.Point(i, i) for i in range(size)]
    ga = from_shapely(a)
    return ga


@pytest.fixture
def data():
    """Length-100 array for this type.

    * data[0] and data[1] should both be non missing
    * data[0] and data[1] should not be equal
    """
    return make_data()


@pytest.fixture
def data_for_twos():
    """Length-100 array in which all the elements are two."""
    raise NotImplementedError


@pytest.fixture
def data_missing():
    """Length-2 array with [NA, Valid]"""
    return from_shapely([None, shapely.geometry.Point(1, 1)])


@pytest.fixture(params=["data", "data_missing"])
def all_data(request, data, data_missing):
    """Parametrized fixture giving 'data' and 'data_missing'"""
    if request.param == "data":
        return data
    elif request.param == "data_missing":
        return data_missing


@pytest.fixture
def data_repeated(data):
    """
    Generate many datasets.

    Parameters
    ----------
    data : fixture implementing `data`

    Returns
    -------
    Callable[[int], Generator]:
        A callable that takes a `count` argument and
        returns a generator yielding `count` datasets.
    """

    def gen(count):
        for _ in range(count):
            yield data

    return gen


@pytest.fixture
def data_for_sorting():
    """Length-3 array with a known sort order.

    This should be three items [B, C, A] with
    A < B < C
    """
    return from_shapely([Point(0, 1), Point(1, 1), Point(0, 0)])


@pytest.fixture
def data_missing_for_sorting():
    """Length-3 array with a known sort order.

    This should be three items [B, NA, A] with
    A < B and NA missing.
    """
    return from_shapely([Point(1, 2), None, Point(0, 0)])


@pytest.fixture
def na_cmp():
    """Binary operator for comparing NA values.
    Should return a function of two arguments that returns
    True if both arguments are (scalar) NA for your type.
    By default, uses ``operator.or``
    """
    return lambda x, y: x is None and y is None


@pytest.fixture
def na_value():
    """The scalar missing value for this type. Default 'None'"""
    return None


@pytest.fixture
def data_for_grouping():
    """Data for factorization, grouping, and unique tests.

    Expected to be like [B, B, NA, NA, A, A, B, C]

    Where A < B < C and NA is missing
    """
    return from_shapely(
        [
            shapely.geometry.Point(1, 1),
            shapely.geometry.Point(1, 1),
            None,
            None,
            shapely.geometry.Point(0, 0),
            shapely.geometry.Point(0, 0),
            shapely.geometry.Point(1, 1),
            shapely.geometry.Point(2, 2),
        ]
    )


@pytest.fixture(params=[True, False])
def box_in_series(request):
    """Whether to box the data in a Series"""
    return request.param


@pytest.fixture(
    params=[
        lambda x: 1,
        lambda x: [1] * len(x),
        lambda x: pd.Series([1] * len(x)),
        lambda x: x,
    ],
    ids=["scalar", "list", "series", "object"],
)
def groupby_apply_op(request):
    """
    Functions to test groupby.apply().
    """
    return request.param


@pytest.fixture(params=[True, False])
def as_frame(request):
    """
    Boolean fixture to support Series and Series.to_frame() comparison testing.
    """
    return request.param


@pytest.fixture(params=[True, False])
def as_series(request):
    """
    Boolean fixture to support arr and Series(arr) comparison testing.
    """
    return request.param


@pytest.fixture(params=[True, False])
def use_numpy(request):
    """
    Boolean fixture to support comparison testing of ExtensionDtype array
    and numpy array.
    """
    return request.param


@pytest.fixture(params=["ffill", "bfill"])
def fillna_method(request):
    """
    Parametrized fixture giving method parameters 'ffill' and 'bfill' for
    Series.fillna(method=<method>) testing.
    """
    return request.param


@pytest.fixture(params=[True, False])
def as_array(request):
    """
    Boolean fixture to support ExtensionDtype _from_sequence method testing.
    """
    return request.param


@pytest.fixture
def invalid_scalar(data):
    """
    A scalar that *cannot* be held by this ExtensionArray.

    The default should work for most subclasses, but is not guaranteed.

    If the array can hold any item (i.e. object dtype), then use pytest.skip.
    """
    return object.__new__(object)


# Fixtures defined in pandas/conftest.py that are also needed: defining them
# here instead of importing for compatibility


@pytest.fixture(
    params=["sum", "max", "min", "mean", "prod", "std", "var", "median", "kurt", "skew"]
)
def all_numeric_reductions(request):
    """
    Fixture for numeric reduction names
    """
    return request.param


@pytest.fixture(params=["all", "any"])
def all_boolean_reductions(request):
    """
    Fixture for boolean reduction names
    """
    return request.param


# only == and != are support for GeometryArray
# @pytest.fixture(params=["__eq__", "__ne__", "__le__", "__lt__", "__ge__", "__gt__"])
@pytest.fixture(params=["__eq__", "__ne__"])
def all_compare_operators(request):
    """
    Fixture for dunder names for common compare operations

    * >=
    * >
    * ==
    * !=
    * <
    * <=
    """
    return request.param


@pytest.fixture(params=[None, lambda x: x])
def sort_by_key(request):
    """
    Simple fixture for testing keys in sorting methods.
    Tests None (no key) and the identity key.
    """
    return request.param


# -----------------------------------------------------------------------------
# Inherited tests
# -----------------------------------------------------------------------------


class TestDtype(extension_tests.BaseDtypeTests):
    # additional tests

    def test_array_type_with_arg(self, data, dtype):
        assert dtype.construct_array_type() is GeometryArray

    def test_registry(self, data, dtype):
        s = pd.Series(np.asarray(data), dtype=object)
        result = s.astype("geometry")
        assert isinstance(result.array, GeometryArray)
        expected = pd.Series(data)
        assert_series_equal(result, expected)


class TestInterface(extension_tests.BaseInterfaceTests):
    def test_contains(self, data, data_missing):
        # overridden due to the inconsistency between
        # GeometryDtype.na_value = np.nan
        # and None being used as NA in array

        # ensure data without missing values
        data = data[~data.isna()]

        # first elements are non-missing
        assert data[0] in data
        assert data_missing[0] in data_missing

        assert None in data_missing
        assert None not in data
        assert pd.NaT not in data_missing


class TestConstructors(extension_tests.BaseConstructorsTests):
    pass


class TestReshaping(extension_tests.BaseReshapingTests):
    # NOTE: this test is copied from pandas/tests/extension/base/reshaping.py
    # because starting with pandas 3.0 the assert_frame_equal is strict regarding
    # the exact missing value (None vs NaN)
    # Our `result` uses None, but the way the `expected` is created results in
    # NaNs (and specifying to use None as fill value in unstack also does not
    # help)
    # -> the only change compared to the upstream test is marked
    @pytest.mark.parametrize(
        "index",
        [
            # Two levels, uniform.
            pd.MultiIndex.from_product(([["A", "B"], ["a", "b"]]), names=["a", "b"]),
            # non-uniform
            pd.MultiIndex.from_tuples([("A", "a"), ("A", "b"), ("B", "b")]),
            # three levels, non-uniform
            pd.MultiIndex.from_product([("A", "B"), ("a", "b"), (0, 1)])
            if PANDAS_GE_30
            else pd.MultiIndex.from_product([("A", "B"), ("a", "b", "c"), (0, 1, 2)]),
            pd.MultiIndex.from_tuples(
                [
                    ("A", "a", 1),
                    ("A", "b", 0),
                    ("A", "a", 0),
                    ("B", "a", 0),
                    ("B", "c", 1),
                ]
            ),
        ],
    )
    @pytest.mark.parametrize("obj", ["series", "frame"])
    def test_unstack(self, data, index, obj):
        data = data[: len(index)]
        if obj == "series":
            ser = pd.Series(data, index=index)
        else:
            ser = pd.DataFrame({"A": data, "B": data}, index=index)

        n = index.nlevels
        levels = list(range(n))
        # [0, 1, 2]
        # [(0,), (1,), (2,), (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)]
        combinations = itertools.chain.from_iterable(
            itertools.permutations(levels, i) for i in range(1, n)
        )

        for level in combinations:
            result = ser.unstack(level=level)
            assert all(
                isinstance(result[col].array, type(data)) for col in result.columns
            )

            if obj == "series":
                # We should get the same result with to_frame+unstack+droplevel
                df = ser.to_frame()

                alt = df.unstack(level=level).droplevel(0, axis=1)
                assert_frame_equal(result, alt)

            obj_ser = ser.astype(object)

            expected = obj_ser.unstack(level=level, fill_value=data.dtype.na_value)
            if obj == "series":
                assert all(is_object_dtype(x) for x in expected.dtypes)
            # <------------ next line is added
            expected[expected.isna()] = None
            # ------------->

            result = result.astype(object)
            assert_frame_equal(result, expected)


class TestGetitem(extension_tests.BaseGetitemTests):
    @pytest.mark.xfail(reason="read-only not yet implemented")
    def test_getitem_propagates_readonly_property(self, data):
        super().test_getitem_propagates_readonly_property(data)


class TestSetitem(extension_tests.BaseSetitemTests):
    @pytest.mark.xfail(reason="read-only not yet implemented")
    def test_readonly_property(self, data):
        super().test_readonly_property(data)

    @pytest.mark.xfail(reason="read-only not yet implemented")
    def test_readonly_propagates_to_numpy_array(self, data):
        super().test_readonly_propagates_to_numpy_array(data)


class TestMissing(extension_tests.BaseMissingTests):
    def test_fillna_series(self, data_missing):
        fill_value = data_missing[1]
        ser = pd.Series(data_missing)

        # Fill with a scalar
        result = ser.fillna(fill_value)
        expected = pd.Series(data_missing._from_sequence([fill_value, fill_value]))
        assert_series_equal(result, expected)

        # Fill with a series
        filler = pd.Series(
            from_shapely(
                [
                    shapely.geometry.Point(1, 1),
                    shapely.geometry.Point(2, 2),
                ],
            )
        )
        result = ser.fillna(filler)
        expected = pd.Series(data_missing._from_sequence([fill_value, fill_value]))
        assert_series_equal(result, expected)

        # Fill with a series not affecting the missing values
        filler = pd.Series(
            from_shapely(
                [
                    shapely.geometry.Point(2, 2),
                    shapely.geometry.Point(1, 1),
                ]
            ),
            index=[10, 11],
        )
        result = ser.fillna(filler)
        assert_series_equal(result, ser)

        # More `GeoSeries.fillna` testcases are in
        # `geopandas\tests\test_pandas_methods.py::test_fillna_scalar`
        # and `geopandas\tests\test_pandas_methods.py::test_fillna_series`.

    @pytest.mark.skipif(
        not PANDAS_GE_21, reason="fillna method not supported with older pandas"
    )
    def test_fillna_limit_pad(self, data_missing):
        super().test_fillna_limit_pad(data_missing)

    @pytest.mark.skipif(
        not PANDAS_GE_21, reason="fillna method not supported with older pandas"
    )
    def test_fillna_limit_backfill(self, data_missing):
        super().test_fillna_limit_backfill(data_missing)

    @pytest.mark.skipif(
        not PANDAS_GE_21, reason="fillna method not supported with older pandas"
    )
    def test_fillna_series_method(self, data_missing, fillna_method):
        super().test_fillna_series_method(data_missing, fillna_method)

    @pytest.mark.skipif(
        not PANDAS_GE_21, reason="fillna method not supported with older pandas"
    )
    def test_fillna_no_op_returns_copy(self, data):
        super().test_fillna_no_op_returns_copy(data)

    @pytest.mark.xfail(reason="read-only not yet implemented")
    def test_fillna_readonly(self, data_missing):
        super().test_fillna_readonly(data_missing)


if PANDAS_GE_22:
    from pandas.tests.extension.base import BaseReduceTests
else:
    from pandas.tests.extension.base import BaseNoReduceTests as BaseReduceTests


class TestReduce(BaseReduceTests):
    @pytest.mark.skip("boolean reduce (any/all) tested in test_pandas_methods")
    def test_reduce_series_boolean(self):
        pass


_all_arithmetic_operators = [
    "__add__",
    "__radd__",
    # '__sub__', '__rsub__',
    "__mul__",
    "__rmul__",
    "__floordiv__",
    "__rfloordiv__",
    "__truediv__",
    "__rtruediv__",
    "__pow__",
    "__rpow__",
    "__mod__",
    "__rmod__",
]


@pytest.fixture(params=_all_arithmetic_operators)
def all_arithmetic_operators(request):
    """
    Fixture for dunder names for common arithmetic operations

    Adapted to exclude __sub__, as this is implemented as "difference".
    """
    return request.param


# an inherited test from pandas creates a Series from a list of geometries, which
# triggers the warning from Shapely, out of control of GeoPandas, so ignoring here
@pytest.mark.filterwarnings(
    "ignore:The array interface is deprecated and will no longer work in Shapely 2.0"
)
class TestArithmeticOps(extension_tests.BaseArithmeticOpsTests):
    @pytest.mark.skip(reason="not applicable")
    def test_divmod_series_array(self, data, data_for_twos):
        pass

    @pytest.mark.skip(reason="not applicable")
    def test_add_series_with_extension_array(self, data):
        pass


# an inherited test from pandas creates a Series from a list of geometries, which
# triggers the warning from Shapely, out of control of GeoPandas, so ignoring here
@pytest.mark.filterwarnings(
    "ignore:The array interface is deprecated and will no longer work in Shapely 2.0"
)
class TestComparisonOps(extension_tests.BaseComparisonOpsTests):
    def _compare_other(self, s, data, op_name, other):
        op = getattr(operator, op_name.strip("_"))
        result = op(s, other)
        expected = s.combine(other, op)
        assert_series_equal(result, expected)

    def test_compare_scalar(self, data, all_compare_operators):
        op_name = all_compare_operators
        s = pd.Series(data)
        self._compare_other(s, data, op_name, data[0])

    def test_compare_array(self, data, all_compare_operators):
        op_name = all_compare_operators
        s = pd.Series(data)
        other = pd.Series([data[0]] * len(data))
        self._compare_other(s, data, op_name, other)


class TestMethods(extension_tests.BaseMethodsTests):
    @pytest.mark.parametrize("dropna", [True, False])
    def test_value_counts(self, all_data, dropna):
        pass

    def test_value_counts_with_normalize(self, data):
        pass

    @pytest.mark.parametrize("ascending", [True, False])
    def test_sort_values_frame(self, data_for_sorting, ascending):
        super().test_sort_values_frame(data_for_sorting, ascending)

    @pytest.mark.skip(reason="searchsorted not supported")
    def test_searchsorted(self, data_for_sorting, as_series):
        pass

    @not_yet_implemented
    def test_combine_le(self):
        pass

    @pytest.mark.skip(reason="addition not supported")
    def test_combine_add(self):
        pass

    @not_yet_implemented
    def test_fillna_length_mismatch(self, data_missing):
        msg = "Length of 'value' does not match."
        with pytest.raises(ValueError, match=msg):
            data_missing.fillna(data_missing.take([1]))

    @no_minmax
    def test_argmin_argmax(self):
        pass

    @no_minmax
    def test_argmin_argmax_empty_array(self):
        pass

    @no_minmax
    def test_argmin_argmax_all_na(self):
        pass

    @no_minmax
    def test_argreduce_series(self):
        pass

    @no_minmax
    def test_argmax_argmin_no_skipna_notimplemented(self):
        pass


class TestCasting(extension_tests.BaseCastingTests):
    pass


class TestGroupby(extension_tests.BaseGroupbyTests):
    @pytest.mark.parametrize("as_index", [True, False])
    def test_groupby_extension_agg(self, as_index, data_for_grouping):
        super().test_groupby_extension_agg(as_index, data_for_grouping)

    def test_groupby_extension_transform(self, data_for_grouping):
        super().test_groupby_extension_transform(data_for_grouping)

    @pytest.mark.parametrize(
        "op",
        [
            lambda x: 1,
            lambda x: [1] * len(x),
            lambda x: pd.Series([1] * len(x)),
            lambda x: x,
        ],
        ids=["scalar", "list", "series", "object"],
    )
    def test_groupby_extension_apply(self, data_for_grouping, op):
        super().test_groupby_extension_apply(data_for_grouping, op)


class TestPrinting(extension_tests.BasePrintingTests):
    pass


@not_yet_implemented
class TestParsing(extension_tests.BaseParsingTests):
    pass
