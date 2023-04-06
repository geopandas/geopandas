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
import operator

import numpy as np
from numpy.testing import assert_array_equal
import pandas as pd
from pandas.tests.extension import base as extension_tests

import shapely.geometry
from shapely.geometry import Point

from geopandas.array import GeometryArray, GeometryDtype, from_shapely
from geopandas._compat import ignore_shapely2_warnings, SHAPELY_GE_20, PANDAS_GE_15

import pytest

# -----------------------------------------------------------------------------
# Compat with extension tests in older pandas versions
# -----------------------------------------------------------------------------


not_yet_implemented = pytest.mark.skip(reason="Not yet implemented")
no_minmax = pytest.mark.skip(reason="Min/max not supported")
requires_shapely2 = pytest.mark.skipif(
    not SHAPELY_GE_20, reason="Requires hashable geometries"
)


# -----------------------------------------------------------------------------
# Required fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def dtype():
    """A fixture providing the ExtensionDtype to validate."""
    return GeometryDtype()


def make_data():
    a = np.empty(100, dtype=object)
    with ignore_shapely2_warnings():
        a[:] = [shapely.geometry.Point(i, i) for i in range(100)]
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
    return from_shapely([Point(0, 1), None, Point(0, 0)])


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
        self.assert_series_equal(result, expected)


class TestInterface(extension_tests.BaseInterfaceTests):
    def test_array_interface(self, data):
        # we are overriding this base test because the creation of `expected`
        # potentially doesn't work for shapely geometries
        # TODO can be removed with Shapely 2.0
        result = np.array(data)
        assert result[0] == data[0]

        result = np.array(data, dtype=object)
        # expected = np.array(list(data), dtype=object)
        expected = np.empty(len(data), dtype=object)
        with ignore_shapely2_warnings():
            expected[:] = list(data)
        assert_array_equal(result, expected)

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
    pass


class TestGetitem(extension_tests.BaseGetitemTests):
    pass


class TestSetitem(extension_tests.BaseSetitemTests):
    pass


class TestMissing(extension_tests.BaseMissingTests):
    def test_fillna_series(self, data_missing):
        fill_value = data_missing[1]
        ser = pd.Series(data_missing)

        # Fill with a scalar
        result = ser.fillna(fill_value)
        expected = pd.Series(data_missing._from_sequence([fill_value, fill_value]))
        self.assert_series_equal(result, expected)

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
        self.assert_series_equal(result, expected)

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
        self.assert_series_equal(result, ser)

        # More `GeoSeries.fillna` testcases are in
        # `geopandas\tests\test_pandas_methods.py::test_fillna_scalar`
        # and `geopandas\tests\test_pandas_methods.py::test_fillna_series`.

    @pytest.mark.skip("fillna method not supported")
    def test_fillna_limit_pad(self, data_missing):
        pass

    @pytest.mark.skip("fillna method not supported")
    def test_fillna_limit_backfill(self, data_missing):
        pass

    @pytest.mark.skip("fillna method not supported")
    def test_fillna_series_method(self, data_missing, method):
        pass

    @pytest.mark.skip("fillna method not supported")
    def test_fillna_no_op_returns_copy(self, data):
        pass


class TestReduce(extension_tests.BaseNoReduceTests):
    @pytest.mark.skip("boolean reduce (any/all) tested in test_pandas_methods")
    def test_reduce_series_boolean():
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
        self.assert_series_equal(result, expected)

    def test_compare_scalar(self, data, all_compare_operators):  # noqa
        op_name = all_compare_operators
        s = pd.Series(data)
        self._compare_other(s, data, op_name, data[0])

    def test_compare_array(self, data, all_compare_operators):  # noqa
        op_name = all_compare_operators
        s = pd.Series(data)
        other = pd.Series([data[0]] * len(data))
        self._compare_other(s, data, op_name, other)


class TestMethods(extension_tests.BaseMethodsTests):
    @pytest.mark.skipif(
        not PANDAS_GE_15, reason="sorting index not yet working with older pandas"
    )
    @pytest.mark.parametrize("dropna", [True, False])
    def test_value_counts(self, all_data, dropna):
        pass

    @pytest.mark.skipif(
        not PANDAS_GE_15, reason="sorting index not yet working with older pandas"
    )
    def test_value_counts_with_normalize(self, data):
        pass

    @requires_shapely2
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
    @requires_shapely2
    @pytest.mark.parametrize("as_index", [True, False])
    def test_groupby_extension_agg(self, as_index, data_for_grouping):
        super().test_groupby_extension_agg(as_index, data_for_grouping)

    @requires_shapely2
    def test_groupby_extension_transform(self, data_for_grouping):
        super().test_groupby_extension_transform(data_for_grouping)

    @requires_shapely2
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
