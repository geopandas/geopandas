import operator

import numpy as np
import pandas as pd

import shapely.geometry

from geopandas.array import GeometryArray, GeometryDtype, from_shapely

import pytest

from pandas.tests.extension import base


@pytest.fixture
def dtype():
    """A fixture providing the ExtensionDtype to validate."""
    return GeometryDtype()


def make_data():
    a = np.array([shapely.geometry.Point(i, i) for i in range(100)],
                 dtype=object)
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


@pytest.fixture(params=['data', 'data_missing'])
def all_data(request, data, data_missing):
    """Parametrized fixture giving 'data' and 'data_missing'"""
    if request.param == 'data':
        return data
    elif request.param == 'data_missing':
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
    raise NotImplementedError


@pytest.fixture
def data_missing_for_sorting():
    """Length-3 array with a known sort order.

    This should be three items [B, NA, A] with
    A < B and NA missing.
    """
    raise NotImplementedError


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
        [shapely.geometry.Point(1, 1),
         shapely.geometry.Point(1, 1),
         None,
         None,
         shapely.geometry.Point(0, 0),
         shapely.geometry.Point(0, 0),
         shapely.geometry.Point(1, 1),
         shapely.geometry.Point(2, 2)])


@pytest.fixture(params=[True, False])
def box_in_series(request):
    """Whether to box the data in a Series"""
    return request.param


@pytest.fixture(params=[
    lambda x: 1,
    lambda x: [1] * len(x),
    lambda x: pd.Series([1] * len(x)),
    lambda x: x,
], ids=['scalar', 'list', 'series', 'object'])
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


@pytest.fixture(params=['ffill', 'bfill'])
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


class TestDtype(base.BaseDtypeTests):

    def test_array_type_with_arg(self, data, dtype):
        assert dtype.construct_array_type() is GeometryArray


class TestInterface(base.BaseInterfaceTests):
    pass


class TestConstructors(base.BaseConstructorsTests):
    pass


class TestReshaping(base.BaseReshapingTests):
    pass


class TestGetitem(base.BaseGetitemTests):
    pass


class TestSetitem(base.BaseSetitemTests):
    pass


class TestMissing(base.BaseMissingTests):

    def test_fillna_series(self, data_missing):
        fill_value = data_missing[1]
        ser = pd.Series(data_missing)

        result = ser.fillna(fill_value)
        expected = pd.Series(data_missing._from_sequence(
            [fill_value, fill_value]))
        self.assert_series_equal(result, expected)

        # filling with array-like not yet supported

        # # Fill with a series
        # result = ser.fillna(expected)
        # self.assert_series_equal(result, expected)

        # # Fill with a series not affecting the missing values
        # result = ser.fillna(ser)
        # self.assert_series_equal(result, ser)

    @pytest.mark.skip("fillna method not supported")
    def test_fillna_limit_pad(self, data_missing):
        pass

    @pytest.mark.skip("fillna method not supported")
    def test_fillna_limit_backfill(self, data_missing):
        pass

    @pytest.mark.skip("fillna method not supported")
    def test_fillna_series_method(self, data_missing, method):
        pass


class TestCasting(base.BaseCastingTests):
    pass


class TestMethods(base.BaseMethodsTests):

    @pytest.mark.skip(reason="Sorting not supported")
    @pytest.mark.parametrize('dropna', [True, False])
    def test_value_counts(self, all_data, dropna):
        pass

    @pytest.mark.skip(reason="Sorting not supported")
    def test_argsort(self, data_for_sorting):
        result = pd.Series(data_for_sorting).argsort()
        expected = pd.Series(np.array([2, 0, 1], dtype=np.int64))
        self.assert_series_equal(result, expected)

    @pytest.mark.skip(reason="Sorting not supported")
    def test_argsort_missing(self, data_missing_for_sorting):
        result = pd.Series(data_missing_for_sorting).argsort()
        expected = pd.Series(np.array([1, -1, 0], dtype=np.int64))
        self.assert_series_equal(result, expected)

    @pytest.mark.skip(reason="Sorting not supported")
    @pytest.mark.parametrize('ascending', [True, False])
    def test_sort_values(self, data_for_sorting, ascending):
        ser = pd.Series(data_for_sorting)
        result = ser.sort_values(ascending=ascending)
        expected = ser.iloc[[2, 0, 1]]
        if not ascending:
            expected = expected[::-1]

        self.assert_series_equal(result, expected)

    @pytest.mark.skip(reason="Sorting not supported")
    @pytest.mark.parametrize('ascending', [True, False])
    def test_sort_values_missing(self, data_missing_for_sorting, ascending):
        ser = pd.Series(data_missing_for_sorting)
        result = ser.sort_values(ascending=ascending)
        if ascending:
            expected = ser.iloc[[2, 0, 1]]
        else:
            expected = ser.iloc[[0, 2, 1]]
        self.assert_series_equal(result, expected)

    @pytest.mark.skip(reason="Sorting not supported")
    @pytest.mark.parametrize('ascending', [True, False])
    def test_sort_values_frame(self, data_for_sorting, ascending):
        df = pd.DataFrame({"A": [1, 2, 1],
                           "B": data_for_sorting})
        result = df.sort_values(['A', 'B'])
        expected = pd.DataFrame({"A": [1, 1, 2],
                                 'B': data_for_sorting.take([2, 0, 1])},
                                index=[2, 0, 1])
        self.assert_frame_equal(result, expected)

    @pytest.mark.skip(reason="comparison not supported")
    def test_combine_le(self):
        pass

    @pytest.mark.skip(reason="addition not supported")
    def test_combine_add(self):
        pass


class TestGroupby(base.BaseGroupbyTests):

    @pytest.mark.skip(reason="Sorting not supported")
    @pytest.mark.parametrize('as_index', [True, False])
    def test_groupby_extension_agg(self, as_index, data_for_grouping):
        pass

    @pytest.mark.skip(reason="Sorting not supported")
    def test_groupby_extension_transform(self, data_for_grouping):
        pass

    @pytest.mark.skip(reason="Sorting not supported")
    @pytest.mark.parametrize('op', [
        lambda x: 1,
        lambda x: [1] * len(x),
        lambda x: pd.Series([1] * len(x)),
        lambda x: x,
    ], ids=['scalar', 'list', 'series', 'object'])
    def test_groupby_extension_apply(self, data_for_grouping, op):
        pass



class TestPrinting(base.BasePrintingTests):
    pass


# class TestParsing(base.BaseParsingTests):
#     pass



# TODO add ops tests

# _all_arithmetic_operators = ['__add__', '__radd__',
#                              '__sub__', '__rsub__',
#                              '__mul__', '__rmul__',
#                              '__floordiv__', '__rfloordiv__',
#                              '__truediv__', '__rtruediv__',
#                              '__pow__', '__rpow__',
#                              '__mod__', '__rmod__']
# # if not PY3:
# #     _all_arithmetic_operators.extend(['__div__', '__rdiv__'])


# @pytest.fixture(params=_all_arithmetic_operators)
# def all_arithmetic_operators(request):
#     """
#     Fixture for dunder names for common arithmetic operations
#     """
#     return request.param


# class TestArithmeticOps(base.BaseArithmeticOpsTests):

#     pass
