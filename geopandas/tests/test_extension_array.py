import operator

import numpy as np
import pandas as pd

import shapely.geometry

from geopandas.array import GeometryDtype, from_shapely

import pytest
from pandas.tests.extension import base


@pytest.fixture
def dtype():
    """A fixture providing the ExtensionDtype to validate."""
    return GeometryDtype()


@pytest.fixture
def data():
    """Length-100 array for this type."""
    a = np.array([shapely.geometry.Point(i, i) for i in range(100)],
                 dtype=object)
    ga = from_shapely(a)
    return ga


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


class TestDtype(base.BaseDtypeTests):
    pass


class TestInterface(base.BaseInterfaceTests):
    pass


class TestConstructors(base.BaseConstructorsTests):
    pass


class TestReshaping(base.BaseReshapingTests):
    pass


class TestGetitem(base.BaseGetitemTests):
    pass


class TestSetitem(base.BaseSetitemTests):

    @pytest.mark.skip(reason="Setting not supported")
    def test_setitem_scalar_series(self, data):
        arr = pd.Series(data)
        arr[0] = data[1]
        assert arr[0] == data[1]

    @pytest.mark.skip(reason="Setting not supported")
    def test_setitem_sequence(self, data):
        arr = pd.Series(data)
        original = data.copy()

        arr[[0, 1]] = [data[1], data[0]]
        assert arr[0] == original[1]
        assert arr[1] == original[0]

    @pytest.mark.skip(reason="Setting not supported")
    def test_setitem_empty_indxer(self, data):
        ser = pd.Series(data)
        original = ser.copy()
        ser[[]] = []
        self.assert_series_equal(ser, original)

    @pytest.mark.skip(reason="Setting not supported")
    def test_setitem_sequence_broadcasts(self, data):
        arr = pd.Series(data)

        arr[[0, 1]] = data[2]
        assert arr[0] == data[2]
        assert arr[1] == data[2]

    @pytest.mark.skip(reason="Setting not supported")
    @pytest.mark.parametrize('setter', ['loc', 'iloc'])
    def test_setitem_scalar(self, data, setter):
        arr = pd.Series(data)
        setter = getattr(arr, setter)
        operator.setitem(setter, 0, data[1])
        assert arr[0] == data[1]

    @pytest.mark.skip(reason="Setting not supported")
    def test_setitem_loc_scalar_mixed(self, data):
        df = pd.DataFrame({"A": np.arange(len(data)), "B": data})
        df.loc[0, 'B'] = data[1]
        assert df.loc[0, 'B'] == data[1]

    @pytest.mark.skip(reason="Setting not supported")
    def test_setitem_loc_scalar_single(self, data):
        df = pd.DataFrame({"B": data})
        df.loc[10, 'B'] = data[1]
        assert df.loc[10, 'B'] == data[1]

    @pytest.mark.skip(reason="Setting not supported")
    def test_setitem_loc_scalar_multiple_homogoneous(self, data):
        df = pd.DataFrame({"A": data, "B": data})
        df.loc[10, 'B'] = data[1]
        assert df.loc[10, 'B'] == data[1]

    @pytest.mark.skip(reason="Setting not supported")
    def test_setitem_iloc_scalar_mixed(self, data):
        df = pd.DataFrame({"A": np.arange(len(data)), "B": data})
        df.iloc[0, 1] = data[1]
        assert df.loc[0, 'B'] == data[1]

    @pytest.mark.skip(reason="Setting not supported")
    def test_setitem_iloc_scalar_single(self, data):
        df = pd.DataFrame({"B": data})
        df.iloc[10, 0] = data[1]
        assert df.loc[10, 'B'] == data[1]

    @pytest.mark.skip(reason="Setting not supported")
    def test_setitem_iloc_scalar_multiple_homogoneous(self, data):
        df = pd.DataFrame({"A": data, "B": data})
        df.iloc[10, 1] = data[1]
        assert df.loc[10, 'B'] == data[1]

    @pytest.mark.skip(reason="Setting not supported")
    @pytest.mark.parametrize('as_callable', [True, False])
    @pytest.mark.parametrize('setter', ['loc', None])
    def test_setitem_mask_aligned(self, data, as_callable, setter):
        ser = pd.Series(data)
        mask = np.zeros(len(data), dtype=bool)
        mask[:2] = True

        if as_callable:
            mask2 = lambda x: mask
        else:
            mask2 = mask

        if setter:
            # loc
            target = getattr(ser, setter)
        else:
            # Series.__setitem__
            target = ser

        operator.setitem(target, mask2, data[5:7])

        ser[mask2] = data[5:7]
        assert ser[0] == data[5]
        assert ser[1] == data[6]

    @pytest.mark.skip(reason="Setting not supported")
    @pytest.mark.parametrize('setter', ['loc', None])
    def test_setitem_mask_broadcast(self, data, setter):
        ser = pd.Series(data)
        mask = np.zeros(len(data), dtype=bool)
        mask[:2] = True

        if setter:   # loc
            target = getattr(ser, setter)
        else:  # __setitem__
            target = ser

        operator.setitem(target, mask, data[10])
        assert ser[0] == data[10]
        assert ser[1] == data[10]


class TestMissing(base.BaseMissingTests):
    
    def test_fillna_series(self, data_missing):
        fill_value = data_missing[1]
        ser = pd.Series(data_missing)

        result = ser.fillna(fill_value)
        expected = pd.Series(type(data_missing)([fill_value, fill_value]))
        self.assert_series_equal(result, expected)

        # filling with array-like not yet supported

        # # Fill with a series
        # result = ser.fillna(expected)
        # self.assert_series_equal(result, expected)

        # # Fill with a series not affecting the missing values
        # result = ser.fillna(ser)
        # self.assert_series_equal(result, ser)


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
