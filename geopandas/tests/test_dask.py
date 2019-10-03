import pandas as pd
import pytest
from geopandas.array import from_shapely
from shapely.geometry import Point

dd = pytest.importorskip("dask.dataframe")


def test_from_pandas():
    s = pd.Series(from_shapely([Point(0, 0), Point(1, 1)]))
    ds = dd.from_pandas(s, 2)

    assert ds.dtype == s.dtype
    dd.utils.assert_eq(s, ds)

    df = pd.DataFrame({"A": s})
    ddf = dd.from_pandas(df, 2)
    assert ddf.dtypes["A"] == s.dtype
    dd.utils.assert_eq(df, ddf)
