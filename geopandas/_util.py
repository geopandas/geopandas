import numpy as np
import pandas as pd

from . import _compat as compat


def isna(value):
    """
    Check if scalar value is NA-like (None, np.nan or pd.NA).

    Custom version that only works for scalars (returning True or False),
    as `pd.isna` also works for array-like input returning a boolean array.
    """
    if value is None:
        return True
    elif isinstance(value, float) and np.isnan(value):
        return True
    elif compat.PANDAS_GE_10 and value is pd.NA:
        return True
    else:
        return False
