from distutils.version import LooseVersion

import six

import pandas as pd


# -----------------------------------------------------------------------------
# pandas compat
# -----------------------------------------------------------------------------

PANDAS_GE_024 = pd.__version__ >= LooseVersion('0.24.0')

from pandas.tests.extension import base as extension_tests  # noqa

if not PANDAS_GE_024:
    extension_tests.BaseNoReduceTests = object
    extension_tests.BaseArithmeticOpsTests = object
    extension_tests.BaseComparisonOpsTests = object
    extension_tests.BasePrintingTests = object
    extension_tests.BaseParsingTests = object


# -----------------------------------------------------------------------------
# Python 2/3 compat
# -----------------------------------------------------------------------------

if six.PY2:
    from collections import Iterable  # noqa
else:
    from collections.abc import Iterable  # noqa
