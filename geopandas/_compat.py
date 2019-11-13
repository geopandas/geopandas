from distutils.version import LooseVersion

import pandas as pd
import six

# -----------------------------------------------------------------------------
# pandas compat
# -----------------------------------------------------------------------------

PANDAS_GE_024 = str(pd.__version__) >= LooseVersion("0.24.0")
PANDAS_GE_025 = str(pd.__version__) >= LooseVersion("0.25.0")


# -----------------------------------------------------------------------------
# Python 2/3 compat
# -----------------------------------------------------------------------------

if six.PY2:
    from collections import Iterable  # noqa
else:
    from collections.abc import Iterable  # noqa
