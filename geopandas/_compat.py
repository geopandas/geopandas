from distutils.version import LooseVersion
import os

import pandas as pd

# -----------------------------------------------------------------------------
# pandas compat
# -----------------------------------------------------------------------------

PANDAS_GE_024 = str(pd.__version__) >= LooseVersion("0.24.0")
PANDAS_GE_025 = str(pd.__version__) >= LooseVersion("0.25.0")
PANDAS_GE_10 = str(pd.__version__) >= LooseVersion("0.26.0.dev")


# -----------------------------------------------------------------------------
# Shapely / PyGEOS compat
# -----------------------------------------------------------------------------

USE_PYGEOS = None

if USE_PYGEOS is None:
    try:
        import pygeos  # noqa

        USE_PYGEOS = True
    except ImportError:
        USE_PYGEOS = False

    env_use_pygeos = os.getenv("USE_PYGEOS", None)
    if env_use_pygeos is not None:
        USE_PYGEOS = bool(int(env_use_pygeos))
