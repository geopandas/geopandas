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


def set_use_pygeos(val=None):
    """
    Set the global configuration on whether to use PyGEOS or not.

    The default is use PyGEOS if it is installed. This can be overridden
    with an environment variable USE_PYGEOS (this is only checked at
    first import, cannot be changed during interactive session).

    Alternatively, pass a value here to force a True/False value.
    """
    global USE_PYGEOS

    if val is not None:
        USE_PYGEOS = bool(val)
    else:
        if USE_PYGEOS is None:
            try:
                import pygeos  # noqa

                USE_PYGEOS = True
            except ImportError:
                USE_PYGEOS = False

            env_use_pygeos = os.getenv("USE_PYGEOS", None)
            if env_use_pygeos is not None:
                USE_PYGEOS = bool(int(env_use_pygeos))


set_use_pygeos()
