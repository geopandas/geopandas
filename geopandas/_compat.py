from distutils.version import LooseVersion
import importlib
import os
import warnings

import pandas as pd
import shapely

# -----------------------------------------------------------------------------
# pandas compat
# -----------------------------------------------------------------------------

PANDAS_GE_024 = str(pd.__version__) >= LooseVersion("0.24.0")
PANDAS_GE_025 = str(pd.__version__) >= LooseVersion("0.25.0")
PANDAS_GE_10 = str(pd.__version__) >= LooseVersion("0.26.0.dev")
PANDAS_GE_11 = str(pd.__version__) >= LooseVersion("1.1.0.dev")


# -----------------------------------------------------------------------------
# Shapely / PyGEOS compat
# -----------------------------------------------------------------------------


SHAPELY_GE_17 = str(shapely.__version__) >= LooseVersion("1.7.0")

HAS_PYGEOS = None
USE_PYGEOS = None
PYGEOS_SHAPELY_COMPAT = None

try:
    import pygeos  # noqa

    HAS_PYGEOS = True
except ImportError:
    HAS_PYGEOS = False


def set_use_pygeos(val=None):
    """
    Set the global configuration on whether to use PyGEOS or not.

    The default is use PyGEOS if it is installed. This can be overridden
    with an environment variable USE_PYGEOS (this is only checked at
    first import, cannot be changed during interactive session).

    Alternatively, pass a value here to force a True/False value.
    """
    global USE_PYGEOS
    global PYGEOS_SHAPELY_COMPAT

    if val is not None:
        USE_PYGEOS = bool(val)
    else:
        if USE_PYGEOS is None:

            USE_PYGEOS = HAS_PYGEOS

            env_use_pygeos = os.getenv("USE_PYGEOS", None)
            if env_use_pygeos is not None:
                USE_PYGEOS = bool(int(env_use_pygeos))

    # validate the pygeos version
    if USE_PYGEOS:
        try:
            import pygeos  # noqa

            # validate the pygeos version
            if not str(pygeos.__version__) >= LooseVersion("0.6"):
                raise ImportError(
                    "PyGEOS >= 0.6 is required, version {0} is installed".format(
                        pygeos.__version__
                    )
                )

            # Check whether Shapely and PyGEOS use the same GEOS version.
            # Based on PyGEOS from_shapely implementation.

            from shapely.geos import geos_version_string as shapely_geos_version
            from pygeos import geos_capi_version_string

            # shapely has something like: "3.6.2-CAPI-1.10.2 4d2925d6"
            # pygeos has something like: "3.6.2-CAPI-1.10.2"
            if not shapely_geos_version.startswith(geos_capi_version_string):
                warnings.warn(
                    "The Shapely GEOS version ({}) is incompatible with the GEOS "
                    "version PyGEOS was compiled with ({}). Conversions between both "
                    "will be slow.".format(
                        shapely_geos_version, geos_capi_version_string
                    )
                )
                PYGEOS_SHAPELY_COMPAT = False
            else:
                PYGEOS_SHAPELY_COMPAT = True

        except ImportError:
            raise ImportError(
                "To use the PyGEOS speed-ups within GeoPandas, you need to install "
                "PyGEOS: 'conda install pygeos' or 'pip install pygeos'"
            )


set_use_pygeos()


def import_optional_dependency(name: str, extra: str = ""):
    """
    Import an optional dependency.

    Adapted from pandas.compat._optional::import_optional_dependency

    Raises a formatted ImportError if the module is not present.

    Parameters
    ----------
    name : str
        The module name.
    extra : str
        Additional text to include in the ImportError message.
    Returns
    -------
    module
    """
    msg = """Missing optional dependency '{name}'. {extra}  "
        "Use pip or conda to install {name}.""".format(
        name=name, extra=extra
    )

    if not isinstance(name, str):
        raise ValueError(
            "Invalid module name: '{name}'; must be a string".format(name=name)
        )

    try:
        module = importlib.import_module(name)

    except ImportError:
        raise ImportError(msg) from None

    return module


# -----------------------------------------------------------------------------
# RTree compat
# -----------------------------------------------------------------------------

HAS_RTREE = None
RTREE_GE_094 = False
try:
    import rtree  # noqa

    HAS_RTREE = True
except ImportError:
    HAS_RTREE = False
