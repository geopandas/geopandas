from distutils.version import LooseVersion
import os
import warnings

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
PYGEOS_SHAPELY_COMPAT = None


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
            try:
                import pygeos  # noqa

                USE_PYGEOS = True
            except ImportError:
                USE_PYGEOS = False

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
