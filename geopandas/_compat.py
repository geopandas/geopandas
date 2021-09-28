import contextlib
from distutils.version import LooseVersion
import importlib
import os
import warnings

import numpy as np
import pandas as pd
import pyproj
import shapely
import shapely.geos


# -----------------------------------------------------------------------------
# pandas compat
# -----------------------------------------------------------------------------

PANDAS_GE_10 = str(pd.__version__) >= LooseVersion("1.0.0")
PANDAS_GE_11 = str(pd.__version__) >= LooseVersion("1.1.0")
PANDAS_GE_115 = str(pd.__version__) >= LooseVersion("1.1.5")
PANDAS_GE_12 = str(pd.__version__) >= LooseVersion("1.2.0")


# -----------------------------------------------------------------------------
# Shapely / PyGEOS compat
# -----------------------------------------------------------------------------


SHAPELY_GE_17 = str(shapely.__version__) >= LooseVersion("1.7.0")
SHAPELY_GE_18 = str(shapely.__version__) >= LooseVersion("1.8")
SHAPELY_GE_20 = str(shapely.__version__) >= LooseVersion("2.0")

GEOS_GE_390 = shapely.geos.geos_version >= (3, 9, 0)


HAS_PYGEOS = None
USE_PYGEOS = None
PYGEOS_SHAPELY_COMPAT = None

PYGEOS_GE_09 = None
PYGEOS_GE_010 = None

INSTALL_PYGEOS_ERROR = "To use PyGEOS within GeoPandas, you need to install PyGEOS: \
'conda install pygeos' or 'pip install pygeos'"

try:
    import pygeos  # noqa

    # only automatically use pygeos if version is high enough
    if str(pygeos.__version__) >= LooseVersion("0.8"):
        HAS_PYGEOS = True
        PYGEOS_GE_09 = str(pygeos.__version__) >= LooseVersion("0.9")
        PYGEOS_GE_010 = str(pygeos.__version__) >= LooseVersion("0.10")
    else:
        warnings.warn(
            "The installed version of PyGEOS is too old ({0} installed, 0.8 required),"
            " and thus GeoPandas will not use PyGEOS.".format(pygeos.__version__),
            UserWarning,
        )
        HAS_PYGEOS = False
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
            if not str(pygeos.__version__) >= LooseVersion("0.8"):
                raise ImportError(
                    "PyGEOS >= 0.8 is required, version {0} is installed".format(
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
            raise ImportError(INSTALL_PYGEOS_ERROR)


set_use_pygeos()


# compat related to deprecation warnings introduced in Shapely 1.8
# -> creating a numpy array from a list-like of Multi-part geometries,
# although doing the correct thing (not expanding in its parts), still raises
# the warning about iteration being deprecated
# This adds a context manager to explicitly ignore this warning


try:
    from shapely.errors import ShapelyDeprecationWarning as shapely_warning
except ImportError:
    shapely_warning = None


if shapely_warning is not None and not SHAPELY_GE_20:

    @contextlib.contextmanager
    def ignore_shapely2_warnings():
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", "Iteration|The array interface|__len__", shapely_warning
            )
            yield


elif (str(np.__version__) >= LooseVersion("1.21")) and not SHAPELY_GE_20:

    @contextlib.contextmanager
    def ignore_shapely2_warnings():
        with warnings.catch_warnings():
            # warning from numpy for existing Shapely releases (this is fixed
            # with Shapely 1.8)
            warnings.filterwarnings(
                "ignore", "An exception was ignored while fetching", DeprecationWarning
            )
            yield


else:

    @contextlib.contextmanager
    def ignore_shapely2_warnings():
        yield


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

# -----------------------------------------------------------------------------
# pyproj compat
# -----------------------------------------------------------------------------

PYPROJ_LT_3 = LooseVersion(pyproj.__version__) < LooseVersion("3")
PYPROJ_GE_31 = LooseVersion(pyproj.__version__) >= LooseVersion("3.1")
