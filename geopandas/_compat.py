import contextlib
from packaging.version import Version
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

PANDAS_GE_115 = Version(pd.__version__) >= Version("1.1.5")
PANDAS_GE_12 = Version(pd.__version__) >= Version("1.2.0")
PANDAS_GE_13 = Version(pd.__version__) >= Version("1.3.0")
PANDAS_GE_14 = Version(pd.__version__) >= Version("1.4.0rc0")
PANDAS_GE_15 = Version(pd.__version__) >= Version("1.5.0")
PANDAS_GE_20 = Version(pd.__version__) >= Version("2.0.0")
PANDAS_GE_21 = Version(pd.__version__) >= Version("2.1.0.dev0")


# -----------------------------------------------------------------------------
# Shapely / PyGEOS compat
# -----------------------------------------------------------------------------


SHAPELY_GE_18 = Version(shapely.__version__) >= Version("1.8")
SHAPELY_GE_182 = Version(shapely.__version__) >= Version("1.8.2")
SHAPELY_GE_20 = Version(shapely.__version__) >= Version("2.0.0.dev0")
SHAPELY_G_20a1 = Version(shapely.__version__) > Version("2.0a1")

GEOS_GE_390 = shapely.geos.geos_version >= (3, 9, 0)


HAS_PYGEOS = None
USE_PYGEOS = None
USE_SHAPELY_20 = None
PYGEOS_SHAPELY_COMPAT = None

PYGEOS_GE_09 = None
PYGEOS_GE_010 = None

INSTALL_PYGEOS_ERROR = "To use PyGEOS within GeoPandas, you need to install PyGEOS: \
'conda install pygeos' or 'pip install pygeos'"

try:
    import pygeos  # noqa

    # only automatically use pygeos if version is high enough
    if Version(pygeos.__version__) >= Version("0.8"):
        HAS_PYGEOS = True
        PYGEOS_GE_09 = Version(pygeos.__version__) >= Version("0.9")
        PYGEOS_GE_010 = Version(pygeos.__version__) >= Version("0.10")
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
    global USE_SHAPELY_20
    global PYGEOS_SHAPELY_COMPAT

    env_use_pygeos = os.getenv("USE_PYGEOS", None)

    if val is not None:
        USE_PYGEOS = bool(val)
    else:
        if USE_PYGEOS is None:
            USE_PYGEOS = HAS_PYGEOS

            if env_use_pygeos is not None:
                USE_PYGEOS = bool(int(env_use_pygeos))

    # validate the pygeos version
    if USE_PYGEOS:
        try:
            import pygeos  # noqa

            # validate the pygeos version
            if not Version(pygeos.__version__) >= Version("0.8"):
                if SHAPELY_GE_20:
                    USE_PYGEOS = False
                    warnings.warn(
                        "The PyGEOS version is too old, and Shapely >= 2 is installed, "
                        "thus using Shapely by default and not PyGEOS."
                    )
                else:
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

    if USE_PYGEOS and env_use_pygeos is None and SHAPELY_GE_20:
        warnings.warn(
            "Shapely 2.0 is installed, but because PyGEOS is also installed, "
            "GeoPandas still uses PyGEOS by default. However, starting with "
            "version 0.14, the default will switch to Shapely. To force to "
            "use Shapely 2.0 now, you can either uninstall PyGEOS or set the "
            "environment variable USE_PYGEOS=0. "
            "You can do this before starting the Python process, or in your code "
            "before importing geopandas:"
            "\n\nimport os\nos.environ['USE_PYGEOS'] = '0'\nimport geopandas\n\n"
            "In the next release, GeoPandas will switch to using Shapely by default, "
            "even if PyGEOS is installed. If you only have PyGEOS installed to "
            "get speed-ups, this switch should be smooth. However, if you are "
            "using PyGEOS directly (calling PyGEOS functions on geometries "
            "from GeoPandas), this will then stop working and you are encouraged to "
            "migrate from PyGEOS to Shapely 2.0 "
            "(https://shapely.readthedocs.io/en/latest/migration_pygeos.html).",
            DeprecationWarning,
            stacklevel=6,
        )

    USE_SHAPELY_20 = (not USE_PYGEOS) and SHAPELY_GE_20


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

elif (Version(np.__version__) >= Version("1.21")) and not SHAPELY_GE_20:

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

PYPROJ_GE_31 = Version(pyproj.__version__) >= Version("3.1")
PYPROJ_GE_32 = Version(pyproj.__version__) >= Version("3.2")
