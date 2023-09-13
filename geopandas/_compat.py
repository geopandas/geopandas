from packaging.version import Version
import importlib

import pandas as pd
import shapely
import shapely.geos


# -----------------------------------------------------------------------------
# pandas compat
# -----------------------------------------------------------------------------

PANDAS_GE_14 = Version(pd.__version__) >= Version("1.4.0rc0")
PANDAS_GE_15 = Version(pd.__version__) >= Version("1.5.0")
PANDAS_GE_20 = Version(pd.__version__) >= Version("2.0.0")
PANDAS_GE_21 = Version(pd.__version__) >= Version("2.1.0")


# -----------------------------------------------------------------------------
# Shapely / GEOS compat
# -----------------------------------------------------------------------------


GEOS_GE_390 = shapely.geos.geos_version >= (3, 9, 0)


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
    import rtree  # noqa: F401

    HAS_RTREE = True
except ImportError:
    HAS_RTREE = False


# -----------------------------------------------------------------------------
# pyproj compat
# -----------------------------------------------------------------------------
