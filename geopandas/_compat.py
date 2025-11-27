import importlib
from packaging.version import Version

import pandas as pd

import shapely

# -----------------------------------------------------------------------------
# pandas compat
# -----------------------------------------------------------------------------

PANDAS_GE_202 = Version(pd.__version__) >= Version("2.0.2")
PANDAS_GE_21 = Version(pd.__version__) >= Version("2.1.0")
PANDAS_GE_22 = Version(pd.__version__) >= Version("2.2.0")
PANDAS_GE_23 = Version(pd.__version__) >= Version("2.3.0")
PANDAS_GE_30 = Version(pd.__version__) >= Version("3.0.0.dev0")
PANDAS_INFER_STR = PANDAS_GE_23 and pd.options.future.infer_string


# -----------------------------------------------------------------------------
# Shapely / GEOS compat
# -----------------------------------------------------------------------------

SHAPELY_GE_204 = Version(shapely.__version__) >= Version("2.0.4")
SHAPELY_GE_21 = Version(shapely.__version__) >= Version("2.1rc1")

GEOS_GE_390 = shapely.geos_version >= (3, 9, 0)
GEOS_GE_310 = shapely.geos_version >= (3, 10, 0)
GEOS_GE_312 = shapely.geos_version >= (3, 12, 0)


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
    msg = f"""Missing optional dependency '{name}'. {extra}  "
        "Use pip or conda to install {name}."""

    if not isinstance(name, str):
        raise ValueError(f"Invalid module name: '{name}'; must be a string")

    try:
        module = importlib.import_module(name)

    except ImportError:
        raise ImportError(msg) from None

    return module


# -----------------------------------------------------------------------------
# pyproj compat
# -----------------------------------------------------------------------------
try:
    import pyproj  # noqa: F401

    HAS_PYPROJ = True

except ImportError as err:
    HAS_PYPROJ = False
    pyproj_import_error = str(err)


def requires_pyproj(func):
    def wrapper(*args, **kwargs):
        if not HAS_PYPROJ:
            raise ImportError(
                f"The 'pyproj' package is required for {func.__name__} to work. "
                "Install it and initialize the object with a CRS before using it."
                f"\nImporting pyproj resulted in: {pyproj_import_error}"
            )
        return func(*args, **kwargs)

    return wrapper
