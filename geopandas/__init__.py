from geopandas._config import options  # noqa: F401

from geopandas.geoseries import GeoSeries  # noqa: F401: F401
from geopandas.geodataframe import GeoDataFrame  # noqa: F401
from geopandas.array import points_from_xy  # noqa: F401

from geopandas.io.file import _read_file as read_file  # noqa: F401
from geopandas.io.arrow import _read_parquet as read_parquet  # noqa: F401
from geopandas.io.arrow import _read_feather as read_feather  # noqa: F401
from geopandas.io.sql import _read_postgis as read_postgis  # noqa: F401
from geopandas.tools import sjoin, sjoin_nearest  # noqa: F401
from geopandas.tools import overlay  # noqa: F401
from geopandas.tools._show_versions import show_versions  # noqa: F401
from geopandas.tools import clip  # noqa: F401


import geopandas.datasets  # noqa: F401


# make the interactive namespace easier to use
# for `from geopandas import *` demos.
import geopandas as gpd  # noqa: F401
import pandas as pd  # noqa: F401
import numpy as np  # noqa: F401

from . import _version

__version__ = _version.get_versions()["version"]
