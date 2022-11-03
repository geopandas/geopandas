from geopandas._config import options  # noqa

from geopandas.geoseries import GeoSeries  # noqa
from geopandas.geodataframe import GeoDataFrame  # noqa
from geopandas.array import points_from_xy  # noqa

from geopandas.io.file import _read_file as read_file  # noqa
from geopandas.io.arrow import _read_parquet as read_parquet  # noqa
from geopandas.io.arrow import _read_feather as read_feather  # noqa
from geopandas.io.sql import _read_postgis as read_postgis  # noqa
from geopandas.tools import sjoin, sjoin_nearest  # noqa
from geopandas.tools import overlay  # noqa
from geopandas.tools._show_versions import show_versions  # noqa
from geopandas.tools import clip  # noqa


import geopandas.datasets  # noqa


# make the interactive namespace easier to use
# for `from geopandas import *` demos.
import geopandas as gpd  # noqa
import pandas as pd  # noqa
import numpy as np  # noqa

from . import _version

__version__ = _version.get_versions()["version"]
