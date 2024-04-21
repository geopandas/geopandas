from geopandas._config import options

from geopandas.geoseries import GeoSeries
from geopandas.geodataframe import GeoDataFrame
from geopandas.array import points_from_xy

from geopandas.io.file import _read_file as read_file
from geopandas.io.file import _list_layers as list_layers
from geopandas.io.arrow import _read_parquet as read_parquet
from geopandas.io.arrow import _read_feather as read_feather
from geopandas.io.sql import _read_postgis as read_postgis
from geopandas.tools import sjoin, sjoin_nearest
from geopandas.tools import overlay
from geopandas.tools._show_versions import show_versions
from geopandas.tools import clip


import geopandas.datasets


# make the interactive namespace easier to use
# for `from geopandas import *` demos.
import geopandas as gpd
import pandas as pd
import numpy as np

from . import _version

__version__ = _version.get_versions()["version"]
