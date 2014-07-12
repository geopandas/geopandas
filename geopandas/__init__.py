try:
    from geopandas.version import version as __version__
except ImportError:
    __version__ = '0.1.0.dev-unknown'

from geopandas.geoseries import GeoSeries
from geopandas.geodataframe import GeoDataFrame

from geopandas.io.file import read_file
from geopandas.io.sql import read_postgis

# make the interactive namespace easier to use
# for `from geopandas import *` demos.
import geopandas as gpd
import pandas as pd
import numpy as np
