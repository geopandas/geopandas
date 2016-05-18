try:
    from geopandas.version import version as __version__
except ImportError:
    __version__ = '0.2.0.dev-unknown'

from geopandas.geoseries import GeoSeries
from geopandas.geodataframe import GeoDataFrame

from geopandas.io.file import read_file
from geopandas.io.sql import read_postgis
from geopandas.tools import sjoin
from geopandas.tools import overlay

import geopandas.datasets

# make the interactive namespace easier to use
# for `from geopandas import *` demos.
import geopandas as gpd
import pandas as pd
import numpy as np
