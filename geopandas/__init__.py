from __future__ import absolute_import, division, print_function

from geopandas.geoseries import GeoSeries
from geopandas.geodataframe import GeoDataFrame, concat

from geopandas.io.file import read_file
from geopandas.io.sql import read_postgis
from geopandas.tools import sjoin
from geopandas.tools import overlay

from shapely.geometry.base import CAP_STYLE, JOIN_STYLE

import geopandas.datasets

# make the interactive namespace easier to use
# for `from geopandas import *` demos.
import geopandas as gpd
import pandas as pd
import numpy as np

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
