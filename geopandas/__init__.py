from geopandas.geoseries import GeoSeries  # noqa
from geopandas.geodataframe import GeoDataFrame  # noqa
from geopandas.array import _points_from_xy as points_from_xy  # noqa

from geopandas.io.file import read_file  # noqa
from geopandas.io.sql import read_postgis  # noqa
from geopandas.tools import sjoin  # noqa
from geopandas.tools import overlay  # noqa
from geopandas.tools._show_versions import show_versions  # noqa

import geopandas.datasets  # noqa

from geopandas._config import options  # noqa

# make the interactive namespace easier to use
# for `from geopandas import *` demos.
import geopandas as gpd  # noqa
import pandas as pd  # noqa
import numpy as np  # noqa

from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions
