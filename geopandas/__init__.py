try:
    from geopandas.version import version as __version__
except ImportError:
    __version__ = '0.1.0.dev-unknown'
from geoseries import GeoSeries
from geodataframe import GeoDataFrame
