import geopandas as gpd
import os
import sys


def included_data(data):

    """
    Retrieve geopandas object from library of included spatial data.

    Parameters
    ----------
    data : str
         name of spatial data to retrieve.

         - 'countries': GeoDataFrame of Countries with
            basic information from Natural Earth.
            http://www.naturalearthdata.com/downloads/110m-cultural-vectors/110m-admin-0-countries/
    """

    datasets = {'countries': 'naturalearth_lowres.shp'}

    d = os.path.dirname(sys.modules['geopandas'].__file__)
    path = os.path.join(d, 'included_data', datasets[data])

    return gpd.GeoDataFrame().from_file(path)
