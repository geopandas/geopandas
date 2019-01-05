"""
Script that generates the included dataset 'naturalearth_lowres.shp'.

Raw data: https://www.naturalearthdata.com/downloads/10m-cultural-vectors/10m-admin-0-countries/
Current version used: version 4.1.0
"""

import geopandas as gpd

# assumes zipfile from naturalearthdata was downloaded to current directory
world_raw = gpd.read_file("zip://./ne_110m_admin_0_countries.zip")
# subsets columns of interest for geopandas examples
world_df = world_raw[['POP_EST', 'CONTINENT', 'NAME', 'ISO_A3',
                      'GDP_MD_EST', 'geometry']]
world_df.columns = world_df.columns.str.lower()
world_df.to_file(driver='ESRI Shapefile',
                 filename='./naturalearth_lowres/naturalearth_lowres.shp')
