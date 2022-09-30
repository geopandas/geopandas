"""
Script that generates the included dataset 'naturalearth_lowres.shp'.

Raw data: https://www.naturalearthdata.com/http//www.naturalearthdata.com/download/110m/cultural/ne_110m_admin_0_countries.zip
Current version used: version 5.0.1
"""  # noqa (E501 link is longer than max line length)

import geopandas as gpd

# assumes zipfile from naturalearthdata was downloaded to current directory
world_raw = gpd.read_file("zip://./ne_110m_admin_0_countries.zip")

# not ideal - fix some country codes
mask = world_raw["ISO_A3"].eq("-99") & world_raw["TYPE"].isin(
    ["Sovereign country", "Country"]
)
world_raw.loc[mask, "ISO_A3"] = world_raw.loc[mask, "ADM0_A3"]

# subsets columns of interest for geopandas examples
world_df = world_raw[
    ["POP_EST", "CONTINENT", "NAME", "ISO_A3", "GDP_MD", "geometry"]
].rename(
    columns={"GDP_MD": "GDP_MD_EST"}
)  # column has changed name...
world_df.columns = world_df.columns.str.lower()

world_df.to_file(
    driver="ESRI Shapefile", filename="./naturalearth_lowres/naturalearth_lowres.shp"
)
