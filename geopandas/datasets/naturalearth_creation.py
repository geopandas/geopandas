import geopandas as gpd
url = 'http://d2ad6b4ur7yvpq.cloudfront.net/naturalearth-3.3.0/ne_110m_admin_0_countries.geojson'
#url for the naturalearth.com earth dataset as a geojson
world_raw = gpd.read_file(url) #reads the geojson file from the url as a geodataframe
world_raw[['pop_est', 'continent', 'name', 'iso_a3', 'gdp_md_est', 'geometry']].to_file(driver='ESRI Shapefile', filename='./naturalearth_lowres/naturalearth_lowres.shp')
#selects the columns used in the example dataset and writes them to the example dataset
