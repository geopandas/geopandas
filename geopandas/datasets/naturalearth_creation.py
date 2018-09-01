import geopandas as gpd
world_raw = gpd.read_file("zip://./ne_110m_admin_0_countries.zip") #reads the zip file from https://www.naturalearthdata.com/http//www.naturalearthdata.com/download/10m/cultural/ne_10m_admin_0_countries.zip 
#current version 4.1.0
select_columns = world_raw[['POP_EST', 'CONTINENT', 'NAME', 'ISO_A3', 'GDP_MD_EST', 'geometry']] #selects the columns used in the example dataset and writes them to the example dataset
select_columns.columns = select_columns.columns.str.lower() #changes column names to lowercase
select_columns.to_file(driver='ESRI Shapefile', filename='./naturalearth_lowres/naturalearth_lowres.shp') #writes the data as a shapefile