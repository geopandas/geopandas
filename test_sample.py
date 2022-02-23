import geopandas


# types
multipolygons = geopandas.read_file(geopandas.datasets.get_path("nybb"))
polygons = multipolygons.explode()
multilinestrings = multipolygons.boundary
linestrings = polygons.boundary