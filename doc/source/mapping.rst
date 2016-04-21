.. currentmodule:: geopandas

.. ipython:: python
   :suppress:

   import geopandas as gpd
   world = gpd.GeoDataFrame().from_file('_example_data/naturalearth_lowres.shp')



Mapping Tools
=========================================

mapping!


Chloropleth Maps
-----------------

.. ipython:: python

    # Examine country GeoDataFrame
    world.head()

    # Basic plot, random colors
    @savefig world_randomcolors.png width=5in
    world.plot();

    # Plot by GDP per capta
    world['gdp_per_cap'] = world.gdp_md_est / world.pop_est
    @savefig world_gdp_per_cap.png width=5in
    world.plot(column='gdp_per_cap');


Adding Basemaps
-----------------
