.. currentmodule:: geopandas

.. ipython:: python
   :suppress:

   import geopandas as gpd
   world = gpd.GeoDataFrame().from_file('_example_data/naturalearth_lowres.shp')



Mapping Tools
=========================================


*geopandas* provides a high-level interface to the ``matplotlib`` library to make making maps easy. Mapping shapes is as easy as using the ``plot()`` method on a ``GeoSeries`` or ``GeoDataFrame``:

.. ipython:: python
    
    # Examine country GeoDataFrame
    world.head()

    # Basic plot, random colors
    @savefig world_randomcolors.png width=5in
    world.plot();



Chloropleth Maps
-----------------

*geopandas* makes it easy to create Chloropleth maps (maps where the color of each shape is based on the value of an associated variable). Simply use the plot command with the ``column`` argument set to the column whose values you want used to assign colors. 

.. ipython:: python

    # Examine country GeoDataFrame
    world.head()

    # Basic plot, random colors
    @savefig world_randomcolors.png width=5in
    world.plot();

    # Plot by GDP per capta
    world = world[(world.pop_est>0) & (world.name!="Antarctica")]
    @savefig world_gdp_per_cap.png width=5in
    world.plot(column='gdp_per_cap');


Color Maps
~~~~~~~~~~

One can also modify the colors used by ``plot`` with the ``cmap`` option (for a full list of colormaps, see the `matplotlib website <http://matplotlib.org/users/colormaps.html>`_):

.. ipython:: python
    @savefig world_gdp_per_cap_red.png width=5in
    world.plot(column='gdp_per_cap', cmap='OrRd')


The way color maps are scaled can also be manipulated with the ``scheme`` option. By default, ``scheme`` is set to 'equal_intervals', but it can also be adjusted to any other `pysal option <http://pysal.org/1.2/library/esda/mapclassify.html>`_, like 'quantiles', 'percentiles', etc.  

.. ipython:: python
    @savefig world_gdp_per_cap_quantiles.png width=5in
    world.plot(column='gdp_per_cap', cmap='OrRd', scheme='quantiles')


Maps with Layers
-----------------

[anyone know how to do this?]

Other Resources
-----------------
Links to jupyter Notebooks for different mapping tasks:

`Making Heat Maps <http://nbviewer.jupyter.org/gist/perrygeo/c426355e40037c452434>`_
