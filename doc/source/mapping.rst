.. currentmodule:: geopandas

.. ipython:: python
   :suppress:

   import geopandas as gpd


Mapping Tools
=========================================


*geopandas* provides a high-level interface to the ``matplotlib`` library for making maps. Mapping shapes is as easy as using the ``plot()`` method on a ``GeoSeries`` or ``GeoDataFrame``.

Loading some example data:

.. ipython:: python

    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    cities = gpd.read_file(gpd.datasets.get_path('naturalearth_cities'))

We can now plot those GeoDataFrames:

.. ipython:: python

    # Examine country GeoDataFrame
    world.head()

    # Basic plot, random colors
    @savefig world_randomcolors.png width=5in
    world.plot();

Note that in general, any options one can pass to `pyplot <http://matplotlib.org/api/pyplot_api.html>`_ in ``matplotlib`` (or `style options that work for lines <http://matplotlib.org/api/lines_api.html>`_) can be passed to the ``plot()`` method.


Chloropleth Maps
-----------------

*geopandas* makes it easy to create Chloropleth maps (maps where the color of each shape is based on the value of an associated variable). Simply use the plot command with the ``column`` argument set to the column whose values you want used to assign colors.

.. ipython:: python

    # Plot by GDP per capta
    world = world[(world.pop_est>0) & (world.name!="Antarctica")]
    world['gdp_per_cap'] = world.gdp_md_est / world.pop_est
    @savefig world_gdp_per_cap.png width=5in
    world.plot(column='gdp_per_cap');


Choosing colors
~~~~~~~~~~~~~~~~

One can also modify the colors used by ``plot`` with the ``cmap`` option (for a full list of colormaps, see the `matplotlib website <http://matplotlib.org/users/colormaps.html>`_):

.. ipython:: python

    @savefig world_gdp_per_cap_red.png width=5in
    world.plot(column='gdp_per_cap', cmap='OrRd');


The way color maps are scaled can also be manipulated with the ``scheme`` option (if you have ``pysal`` installed, which can be accomplished via ``conda install pysal``). By default, ``scheme`` is set to 'equal_intervals', but it can also be adjusted to any other `pysal option <http://pysal.org/1.2/library/esda/mapclassify.html>`_, like 'quantiles', 'percentiles', etc.

.. ipython:: python

    @savefig world_gdp_per_cap_quantiles.png width=5in
    world.plot(column='gdp_per_cap', cmap='OrRd', scheme='quantiles');


Maps with Layers
-----------------

There are two strategies for making a map with multiple layers -- one more succinct, and one that is a littel more flexible.

Before combining maps, however, remember to always ensure they share a common CRS (so they will align).

.. ipython:: python

    # Look at capitals
    # Note use of standard `pyplot` line style options
    @savefig capitals.png width=5in
    cities.plot(marker='*', color='green', markersize=5);

    # Check crs
    cities = cities.to_crs(world.crs)

    # Now we can overlay over country outlines
    # And yes, there are lots of island capitals
    # apparently in the middle of the ocean!

**Method 1**

.. ipython:: python

    base = world.plot(color='white')
    @savefig capitals_over_countries_1.png width=5in
    cities.plot(ax=base, marker='o', color='red', markersize=5);

**Method 2: Using matplotlib objects**

.. ipython:: python

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()

    # set aspect to equal. This is done automatically
    # when using *geopandas* plot on it's own, but not when
    # working with pyplot directly.
    ax.set_aspect('equal')

    world.plot(ax=ax, color='white')
    cities.plot(ax=ax, marker='o', color='red', markersize=5)
    @savefig capitals_over_countries_2.png width=5in
    plt.show();


Other Resources
-----------------
Links to jupyter Notebooks for different mapping tasks:

`Making Heat Maps <http://nbviewer.jupyter.org/gist/perrygeo/c426355e40037c452434>`_
