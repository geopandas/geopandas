.. ipython:: python
   :suppress:

   import geopandas as gpd


Aggregation with dissolve
=============================

It is often the case that we find ourselves working with spatial data that is more granular than we need. For example, we might have data on sub-national units, but we're actually interested in studying patterns at the level of countries.

In a non-spatial setting, we aggregate our data using the ``groupby`` function. But when working with spatial data, we need a special tool that can also aggregate geometric features. In the *geopandas* library, that functionality is provided by the ``dissolve`` function.

``dissolve`` can be thought of as doing three things: (a) it dissolves all the geometries within a given group together into a single geometric feature (using the ``unary_union`` method), and (b) it aggregates all the rows of data in a group using ``groupby.aggregate()``, and (c) it combines those two results.

``dissolve`` Example
~~~~~~~~~~~~~~~~~~~~~

Suppose we are interested in studying continents, but we only have country-level data like the country dataset included in *geopandas*. We can easily convert this to a continent-level dataset.


First, let's look at the most simple case where we just want continent shapes and names. By default, ``dissolve`` will pass ``'first'`` to ``groupby.aggregate``.

.. ipython:: python

    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    world = world[['continent', 'geometry']]
    continents = world.dissolve(by='continent')

    @savefig continents.png width=5in
    continents.plot();

    continents.head()

If we are interested in aggregate populations, however, we can pass different functions to the ``dissolve`` method to aggregate populations:

.. ipython:: python

   world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
   world = world[['continent', 'geometry', 'pop_est']]
   continents = world.dissolve(by='continent', aggfunc='sum')

   @savefig continents.png width=5in
   continents.plot(column = 'pop_est', scheme='quantiles', cmap='YlOrRd');

   continents.head()



.. toctree::
   :maxdepth: 2
