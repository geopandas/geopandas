.. ipython:: python
   :suppress:

   import geopandas
   import matplotlib
   orig = matplotlib.rcParams['figure.figsize']
   matplotlib.rcParams['figure.figsize'] = [orig[0] * 1.5, orig[1]]


Aggregation with dissolve
=============================

Spatial data are often more granular than we need. For example, we might have data on sub-national units, but we're actually interested in studying patterns at the level of countries.

In a non-spatial setting, when all we need are summary statistics of the data, we aggregate our data using the ``groupby`` function. But for spatial data, we sometimes also need to aggregate geometric features. In the *geopandas* library, we can aggregate geometric features using the ``dissolve`` function.

``dissolve`` can be thought of as doing three things: (a) it dissolves all the geometries within a given group together into a single geometric feature (using the ``unary_union`` method), and (b) it aggregates all the rows of data in a group using ``groupby.aggregate()``, and (c) it combines those two results.

``dissolve`` Example
~~~~~~~~~~~~~~~~~~~~~

Suppose we are interested in studying continents, but we only have country-level data like the country dataset included in *geopandas*. We can easily convert this to a continent-level dataset.


First, let's look at the most simple case where we just want continent shapes and names. By default, ``dissolve`` will pass ``'first'`` to ``groupby.aggregate``.

.. ipython:: python

    world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
    world = world[['continent', 'geometry']]
    continents = world.dissolve(by='continent')

    @savefig continents1.png
    continents.plot();

    continents.head()

If we are interested in aggregate populations, however, we can pass different functions to the ``dissolve`` method to aggregate populations using the ``aggfunc =`` argument:

.. ipython:: python

   world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
   world = world[['continent', 'geometry', 'pop_est']]
   continents = world.dissolve(by='continent', aggfunc='sum')

   @savefig continents2.png
   continents.plot(column = 'pop_est', scheme='quantiles', cmap='YlOrRd');

   continents.head()


.. ipython:: python
    :suppress:

    matplotlib.rcParams['figure.figsize'] = orig


.. toctree::
   :maxdepth: 2

Dissolve Arguments
~~~~~~~~~~~~~~~~~~

The ``aggfunc =`` argument defaults to 'first' which means that the first row of attributes values found in the dissolve routine will be assigned to the resultant dissolved geodataframe.
However it also accepts other summary statistic options as allowed by ``pandas.groupby()`` including:

* 'first'
* 'last'
* 'min'
* 'max'
* 'sum'
* 'mean'
* 'median'
