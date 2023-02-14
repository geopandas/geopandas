.. ipython:: python
   :suppress:

   import geopandas
   import matplotlib
   orig = matplotlib.rcParams['figure.figsize']
   matplotlib.rcParams['figure.figsize'] = [orig[0] * 1.5, orig[1]]


Aggregation with dissolve
=============================

Spatial data are often more granular than we need. For example, we might have data on sub-national units, but we're actually interested in studying patterns at the level of countries.

In a non-spatial setting, when all we need are summary statistics of the data, we aggregate our data using the :meth:`~pandas.DataFrame.groupby` function. But for spatial data, we sometimes also need to aggregate geometric features. In the GeoPandas library, we can aggregate geometric features using the :meth:`~geopandas.GeoDataFrame.dissolve` function.

:meth:`~geopandas.GeoDataFrame.dissolve` can be thought of as doing three things:

(a) it dissolves all the geometries within a given group together into a single geometric feature (using the :attr:`~geopandas.GeoSeries.unary_union` method), and
(b) it aggregates all the rows of data in a group using :ref:`groupby.aggregate <groupby.aggregate>`, and
(c) it combines those two results.

:meth:`~geopandas.GeoDataFrame.dissolve` Example
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Suppose we are interested in Nepalese zone, but we only have Nepalese district-level data like the `geoda.nepal` dataset included in `geodatasets`. We can easily convert this to a zone-level dataset.


First, let's look at the most simple case where we just want zone shapes and names. By default, :meth:`~geopandas.GeoDataFrame.dissolve` will pass ``'first'`` to :ref:`groupby.aggregate <groupby.aggregate>`.

.. ipython:: python

    nepal = geopandas.read_file(geodatasets.get_path('geoda nepal'))
    nepal = nepal[['name_2', 'geometry']]  # name_2 contains zone names
    zones = nepal.dissolve(by='name_2')

    @savefig zones1.png
    zones.plot();

    zones.head()

If we are interested in aggregate populations, however, we can pass different functions to the :meth:`~geopandas.GeoDataFrame.dissolve` method to aggregate populations using the ``aggfunc =`` argument:

.. ipython:: python

   nepal = geopandas.read_file(geodatasets.get_path('geoda nepal'))
   nepal = nepal[['name_2', 'geometry', 'population']]  # name_2 contains zone names
   zones = nepal.dissolve(by='name_2', aggfunc='sum')

   @savefig zones2.png
   zones.plot(column = 'population', scheme='quantiles', cmap='YlOrRd');

   zones.head()


.. ipython:: python
    :suppress:

    matplotlib.rcParams['figure.figsize'] = orig


.. toctree::
   :maxdepth: 2

Dissolve arguments
~~~~~~~~~~~~~~~~~~

The ``aggfunc =`` argument defaults to 'first' which means that the first row of attributes values found in the dissolve routine will be assigned to the resultant dissolved geodataframe.
However it also accepts other summary statistic options as allowed by :meth:`pandas.groupby <pandas.DataFrame.groupby>` including:

* 'first'
* 'last'
* 'min'
* 'max'
* 'sum'
* 'mean'
* 'median'
* function
* string function name
* list of functions and/or function names, e.g. [np.sum, 'mean']
* dict of axis labels -> functions, function names or list of such.

For example, to get the number of countries on each continent,
as well as the populations of the largest and smallest country of each,
we can aggregate the ``'name'`` column using ``'count'``,
and the ``'pop_est'`` column using ``'min'`` and ``'max'``:

.. ipython:: python

    nepal = geopandas.read_file(geodatasets.get_path('geoda nepal'))
    zones = nepal.dissolve(
        by="name_2",
        aggfunc={
            "district": "count",
            "population": ["min", "max"],
        },
    )
   zones.head()
