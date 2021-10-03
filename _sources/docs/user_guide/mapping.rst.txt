.. currentmodule:: geopandas

.. ipython:: python
   :suppress:

   import geopandas
   import matplotlib
   orig = matplotlib.rcParams['figure.figsize']
   matplotlib.rcParams['figure.figsize'] = [orig[0] * 1.5, orig[1]]
   import matplotlib.pyplot as plt
   plt.close('all')


Mapping and Plotting Tools
=========================================


*geopandas* provides a high-level interface to the matplotlib_ library for making maps. Mapping shapes is as easy as using the :meth:`~GeoDataFrame.plot()` method on a :class:`GeoSeries` or :class:`GeoDataFrame`.

.. _matplotlib: https://matplotlib.org/stable/

Loading some example data:

.. ipython:: python

    world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
    cities = geopandas.read_file(geopandas.datasets.get_path('naturalearth_cities'))

We can now plot those GeoDataFrames:

.. ipython:: python

    # Examine country GeoDataFrame
    world.head()

    # Basic plot, random colors
    @savefig world_randomcolors.png
    world.plot();

Note that in general, any options one can pass to `pyplot <http://matplotlib.org/api/pyplot_api.html>`_ in matplotlib_ (or `style options that work for lines <http://matplotlib.org/api/lines_api.html>`_) can be passed to the :meth:`~GeoDataFrame.plot` method.


Choropleth Maps
-----------------

*geopandas* makes it easy to create Choropleth maps (maps where the color of each shape is based on the value of an associated variable). Simply use the plot command with the ``column`` argument set to the column whose values you want used to assign colors.

.. ipython:: python
   :okwarning:

    # Plot by GDP per capita
    world = world[(world.pop_est>0) & (world.name!="Antarctica")]
    world['gdp_per_cap'] = world.gdp_md_est / world.pop_est
    @savefig world_gdp_per_cap.png
    world.plot(column='gdp_per_cap');


Creating a legend
~~~~~~~~~~~~~~~~~

When plotting a map, one can enable a legend using the ``legend`` argument:

.. ipython:: python

    # Plot population estimates with an accurate legend
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 1)
    @savefig world_pop_est.png
    world.plot(column='pop_est', ax=ax, legend=True)

However, the default appearance of the legend and plot axes may not be desirable. One can define the plot axes (with ``ax``) and the legend axes (with ``cax``) and then pass those in to the :meth:`~GeoDataFrame.plot` call. The following example uses ``mpl_toolkits`` to vertically align the plot axes and the legend axes:

.. ipython:: python

    # Plot population estimates with an accurate legend
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    fig, ax = plt.subplots(1, 1)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    @savefig world_pop_est_fixed_legend_height.png
    world.plot(column='pop_est', ax=ax, legend=True, cax=cax)


And the following example plots the color bar below the map and adds its label using ``legend_kwds``:

.. ipython:: python

    # Plot population estimates with an accurate legend
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 1)
    @savefig world_pop_est_horizontal.png
    world.plot(column='pop_est',
               ax=ax,
               legend=True,
               legend_kwds={'label': "Population by Country",
                            'orientation': "horizontal"})


Choosing colors
~~~~~~~~~~~~~~~~

One can also modify the colors used by :meth:`~GeoDataFrame.plot` with the ``cmap`` option (for a full list of colormaps, see the `matplotlib website <http://matplotlib.org/users/colormaps.html>`_):

.. ipython:: python

    @savefig world_gdp_per_cap_red.png
    world.plot(column='gdp_per_cap', cmap='OrRd');


To make the color transparent for when you just want to show the boundary, you have two options. One option is to do ``world.plot(facecolor="none", edgecolor="black")``. However, this can cause a lot of confusion because ``"none"``  and ``None`` are different in the context of using ``facecolor`` and they do opposite things. ``None`` does the "default behavior" based on matplotlib, and if you use it for ``facecolor``, it actually adds a color. The second option is to use ``world.boundary.plot()``. This option is more explicit and clear.:

.. ipython:: python

    @savefig world_gdp_per_cap_transparent.png
    world.boundary.plot();


The way color maps are scaled can also be manipulated with the ``scheme`` option (if you have ``mapclassify`` installed, which can be accomplished via ``conda install -c conda-forge mapclassify``). The ``scheme`` option can be set to any scheme provided by mapclassify (e.g. 'box_plot', 'equal_interval',
'fisher_jenks', 'fisher_jenks_sampled', 'headtail_breaks', 'jenks_caspall', 'jenks_caspall_forced', 'jenks_caspall_sampled', 'max_p_classifier', 'maximum_breaks', 'natural_breaks', 'quantiles', 'percentiles', 'std_mean' or 'user_defined'). Arguments can be passed in classification_kwds dict. See the `mapclassify documentation <https://pysal.org/mapclassify>`_ for further details about these map classification schemes.

.. ipython:: python

    @savefig world_gdp_per_cap_quantiles.png
    world.plot(column='gdp_per_cap', cmap='OrRd', scheme='quantiles');


Missing data
~~~~~~~~~~~~

In some cases one may want to plot data which contains missing values - for some features one simply does not know the value. Geopandas (from the version 0.7) by defaults ignores such features.

.. ipython:: python

    import numpy as np
    world.loc[np.random.choice(world.index, 40), 'pop_est'] = np.nan
    @savefig missing_vals.png
    world.plot(column='pop_est');

However, passing ``missing_kwds`` one can specify the style and label of features containing None or NaN.

.. ipython:: python

    @savefig missing_vals_grey.png
    world.plot(column='pop_est', missing_kwds={'color': 'lightgrey'});

    @savefig missing_vals_hatch.png
    world.plot(
        column="pop_est",
        legend=True,
        scheme="quantiles",
        figsize=(15, 10),
        missing_kwds={
            "color": "lightgrey",
            "edgecolor": "red",
            "hatch": "///",
            "label": "Missing values",
        },
    );

Other map customizations
~~~~~~~~~~~~~~~~~~~~~~~~

Maps usually do not have to have axis labels. You can turn them off using ``set_axis_off()`` or ``axis("off")`` axis methods.

.. ipython:: python

    ax = world.plot()
    @savefig set_axis_off.png
    ax.set_axis_off();

Maps with Layers
-----------------

There are two strategies for making a map with multiple layers -- one more succinct, and one that is a little more flexible.

Before combining maps, however, remember to always ensure they share a common CRS (so they will align).

.. ipython:: python

    # Look at capitals
    # Note use of standard `pyplot` line style options
    @savefig capitals.png
    cities.plot(marker='*', color='green', markersize=5);

    # Check crs
    cities = cities.to_crs(world.crs)

    # Now we can overlay over country outlines
    # And yes, there are lots of island capitals
    # apparently in the middle of the ocean!

**Method 1**

.. ipython:: python

    base = world.plot(color='white', edgecolor='black')
    @savefig capitals_over_countries_1.png
    cities.plot(ax=base, marker='o', color='red', markersize=5);

**Method 2: Using matplotlib objects**

.. ipython:: python

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()

    # set aspect to equal. This is done automatically
    # when using *geopandas* plot on it's own, but not when
    # working with pyplot directly.
    ax.set_aspect('equal')

    world.plot(ax=ax, color='white', edgecolor='black')
    cities.plot(ax=ax, marker='o', color='red', markersize=5)
    @savefig capitals_over_countries_2.png
    plt.show();

Control the order of multiple layers in a plot
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When plotting multiple layers, use ``zorder`` to take control of the order of layers being plotted.
The lower the ``zorder`` is, the lower the layer is on the map and vice versa.

Without specified ``zorder``, cities (Points) gets plotted below world (Polygons), following the default order based on geometry types.

.. ipython:: python

    ax = cities.plot(color='k')
    @savefig zorder_default.png
    world.plot(ax=ax);

We can set the ``zorder`` for cities higher than for world to move it of top.

.. ipython:: python

    ax = cities.plot(color='k', zorder=2)
    @savefig zorder_set.png
    world.plot(ax=ax, zorder=1);


Pandas Plots
-----------------

Plotting methods also allow for different plot styles from pandas
along with the default ``geo`` plot. These methods can be accessed using
the ``kind`` keyword argument in :meth:`~GeoDataFrame.plot`, and include:

* ``geo`` for mapping
* ``line`` for line plots
* ``bar`` or ``barh`` for bar plots
* ``hist`` for histogram
* ``box`` for boxplot
* ``kde`` or ``density`` for density plots
* ``area``  for area plots
* ``scatter`` for scatter plots
* ``hexbin`` for hexagonal bin plots
* ``pie`` for pie plots

.. ipython:: python

    gdf = world.head(10)
    @savefig pandas_line_plot.png
    gdf.plot(kind='scatter', x="pop_est", y="gdp_md_est")

You can also create these other plots using the ``GeoDataFrame.plot.<kind>`` accessor methods instead of providing the ``kind`` keyword argument.

.. ipython:: python

    @savefig pandas_bar_plot.png
    gdf.plot.bar()

For more information check out the `pandas documentation <https://pandas.pydata.org/pandas-docs/stable/user_guide/visualization.html>`_.


Other Resources
-----------------
Links to jupyter Notebooks for different mapping tasks:

`Making Heat Maps <http://nbviewer.jupyter.org/gist/perrygeo/c426355e40037c452434>`_


.. ipython:: python
    :suppress:

    matplotlib.rcParams['figure.figsize'] = orig


.. ipython:: python
    :suppress:

    import matplotlib.pyplot as plt
    plt.close('all')
