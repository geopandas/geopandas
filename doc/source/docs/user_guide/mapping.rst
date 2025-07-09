.. currentmodule:: geopandas

.. ipython:: python
   :suppress:

   import geopandas
   import matplotlib
   orig = matplotlib.rcParams['figure.figsize']
   matplotlib.rcParams['figure.figsize'] = [orig[0] * 1.5, orig[1] * 1.5]
   import matplotlib.pyplot as plt
   plt.close('all')


Mapping and plotting tools
=========================================


GeoPandas provides a high-level interface to the matplotlib_ library for making maps. Mapping shapes is as easy as using the :meth:`~GeoDataFrame.plot()` method on a :class:`GeoSeries` or :class:`GeoDataFrame`.

.. _matplotlib: https://matplotlib.org/stable/

Loading some example data:

.. ipython:: python

    import geodatasets

    chicago = geopandas.read_file(geodatasets.get_path("geoda.chicago_commpop"))
    groceries = geopandas.read_file(geodatasets.get_path("geoda.groceries"))

You can now plot those GeoDataFrames:

.. ipython:: python

    # Examine the chicago GeoDataFrame
    chicago.head()

    # Basic plot, single color
    @savefig chicago_singlecolor.png
    chicago.plot();

Note that in general, any options one can pass to `pyplot <http://matplotlib.org/api/pyplot_api.html>`_ in matplotlib_ (or `style options that work for lines <http://matplotlib.org/api/lines_api.html>`_) can be passed to the :meth:`~GeoDataFrame.plot` method.


Choropleth maps
-----------------

GeoPandas makes it easy to create Choropleth maps (maps where the color of each shape is based on the value of an associated variable). Simply use the plot command with the ``column`` argument set to the column whose values you want used to assign colors.

.. ipython:: python
   :okwarning:

    # Plot by population
    @savefig chicago_population.png
    chicago.plot(column="POP2010");


Creating a legend
~~~~~~~~~~~~~~~~~

When plotting a map, one can enable a legend using the ``legend`` argument:

.. ipython:: python

    # Plot population estimates with an accurate legend
    @savefig chicago_choro.png
    chicago.plot(column='POP2010', legend=True);

The following example plots the color bar below the map and adds its label using ``legend_kwds``:

.. ipython:: python

    # Plot population estimates with an accurate legend
    @savefig chicago_horizontal.png
    chicago.plot(
        column="POP2010",
        legend=True,
        legend_kwds={"label": "Population in 2010", "orientation": "horizontal"},
    );

However, the default appearance of the legend and plot axes may not be desirable. One can define the plot axes (with ``ax``) and the legend axes (with ``cax``) and then pass those in to the :meth:`~GeoDataFrame.plot` call. The following example uses ``mpl_toolkits`` to horizontally align the plot axes and the legend axes and change the width:

.. ipython:: python

    # Plot population estimates with an accurate legend
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    fig, ax = plt.subplots(1, 1)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("bottom", size="5%", pad=0.1)
    @savefig chicago_cax.png
    chicago.plot(
        column="POP2010",
        ax=ax,
        legend=True,
        cax=cax,
        legend_kwds={"label": "Population in 2010", "orientation": "horizontal"},
    );


Choosing colors
~~~~~~~~~~~~~~~~

You can also modify the colors used by :meth:`~GeoDataFrame.plot` with the ``cmap`` option. For a full list of colormaps, see `Choosing Colormaps in Matplotlib <https://matplotlib.org/stable/tutorials/colors/colormaps.html>`_.

.. ipython:: python

    @savefig chicago_red.png
    chicago.plot(column='POP2010', cmap='OrRd');


To make the color transparent for when you just want to show the boundary, you have two options. One option is to do ``chicago.plot(facecolor="none", edgecolor="black")``. However, this can cause a lot of confusion because ``"none"``  and ``None`` are different in the context of using ``facecolor`` and they do opposite things. ``None`` does the "default behavior" based on matplotlib, and if you use it for ``facecolor``, it actually adds a color. The second option is to use ``chicago.boundary.plot()``. This option is more explicit and clear.:

.. ipython:: python

    @savefig chicago_transparent.png
    chicago.boundary.plot();


The way color maps are scaled can also be manipulated with the ``scheme`` option (if you have ``mapclassify`` installed, which can be accomplished via ``conda install -c conda-forge mapclassify``). The ``scheme`` option can be set to any scheme provided by mapclassify (e.g. 'box_plot', 'equal_interval',
'fisher_jenks', 'fisher_jenks_sampled', 'headtail_breaks', 'jenks_caspall', 'jenks_caspall_forced', 'jenks_caspall_sampled', 'max_p_classifier', 'maximum_breaks', 'natural_breaks', 'quantiles', 'percentiles', 'std_mean' or 'user_defined'). Arguments can be passed in classification_kwds dict. See the `mapclassify documentation <https://pysal.org/mapclassify>`_ for further details about these map classification schemes.

.. ipython:: python

    @savefig chicago_quantiles.png
    chicago.plot(column='POP2010', cmap='OrRd', scheme='quantiles');


Missing data
~~~~~~~~~~~~

In some cases one may want to plot data which contains missing values - for some features one simply does not know the value. Geopandas (from the version 0.7) by defaults ignores such features.

.. ipython:: python

    import numpy as np
    chicago.loc[np.random.choice(chicago.index, 30), 'POP2010'] = np.nan
    @savefig missing_vals.png
    chicago.plot(column='POP2010');

However, passing ``missing_kwds`` one can specify the style and label of features containing None or NaN.

.. ipython:: python

    @savefig missing_vals_grey.png
    chicago.plot(column='POP2010', missing_kwds={'color': 'lightgrey'});

    @savefig missing_vals_hatch.png
    chicago.plot(
        column="POP2010",
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

    ax = chicago.plot()
    @savefig set_axis_off.png
    ax.set_axis_off();

Maps with layers
-----------------

There are two strategies for making a map with multiple layers -- one more succinct, and one that is a little more flexible.

Before combining maps, however, remember to always ensure they share a common CRS (so they will align).

.. ipython:: python

    # Look at capitals
    # Note use of standard `pyplot` line style options
    @savefig capitals.png
    groceries.plot(marker='*', color='green', markersize=5);

    # Check crs
    groceries = groceries.to_crs(chicago.crs)

    # Now you can overlay over the outlines

**Method 1**

.. ipython:: python

    base = chicago.plot(color='white', edgecolor='black')
    @savefig groceries_over_chicago_1.png
    groceries.plot(ax=base, marker='o', color='red', markersize=5);

**Method 2: Using matplotlib objects**

.. ipython:: python

    fig, ax = plt.subplots()

    chicago.plot(ax=ax, color='white', edgecolor='black')
    groceries.plot(ax=ax, marker='o', color='red', markersize=5)
    @savefig groceries_over_chicago_2.png
    plt.show();

Control the order of multiple layers in a plot
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When plotting multiple layers, use ``zorder`` to take control of the order of layers being plotted.
The lower the ``zorder`` is, the lower the layer is on the map and vice versa.

Without specified ``zorder``, cities (Points) gets plotted below world (Polygons), following the default order based on geometry types.

.. ipython:: python

    ax = groceries.plot(color='k')
    @savefig zorder_default.png
    chicago.plot(ax=ax);

You can set the ``zorder`` for cities higher than for world to move it of top.

.. ipython:: python

    ax = groceries.plot(color='k', zorder=2)
    @savefig zorder_set.png
    chicago.plot(ax=ax, zorder=1);


Pandas plots
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

    @savefig pandas_line_plot.png
    chicago.plot(kind="scatter", x="POP2010", y="POP2000")

You can also create these other plots using the ``GeoDataFrame.plot.<kind>`` accessor methods instead of providing the ``kind`` keyword argument.
For example, ``hist``, can be used to plot histograms of population for two different years from the Chicago dataset.

.. ipython:: python

    @savefig pandas_hist_plot.png
    chicago[["POP2000", "POP2010", "geometry"]].plot.hist(alpha=.4)

For more information, see `Chart visualization <https://pandas.pydata.org/pandas-docs/stable/user_guide/visualization.html>`_ in the pandas documentation.


Other resources
-----------------
Links to Jupyter Notebooks for different mapping tasks:

`Making Heat Maps <http://nbviewer.jupyter.org/gist/perrygeo/c426355e40037c452434>`_


.. ipython:: python
    :suppress:

    matplotlib.rcParams['figure.figsize'] = orig


.. ipython:: python
    :suppress:

    import matplotlib.pyplot as plt
    plt.close('all')
