.. currentmodule:: geopandas

.. ipython:: python
   :suppress:

   import geopandas
   import matplotlib.pyplot as plt
   plt.close('all')


Choropleth maps
===============

Choropleth maps represent a continuous or ordinal variable using a color assigned to each geometry. GeoPandas can create choropleth maps natively by passing the column containing those values to the ``column`` argument of the :meth:`~GeoDataFrame.plot` method.

Loading some example data:

.. ipython:: python

    import geodatasets

    south = geopandas.read_file(
        geodatasets.get_path("geoda.south")
    )

Default behavior
----------------

The dataset contains counties in southern US states, with sociodemographic data associated with each county. To plot the percentage of female-headed households in 1990, pass its column name, ``"FH90"``, to ``column``:

.. ipython:: python

    @savefig south.png
    south.plot('FH90');

If you would like to show a legend, pass ``legend=True``. By default, this will create a colorbar.

.. ipython:: python

    @savefig south_legend.png
    south.plot('FH90', legend=True);

The default color mapping stretches the colors from the minimum to the maximum of the observed values. These bounds can be customized with the ``vmin`` and ``vmax`` keywords, which specify the minimum and maximum values respectively. In that case, the colorbar indicates that some values go beyond the extremes.

.. ipython:: python

    @savefig south_vmax.png
    south.plot('FH90', legend=True, vmax=30);

Classification schemes
----------------------

The default mapping of values to colors is linear, which may not provide an optimal cartographic visualization of the data. The mapping can be adjusted using the ``scheme`` keyword, which specifies a classification scheme. You can, for example, map colors to quantiles, where each color bin contains the same number of geometries. Note that using ``scheme`` requires the optional ``mapclassify`` dependency.


.. ipython:: python

    @savefig south_quantiles.png
    south.plot('FH90', legend=True, scheme='quantiles');

This changes the mapping of colors but, by default, also changes the style of the legend. This default exists for legacy reasons and can be useful on its own. However, a more suitable option is often to keep the colorbar, which you can request with ``legend_kwds``.

.. ipython:: python

    @savefig south_scheme_cbar.png
    south.plot(
        'FH90',
        legend=True,
        scheme='quantiles',
        legend_kwds={"colorbar": True},
    );

There are two options for spacing colors in the colorbar. The default is proportional spacing, where each bin gets a portion of the colorbar proportional to the range of values it represents. When bin edges are densely packed around the same values, this can yield a suboptimal result. In that case, it may be more useful to use uniform spacing, where each bin gets an equal portion of the colorbar and each color has enough space.

.. ipython:: python

    @savefig south_scheme_cbar_uniform.png
    south.plot(
        'FH90',
        legend=True,
        scheme='quantiles',
        legend_kwds={"colorbar": True, "spacing": "uniform"},
    );

Most schemes support the ``k`` argument, which indicates the number of bins to use. Larger values are recommended when using a colorbar legend.

.. ipython:: python

    @savefig south_scheme_k.png
    south.plot(
        'FH90',
        legend=True,
        scheme='quantiles',
        k=10,
        legend_kwds={"colorbar": True},
    );

Classification schemes are provided by the ``mapclassify`` package, and any of its supported schemes can be used here. The current options include:

* ``"BoxPlot"``
* ``"EqualInterval"``
* ``"FisherJenks"``
* ``"FisherJenksSampled"``
* ``"HeadTailBreaks"``
* ``"JenksCaspall"``
* ``"JenksCaspallForced"``
* ``"JenksCaspallSampled"``
* ``"MaxP"``
* ``"MaximumBreaks"``
* ``"NaturalBreaks"``
* ``"Percentiles"``
* ``"PrettyBreaks"``
* ``"Quantiles"``
* ``"StdMean"``
* ``"UserDefined"``

Different classification schemes can emphasize different patterns in the same data.

.. ipython:: python

    fig, axs = plt.subplots(2, 2)

    schemes = ["EqualInterval", "Quantiles", "NaturalBreaks", "FisherJenks"]

    for scheme, ax in zip(schemes, axs.flat):
        south.plot(
            "FH90",
            scheme=scheme,
            legend=True,
            legend_kwds={"colorbar": True},
            ax=ax)
        ax.set_title(scheme)
        ax.set_axis_off()
    @savefig schemes.png
    plt.tight_layout()

GeoPandas also supports ``scheme="greedy"``, which uses :func:`mapclassify.greedy` to derive a topological coloring where neighboring geometries do not share the same color. This scheme cannot be used together with ``column`` because it does not classify attribute values.

.. ipython:: python

    @savefig greedy.png
    south.plot(scheme="greedy", cmap="Set3");

Arguments for the classification scheme can be passed as a dictionary to ``classification_kwds``. See the `mapclassify documentation <https://pysal.org/mapclassify/>`__ for details on the available schemes and their parameters.

.. ipython:: python

    @savefig class_kwds.png
    south.plot(
        "FH90",
        legend=True,
        scheme="UserDefined",
        classification_kwds={
            "bins": [10, 20, 30],
            "lowest": 0
        },
        legend_kwds={"colorbar": True},
    );

Colormaps
---------

The colormap used in the choropleth map can be any string recognized by Matplotlib or any :class:`matplotlib.colors.Colormap` object.

.. ipython:: python

    fig, axs = plt.subplots(2, 2)

    cmaps = ["viridis", "cividis", "plasma", "managua"]

    for cmap, ax in zip(cmaps, axs.flat):
        south.plot('FH90', cmap=cmap, ax=ax)
        ax.set_title(cmap)
    @savefig cmaps.png
    plt.tight_layout()

.. ipython:: python
    :suppress:

    plt.close('all')
