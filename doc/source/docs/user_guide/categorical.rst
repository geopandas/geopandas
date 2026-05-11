.. currentmodule:: geopandas

.. ipython:: python
   :suppress:

   import geopandas
   import matplotlib.pyplot as plt
   plt.close('all')


Categorical maps
================

Categorical maps visualize distinct categories linked to individual observations. As such, they usually use different colormaps and users have different expectations about their behavior. GeoPandas can create categorical maps natively by passing the column containing those values to the ``column`` argument of the :meth:`~GeoDataFrame.plot` method. The method infers whether the value is categorical based on its dtype. Alternatively, you can force a numeric column to be treated as categorical by passing ``categorical=True``.

.. ipython:: python

    import geodatasets

    guerry = geopandas.read_file(
        geodatasets.get_path("geoda.guerry")
    )

Default behavior
----------------

By default, when GeoPandas detects categorical data, it switches the default colormap to ``"tab10"`` (or ``"tab20"`` for more than 10 unique categories) for better distinction between the classes.

.. ipython:: python

    @savefig guerry.png
    guerry.plot("Region");

You can request a legend using ``legend=True``, which shows one legend item for each unique category, with a handle reflecting the style mapped to the class. Additional keywords to adapt the legend can be passed in a ``legend_kwds`` dictionary.

.. ipython:: python

    @savefig guerry_legend.png
    guerry.plot("Region", legend=True, legend_kwds={"loc": "lower left"});

Alternatively, you can use Matplotlib's tooling to generate the legend from the figure using :func:`matplotlib.pyplot.legend`.

.. ipython:: python

    guerry.plot("Region")
    @savefig guerry_legend_plt.png
    plt.legend(loc="lower left");

Colormap treatment
------------------

While GeoPandas uses a categorical colormap by default, any colormap understood by Matplotlib can be used. GeoPandas attempts to detect whether the colormap is categorical or continuous and adapts the mapping of categories to colors accordingly. With categorical colormaps (those with less than 32 unique colors), it iterates over colors sequentially. For continuous colormaps, it retrieves colors proportionally from the entire extent of the colormap for optimal contrast.

.. ipython:: python

    fig, axs = plt.subplots(1, 2)

    cmaps = [
        "Set2",  # categorical
        "Reds",  # continuous
    ]

    @savefig guerry_color_mapping.png
    for cmap, ax in zip(cmaps, axs):
        guerry.plot("Region", cmap=cmap, ax=ax, legend=True)
        ax.set_title(cmap)
        ax.set_axis_off()

Mapping styles to categories
----------------------------

However, a common use case is mapping custom colors to individual categories. This can be done using a dictionary with unique categories as keys and colors as values.

.. ipython:: python

    colors = {
        'C': '#f8da20',
        'E': '#924b4e',
        'N': '#99cfba',
        'S': '#332737',
        'W': '#aaa08c',
    }

    @savefig guerry_custom_cmap.png
    guerry.plot("Region", cmap=colors, legend=True, legend_kwds={"loc": "lower left"});

Other styles can be mapped in an analogous way. Another option is to pass an array of the same length as the GeoDataFrame to be mapped to individual geometries.

.. ipython:: python

    hatches = {
        'C': '//',
        'E': r'\\',
        'N': '||',
        'S': 'O',
        'W': 'o',
    }

    edgecolors = {
        'C': 'black',
        'E': 'black',
        'N': 'white',
        'S': 'white',
        'W': 'white',
    }

    @savefig guerry_custom_mapping.png
    guerry.plot(
        "Region",
        cmap=colors,
        hatch=hatches,
        edgecolor=edgecolors,
        linewidth=0,
        legend=True,
        legend_kwds={"loc": "lower left"},
    );

Maps with multiple layers
-------------------------

When creating a map with multiple layers, legend entries for all data plotted on the same Axes are merged. You can also see that the legend handles reflect the geometry types and their specific symbology for easier identification.

.. ipython:: python

    ax = guerry.plot("Region", cmap=colors)
    ax = guerry.set_geometry(guerry.centroid).iloc[:10].plot(
        "Dprtmnt",
        ax=ax,
        cmap="tab10",
        edgecolor="k",
        markersize="Wealth",
        marker="X",
    )
    @savefig guerry_layers.png
    ax.legend(bbox_to_anchor=(1.45, 1), frameon=False);
