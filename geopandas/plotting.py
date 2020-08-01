import warnings

import numpy as np
import pandas as pd

import geopandas

from distutils.version import LooseVersion


def deprecated(new):
    """Helper to provide deprecation warning."""

    def old(*args, **kwargs):
        warnings.warn(
            "{} is intended for internal ".format(new.__name__[1:])
            + "use only, and will be deprecated.",
            DeprecationWarning,
            stacklevel=2,
        )
        new(*args, **kwargs)

    return old


def _flatten_multi_geoms(geoms, prefix="Multi"):
    """
    Returns Series like geoms and index, except that any Multi geometries
    are split into their components and indices are repeated for all component
    in the same Multi geometry.  Maintains 1:1 matching of geometry to value.

    Prefix specifies type of geometry to be flatten. 'Multi' for MultiPoint and similar,
    "Geom" for GeometryCollection.

    Returns
    -------
    components : list of geometry

    component_index : index array
        indices are repeated for all components in the same Multi geometry
    """
    components, component_index = [], []

    if not geoms.geom_type.str.startswith(prefix).any():
        return geoms, np.arange(len(geoms))

    for ix, geom in enumerate(geoms):
        if geom.type.startswith(prefix):
            for poly in geom:
                components.append(poly)
                component_index.append(ix)
        else:
            components.append(geom)
            component_index.append(ix)

    return components, np.array(component_index)


def _expand_kwargs(kwargs, multiindex):
    """
    Most arguments to the plot functions must be a (single) value, or a sequence
    of values. This function checks each key-value pair in 'kwargs' and expands
    it (in place) to the correct length/formats with help of 'multiindex', unless
    the value appears to already be a valid (single) value for the key.
    """
    from matplotlib.colors import is_color_like
    from typing import Iterable

    for att, value in kwargs.items():
        if "color" in att:  # color(s), edgecolor(s), facecolor(s)
            if is_color_like(value):
                continue
        elif "linestyle" in att:  # linestyle(s)
            # A single linestyle can be 2-tuple of a number and an iterable.
            if (
                isinstance(value, tuple)
                and len(value) == 2
                and isinstance(value[1], Iterable)
            ):
                continue
        elif att in ["marker", "alpha"]:
            # For these attributes, only a single value is allowed, so never expand.
            continue

        if pd.api.types.is_list_like(value):
            kwargs[att] = np.take(value, multiindex, axis=0)


def _plot_polygon_collection(
    ax, geoms, values=None, color=None, cmap=None, vmin=None, vmax=None, **kwargs
):
    """
    Plots a collection of Polygon and MultiPolygon geometries to `ax`

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        where shapes will be plotted
    geoms : a sequence of `N` Polygons and/or MultiPolygons (can be mixed)

    values : a sequence of `N` values, optional
        Values will be mapped to colors using vmin/vmax/cmap. They should
        have 1:1 correspondence with the geometries (not their components).
        Otherwise follows `color` / `facecolor` kwargs.
    edgecolor : single color or sequence of `N` colors
        Color for the edge of the polygons
    facecolor : single color or sequence of `N` colors
        Color to fill the polygons. Cannot be used together with `values`.
    color : single color or sequence of `N` colors
        Sets both `edgecolor` and `facecolor`
    **kwargs
        Additional keyword arguments passed to the collection

    Returns
    -------
    collection : matplotlib.collections.Collection that was plotted
    """

    try:
        from descartes.patch import PolygonPatch
    except ImportError:
        raise ImportError(
            "The descartes package is required for plotting polygons in geopandas. "
            "You can install it using 'conda install -c conda-forge descartes' or "
            "'pip install descartes'."
        )
    from matplotlib.collections import PatchCollection

    geoms, multiindex = _flatten_multi_geoms(geoms)
    if values is not None:
        values = np.take(values, multiindex, axis=0)

    # PatchCollection does not accept some kwargs.
    kwargs = {
        att: value
        for att, value in kwargs.items()
        if att not in ["markersize", "marker"]
    }

    # Add to kwargs for easier checking below.
    if color is not None:
        kwargs["color"] = color

    _expand_kwargs(kwargs, multiindex)

    collection = PatchCollection([PolygonPatch(poly) for poly in geoms], **kwargs)

    if values is not None:
        collection.set_array(np.asarray(values))
        collection.set_cmap(cmap)
        if "norm" not in kwargs:
            collection.set_clim(vmin, vmax)

    ax.add_collection(collection, autolim=True)
    ax.autoscale_view()
    return collection


plot_polygon_collection = deprecated(_plot_polygon_collection)


def _plot_linestring_collection(
    ax, geoms, values=None, color=None, cmap=None, vmin=None, vmax=None, **kwargs
):
    """
    Plots a collection of LineString and MultiLineString geometries to `ax`

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        where shapes will be plotted
    geoms : a sequence of `N` LineStrings and/or MultiLineStrings (can be
            mixed)
    values : a sequence of `N` values, optional
        Values will be mapped to colors using vmin/vmax/cmap. They should
        have 1:1 correspondence with the geometries (not their components).
    color : single color or sequence of `N` colors
        Cannot be used together with `values`.

    Returns
    -------
    collection : matplotlib.collections.Collection that was plotted
    """
    from matplotlib.collections import LineCollection

    geoms, multiindex = _flatten_multi_geoms(geoms)
    if values is not None:
        values = np.take(values, multiindex, axis=0)

    # LineCollection does not accept some kwargs.
    kwargs = {
        att: value
        for att, value in kwargs.items()
        if att not in ["markersize", "marker"]
    }

    # Add to kwargs for easier checking below.
    if color is not None:
        kwargs["color"] = color

    _expand_kwargs(kwargs, multiindex)

    segments = [np.array(linestring)[:, :2] for linestring in geoms]
    collection = LineCollection(segments, **kwargs)

    if values is not None:
        collection.set_array(np.asarray(values))
        collection.set_cmap(cmap)
        if "norm" not in kwargs:
            collection.set_clim(vmin, vmax)

    ax.add_collection(collection, autolim=True)
    ax.autoscale_view()
    return collection


plot_linestring_collection = deprecated(_plot_linestring_collection)


def _plot_point_collection(
    ax,
    geoms,
    values=None,
    color=None,
    cmap=None,
    vmin=None,
    vmax=None,
    marker="o",
    markersize=None,
    **kwargs
):
    """
    Plots a collection of Point and MultiPoint geometries to `ax`

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        where shapes will be plotted
    geoms : sequence of `N` Points or MultiPoints

    values : a sequence of `N` values, optional
        Values mapped to colors using vmin, vmax, and cmap.
        Cannot be specified together with `color`.
    markersize : scalar or array-like, optional
        Size of the markers. Note that under the hood ``scatter`` is
        used, so the specified value will be proportional to the
        area of the marker (size in points^2).

    Returns
    -------
    collection : matplotlib.collections.Collection that was plotted
    """
    if values is not None and color is not None:
        raise ValueError("Can only specify one of 'values' and 'color' kwargs")

    geoms, multiindex = _flatten_multi_geoms(geoms)
    if values is not None:
        values = np.take(values, multiindex, axis=0)

    x = [p.x for p in geoms]
    y = [p.y for p in geoms]

    # matplotlib 1.4 does not support c=None, and < 2.0 does not support s=None
    if values is not None:
        kwargs["c"] = values
    if markersize is not None:
        kwargs["s"] = markersize

    # Add to kwargs for easier checking below.
    if color is not None:
        kwargs["color"] = color
    if marker is not None:
        kwargs["marker"] = marker
    _expand_kwargs(kwargs, multiindex)

    if "norm" not in kwargs:
        collection = ax.scatter(x, y, vmin=vmin, vmax=vmax, cmap=cmap, **kwargs)
    else:
        collection = ax.scatter(x, y, cmap=cmap, **kwargs)

    return collection


plot_point_collection = deprecated(_plot_point_collection)


def plot_series(
    s, cmap=None, color=None, ax=None, figsize=None, aspect="auto", **style_kwds
):
    """
    Plot a GeoSeries.

    Generate a plot of a GeoSeries geometry with matplotlib.

    Parameters
    ----------
    s : Series
        The GeoSeries to be plotted. Currently Polygon,
        MultiPolygon, LineString, MultiLineString and Point
        geometries can be plotted.
    cmap : str (default None)
        The name of a colormap recognized by matplotlib. Any
        colormap will work, but categorical colormaps are
        generally recommended. Examples of useful discrete
        colormaps include:

            tab10, tab20, Accent, Dark2, Paired, Pastel1, Set1, Set2

    color : str (default None)
        If specified, all objects will be colored uniformly.
    ax : matplotlib.pyplot.Artist (default None)
        axes on which to draw the plot
    figsize : pair of floats (default None)
        Size of the resulting matplotlib.figure.Figure. If the argument
        ax is given explicitly, figsize is ignored.
    aspect : 'auto', 'equal' or float (default 'auto')
        Set aspect of axis. If 'auto', the default aspect for map plots is 'equal'; if
        however data are not projected (coordinates are long/lat), the aspect is by
        default set to 1/cos(s_y * pi/180) with s_y the y coordinate of the middle of
        the GeoSeries (the mean of the y range of bounding box) so that a long/lat
        square appears square in the middle of the plot. This implies an
        Equirectangular projection. It can also be set manually (float) as the ratio
        of y-unit to x-unit.
    **style_kwds : dict
        Color options to be passed on to the actual plot function, such
        as ``edgecolor``, ``facecolor``, ``linewidth``, ``markersize``,
        ``alpha``.

    Returns
    -------
    ax : matplotlib axes instance
    """
    if "colormap" in style_kwds:
        warnings.warn(
            "'colormap' is deprecated, please use 'cmap' instead "
            "(for consistency with matplotlib)",
            FutureWarning,
        )
        cmap = style_kwds.pop("colormap")
    if "axes" in style_kwds:
        warnings.warn(
            "'axes' is deprecated, please use 'ax' instead "
            "(for consistency with pandas)",
            FutureWarning,
        )
        ax = style_kwds.pop("axes")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError(
            "The matplotlib package is required for plotting in geopandas. "
            "You can install it using 'conda install -c conda-forge matplotlib' or "
            "'pip install matplotlib'."
        )

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    if aspect == "auto":
        if s.crs and s.crs.is_geographic:
            bounds = s.total_bounds
            y_coord = np.mean([bounds[1], bounds[3]])
            ax.set_aspect(1 / np.cos(y_coord * np.pi / 180))
            # formula ported from R package sp
            # https://github.com/edzer/sp/blob/master/R/mapasp.R
        else:
            ax.set_aspect("equal")
    else:
        ax.set_aspect(aspect)

    if s.empty:
        warnings.warn(
            "The GeoSeries you are attempting to plot is "
            "empty. Nothing has been displayed.",
            UserWarning,
        )
        return ax

    # if cmap is specified, create range of colors based on cmap
    values = None
    if cmap is not None:
        values = np.arange(len(s))
        if hasattr(cmap, "N"):
            values = values % cmap.N
        style_kwds["vmin"] = style_kwds.get("vmin", values.min())
        style_kwds["vmax"] = style_kwds.get("vmax", values.max())

    # decompose GeometryCollections
    geoms, multiindex = _flatten_multi_geoms(s.geometry, prefix="Geom")
    values = np.take(values, multiindex, axis=0) if cmap else None
    expl_series = geopandas.GeoSeries(geoms)

    geom_types = expl_series.type
    poly_idx = np.asarray((geom_types == "Polygon") | (geom_types == "MultiPolygon"))
    line_idx = np.asarray(
        (geom_types == "LineString")
        | (geom_types == "MultiLineString")
        | (geom_types == "LinearRing")
    )
    point_idx = np.asarray((geom_types == "Point") | (geom_types == "MultiPoint"))

    # plot all Polygons and all MultiPolygon components in the same collection
    polys = expl_series[poly_idx]
    if not polys.empty:
        # color overrides both face and edgecolor. As we want people to be
        # able to use edgecolor as well, pass color to facecolor
        facecolor = style_kwds.pop("facecolor", None)
        if color is not None:
            facecolor = color

        values_ = values[poly_idx] if cmap else None
        _plot_polygon_collection(
            ax, polys, values_, facecolor=facecolor, cmap=cmap, **style_kwds
        )

    # plot all LineStrings and MultiLineString components in same collection
    lines = expl_series[line_idx]
    if not lines.empty:
        values_ = values[line_idx] if cmap else None
        _plot_linestring_collection(
            ax, lines, values_, color=color, cmap=cmap, **style_kwds
        )

    # plot all Points in the same collection
    points = expl_series[point_idx]
    if not points.empty:
        values_ = values[point_idx] if cmap else None
        _plot_point_collection(
            ax, points, values_, color=color, cmap=cmap, **style_kwds
        )

    plt.draw()
    return ax


def plot_dataframe(
    df,
    column=None,
    cmap=None,
    color=None,
    ax=None,
    cax=None,
    categorical=False,
    legend=False,
    scheme=None,
    k=5,
    vmin=None,
    vmax=None,
    markersize=None,
    figsize=None,
    legend_kwds=None,
    categories=None,
    classification_kwds=None,
    missing_kwds=None,
    aspect="auto",
    **style_kwds
):
    """
    Plot a GeoDataFrame.

    Generate a plot of a GeoDataFrame with matplotlib.  If a
    column is specified, the plot coloring will be based on values
    in that column.

    Parameters
    ----------
    df : GeoDataFrame
        The GeoDataFrame to be plotted.  Currently Polygon,
        MultiPolygon, LineString, MultiLineString and Point
        geometries can be plotted.
    column : str, np.array, pd.Series (default None)
        The name of the dataframe column, np.array, or pd.Series to be plotted.
        If np.array or pd.Series are used then it must have same length as
        dataframe. Values are used to color the plot. Ignored if `color` is
        also set.
    cmap : str (default None)
        The name of a colormap recognized by matplotlib.
    color : str (default None)
        If specified, all objects will be colored uniformly.
    ax : matplotlib.pyplot.Artist (default None)
        axes on which to draw the plot
    cax : matplotlib.pyplot Artist (default None)
        axes on which to draw the legend in case of color map.
    categorical : bool (default False)
        If False, cmap will reflect numerical values of the
        column being plotted.  For non-numerical columns, this
        will be set to True.
    legend : bool (default False)
        Plot a legend. Ignored if no `column` is given, or if `color` is given.
    scheme : str (default None)
        Name of a choropleth classification scheme (requires mapclassify).
        A mapclassify.MapClassifier object will be used
        under the hood. Supported are all schemes provided by mapclassify (e.g.
        'BoxPlot', 'EqualInterval', 'FisherJenks', 'FisherJenksSampled',
        'HeadTailBreaks', 'JenksCaspall', 'JenksCaspallForced',
        'JenksCaspallSampled', 'MaxP', 'MaximumBreaks',
        'NaturalBreaks', 'Quantiles', 'Percentiles', 'StdMean',
        'UserDefined'). Arguments can be passed in classification_kwds.
    k : int (default 5)
        Number of classes (ignored if scheme is None)
    vmin : None or float (default None)
        Minimum value of cmap. If None, the minimum data value
        in the column to be plotted is used.
    vmax : None or float (default None)
        Maximum value of cmap. If None, the maximum data value
        in the column to be plotted is used.
    markersize : str or float or sequence (default None)
        Only applies to point geometries within a frame.
        If a str, will use the values in the column of the frame specified
        by markersize to set the size of markers. Otherwise can be a value
        to apply to all points, or a sequence of the same length as the
        number of points.
    figsize : tuple of integers (default None)
        Size of the resulting matplotlib.figure.Figure. If the argument
        axes is given explicitly, figsize is ignored.
    legend_kwds : dict (default None)
        Keyword arguments to pass to matplotlib.pyplot.legend() or
        matplotlib.pyplot.colorbar().
        Additional accepted keywords when `scheme` is specified:

        fmt : string
            A formatting specification for the bin edges of the classes in the
            legend. For example, to have no decimals: ``{"fmt": "{:.0f}"}``.
        labels : list-like
            A list of legend labels to override the auto-generated labels.
            Needs to have the same number of elements as the number of
            classes (`k`).
    categories : list-like
        Ordered list-like object of categories to be used for categorical plot.
    classification_kwds : dict (default None)
        Keyword arguments to pass to mapclassify
    missing_kwds : dict (default None)
        Keyword arguments specifying color options (as style_kwds)
        to be passed on to geometries with missing values in addition to
        or overwriting other style kwds. If None, geometries with missing
        values are not plotted.
    aspect : 'auto', 'equal' or float (default 'auto')
        Set aspect of axis. If 'auto', the default aspect for map plots is 'equal'; if
        however data are not projected (coordinates are long/lat), the aspect is by
        default set to 1/cos(df_y * pi/180) with df_y the y coordinate of the middle of
        the GeoDataFrame (the mean of the y range of bounding box) so that a long/lat
        square appears square in the middle of the plot. This implies an
        Equirectangular projection. It can also be set manually (float) as the ratio
        of y-unit to x-unit.

    **style_kwds : dict
        Style options to be passed on to the actual plot function, such
        as ``edgecolor``, ``facecolor``, ``linewidth``, ``markersize``,
        ``alpha``.

    Returns
    -------
    ax : matplotlib axes instance

    """
    if "colormap" in style_kwds:
        warnings.warn(
            "'colormap' is deprecated, please use 'cmap' instead "
            "(for consistency with matplotlib)",
            FutureWarning,
        )
        cmap = style_kwds.pop("colormap")
    if "axes" in style_kwds:
        warnings.warn(
            "'axes' is deprecated, please use 'ax' instead "
            "(for consistency with pandas)",
            FutureWarning,
        )
        ax = style_kwds.pop("axes")
    if column is not None and color is not None:
        warnings.warn(
            "Only specify one of 'column' or 'color'. Using 'color'.", UserWarning
        )
        column = None

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError(
            "The matplotlib package is required for plotting in geopandas. "
            "You can install it using 'conda install -c conda-forge matplotlib' or "
            "'pip install matplotlib'."
        )

    if ax is None:
        if cax is not None:
            raise ValueError("'ax' can not be None if 'cax' is not.")
        fig, ax = plt.subplots(figsize=figsize)

    if aspect == "auto":
        if df.crs and df.crs.is_geographic:
            bounds = df.total_bounds
            y_coord = np.mean([bounds[1], bounds[3]])
            ax.set_aspect(1 / np.cos(y_coord * np.pi / 180))
            # formula ported from R package sp
            # https://github.com/edzer/sp/blob/master/R/mapasp.R
        else:
            ax.set_aspect("equal")
    else:
        ax.set_aspect(aspect)

    if df.empty:
        warnings.warn(
            "The GeoDataFrame you are attempting to plot is "
            "empty. Nothing has been displayed.",
            UserWarning,
        )
        return ax

    if isinstance(markersize, str):
        markersize = df[markersize].values

    if column is None:
        return plot_series(
            df.geometry,
            cmap=cmap,
            color=color,
            ax=ax,
            figsize=figsize,
            markersize=markersize,
            aspect=aspect,
            **style_kwds
        )

    # To accept pd.Series and np.arrays as column
    if isinstance(column, (np.ndarray, pd.Series)):
        if column.shape[0] != df.shape[0]:
            raise ValueError(
                "The dataframe and given column have different number of rows."
            )
        else:
            values = column
    else:
        values = df[column]

    if pd.api.types.is_categorical_dtype(values.dtype):
        if categories is not None:
            raise ValueError(
                "Cannot specify 'categories' when column has categorical dtype"
            )
        categorical = True
    elif values.dtype is np.dtype("O") or categories:
        categorical = True

    nan_idx = np.asarray(pd.isna(values), dtype="bool")

    # Define `values` as a Series
    if categorical:
        if cmap is None:
            cmap = "tab10"

        cat = pd.Categorical(values, categories=categories)
        categories = list(cat.categories)

        # values missing in the Categorical but not in original values
        missing = list(np.unique(values[~nan_idx & cat.isna()]))
        if missing:
            raise ValueError(
                "Column contains values not listed in categories. "
                "Missing categories: {}.".format(missing)
            )

        values = cat.codes[~nan_idx]
        vmin = 0 if vmin is None else vmin
        vmax = len(categories) - 1 if vmax is None else vmax

    if scheme is not None:
        if classification_kwds is None:
            classification_kwds = {}
        if "k" not in classification_kwds:
            classification_kwds["k"] = k

        binning = _mapclassify_choro(values[~nan_idx], scheme, **classification_kwds)
        # set categorical to True for creating the legend
        categorical = True
        if legend_kwds is not None and "labels" in legend_kwds:
            if len(legend_kwds["labels"]) != binning.k:
                raise ValueError(
                    "Number of labels must match number of bins, "
                    "received {} labels for {} bins".format(
                        len(legend_kwds["labels"]), binning.k
                    )
                )
            else:
                categories = list(legend_kwds.pop("labels"))
        else:
            fmt = "{:.2f}"
            if legend_kwds is not None and "fmt" in legend_kwds:
                fmt = legend_kwds.pop("fmt")
            categories = binning.get_legend_classes(fmt)
        values = np.array(binning.yb)

    # fill values with placeholder where were NaNs originally to map them properly
    # (after removing them in categorical or scheme)
    if categorical:
        for n in np.where(nan_idx)[0]:
            values = np.insert(values, n, values[0])

    mn = values[~np.isnan(values)].min() if vmin is None else vmin
    mx = values[~np.isnan(values)].max() if vmax is None else vmax

    # decompose GeometryCollections
    geoms, multiindex = _flatten_multi_geoms(df.geometry, prefix="Geom")
    values = np.take(values, multiindex, axis=0)
    nan_idx = np.take(nan_idx, multiindex, axis=0)
    expl_series = geopandas.GeoSeries(geoms)

    geom_types = expl_series.type
    poly_idx = np.asarray((geom_types == "Polygon") | (geom_types == "MultiPolygon"))
    line_idx = np.asarray(
        (geom_types == "LineString")
        | (geom_types == "MultiLineString")
        | (geom_types == "LinearRing")
    )
    point_idx = np.asarray((geom_types == "Point") | (geom_types == "MultiPoint"))

    # plot all Polygons and all MultiPolygon components in the same collection
    polys = expl_series[poly_idx & np.invert(nan_idx)]
    subset = values[poly_idx & np.invert(nan_idx)]
    if not polys.empty:
        _plot_polygon_collection(
            ax, polys, subset, vmin=mn, vmax=mx, cmap=cmap, **style_kwds
        )

    # plot all LineStrings and MultiLineString components in same collection
    lines = expl_series[line_idx & np.invert(nan_idx)]
    subset = values[line_idx & np.invert(nan_idx)]
    if not lines.empty:
        _plot_linestring_collection(
            ax, lines, subset, vmin=mn, vmax=mx, cmap=cmap, **style_kwds
        )

    # plot all Points in the same collection
    points = expl_series[point_idx & np.invert(nan_idx)]
    subset = values[point_idx & np.invert(nan_idx)]
    if not points.empty:
        if isinstance(markersize, np.ndarray):
            markersize = np.take(markersize, multiindex, axis=0)
            markersize = markersize[point_idx & np.invert(nan_idx)]
        _plot_point_collection(
            ax,
            points,
            subset,
            vmin=mn,
            vmax=mx,
            markersize=markersize,
            cmap=cmap,
            **style_kwds
        )

    if missing_kwds is not None:
        if color:
            if "color" not in missing_kwds:
                missing_kwds["color"] = color

        merged_kwds = style_kwds.copy()
        merged_kwds.update(missing_kwds)

        plot_series(expl_series[nan_idx], ax=ax, **merged_kwds)

    if legend and not color:

        if legend_kwds is None:
            legend_kwds = {}
        if "fmt" in legend_kwds:
            legend_kwds.pop("fmt")

        from matplotlib.lines import Line2D
        from matplotlib.colors import Normalize
        from matplotlib import cm

        norm = style_kwds.get("norm", None)
        if not norm:
            norm = Normalize(vmin=mn, vmax=mx)
        n_cmap = cm.ScalarMappable(norm=norm, cmap=cmap)
        if categorical:
            patches = []
            for value, cat in enumerate(categories):
                patches.append(
                    Line2D(
                        [0],
                        [0],
                        linestyle="none",
                        marker="o",
                        alpha=style_kwds.get("alpha", 1),
                        markersize=10,
                        markerfacecolor=n_cmap.to_rgba(value),
                        markeredgewidth=0,
                    )
                )
            if missing_kwds is not None:
                if "color" in merged_kwds:
                    merged_kwds["facecolor"] = merged_kwds["color"]
                patches.append(
                    Line2D(
                        [0],
                        [0],
                        linestyle="none",
                        marker="o",
                        alpha=merged_kwds.get("alpha", 1),
                        markersize=10,
                        markerfacecolor=merged_kwds.get("facecolor", None),
                        markeredgecolor=merged_kwds.get("edgecolor", None),
                        markeredgewidth=merged_kwds.get(
                            "linewidth", 1 if merged_kwds.get("edgecolor", False) else 0
                        ),
                    )
                )
                categories.append(merged_kwds.get("label", "NaN"))
            legend_kwds.setdefault("numpoints", 1)
            legend_kwds.setdefault("loc", "best")
            ax.legend(patches, categories, **legend_kwds)
        else:

            if cax is not None:
                legend_kwds.setdefault("cax", cax)
            else:
                legend_kwds.setdefault("ax", ax)

            n_cmap.set_array([])
            ax.get_figure().colorbar(n_cmap, **legend_kwds)

    plt.draw()
    return ax


def _mapclassify_choro(values, scheme, **classification_kwds):
    """
    Wrapper for choropleth schemes from mapclassify for use with plot_dataframe

    Parameters
    ----------
    values
        Series to be plotted
    scheme : str
        One of mapclassify classification schemes
        Options are BoxPlot, EqualInterval, FisherJenks,
        FisherJenksSampled, HeadTailBreaks, JenksCaspall,
        JenksCaspallForced, JenksCaspallSampled, MaxP,
        MaximumBreaks, NaturalBreaks, Quantiles, Percentiles, StdMean,
        UserDefined

    **classification_kwds : dict
        Keyword arguments for classification scheme
        For details see mapclassify documentation:
        https://mapclassify.readthedocs.io/en/latest/api.html

    Returns
    -------
    binning
        Binning objects that holds the Series with values replaced with
        class identifier and the bins.
    """
    try:
        import mapclassify.classifiers as classifiers

    except ImportError:
        raise ImportError(
            "The 'mapclassify' >= 2.2.0 package is required to use the 'scheme' keyword"
        )
    from mapclassify import __version__ as mc_version

    if mc_version < LooseVersion("2.2.0"):
        raise ImportError(
            "The 'mapclassify' >= 2.2.0 package is required to "
            "use the 'scheme' keyword"
        )
    schemes = {}
    for classifier in classifiers.CLASSIFIERS:
        schemes[classifier.lower()] = getattr(classifiers, classifier)

    scheme = scheme.lower()

    # mapclassify < 2.1 cleaned up the scheme names (removing underscores)
    # trying both to keep compatibility with older versions and provide
    # compatibility with newer versions of mapclassify
    oldnew = {
        "Box_Plot": "BoxPlot",
        "Equal_Interval": "EqualInterval",
        "Fisher_Jenks": "FisherJenks",
        "Fisher_Jenks_Sampled": "FisherJenksSampled",
        "HeadTail_Breaks": "HeadTailBreaks",
        "Jenks_Caspall": "JenksCaspall",
        "Jenks_Caspall_Forced": "JenksCaspallForced",
        "Jenks_Caspall_Sampled": "JenksCaspallSampled",
        "Max_P_Plassifier": "MaxP",
        "Maximum_Breaks": "MaximumBreaks",
        "Natural_Breaks": "NaturalBreaks",
        "Std_Mean": "StdMean",
        "User_Defined": "UserDefined",
    }
    scheme_names_mapping = {}
    scheme_names_mapping.update(
        {old.lower(): new.lower() for old, new in oldnew.items()}
    )
    scheme_names_mapping.update(
        {new.lower(): old.lower() for old, new in oldnew.items()}
    )

    try:
        scheme_class = schemes[scheme]
    except KeyError:
        scheme = scheme_names_mapping.get(scheme, scheme)
        try:
            scheme_class = schemes[scheme]
        except KeyError:
            raise ValueError(
                "Invalid scheme. Scheme must be in the set: %r" % schemes.keys()
            )

    if classification_kwds["k"] is not None:
        from inspect import getfullargspec as getspec

        spec = getspec(scheme_class.__init__)
        if "k" not in spec.args:
            del classification_kwds["k"]
    try:
        binning = scheme_class(np.asarray(values), **classification_kwds)
    except TypeError:
        raise TypeError("Invalid keyword argument for %r " % scheme)
    return binning
