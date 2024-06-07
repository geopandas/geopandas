import warnings
from packaging.version import Version

import numpy as np
import pandas as pd
from pandas import CategoricalDtype
from pandas.plotting import PlotAccessor

import geopandas

from ._decorator import doc


def _sanitize_geoms(geoms, prefix="Multi"):
    """
    Returns Series like geoms and index, except that any Multi geometries
    are split into their components and indices are repeated for all component
    in the same Multi geometry. At the same time, empty or missing geometries are
    filtered out.  Maintains 1:1 matching of geometry to value.

    Prefix specifies type of geometry to be flatten. 'Multi' for MultiPoint and similar,
    "Geom" for GeometryCollection.

    Returns
    -------
    components : list of geometry

    component_index : index array
        indices are repeated for all components in the same Multi geometry
    """
    # TODO(shapely) look into simplifying this with
    # shapely.get_parts(geoms, return_index=True) from shapely 2.0
    components, component_index = [], []

    if (
        not geoms.geom_type.str.startswith(prefix).any()
        and not geoms.is_empty.any()
        and not geoms.isna().any()
    ):
        return geoms, np.arange(len(geoms))

    for ix, geom in enumerate(geoms):
        if geom is not None and geom.geom_type.startswith(prefix) and not geom.is_empty:
            for poly in geom.geoms:
                components.append(poly)
                component_index.append(ix)
        elif geom is None or geom.is_empty:
            continue
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
    from typing import Iterable

    from matplotlib.colors import is_color_like

    scalar_kwargs = ["marker", "path_effects"]
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
        elif att in scalar_kwargs:
            # For these attributes, only a single value is allowed, so never expand.
            continue

        if pd.api.types.is_list_like(value):
            kwargs[att] = np.take(value, multiindex, axis=0)


def _PolygonPatch(polygon, **kwargs):
    """Constructs a matplotlib patch from a Polygon geometry

    The `kwargs` are those supported by the matplotlib.patches.PathPatch class
    constructor. Returns an instance of matplotlib.patches.PathPatch.

    Example (using Shapely Point and a matplotlib axes)::

        b = shapely.geometry.Point(0, 0).buffer(1.0)
        patch = _PolygonPatch(b, fc='blue', ec='blue', alpha=0.5)
        ax.add_patch(patch)

    GeoPandas originally relied on the descartes package by Sean Gillies
    (BSD license, https://pypi.org/project/descartes) for PolygonPatch, but
    this dependency was removed in favor of the below matplotlib code.
    """
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path

    path = Path.make_compound_path(
        Path(np.asarray(polygon.exterior.coords)[:, :2]),
        *[Path(np.asarray(ring.coords)[:, :2]) for ring in polygon.interiors],
    )
    return PathPatch(path, **kwargs)


def _plot_polygon_collection(
    ax,
    geoms,
    values=None,
    color=None,
    cmap=None,
    vmin=None,
    vmax=None,
    autolim=True,
    **kwargs,
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
    autolim : bool (default True)
        Update axes data limits to contain the new geometries.
    **kwargs
        Additional keyword arguments passed to the collection

    Returns
    -------
    collection : matplotlib.collections.Collection that was plotted
    """
    from matplotlib.collections import PatchCollection

    geoms, multiindex = _sanitize_geoms(geoms)
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

    collection = PatchCollection([_PolygonPatch(poly) for poly in geoms], **kwargs)

    if values is not None:
        collection.set_array(np.asarray(values))
        collection.set_cmap(cmap)
        if "norm" not in kwargs:
            collection.set_clim(vmin, vmax)

    ax.add_collection(collection, autolim=autolim)
    ax.autoscale_view()
    return collection


def _plot_linestring_collection(
    ax,
    geoms,
    values=None,
    color=None,
    cmap=None,
    vmin=None,
    vmax=None,
    autolim=True,
    **kwargs,
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
    autolim : bool (default True)
        Update axes data limits to contain the new geometries.

    Returns
    -------
    collection : matplotlib.collections.Collection that was plotted
    """
    from matplotlib.collections import LineCollection

    geoms, multiindex = _sanitize_geoms(geoms)
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

    segments = [np.array(linestring.coords)[:, :2] for linestring in geoms]
    collection = LineCollection(segments, **kwargs)

    if values is not None:
        collection.set_array(np.asarray(values))
        collection.set_cmap(cmap)
        if "norm" not in kwargs:
            collection.set_clim(vmin, vmax)

    ax.add_collection(collection, autolim=autolim)
    ax.autoscale_view()
    return collection


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
    **kwargs,
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

    geoms, multiindex = _sanitize_geoms(geoms)
    # values are expanded below as kwargs["c"]

    x = [p.x if not p.is_empty else None for p in geoms]
    y = [p.y if not p.is_empty else None for p in geoms]

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


def plot_series(
    s,
    cmap=None,
    color=None,
    ax=None,
    figsize=None,
    aspect="auto",
    autolim=True,
    **style_kwds,
):
    """
    Plot a GeoSeries.

    Generate a plot of a GeoSeries geometry with matplotlib.

    Parameters
    ----------
    s : Series
        The GeoSeries to be plotted. Currently Polygon,
        MultiPolygon, LineString, MultiLineString, Point and MultiPoint
        geometries can be plotted.
    cmap : str (default None)
        The name of a colormap recognized by matplotlib. Any
        colormap will work, but categorical colormaps are
        generally recommended. Examples of useful discrete
        colormaps include:

            tab10, tab20, Accent, Dark2, Paired, Pastel1, Set1, Set2

    color : str, np.array, pd.Series, List (default None)
        If specified, all objects will be colored uniformly.
    ax : matplotlib.pyplot.Artist (default None)
        axes on which to draw the plot
    figsize : pair of floats (default None)
        Size of the resulting matplotlib.figure.Figure. If the argument
        ax is given explicitly, figsize is ignored.
    aspect : 'auto', 'equal', None or float (default 'auto')
        Set aspect of axis. If 'auto', the default aspect for map plots is 'equal'; if
        however data are not projected (coordinates are long/lat), the aspect is by
        default set to 1/cos(s_y * pi/180) with s_y the y coordinate of the middle of
        the GeoSeries (the mean of the y range of bounding box) so that a long/lat
        square appears square in the middle of the plot. This implies an
        Equirectangular projection. If None, the aspect of `ax` won't be changed. It can
        also be set manually (float) as the ratio of y-unit to x-unit.
    autolim : bool (default True)
        Update axes data limits to contain the new geometries.
    **style_kwds : dict
        Color options to be passed on to the actual plot function, such
        as ``edgecolor``, ``facecolor``, ``linewidth``, ``markersize``,
        ``alpha``.

    Returns
    -------
    ax : matplotlib axes instance
    """

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
    elif aspect is not None:
        ax.set_aspect(aspect)

    if s.empty:
        warnings.warn(
            "The GeoSeries you are attempting to plot is "
            "empty. Nothing has been displayed.",
            UserWarning,
            stacklevel=3,
        )
        return ax

    if s.is_empty.all():
        warnings.warn(
            "The GeoSeries you are attempting to plot is "
            "composed of empty geometries. Nothing has been displayed.",
            UserWarning,
            stacklevel=3,
        )
        return ax

    # have colors been given for all geometries?
    color_given = pd.api.types.is_list_like(color) and len(color) == len(s)

    # if cmap is specified, create range of colors based on cmap
    values = None
    if cmap is not None:
        values = np.arange(len(s))
        if hasattr(cmap, "N"):
            values = values % cmap.N
        style_kwds["vmin"] = style_kwds.get("vmin", values.min())
        style_kwds["vmax"] = style_kwds.get("vmax", values.max())

    # decompose GeometryCollections
    geoms, multiindex = _sanitize_geoms(s.geometry, prefix="Geom")
    values = np.take(values, multiindex, axis=0) if cmap else None
    # ensure indexes are consistent
    if color_given and isinstance(color, pd.Series):
        color = color.reindex(s.index)
    expl_color = np.take(color, multiindex, axis=0) if color_given else color
    expl_series = geopandas.GeoSeries(geoms)

    geom_types = expl_series.geom_type
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
        color_ = expl_color[poly_idx] if color_given else color
        if color is not None:
            facecolor = color_

        values_ = values[poly_idx] if cmap else None
        _plot_polygon_collection(
            ax,
            polys,
            values_,
            facecolor=facecolor,
            cmap=cmap,
            autolim=autolim,
            **style_kwds,
        )

    # plot all LineStrings and MultiLineString components in same collection
    lines = expl_series[line_idx]
    if not lines.empty:
        values_ = values[line_idx] if cmap else None
        color_ = expl_color[line_idx] if color_given else color

        _plot_linestring_collection(
            ax, lines, values_, color=color_, cmap=cmap, autolim=autolim, **style_kwds
        )

    # plot all Points in the same collection
    points = expl_series[point_idx]
    if not points.empty:
        values_ = values[point_idx] if cmap else None
        color_ = expl_color[point_idx] if color_given else color

        _plot_point_collection(
            ax, points, values_, color=color_, cmap=cmap, **style_kwds
        )

    ax.figure.canvas.draw_idle()
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
    autolim=True,
    **style_kwds,
):
    """
    Plot a GeoDataFrame.

    Generate a plot of a GeoDataFrame with matplotlib.  If a
    column is specified, the plot coloring will be based on values
    in that column.

    Parameters
    ----------
    column : str, np.array, pd.Series (default None)
        The name of the dataframe column, np.array, or pd.Series to be plotted.
        If np.array or pd.Series are used then it must have same length as
        dataframe. Values are used to color the plot. Ignored if `color` is
        also set.
    kind: str
        The kind of plots to produce. The default is to create a map ("geo").
        Other supported kinds of plots from pandas:

        - 'line' : line plot
        - 'bar' : vertical bar plot
        - 'barh' : horizontal bar plot
        - 'hist' : histogram
        - 'box' : BoxPlot
        - 'kde' : Kernel Density Estimation plot
        - 'density' : same as 'kde'
        - 'area' : area plot
        - 'pie' : pie plot
        - 'scatter' : scatter plot
        - 'hexbin' : hexbin plot.
    cmap : str (default None)
        The name of a colormap recognized by matplotlib.
    color : str, np.array, pd.Series (default None)
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
        Keyword arguments to pass to :func:`matplotlib.pyplot.legend` or
        :func:`matplotlib.pyplot.colorbar`.
        Additional accepted keywords when `scheme` is specified:

        fmt : string
            A formatting specification for the bin edges of the classes in the
            legend. For example, to have no decimals: ``{"fmt": "{:.0f}"}``.
        labels : list-like
            A list of legend labels to override the auto-generated labels.
            Needs to have the same number of elements as the number of
            classes (`k`).
        interval : boolean (default False)
            An option to control brackets from mapclassify legend.
            If True, open/closed interval brackets are shown in the legend.
    categories : list-like
        Ordered list-like object of categories to be used for categorical plot.
    classification_kwds : dict (default None)
        Keyword arguments to pass to mapclassify
    missing_kwds : dict (default None)
        Keyword arguments specifying color options (as style_kwds)
        to be passed on to geometries with missing values in addition to
        or overwriting other style kwds. If None, geometries with missing
        values are not plotted.
    aspect : 'auto', 'equal', None or float (default 'auto')
        Set aspect of axis. If 'auto', the default aspect for map plots is 'equal'; if
        however data are not projected (coordinates are long/lat), the aspect is by
        default set to 1/cos(df_y * pi/180) with df_y the y coordinate of the middle of
        the GeoDataFrame (the mean of the y range of bounding box) so that a long/lat
        square appears square in the middle of the plot. This implies an
        Equirectangular projection. If None, the aspect of `ax` won't be changed. It can
        also be set manually (float) as the ratio of y-unit to x-unit.
    autolim : bool (default True)
        Update axes data limits to contain the new geometries.
    **style_kwds : dict
        Style options to be passed on to the actual plot function, such
        as ``edgecolor``, ``facecolor``, ``linewidth``, ``markersize``,
        ``alpha``.

    Returns
    -------
    ax : matplotlib axes instance

    Examples
    --------
    >>> import geodatasets
    >>> df = geopandas.read_file(geodatasets.get_path("nybb"))
    >>> df.head()  # doctest: +SKIP
       BoroCode  ...                                           geometry
    0         5  ...  MULTIPOLYGON (((970217.022 145643.332, 970227....
    1         4  ...  MULTIPOLYGON (((1029606.077 156073.814, 102957...
    2         3  ...  MULTIPOLYGON (((1021176.479 151374.797, 102100...
    3         1  ...  MULTIPOLYGON (((981219.056 188655.316, 980940....
    4         2  ...  MULTIPOLYGON (((1012821.806 229228.265, 101278...

    >>> df.plot("BoroName", cmap="Set1")  # doctest: +SKIP

    See the User Guide page :doc:`../../user_guide/mapping` for details.

    """
    if column is not None and color is not None:
        warnings.warn(
            "Only specify one of 'column' or 'color'. Using 'color'.",
            UserWarning,
            stacklevel=3,
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
    elif aspect is not None:
        ax.set_aspect(aspect)

    # GH 1555
    # if legend_kwds set, copy so we don't update it in place
    if legend_kwds is not None:
        legend_kwds = legend_kwds.copy()

    if df.empty:
        warnings.warn(
            "The GeoDataFrame you are attempting to plot is "
            "empty. Nothing has been displayed.",
            UserWarning,
            stacklevel=3,
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
            autolim=autolim,
            **style_kwds,
        )

    # To accept pd.Series and np.arrays as column
    if isinstance(column, (np.ndarray, pd.Series)):
        if column.shape[0] != df.shape[0]:
            raise ValueError(
                "The dataframe and given column have different number of rows."
            )
        else:
            values = column

            # Make sure index of a Series matches index of df
            if isinstance(values, pd.Series):
                values = values.reindex(df.index)
    else:
        values = df[column]

    if isinstance(values.dtype, CategoricalDtype):
        if categories is not None:
            raise ValueError(
                "Cannot specify 'categories' when column has categorical dtype"
            )
        categorical = True
    elif (
        pd.api.types.is_object_dtype(values.dtype)
        or pd.api.types.is_bool_dtype(values.dtype)
        or pd.api.types.is_string_dtype(values.dtype)
        or categories
    ):
        categorical = True

    nan_idx = np.asarray(pd.isna(values), dtype="bool")

    if scheme is not None:
        mc_err = (
            "The 'mapclassify' package (>= 2.4.0) is "
            "required to use the 'scheme' keyword."
        )
        try:
            import mapclassify

        except ImportError:
            raise ImportError(mc_err)

        if Version(mapclassify.__version__) < Version("2.4.0"):
            raise ImportError(mc_err)

        if classification_kwds is None:
            classification_kwds = {}
        if "k" not in classification_kwds:
            classification_kwds["k"] = k

        binning = mapclassify.classify(
            np.asarray(values[~nan_idx]), scheme, **classification_kwds
        )
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
                labels = list(legend_kwds.pop("labels"))
        else:
            fmt = "{:.2f}"
            if legend_kwds is not None and "fmt" in legend_kwds:
                fmt = legend_kwds.pop("fmt")

            labels = binning.get_legend_classes(fmt)
            if legend_kwds is not None:
                show_interval = legend_kwds.pop("interval", False)
            else:
                show_interval = False
            if not show_interval:
                labels = [c[1:-1] for c in labels]

        values = pd.Categorical(
            [np.nan] * len(values), categories=binning.bins, ordered=True
        )
        values[~nan_idx] = pd.Categorical.from_codes(
            binning.yb, categories=binning.bins, ordered=True
        )
        if cmap is None:
            cmap = "viridis"

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

    # fill values with placeholder where were NaNs originally to map them properly
    # (after removing them in categorical or scheme)
    if categorical:
        for n in np.where(nan_idx)[0]:
            values = np.insert(values, n, values[0])

    mn = values[~np.isnan(values)].min() if vmin is None else vmin
    mx = values[~np.isnan(values)].max() if vmax is None else vmax

    # decompose GeometryCollections
    geoms, multiindex = _sanitize_geoms(df.geometry, prefix="Geom")
    values = np.take(values, multiindex, axis=0)
    nan_idx = np.take(nan_idx, multiindex, axis=0)
    expl_series = geopandas.GeoSeries(geoms)

    geom_types = expl_series.geom_type
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
            ax,
            polys,
            subset,
            vmin=mn,
            vmax=mx,
            cmap=cmap,
            autolim=autolim,
            **style_kwds,
        )

    # plot all LineStrings and MultiLineString components in same collection
    lines = expl_series[line_idx & np.invert(nan_idx)]
    subset = values[line_idx & np.invert(nan_idx)]
    if not lines.empty:
        _plot_linestring_collection(
            ax,
            lines,
            subset,
            vmin=mn,
            vmax=mx,
            cmap=cmap,
            autolim=autolim,
            **style_kwds,
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
            **style_kwds,
        )

    missing_data = not expl_series[nan_idx].empty
    if missing_kwds is not None and missing_data:
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

        from matplotlib import cm
        from matplotlib.colors import Normalize
        from matplotlib.lines import Line2D

        norm = style_kwds.get("norm", None)
        if not norm:
            norm = Normalize(vmin=mn, vmax=mx)
        n_cmap = cm.ScalarMappable(norm=norm, cmap=cmap)
        if categorical:
            if scheme is not None:
                categories = labels
            patches = []
            for i in range(len(categories)):
                patches.append(
                    Line2D(
                        [0],
                        [0],
                        linestyle="none",
                        marker="o",
                        alpha=style_kwds.get("alpha", 1),
                        markersize=10,
                        markerfacecolor=n_cmap.to_rgba(i),
                        markeredgewidth=0,
                    )
                )
            if missing_kwds is not None and missing_data:
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
            legend_kwds.setdefault("handles", patches)
            legend_kwds.setdefault("labels", categories)
            ax.legend(**legend_kwds)
        else:
            if cax is not None:
                legend_kwds.setdefault("cax", cax)
            else:
                legend_kwds.setdefault("ax", ax)

            n_cmap.set_array(np.array([]))
            ax.get_figure().colorbar(n_cmap, **legend_kwds)

    ax.figure.canvas.draw_idle()
    return ax


@doc(plot_dataframe)
class GeoplotAccessor(PlotAccessor):
    _pandas_kinds = PlotAccessor._all_kinds

    def __call__(self, *args, **kwargs):
        data = self._parent.copy()
        kind = kwargs.pop("kind", "geo")
        if kind == "geo":
            return plot_dataframe(data, *args, **kwargs)
        if kind in self._pandas_kinds:
            # Access pandas plots
            return PlotAccessor(data)(kind=kind, **kwargs)
        else:
            # raise error
            raise ValueError(f"{kind} is not a valid plot kind")

    def geo(self, *args, **kwargs):
        return self(kind="geo", *args, **kwargs)  # noqa: B026
