import warnings

import numpy as np
import pandas as pd
from pandas.plotting import PlotAccessor

from distutils.version import LooseVersion

from ._decorator import doc

_MPL_COLL_KWD_ALIASES = {
    "antialiased": ["antialiaseds", "aa"],
    "edgecolor": ["edgecolors", "ec"],
    "facecolor": ["facecolors", "fc"],
    "linestyle": ["linestyles", "dashes", "ls"],
    "linewidth": ["linewidths", "lw"],
}
_MPL_COLL_KWD_NORM = {
    alias: key
    for key, alias_list in _MPL_COLL_KWD_ALIASES.items()
    for alias in alias_list
}


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
    Returns Series like geoms and index, except that any Multi geometries are
    split into their components and indices are repeated for all component in
    the same Multi geometry. Also gets rid of empty geometries, but
    `component_index` still refers to the original position in `geoms`. This way
    the corresponding values can be extracted.

    Prefix specifies type of geometry to be flatten. "Multi" for MultiPoint and
    similar, "Geom" for GeometryCollection.

    Returns
    -------
    components : GeoSeries

    component_index : index array indices are repeated for all components in the
        same Multi geometry
    """
    from shapely.geometry import Point

    geoms = geoms.fillna(Point())
    has_multi_geom = (geoms.geom_type.str.startswith(prefix) & ~geoms.is_empty).any()

    if has_multi_geom:
        exploded_geoms = geoms.explode()
        not_empty = ~exploded_geoms.is_empty.values
        geo_series = exploded_geoms
        component_index = exploded_geoms.index.codes[1]
        component_index = (component_index == 0).cumsum() - 1

    else:
        not_empty = ~geoms.is_empty
        geo_series = geoms
        component_index = np.arange(len(geoms))

    geo_series = geo_series.loc[not_empty].reset_index(drop=True)
    component_index = component_index[not_empty]
    return geo_series, component_index


def _expand_kwargs(kwargs, multiindex, is_final_expansion=False):
    """
    Most arguments to the plot functions must be a (single) value, or a sequence
    of values. This function checks each key-value pair in 'kwargs' and expands
    it (in place) to the correct length/formats with help of 'multiindex', unless
    the value appears to already be a valid (single) value for the key.
    `multiindex` should be list_like
    """
    import matplotlib
    from matplotlib.colors import is_color_like
    from typing import Iterable

    mpl = matplotlib.__version__
    if not (mpl >= LooseVersion("3.4") or (mpl > LooseVersion("3.3.2") and "+" in mpl)):
        # alpha is supported as array argument with matplotlib 3.4+
        single_value_kwargs = ["hatch", "marker", "path_effects"]
    else:
        single_value_kwargs = ["hatch", "marker", "alpha", "path_effects"]

    to_pop = []
    for att, value in kwargs.items():
        if "color" in att:  # color(s), edgecolor(s), facecolor(s)
            if is_color_like(value):
                continue
        elif att == "linestyle":
            # A single linestyle can be 2-tuple of a number and an iterable.
            if (
                isinstance(value, tuple)
                and len(value) == 2
                and isinstance(value[1], Iterable)
            ):
                continue

        if pd.api.types.is_list_like(value) and multiindex[-1] < len(value):
            value = np.take(value, multiindex, axis=0)
            # If value only contains null values, which can happen for a
            # categorical plot, pop the argument to later retrieve the
            # matplotlib default, which is not necessarily None or np.nan.
            if pd.isnull(value).all():
                to_pop.append(att)
            elif (
                att in single_value_kwargs
                and is_final_expansion
                and np.all(value == value[0])
            ):
                value = value[0]
            # For plain text styles, a single-value array cannot be passed
            # as a linestyle to a Collection.
            elif (
                att == "linestyle"
                and isinstance(value, np.ndarray)
                and is_final_expansion
            ):
                value = value.tolist()
            kwargs[att] = value

    for att in to_pop:
        kwargs.pop(att)


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
    ax, geoms, values=None, color=None, cmap=None, norm=None, **kwargs
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
    color : single color or sequence of `N` colors
        Sets both `edgecolor` and `facecolor`
    **kwargs
        Additional keyword arguments passed to the collection
    """
    from matplotlib.collections import PatchCollection

    # PatchCollection does not accept some kwargs.
    kwargs = {
        att: value
        for att, value in kwargs.items()
        if att not in ["markersize", "marker", "s"]
    }

    geoms, multiindex = _flatten_multi_geoms(geoms)
    poly_patches = geoms.apply(_PolygonPatch)

    if isinstance(values, pd.Categorical):
        # This should never be entered when called through `plot_series`.
        _expand_kwargs(kwargs, multiindex)
        values = values[multiindex]
        codes = values.codes
        ucodes = np.unique(codes)
        categories = values.categories[ucodes]
        # Have to iterate because an Artist.label can only be a (single) str.
        for cat, cat_code in zip(categories, ucodes):
            cat_kwargs = kwargs.copy()
            cat_idx = np.where(codes == cat_code)[0]
            # some properties like hatch can only be a single value, so:
            _expand_kwargs(cat_kwargs, cat_idx[[0]], is_final_expansion=True)
            cat_patches = np.take(poly_patches, cat_idx, axis=0)
            collection = PatchCollection(cat_patches, label=cat, **cat_kwargs)
            collection.set_facecolor(color.get(cat, "none"))
            ax.add_collection(collection, autolim=True)

    else:
        # Add to kwargs for easier checking below.
        if color is not None:
            kwargs["color"] = color
        _expand_kwargs(kwargs, multiindex, is_final_expansion=True)

        collection = PatchCollection(poly_patches, **kwargs)

        if values is not None:
            values = np.take(values, multiindex, axis=0)
            collection.set_array(values)
            collection.set_cmap(cmap)
            collection.set_norm(norm)

        ax.add_collection(collection, autolim=True)

    ax.autoscale_view()
    return collection


plot_polygon_collection = deprecated(_plot_polygon_collection)


def _plot_linestring_collection(
    ax, geoms, values=None, color=None, cmap=None, norm=None, **kwargs
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
    """
    from matplotlib.collections import LineCollection

    # LineCollection does not accept some kwargs.
    kwargs = {
        att: value
        for att, value in kwargs.items()
        if att not in ["markersize", "marker", "s"]
    }

    geoms, multiindex = _flatten_multi_geoms(geoms)
    segments = [np.array(linestring.coords)[:, :2] for linestring in geoms]

    if isinstance(values, pd.Categorical):
        # This should never be entered when called through `plot_series`.
        _expand_kwargs(kwargs, multiindex)
        values = values[multiindex]
        codes = values.codes
        ucodes = np.unique(codes)
        categories = values.categories[ucodes]
        # Have to iterate because an Artist.label can only be a (single) str.
        for cat, cat_code in zip(categories, ucodes):
            cat_kwargs = kwargs.copy()
            cat_idx = np.where(codes == cat_code)[0]
            _expand_kwargs(cat_kwargs, cat_idx[[0]], is_final_expansion=True)
            cat_segments = np.take(segments, cat_idx, axis=0)
            collection = LineCollection(cat_segments, label=cat, **cat_kwargs)
            collection.set_color(color.get(cat, "none"))
            ax.add_collection(collection, autolim=True)

    else:
        # Add to kwargs for easier checking below.
        if color is not None:
            kwargs["color"] = color
        _expand_kwargs(kwargs, multiindex, is_final_expansion=True)

        collection = LineCollection(segments, **kwargs)

        if values is not None:
            values = np.take(values, multiindex, axis=0)
            collection.set_array(np.asarray(values))
            collection.set_cmap(cmap)
            collection.set_norm(norm)

        ax.add_collection(collection, autolim=True)

    ax.autoscale_view()
    return collection


plot_linestring_collection = deprecated(_plot_linestring_collection)


def _plot_point_collection(
    ax, geoms, values=None, color=None, cmap=None, norm=None, **kwargs
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
    """
    # matplotlib < 2.0 does not support s=None.
    if kwargs.get("markersize") is not None:
        # We square to match the units.
        kwargs["s"] = kwargs.pop("markersize") ** 2

    geoms, multiindex = _flatten_multi_geoms(geoms)
    x, y = geoms.x, geoms.y

    if isinstance(values, pd.Categorical):
        # This should never be entered when called through `plot_series`.
        _expand_kwargs(kwargs, multiindex)
        values = values[multiindex]
        codes = values.codes
        ucodes = np.unique(codes)
        categories = values.categories[ucodes]
        # Have to iterate because an Artist.label can only be a (single) str.
        for cat, cat_code in zip(categories, ucodes):
            cat_kwargs = kwargs.copy()
            cat_idx = np.where(codes == cat_code)[0]
            _expand_kwargs(cat_kwargs, cat_idx[[0]], is_final_expansion=True)
            cat_x = np.take(x, cat_idx, axis=0)
            cat_y = np.take(y, cat_idx, axis=0)
            collection = ax.scatter(cat_x, cat_y, label=cat, **cat_kwargs)
            collection.set_color(color.get(cat, "none"))

    else:
        # Add to kwargs for easier checking below.
        if color is not None:
            kwargs["color"] = color
        _expand_kwargs(kwargs, multiindex, is_final_expansion=True)

        # matplotlib 1.4 does not support c=None. `values` has not been expanded
        # by the multiindex so _expand_kwargs should not be called for "c".
        if values is not None:
            kwargs["c"] = np.take(values, multiindex, axis=0)

        collection = ax.scatter(x, y, cmap=cmap, norm=norm, **kwargs)

    return collection


plot_point_collection = deprecated(_plot_point_collection)


def plot_series(
    s,
    cmap=None,
    norm=None,
    color=None,
    ax=None,
    figsize=None,
    aspect="auto",
    **style_kwds,
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
    TODO: norm
    color : str (default None)
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

    for kwd, normed_kwd in _MPL_COLL_KWD_NORM.items():
        if kwd in style_kwds:
            style_kwds[normed_kwd] = style_kwds.pop(kwd)

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
        )
        return ax

    if s.is_empty.all():
        warnings.warn(
            "The GeoSeries you are attempting to plot is "
            "composed of empty geometries. Nothing has been displayed.",
            UserWarning,
        )
        return ax

    # If `cmap` is specified, create range of colors based on `cmap`.
    values = None
    if cmap is not None:
        values = np.arange(len(s))
        if hasattr(cmap, "N"):
            values = values % cmap.N
        if norm is None:
            vmin = style_kwds.get("vmin", values.min())
            vmax = style_kwds.get("vmax", values.max())
            from matplotlib.colors import Normalize

            norm = Normalize(vmin=vmin, vmax=vmax)

    # decompose GeometryCollections
    geoms, multiindex = _flatten_multi_geoms(s.geometry, prefix="Geom")
    values = np.take(values, multiindex, axis=0) if cmap else None
    geom_types = geoms.type
    poly_idx = np.asarray((geom_types == "Polygon") | (geom_types == "MultiPolygon"))
    line_idx = np.asarray(
        (geom_types == "LineString")
        | (geom_types == "MultiLineString")
        | (geom_types == "LinearRing")
    )
    point_idx = np.asarray((geom_types == "Point") | (geom_types == "MultiPoint"))

    if color is not None:
        style_kwds["color"] = color

    # plot all Polygons and all MultiPolygon components in the same collection
    polys = geoms[poly_idx]
    if not polys.empty:
        polys_style_kwds = style_kwds.copy()
        _expand_kwargs(polys_style_kwds, multiindex[poly_idx])
        # `color` overrides both `facecolor` and `edgecolor`. As we want users
        # to be able to use `edgecolor` as well, pass `color` to `facecolor`:
        default_fc = polys_style_kwds.pop("color", None)
        if polys_style_kwds.get("facecolor") is None:
            polys_style_kwds["facecolor"] = default_fc

        values_ = values[poly_idx] if cmap else None
        _plot_polygon_collection(
            ax, polys, values_, cmap=cmap, norm=norm, **polys_style_kwds
        )

    # plot all LineStrings and MultiLineString components in same collection
    lines = geoms[line_idx]
    if not lines.empty:
        lines_style_kwds = style_kwds.copy()
        _expand_kwargs(lines_style_kwds, multiindex[line_idx])

        values_ = values[line_idx] if cmap else None
        _plot_linestring_collection(
            ax, lines, values_, cmap=cmap, norm=norm, **lines_style_kwds
        )

    # plot all Points in the same collection
    points = geoms[point_idx]
    if not points.empty:
        pts_style_kwds = style_kwds.copy()
        _expand_kwargs(pts_style_kwds, multiindex[point_idx])

        values_ = values[point_idx] if cmap else None
        _plot_point_collection(
            ax, points, values_, cmap=cmap, norm=norm, **pts_style_kwds
        )

    plt.draw()
    return ax


def plot_dataframe(
    df,
    column=None,
    cmap=None,
    norm=None,
    color=None,
    ax=None,
    cax=None,
    categorical=False,
    legend=False,
    scheme=None,
    k=5,
    vmin=None,
    vmax=None,
    figsize=None,
    legend_kwds=None,
    categories=None,
    classification_kwds=None,
    missing_kwds=None,
    aspect="auto",
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
        The kind of plots to produce:
         - 'geo': Map (default)
         Pandas Kinds
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
    TODO: norm
    color : str or dict (default None)
        If specified, all objects will be colored uniformly, if str, or
        according to the mapping it defines of values in `column` to colors if
        dict.
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

    **style_kwds : dict
        Style options to be passed on to the actual plot function, such
        as ``edgecolor``, ``facecolor``, ``linewidth``, ``markersize``,
        ``alpha``.

    Returns
    -------
    ax : matplotlib axes instance

    Examples
    --------
    >>> df = geopandas.read_file(geopandas.datasets.get_path("naturalearth_lowres"))
    >>> df.head()  # doctest: +SKIP
        pop_est      continent                      name iso_a3  \
gdp_md_est                                           geometry
    0     920938        Oceania                      Fiji    FJI      8374.0  MULTIPOLY\
GON (((180.00000 -16.06713, 180.00000...
    1   53950935         Africa                  Tanzania    TZA    150600.0  POLYGON (\
(33.90371 -0.95000, 34.07262 -1.05982...
    2     603253         Africa                 W. Sahara    ESH       906.5  POLYGON (\
(-8.66559 27.65643, -8.66512 27.58948...
    3   35623680  North America                    Canada    CAN   1674000.0  MULTIPOLY\
GON (((-122.84000 49.00000, -122.9742...
    4  326625791  North America  United States of America    USA  18560000.0  MULTIPOLY\
GON (((-122.84000 49.00000, -120.0000...

    >>> df.plot("pop_est", cmap="Blues")  # doctest: +SKIP

    See the User Guide page :doc:`../../user_guide/mapping` for details.

    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError(
            "The matplotlib package is required for plotting in geopandas. "
            "You can install it using 'conda install -c conda-forge matplotlib' or "
            "'pip install matplotlib'."
        )

    for kwd, normed_kwd in _MPL_COLL_KWD_NORM.items():
        if kwd in style_kwds:
            style_kwds[normed_kwd] = style_kwds.pop(kwd)
        if missing_kwds is not None and kwd in missing_kwds:
            missing_kwds[normed_kwd] = missing_kwds.pop(kwd)

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

    from matplotlib.colors import is_color_like

    if column is not None and is_color_like(color):
        warnings.warn(
            "Only specify one of 'column' or 'color'. Using 'color'.", UserWarning
        )
        column = None

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
        )
        return ax

    if column is None:
        return plot_series(
            df.geometry,
            cmap=cmap,
            norm=norm,
            color=color,
            ax=ax,
            figsize=figsize,
            aspect=aspect,
            **style_kwds,
        )

    # To accept pd.Series and np.arrays as column
    if isinstance(column, (np.ndarray, pd.Series)):
        if column.shape[0] != df.shape[0]:
            raise ValueError(
                "The dataframe and given column have different number of rows."
            )
        else:
            values = np.asarray(column)

    else:
        values = df[column].values

    if pd.api.types.is_categorical_dtype(values.dtype):
        if categories is not None:
            raise ValueError(
                "Cannot specify 'categories' when column has categorical dtype"
            )
        categorical = True
    elif values.dtype is np.dtype("O") or categories:
        categorical = True

    nan_idx = np.asarray(pd.isna(values), dtype="bool")

    is_color_mapping = isinstance(color, dict)
    categorical = categorical or is_color_mapping

    if scheme is not None:
        mc_err = (
            "The 'mapclassify' package (>= 2.4.0) is "
            "required to use the 'scheme' keyword."
        )
        try:
            import mapclassify

        except ImportError:
            raise ImportError(mc_err)

        if mapclassify.__version__ < LooseVersion("2.4.0"):
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

        values = pd.Categorical([np.nan] * len(values), categories=labels, ordered=True)
        values[~nan_idx] = pd.Categorical.from_codes(
            binning.yb, categories=labels, ordered=True
        )
        if cmap is None:
            cmap = "viridis"

    # Define `values` as a Series
    if categorical:
        # We order to avoid raising an error when taking the min
        cat = pd.Categorical(values, categories=categories).as_ordered()
        categories = list(cat.categories)
        # values missing in the Categorical but not in original values
        missing = list(np.unique(values[~nan_idx & cat.isna()]))
        if missing:
            raise ValueError(
                "Column contains values not listed in categories. "
                "Missing categories: {}.".format(missing)
            )

        values = cat[~nan_idx]
        vmin = 0 if vmin is None else vmin
        vmax = len(categories) - 1 if vmax is None else vmax

        if cmap is None and scheme is None:
            cmap = "tab10"

        for key, value in style_kwds.items():
            if isinstance(value, dict):
                style_kwds[key] = np.asarray(cat.map(value))

        # fill values with placeholder where were NaNs originally to map them
        # properly (after removing them in categorical or scheme)
        values = cat.fillna(values[0])

    # else, raise if given dictionaries and cateforical=False??

    # If `color` is not a mapping, the colors will then depend on a norm.
    if not is_color_mapping:
        # If the `norm` was not provided, generate one either with provided
        # `vmin` and `vmax`, or based on the `values` array.
        if norm is None:
            from matplotlib.colors import Normalize

            # If categorical vmin and vmax cannot be None, so the fact that
            # value.min() is a string does not matter.
            mn = values[~np.isnan(values)].min() if vmin is None else vmin
            mx = values[~np.isnan(values)].max() if vmax is None else vmax
            norm = Normalize(vmin=mn, vmax=mx)

        # For categorical plots in this case we still need to assign a color per
        # category:
        if categorical:
            from matplotlib import cm

            n_cmap = cm.ScalarMappable(norm=norm, cmap=cmap)
            cat_colors = n_cmap.to_rgba(np.arange(len(categories)))
            color = {cat: cat_color for cat, cat_color in zip(categories, cat_colors)}

    geoms, multiindex = _flatten_multi_geoms(df.geometry, prefix="Geom")
    values = values[multiindex]
    nan_idx = np.take(nan_idx, multiindex, axis=0)
    geom_types = geoms.type
    poly_idx = np.asarray((geom_types == "Polygon") | (geom_types == "MultiPolygon"))
    line_idx = np.asarray(
        (geom_types == "LineString")
        | (geom_types == "MultiLineString")
        | (geom_types == "LinearRing")
    )
    point_idx = np.asarray((geom_types == "Point") | (geom_types == "MultiPoint"))

    # plot all Polygons and all MultiPolygon components in the same collection
    polys_mask = poly_idx & np.invert(nan_idx)
    polys = geoms[polys_mask]
    subset = values[polys_mask]

    if not polys.empty:
        polys_style_kwds = style_kwds.copy()
        _expand_kwargs(polys_style_kwds, multiindex[polys_mask])
        _plot_polygon_collection(
            ax, polys, subset, color=color, cmap=cmap, norm=norm, **polys_style_kwds
        )
        if categorical:
            ax.legend = _legend_with_poly_wrapper(ax.legend)
            ax.get_legend_handles_labels = _legend_with_poly_wrapper(
                ax.get_legend_handles_labels,
                handler_map_kwarg_name="legend_handler_map",
            )

    # plot all LineStrings and MultiLineString components in same collection
    lines_mask = line_idx & np.invert(nan_idx)
    lines = geoms[lines_mask]
    subset = values[lines_mask]
    if not lines.empty:
        lines_style_kwds = style_kwds.copy()
        _expand_kwargs(lines_style_kwds, multiindex[lines_mask])
        _plot_linestring_collection(
            ax, lines, subset, color=color, cmap=cmap, norm=norm, **lines_style_kwds
        )

    # plot all Points in the same collection
    pts_mask = point_idx & np.invert(nan_idx)
    points = geoms[pts_mask]
    subset = values[pts_mask]
    if not points.empty:
        pts_style_kwds = style_kwds.copy()
        _expand_kwargs(pts_style_kwds, multiindex[pts_mask])
        _plot_point_collection(
            ax, points, subset, color=color, cmap=cmap, norm=norm, **pts_style_kwds
        )

    if missing_kwds is not None and not geoms[nan_idx].empty:
        if color:
            missing_kwds["color"] = missing_kwds.get("color", color)
        merged_kwds = style_kwds.copy()
        merged_kwds.update(missing_kwds)
        _expand_kwargs(merged_kwds, multiindex[nan_idx])
        merged_kwds["label"] = merged_kwds.get("label", "NaN")
        plot_series(geoms[nan_idx], ax=ax, **merged_kwds)

    if legend:
        if legend_kwds is None:
            legend_kwds = {}
        if "fmt" in legend_kwds:
            legend_kwds.pop("fmt")

        if categorical:
            legend_kwds.setdefault("numpoints", 1)
            legend_kwds.setdefault("loc", "best")
            ax.legend(**legend_kwds)

        else:
            from matplotlib import cm

            if cax is not None:
                legend_kwds.setdefault("cax", cax)
            else:
                legend_kwds.setdefault("ax", ax)
            n_cmap = cm.ScalarMappable(norm=norm, cmap=cmap)
            n_cmap.set_array(np.array([]))
            ax.get_figure().colorbar(n_cmap, **legend_kwds)

    plt.draw()
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
        return self(kind="geo", *args, **kwargs)


def _legend_with_poly_wrapper(fun, handler_map_kwarg_name="handler_map"):
    """
    Decorator for ax.legend that enables `PatchCollection` objects plotted by
    `_plot_polygon_collection` to be rendered correctly in the legend.
    """
    from matplotlib.legend_handler import HandlerPolyCollection
    from matplotlib.collections import PatchCollection

    def legend(*args, **kwargs):
        kwargs[handler_map_kwarg_name] = {PatchCollection: HandlerPolyCollection()}
        return fun(*args, **kwargs)

    return legend
