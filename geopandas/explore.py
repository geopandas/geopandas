import warnings
from statistics import mean

import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype

from shapely.geometry import LineString

import geopandas

_MAP_KWARGS = [
    "location",
    "prefer_canvas",
    "no_touch",
    "disable_3d",
    "png_enabled",
    "zoom_control",
    "crs",
    "zoom_start",
    "left",
    "top",
    "position",
    "min_zoom",
    "max_zoom",
    "min_lat",
    "max_lat",
    "min_lon",
    "max_lon",
    "max_bounds",
]


def _explore(
    df,
    column=None,
    cmap=None,
    color=None,
    m=None,
    tiles="OpenStreetMap",
    attr=None,
    tooltip=True,
    popup=False,
    highlight=True,
    categorical=False,
    legend=True,
    scheme=None,
    k=5,
    vmin=None,
    vmax=None,
    width="100%",
    height="100%",
    categories=None,
    classification_kwds=None,
    control_scale=True,
    marker_type=None,
    marker_kwds={},
    style_kwds={},
    highlight_kwds={},
    missing_kwds={},
    tooltip_kwds={},
    popup_kwds={},
    legend_kwds={},
    map_kwds={},
    **kwargs,
):
    """Explore data in interactive map based on GeoPandas and folium/leaflet.js.

    Generate an interactive leaflet map based on :class:`~geopandas.GeoDataFrame`

    Parameters
    ----------
    column : str, np.array, pd.Series (default None)
        The name of the dataframe column, :class:`numpy.array`,
        or :class:`pandas.Series` to be plotted. If :class:`numpy.array` or
        :class:`pandas.Series` are used then it must have same length as dataframe.
    cmap : str, matplotlib.Colormap, branca.colormap or function (default None)
        The name of a colormap recognized by ``matplotlib``, a list-like of colors,
        :class:`matplotlib.colors.Colormap`, a :class:`branca.colormap.ColorMap` or
        function that returns a named color or hex based on the column
        value, e.g.::

            def my_colormap(value):  # scalar value defined in 'column'
                if value > 1:
                    return "green"
                return "red"

    color : str, array-like (default None)
        Named color or a list-like of colors (named or hex).
    m : folium.Map (default None)
        Existing map instance on which to draw the plot.
    tiles : str, xyzservices.TileProvider (default 'OpenStreetMap Mapnik')
        Map tileset to use. Can choose from the list supported by folium, query a
        :class:`xyzservices.TileProvider` by a name from ``xyzservices.providers``,
        pass :class:`xyzservices.TileProvider` object or pass custom XYZ URL.
        The current list of built-in providers (when ``xyzservices`` is not available):

        ``["OpenStreetMap", "CartoDB positron", “CartoDB dark_matter"]``

        You can pass a custom tileset to Folium by passing a Leaflet-style URL
        to the tiles parameter: ``http://{s}.yourtiles.com/{z}/{x}/{y}.png``.
        Be sure to check their terms and conditions and to provide attribution with
        the ``attr`` keyword.
    attr : str (default None)
        Map tile attribution; only required if passing custom tile URL.
    tooltip : bool, str, int, list (default True)
        Display GeoDataFrame attributes when hovering over the object.
        ``True`` includes all columns. ``False`` removes tooltip. Pass string or list of
        strings to specify a column(s). Integer specifies first n columns to be
        included. Defaults to ``True``.
    popup : bool, str, int, list (default False)
        Input GeoDataFrame attributes for object displayed when clicking.
        ``True`` includes all columns. ``False`` removes popup. Pass string or list of
        strings to specify a column(s). Integer specifies first n columns to be
        included. Defaults to ``False``.
    highlight : bool (default True)
        Enable highlight functionality when hovering over a geometry.
    categorical : bool (default False)
        If ``False``, ``cmap`` will reflect numerical values of the
        column being plotted. For non-numerical columns, this
        will be set to True.
    legend : bool (default True)
        Plot a legend in choropleth plots.
        Ignored if no ``column`` is given.
    scheme : str (default None)
        Name of a choropleth classification scheme (requires ``mapclassify`` >= 2.4.0).
        A :func:`mapclassify.classify` will be used
        under the hood. Supported are all schemes provided by ``mapclassify`` (e.g.
        ``'BoxPlot'``, ``'EqualInterval'``, ``'FisherJenks'``, ``'FisherJenksSampled'``,
        ``'HeadTailBreaks'``, ``'JenksCaspall'``, ``'JenksCaspallForced'``,
        ``'JenksCaspallSampled'``, ``'MaxP'``, ``'MaximumBreaks'``,
        ``'NaturalBreaks'``, ``'Quantiles'``, ``'Percentiles'``, ``'StdMean'``,
        ``'UserDefined'``). Arguments can be passed in ``classification_kwds``.
    k : int (default 5)
        Number of classes
    vmin : None or float (default None)
        Minimum value of ``cmap``. If ``None``, the minimum data value
        in the column to be plotted is used.
    vmax : None or float (default None)
        Maximum value of ``cmap``. If ``None``, the maximum data value
        in the column to be plotted is used.
    width : pixel int or percentage string (default: '100%')
        Width of the folium :class:`~folium.folium.Map`. If the argument
        m is given explicitly, width is ignored.
    height : pixel int or percentage string (default: '100%')
        Height of the folium :class:`~folium.folium.Map`. If the argument
        m is given explicitly, height is ignored.
    categories : list-like
        Ordered list-like object of categories to be used for categorical plot.
    classification_kwds : dict (default None)
        Keyword arguments to pass to mapclassify
    control_scale : bool, (default True)
        Whether to add a control scale on the map.
    marker_type : str, folium.Circle, folium.CircleMarker, folium.Marker (default None)
        Allowed string options are ('marker', 'circle', 'circle_marker'). Defaults to
        folium.CircleMarker.
    marker_kwds: dict (default {})
        Additional keywords to be passed to the selected ``marker_type``, e.g.:

        radius : float (default 2 for ``circle_marker`` and 50 for ``circle``))
            Radius of the circle, in meters (for ``circle``) or pixels
            (for ``circle_marker``).
        fill : bool (default True)
            Whether to fill the ``circle`` or ``circle_marker`` with color.
        icon : folium.map.Icon
            the :class:`folium.map.Icon` object to use to render the marker.
        draggable : bool (default False)
            Set to True to be able to drag the marker around the map.

    style_kwds : dict (default {})
        Additional style to be passed to folium ``style_function``:

        stroke : bool (default True)
            Whether to draw stroke along the path. Set it to ``False`` to
            disable borders on polygons or circles.
        color : str
            Stroke color
        weight : int
            Stroke width in pixels
        opacity : float (default 1.0)
            Stroke opacity
        fill : boolean (default True)
            Whether to fill the path with color. Set it to ``False`` to
            disable filling on polygons or circles.
        fillColor : str
            Fill color. Defaults to the value of the color option
        fillOpacity : float (default 0.5)
            Fill opacity.
        style_function : callable
            Function mapping a GeoJson Feature to a style ``dict``.

            * Style properties :func:`folium.vector_layers.path_options`
            * GeoJson features :class:`GeoDataFrame.__geo_interface__`

            e.g.::

                lambda x: {"color":"red" if x["properties"]["gdp_md_est"]<10**6
                                             else "blue"}

        Plus all supported by :func:`folium.vector_layers.path_options`. See the
        documentation of :class:`folium.features.GeoJson` for details.

    highlight_kwds : dict (default {})
        Style to be passed to folium highlight_function. Uses the same keywords
        as ``style_kwds``. When empty, defaults to ``{"fillOpacity": 0.75}``.
    missing_kwds : dict (default {})
        Additional style for missing values:

        color : str
            Color of missing values. Defaults to ``None``, which uses Folium's default.
        label : str (default "NaN")
            Legend entry for missing values.
    tooltip_kwds : dict (default {})
        Additional keywords to be passed to :class:`folium.features.GeoJsonTooltip`,
        e.g. ``aliases``, ``labels``, or ``sticky``.
    popup_kwds : dict (default {})
        Additional keywords to be passed to :class:`folium.features.GeoJsonPopup`,
        e.g. ``aliases`` or ``labels``.
    legend_kwds : dict (default {})
        Additional keywords to be passed to the legend.

        Currently supported customisation:

        caption : string
            Custom caption of the legend. Defaults to the column name.

        Additional accepted keywords when ``scheme`` is specified:

        colorbar : bool (default True)
            An option to control the style of the legend. If True, continuous
            colorbar will be used. If False, categorical legend will be used for bins.
        scale : bool (default True)
            Scale bins along the colorbar axis according to the bin edges (True)
            or use the equal length for each bin (False)
        fmt : string (default "{:.2f}")
            A formatting specification for the bin edges of the classes in the
            legend. For example, to have no decimals: ``{"fmt": "{:.0f}"}``. Applies
            if ``colorbar=False``.
        labels : list-like
            A list of legend labels to override the auto-generated labels.
            Needs to have the same number of elements as the number of
            classes (`k`). Applies if ``colorbar=False``.
        interval : boolean (default False)
            An option to control brackets from mapclassify legend.
            If True, open/closed interval brackets are shown in the legend.
            Applies if ``colorbar=False``.
        max_labels : int, default 10
            Maximum number of colorbar tick labels (requires branca>=0.5.0)
    map_kwds : dict (default {})
        Additional keywords to be passed to folium :class:`~folium.folium.Map`,
        e.g. ``dragging``, or ``scrollWheelZoom``.


    **kwargs : dict
        Additional options to be passed on to the folium object.

    Returns
    -------
    m : folium.folium.Map
        folium :class:`~folium.folium.Map` instance

    Examples
    --------
    >>> import geodatasets
    >>> df = geopandas.read_file(
    ...     geodatasets.get_path("geoda.chicago_health")
    ... )
    >>> df.head(2)  # doctest: +SKIP
       ComAreaID  ...                                           geometry
    0         35  ...  POLYGON ((-87.60914 41.84469, -87.60915 41.844...
    1         36  ...  POLYGON ((-87.59215 41.81693, -87.59231 41.816...

    [2 rows x 87 columns]

    >>> df.explore("Pop2012", cmap="Blues")  # doctest: +SKIP
    """

    def _colormap_helper(_cmap, n_resample=None, idx=None):
        """Return the color map specified.

        Helper function for MPL deprecation - GH#2596.
        """
        if not n_resample:
            return cm.get_cmap(_cmap)
        else:
            return cm.get_cmap(_cmap).resampled(n_resample)(idx)

    try:
        import re

        import branca as bc
        import folium
        import matplotlib.pyplot as plt
        from mapclassify import classify
        from matplotlib import colormaps as cm
        from matplotlib import colors

    except (ImportError, ModuleNotFoundError):
        raise ImportError(
            "The 'folium>=0.12', 'matplotlib' and 'mapclassify' packages "
            "are required for 'explore()'. You can install them using "
            "'conda install -c conda-forge \"folium>=0.12\" matplotlib mapclassify' "
            "or 'pip install \"folium>=0.12\" matplotlib mapclassify'."
        )

    # xyservices is an optional dependency
    try:
        import xyzservices

        HAS_XYZSERVICES = True
    except (ImportError, ModuleNotFoundError):
        HAS_XYZSERVICES = False

    gdf = df.copy()

    # convert LinearRing to LineString
    rings_mask = df.geom_type == "LinearRing"
    if rings_mask.any():
        gdf.geometry[rings_mask] = gdf.geometry[rings_mask].apply(
            lambda g: LineString(g)
        )
    if isinstance(gdf, geopandas.GeoSeries):
        gdf = gdf.to_frame()

    if gdf.crs is None:
        kwargs["crs"] = "Simple"
        tiles = None
    elif not gdf.crs.equals(4326):
        gdf = gdf.to_crs(4326)

    # Fields which are not JSON serializable are coerced to strings
    json_not_supported_cols = gdf.columns[
        [is_datetime64_any_dtype(gdf[c]) for c in gdf.columns]
    ].union(gdf.columns[gdf.dtypes == "object"])

    if len(json_not_supported_cols) > 0:
        gdf = gdf.astype({c: "string" for c in json_not_supported_cols})

    if not isinstance(gdf.index, pd.MultiIndex) and (
        is_datetime64_any_dtype(gdf.index) or (gdf.index.dtype == "object")
    ):
        gdf.index = gdf.index.astype("string")

    # create folium.Map object
    if m is None:
        # Get bounds to specify location and map extent
        bounds = gdf.total_bounds
        location = kwargs.pop("location", None)
        if location is None and not np.isnan(bounds).all():
            x = mean([bounds[0], bounds[2]])
            y = mean([bounds[1], bounds[3]])
            location = (y, x)
            if "zoom_start" in kwargs.keys():
                fit = False
            else:
                fit = True
        else:
            fit = False

        # get a subset of kwargs to be passed to folium.Map
        for i in _MAP_KWARGS:
            if i in map_kwds:
                raise ValueError(
                    f"'{i}' cannot be specified in 'map_kwds'. "
                    f"Use the '{i}={map_kwds[i]}' argument instead."
                )
        map_kwds = {
            **map_kwds,
            **{i: kwargs[i] for i in kwargs.keys() if i in _MAP_KWARGS},
        }

        if HAS_XYZSERVICES:
            # match provider name string to xyzservices.TileProvider
            if isinstance(tiles, str):
                try:
                    tiles = xyzservices.providers.query_name(tiles)
                except ValueError:
                    pass

            if isinstance(tiles, xyzservices.TileProvider):
                attr = attr if attr else tiles.html_attribution
                if "min_zoom" not in map_kwds:
                    map_kwds["min_zoom"] = tiles.get("min_zoom", 0)
                if "max_zoom" not in map_kwds:
                    map_kwds["max_zoom"] = tiles.get("max_zoom", 18)
                tiles = tiles.build_url(scale_factor="{r}")

        m = folium.Map(
            location=location,
            control_scale=control_scale,
            tiles=tiles,
            attr=attr,
            width=width,
            height=height,
            **map_kwds,
        )

        # fit bounds to get a proper zoom level
        if fit:
            m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])

    if gdf.is_empty.all():
        warnings.warn(
            "The GeoSeries you are attempting to plot is "
            "composed of empty geometries. Nothing has been displayed.",
            UserWarning,
            stacklevel=3,
        )
        return m

    for map_kwd in _MAP_KWARGS:
        kwargs.pop(map_kwd, None)

    nan_idx = None

    if column is not None:
        if pd.api.types.is_list_like(column):
            if len(column) != gdf.shape[0]:
                raise ValueError(
                    "The GeoDataFrame and given column have different number of rows."
                )
            else:
                column_name = "__plottable_column"
                gdf[column_name] = column
                column = column_name
        elif isinstance(gdf[column].dtype, pd.CategoricalDtype):
            if categories is not None:
                raise ValueError(
                    "Cannot specify 'categories' when column has categorical dtype"
                )
            categorical = True
        elif (
            pd.api.types.is_object_dtype(gdf[column])
            or pd.api.types.is_bool_dtype(gdf[column])
            or pd.api.types.is_string_dtype(gdf[column])
            or categories
        ):
            categorical = True

        nan_idx = pd.isna(gdf[column])

        if categorical:
            cat = pd.Categorical(gdf[column][~nan_idx], categories=categories)
            N = len(cat.categories)
            cmap = cmap if cmap else "tab20"

            # colormap exists in matplotlib
            if cmap in plt.colormaps():
                color = np.apply_along_axis(
                    colors.to_hex,
                    1,
                    _colormap_helper(cmap, n_resample=N, idx=cat.codes),
                )
                legend_colors = np.apply_along_axis(
                    colors.to_hex, 1, _colormap_helper(cmap, n_resample=N, idx=range(N))
                )

            # colormap is matplotlib.Colormap
            elif isinstance(cmap, colors.Colormap):
                color = np.apply_along_axis(colors.to_hex, 1, cmap(cat.codes))
                legend_colors = np.apply_along_axis(colors.to_hex, 1, cmap(range(N)))

            # custom list of colors
            elif pd.api.types.is_list_like(cmap):
                if N > len(cmap):
                    cmap = cmap * (N // len(cmap) + 1)
                color = np.take(cmap, cat.codes)
                legend_colors = np.take(cmap, range(N))

            else:
                raise ValueError(
                    "'cmap' is invalid. For categorical plots, pass either valid "
                    "named matplotlib colormap or a list-like of colors."
                )

        elif callable(cmap):
            # List of colors based on Branca colormaps or self-defined functions
            color = [cmap(x) for x in df[column]]

        else:
            vmin = gdf[column].min() if vmin is None else vmin
            vmax = gdf[column].max() if vmax is None else vmax

            # get bins
            if scheme is not None:
                if classification_kwds is None:
                    classification_kwds = {}
                if "k" not in classification_kwds:
                    classification_kwds["k"] = k

                binning = classify(
                    np.asarray(gdf[column][~nan_idx]), scheme, **classification_kwds
                )
                color = np.apply_along_axis(
                    colors.to_hex,
                    1,
                    _colormap_helper(cmap, n_resample=binning.k, idx=binning.yb),
                )

            else:
                bins = np.linspace(vmin, vmax, 257)[1:]
                binning = classify(
                    np.asarray(gdf[column][~nan_idx]), "UserDefined", bins=bins
                )

                color = np.apply_along_axis(
                    colors.to_hex,
                    1,
                    _colormap_helper(cmap, n_resample=256, idx=binning.yb),
                )

    # set default style
    if "fillOpacity" not in style_kwds:
        style_kwds["fillOpacity"] = 0.5
    if "weight" not in style_kwds:
        style_kwds["weight"] = 2
    if "style_function" in style_kwds:
        style_kwds_function = style_kwds["style_function"]
        if not callable(style_kwds_function):
            raise ValueError("'style_function' has to be a callable")
        style_kwds.pop("style_function")
    else:

        def _no_style(x):
            return {}

        style_kwds_function = _no_style

    # specify color
    if color is not None:
        if (
            isinstance(color, str)
            and isinstance(gdf, geopandas.GeoDataFrame)
            and color in gdf.columns
        ):  # use existing column

            def _style_color(x):
                base_style = {
                    "fillColor": x["properties"][color],
                    **style_kwds,
                }
                return {
                    **base_style,
                    **style_kwds_function(x),
                }

            style_function = _style_color
        else:  # assign new column
            if isinstance(gdf, geopandas.GeoSeries):
                gdf = geopandas.GeoDataFrame(geometry=gdf)

            if nan_idx is not None and nan_idx.any():
                nan_color = missing_kwds.pop("color", None)

                gdf["__folium_color"] = nan_color
                gdf.loc[~nan_idx, "__folium_color"] = color
            else:
                gdf["__folium_color"] = color

            stroke_color = style_kwds.pop("color", None)
            if not stroke_color:

                def _style_column(x):
                    base_style = {
                        "fillColor": x["properties"]["__folium_color"],
                        "color": x["properties"]["__folium_color"],
                        **style_kwds,
                    }
                    return {
                        **base_style,
                        **style_kwds_function(x),
                    }

                style_function = _style_column
            else:

                def _style_stroke(x):
                    base_style = {
                        "fillColor": x["properties"]["__folium_color"],
                        "color": stroke_color,
                        **style_kwds,
                    }
                    return {
                        **base_style,
                        **style_kwds_function(x),
                    }

                style_function = _style_stroke
    else:  # use folium default

        def _style_default(x):
            return {**style_kwds, **style_kwds_function(x)}

        style_function = _style_default

    if highlight:
        if "fillOpacity" not in highlight_kwds:
            highlight_kwds["fillOpacity"] = 0.75

        def _style_highlight(x):
            return {**highlight_kwds}

        highlight_function = _style_highlight
    else:
        highlight_function = None

    # define default for points
    if marker_type is None:
        marker_type = "circle_marker"

    marker = marker_type
    if isinstance(marker_type, str):
        if marker_type == "marker":
            marker = folium.Marker(**marker_kwds)
        elif marker_type == "circle":
            marker = folium.Circle(**marker_kwds)
        elif marker_type == "circle_marker":
            marker_kwds["radius"] = marker_kwds.get("radius", 2)
            marker_kwds["fill"] = marker_kwds.get("fill", True)
            marker = folium.CircleMarker(**marker_kwds)
        else:
            raise ValueError(
                "Only 'marker', 'circle', and 'circle_marker' are "
                "supported as marker values"
            )

    # remove additional geometries
    if isinstance(gdf, geopandas.GeoDataFrame):
        non_active_geoms = [
            name
            for name, val in (gdf.dtypes == "geometry").items()
            if val and name != gdf.geometry.name
        ]
        gdf = gdf.drop(columns=non_active_geoms)

    # prepare tooltip and popup
    if isinstance(gdf, geopandas.GeoDataFrame):
        # add named index to the tooltip
        if gdf.index.name is not None:
            gdf = gdf.reset_index()
        # specify fields to show in the tooltip
        tooltip = _tooltip_popup("tooltip", tooltip, gdf, **tooltip_kwds)
        popup = _tooltip_popup("popup", popup, gdf, **popup_kwds)
    else:
        tooltip = None
        popup = None
    # escape the curly braces {{}} for jinja2 templates
    feature_collection = gdf[
        ~(gdf.geometry.isna() | gdf.geometry.is_empty)  # drop missing or empty geoms
    ].__geo_interface__
    for feature in feature_collection["features"]:
        for prop in feature["properties"]:
            # escape the curly braces in values
            if isinstance(feature["properties"][prop], str):
                feature["properties"][prop] = re.sub(
                    r"\{{2,}",
                    lambda x: "{% raw %}" + x.group(0) + "{% endraw %}",
                    feature["properties"][prop],
                )

    # add dataframe to map
    folium.GeoJson(
        feature_collection,
        tooltip=tooltip,
        popup=popup,
        marker=marker,
        style_function=style_function,
        highlight_function=highlight_function,
        **kwargs,
    ).add_to(m)

    if legend:
        # NOTE: overlaps will be resolved in branca #88
        caption = column if not column == "__plottable_column" else ""
        caption = legend_kwds.pop("caption", caption)
        if categorical:
            categories = cat.categories.to_list()
            legend_colors = legend_colors.tolist()

            if nan_idx.any() and nan_color:
                categories.append(missing_kwds.pop("label", "NaN"))
                legend_colors.append(nan_color)

            _categorical_legend(m, caption, categories, legend_colors)
        elif column is not None:
            cbar = legend_kwds.pop("colorbar", True)
            colormap_kwds = {}
            if "max_labels" in legend_kwds:
                colormap_kwds["max_labels"] = legend_kwds.pop("max_labels")
            if scheme:
                cb_colors = np.apply_along_axis(
                    colors.to_hex,
                    1,
                    _colormap_helper(cmap, n_resample=binning.k, idx=range(binning.k)),
                )
                if cbar:
                    if legend_kwds.pop("scale", True):
                        index = [vmin] + binning.bins.tolist()
                    else:
                        index = None
                    colorbar = bc.colormap.StepColormap(
                        cb_colors,
                        vmin=vmin,
                        vmax=vmax,
                        caption=caption,
                        index=index,
                        **colormap_kwds,
                    )
                else:
                    fmt = legend_kwds.pop("fmt", "{:.2f}")
                    if "labels" in legend_kwds:
                        categories = legend_kwds["labels"]
                    else:
                        categories = binning.get_legend_classes(fmt)
                        show_interval = legend_kwds.pop("interval", False)
                        if not show_interval:
                            categories = [c[1:-1] for c in categories]

                    if nan_idx.any() and nan_color:
                        categories.append(missing_kwds.pop("label", "NaN"))
                        cb_colors = np.append(cb_colors, nan_color)
                    _categorical_legend(m, caption, categories, cb_colors)

            else:
                if isinstance(cmap, bc.colormap.ColorMap):
                    colorbar = cmap
                else:
                    mp_cmap = _colormap_helper(cmap)
                    cb_colors = np.apply_along_axis(
                        colors.to_hex, 1, mp_cmap(range(mp_cmap.N))
                    )

                    # linear legend
                    if mp_cmap.N > 20:
                        colorbar = bc.colormap.LinearColormap(
                            cb_colors,
                            vmin=vmin,
                            vmax=vmax,
                            caption=caption,
                            **colormap_kwds,
                        )

                    # steps
                    else:
                        colorbar = bc.colormap.StepColormap(
                            cb_colors,
                            vmin=vmin,
                            vmax=vmax,
                            caption=caption,
                            **colormap_kwds,
                        )

            if cbar:
                if nan_idx.any() and nan_color:
                    _categorical_legend(
                        m, "", [missing_kwds.pop("label", "NaN")], [nan_color]
                    )
                m.add_child(colorbar)

    return m


def _tooltip_popup(type, fields, gdf, **kwds):
    """Get tooltip or popup."""
    import folium

    # specify fields to show in the tooltip
    if fields is False or fields is None or fields == 0:
        return None
    else:
        if fields is True:
            fields = gdf.columns.drop(gdf.geometry.name).to_list()
        elif isinstance(fields, int):
            fields = gdf.columns.drop(gdf.geometry.name).to_list()[:fields]
        elif isinstance(fields, str):
            fields = [fields]

    for field in ["__plottable_column", "__folium_color"]:
        if field in fields:
            fields.remove(field)

    # Cast fields to str
    fields = list(map(str, fields))
    if type == "tooltip":
        return folium.GeoJsonTooltip(fields, **kwds)
    elif type == "popup":
        return folium.GeoJsonPopup(fields, **kwds)


def _categorical_legend(m, title, categories, colors):
    """Add categorical legend to a map.

    The implementation is using the code originally written by Michel Metran
    (@michelmetran) and released on GitHub
    (https://github.com/michelmetran/package_folium) under MIT license.

    Copyright (c) 2020 Michel Metran

    Parameters
    ----------
    m : folium.Map
        Existing map instance on which to draw the plot
    title : str
        title of the legend (e.g. column name)
    categories : list-like
        list of categories
    colors : list-like
        list of colors (in the same order as categories)
    """
    # Header to Add
    head = """
    {% macro header(this, kwargs) %}
    <script src="https://code.jquery.com/ui/1.12.1/jquery-ui.js"></script>
    <script>$( function() {
        $( ".maplegend" ).draggable({
            start: function (event, ui) {
                $(this).css({
                    right: "auto",
                    top: "auto",
                    bottom: "auto"
                });
            }
        });
    });
    </script>
    <style type='text/css'>
      .maplegend {
        position: absolute;
        z-index:9999;
        background-color: rgba(255, 255, 255, .8);
        border-radius: 5px;
        box-shadow: 0 0 15px rgba(0,0,0,0.2);
        padding: 10px;
        font: 12px/14px Arial, Helvetica, sans-serif;
        right: 10px;
        bottom: 20px;
      }
      .maplegend .legend-title {
        text-align: left;
        margin-bottom: 5px;
        font-weight: bold;
        }
      .maplegend .legend-scale ul {
        margin: 0;
        margin-bottom: 0px;
        padding: 0;
        float: left;
        list-style: none;
        }
      .maplegend .legend-scale ul li {
        list-style: none;
        margin-left: 0;
        line-height: 16px;
        margin-bottom: 2px;
        }
      .maplegend ul.legend-labels li span {
        display: block;
        float: left;
        height: 14px;
        width: 14px;
        margin-right: 5px;
        margin-left: 0;
        border: 0px solid #ccc;
        }
      .maplegend .legend-source {
        color: #777;
        clear: both;
        }
      .maplegend a {
        color: #777;
        }
    </style>
    {% endmacro %}
    """
    import branca as bc

    # Add CSS (on Header)
    macro = bc.element.MacroElement()
    macro._template = bc.element.Template(head)
    m.get_root().add_child(macro)

    body = f"""
    <div id='maplegend {title}' class='maplegend'>
        <div class='legend-title'>{title}</div>
        <div class='legend-scale'>
            <ul class='legend-labels'>"""

    # Loop Categories
    for label, color in zip(categories, colors):
        body += f"""
                <li><span style='background:{color}'></span>{label}</li>"""

    body += """
            </ul>
        </div>
    </div>
    """

    # Add Body
    body = bc.element.Element(body, "legend")
    m.get_root().html.add_child(body)


def _explore_geoseries(
    s,
    color=None,
    m=None,
    tiles="OpenStreetMap",
    attr=None,
    highlight=True,
    width="100%",
    height="100%",
    control_scale=True,
    marker_type=None,
    marker_kwds={},
    style_kwds={},
    highlight_kwds={},
    map_kwds={},
    **kwargs,
):
    """Interactive map based on GeoPandas and folium/leaflet.js.

    Generate an interactive leaflet map based on :class:`~geopandas.GeoSeries`

    Parameters
    ----------
    color : str, array-like (default None)
        Named color or a list-like of colors (named or hex).
    m : folium.Map (default None)
        Existing map instance on which to draw the plot.
    tiles : str, xyzservices.TileProvider (default 'OpenStreetMap Mapnik')
        Map tileset to use. Can choose from the list supported by folium, query a
        :class:`xyzservices.TileProvider` by a name from ``xyzservices.providers``,
        pass :class:`xyzservices.TileProvider` object or pass custom XYZ URL.
        The current list of built-in providers (when ``xyzservices`` is not available):

        ``["OpenStreetMap", "CartoDB positron", “CartoDB dark_matter"]``

        You can pass a custom tileset to Folium by passing a Leaflet-style URL
        to the tiles parameter: ``http://{s}.yourtiles.com/{z}/{x}/{y}.png``.
        Be sure to check their terms and conditions and to provide attribution with
        the ``attr`` keyword.
    attr : str (default None)
        Map tile attribution; only required if passing custom tile URL.
    highlight : bool (default True)
        Enable highlight functionality when hovering over a geometry.
    width : pixel int or percentage string (default: '100%')
        Width of the folium :class:`~folium.folium.Map`. If the argument
        m is given explicitly, width is ignored.
    height : pixel int or percentage string (default: '100%')
        Height of the folium :class:`~folium.folium.Map`. If the argument
        m is given explicitly, height is ignored.
    control_scale : bool, (default True)
        Whether to add a control scale on the map.
    marker_type : str, folium.Circle, folium.CircleMarker, folium.Marker (default None)
        Allowed string options are ('marker', 'circle', 'circle_marker'). Defaults to
        folium.Marker.
    marker_kwds: dict (default {})
        Additional keywords to be passed to the selected ``marker_type``, e.g.:

        radius : float
            Radius of the circle, in meters (for ``'circle'``) or pixels
            (for ``circle_marker``).
        icon : folium.map.Icon
            the :class:`folium.map.Icon` object to use to render the marker.
        draggable : bool (default False)
            Set to True to be able to drag the marker around the map.

    style_kwds : dict (default {})
        Additional style to be passed to folium ``style_function``:

        stroke : bool (default True)
            Whether to draw stroke along the path. Set it to ``False`` to
            disable borders on polygons or circles.
        color : str
            Stroke color
        weight : int
            Stroke width in pixels
        opacity : float (default 1.0)
            Stroke opacity
        fill : boolean (default True)
            Whether to fill the path with color. Set it to ``False`` to
            disable filling on polygons or circles.
        fillColor : str
            Fill color. Defaults to the value of the color option
        fillOpacity : float (default 0.5)
            Fill opacity.
        style_function : callable
            Function mapping a GeoJson Feature to a style ``dict``.

            * Style properties :func:`folium.vector_layers.path_options`
            * GeoJson features :class:`GeoSeries.__geo_interface__`

            e.g.::

                lambda x: {"color":"red" if x["properties"]["gdp_md_est"]<10**6
                                             else "blue"}


        Plus all supported by :func:`folium.vector_layers.path_options`. See the
        documentation of :class:`folium.features.GeoJson` for details.

    highlight_kwds : dict (default {})
        Style to be passed to folium highlight_function. Uses the same keywords
        as ``style_kwds``. When empty, defaults to ``{"fillOpacity": 0.75}``.
    map_kwds : dict (default {})
        Additional keywords to be passed to folium :class:`~folium.folium.Map`,
        e.g. ``dragging``, or ``scrollWheelZoom``.

    **kwargs : dict
        Additional options to be passed on to the folium.

    Returns
    -------
    m : folium.folium.Map
        folium :class:`~folium.folium.Map` instance

    """
    return _explore(
        s,
        color=color,
        m=m,
        tiles=tiles,
        attr=attr,
        highlight=highlight,
        width=width,
        height=height,
        control_scale=control_scale,
        marker_type=marker_type,
        marker_kwds=marker_kwds,
        style_kwds=style_kwds,
        highlight_kwds=highlight_kwds,
        map_kwds=map_kwds,
        **kwargs,
    )
