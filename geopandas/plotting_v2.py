import numpy as np

from ._compat import HAS_MATPLOTLIB

if HAS_MATPLOTLIB:
    from matplotlib.collections import PatchCollection

    class GeoPandasPolyCollection(PatchCollection):
        """Subclass to assign handler without overriding one for PatchCollection."""

    from matplotlib.legend import Legend
    from matplotlib.legend_handler import HandlerPolyCollection

    # PatchCollection is not supported by Legend but we can use PolyCollection handler
    # instead in our specific case. Define a subclass and assign a handler.
    Legend.update_default_handler_map(
        {GeoPandasPolyCollection: HandlerPolyCollection()}
    )


def _PolygonPatch(polygon, **kwargs):
    """Construct a matplotlib patch from a (Multi)Polygon geometry.

    The `kwargs` are those supported by the matplotlib.patches.PathPatch class
    constructor. Returns an instance of matplotlib.patches.PathPatch.

    To ensure proper rendering on the matplotlib side, winding order of individual
    rings needs to be normalized as the order is what matplotlib uses to determine
    if a Path represents a patch or a hole.

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

    if polygon.geom_type == "Polygon":
        path = Path.make_compound_path(
            Path(np.asarray(polygon.exterior.coords)[:, :2], closed=True),
            *[
                Path(np.asarray(ring.coords)[:, :2], closed=True)
                for ring in polygon.interiors
            ],
        )
    else:
        paths = []
        for part in polygon.geoms:
            # exteriors
            paths.append(Path(np.asarray(part.exterior.coords)[:, :2], closed=True))
            # interiors
            for ring in part.interiors:
                paths.append(Path(np.asarray(ring.coords)[:, :2], closed=True))
        path = Path.make_compound_path(*paths)

    return PathPatch(path, **kwargs)


def _plot_polygon_collection(
    ax,
    geoms,
    values=None,
    cmap=None,
    vmin=None,
    vmax=None,
    autolim=True,
    **kwargs,
):
    """Plot a collection of Polygon and MultiPolygon geometries to `ax`.

    Note that all style keywords, like ``color`` that can be set as an array in
    matplotlib shall be passed directly via kwargs.

    Parameters
    ----------
    ax : _type_
        _description_
    geoms : _type_
        _description_
    values : _type_, optional
        _description_, by default None
    cmap : _type_, optional
        _description_, by default None
    vmin : _type_, optional
        _description_, by default None
    vmax : _type_, optional
        _description_, by default None
    autolim : bool, optional
        _description_, by default True

    Returns
    -------
    _type_
        _description_
    """
    # GeoPandasPolyCollection does not accept some kwargs.
    kwargs = {
        att: value
        for att, value in kwargs.items()
        if att not in ["markersize", "marker"]
    }

    collection = GeoPandasPolyCollection(
        [_PolygonPatch(poly) for poly in geoms], **kwargs
    )

    if values is not None:
        collection.set_array(np.asarray(values))
        collection.set_cmap(cmap)
        if "norm" not in kwargs:
            collection.set_clim(vmin, vmax)

    ax.add_collection(collection, autolim=autolim)
    ax.autoscale_view()
    return collection
