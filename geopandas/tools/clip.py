"""
geopandas.clip
============

A module to clip vector data using GeoPandas.

"""


import pandas as pd
from geopandas import GeoDataFrame, GeoSeries
import numpy as np
import warnings

from shapely.geometry import Polygon, MultiPolygon


def _clip_points(gdf, poly):
    """Clip point geometry to the clip_obj GeoDataFrame extent.

    Clip an input point GeoDataFrame to the polygon extent of the clip_obj
    parameter. Points that intersect the clip_obj geometry are extracted with
    associated attributes and returned.

    Parameters
    ----------
    gdf : GeoDataFrame
        Composed of point geometry that will be clipped to the clip_obj.

    poly : (Multi)Polygon
        Reference geometry used to spatially clip the data.

    Returns
    -------
    GeoDataFrame
        The returned GeoDataFrame is a subset of gdf that intersects
        with clip_obj.
    """
    spatial_index = gdf.sindex
    bbox = poly.bounds
    sidx = list(spatial_index.intersection(bbox))
    gdf_sub = gdf.iloc[sidx]

    return gdf_sub[gdf_sub.geometry.intersects(poly)]


def _clip_line_poly(gdf, poly):
    """Clip line and polygon geometry to the clip_obj GeoDataFrame extent.

    Clip an input line or polygon to the polygon extent of the clip_obj
    parameter. Lines or Polygons that intersect the clip_obj geometry are
    extracted with associated attributes and returned.

    Parameters
    ----------
    gdf : GeoDataFrame
        Line or polygon geometry that is clipped to clip_obj.

    poly : (Multi)Polygon
        Reference polygon for clipping.

    Returns
    -------
    GeoDataFrame
        The returned GeoDataFrame is a clipped subset of gdf
        that intersects with clip_obj.
    """
    # Create a single polygon object for clipping
    spatial_index = gdf.sindex

    # Create a box for the initial intersection
    bbox = poly.bounds
    # Get a list of id's for each object that overlaps the bounding box and
    # subset the data to just those lines
    sidx = list(spatial_index.intersection(bbox))
    gdf_sub = gdf.iloc[sidx]

    # Clip the data - with these data
    clipped = gdf_sub.copy()
    clipped["geometry"] = gdf_sub.intersection(poly)

    # Return the clipped layer with no null geometry values or empty geometries
    return clipped[~clipped.geometry.is_empty & clipped.geometry.notnull()]


def clip(gdf, clip_obj, drop_slivers=False):
    """Clip points, lines, or polygon geometries to the clip_obj extent.

    Both layers must be in the same Coordinate Reference System (CRS) and will
    be clipped to the full extent of the clip object.

    If there are multiple polygons in clip_obj,
    data from gdf will be clipped to the total boundary of
    all polygons in clip_obj.

    Parameters
    ----------
    gdf : GeoDataFrame
          Vector layer (point, line, polygon) to be clipped to clip_obj.
    clip_obj : GeoDataFrame, GeoSeries, (Multi)Polygon
          Polygon vector layer used to clip gdf.
          The clip_obj's geometry is dissolved into one geometric feature
          and intersected with gdf.
    drop_slivers : Boolean
          If a clip operation returns a GeometryCollection, remove polygon
          slivers and only return polygon features.

    Returns
    -------
    GeoDataFrame
         Vector data (points, lines, polygons) from gdf clipped to
         polygon boundary from clip_obj.

    Examples
    --------
    Clip points (global capital cities) with a polygon (the South American continent):

        >>> import geopandas as gpd
        >>> import geopandas.clip as gc
        >>> path = geopandas.datasets.get_path('naturalearth_lowres')
        >>> world = geopandas.read_file(path)
        >>> south_america = world[world['continent'] == "South America"]
        >>> capitals = gpd.read_file(gpd.datasets.get_path('naturalearth_cities'))
        >>> capitals.shape
        (202, 2)
        >>> sa_capitals = cl.clip(capitals, south_america)
        >>> sa_capitals.shape
        (12, 2)
    """
    if not isinstance(gdf, (GeoDataFrame, GeoSeries)):
        raise TypeError(
            "'gdf' should be GeoDataFrame or GeoSeries, got {}".format(type(gdf))
        )

    if not isinstance(clip_obj, (GeoDataFrame, GeoSeries, Polygon, MultiPolygon)):
        raise TypeError(
            "'clip_obj' should be GeoDataFrame, GeoSeries or"
            "(Multi)Polygon, got {}".format(type(gdf))
        )

    if isinstance(clip_obj, (GeoDataFrame, GeoSeries)):
        xmin, ymin, xmax, ymax = clip_obj.total_bounds
    else:
        xmin, ymin, xmax, ymax = clip_obj.bounds
    if gdf.cx[xmin:xmax, ymin:ymax].empty:
        raise ValueError("Shape and crop extent do not overlap.")

    if isinstance(clip_obj, (GeoDataFrame, GeoSeries)):
        poly = clip_obj.geometry.unary_union
    else:
        poly = clip_obj

    geom_types = gdf.geometry.type
    poly_idx = np.asarray((geom_types == "Polygon") | (geom_types == "MultiPolygon"))
    line_idx = np.asarray(
        (geom_types == "LineString")
        | (geom_types == "LinearRing")
        | (geom_types == "MultiLineString")
    )
    point_idx = np.asarray((geom_types == "Point") | (geom_types == "MultiPoint"))

    points = gdf[point_idx]
    if not points.empty:
        point_gdf = _clip_points(points, poly)
    else:
        point_gdf = None

    polys = gdf[poly_idx]
    if not polys.empty:
        poly_gdf = _clip_line_poly(polys, poly)
    else:
        poly_gdf = None

    lines = gdf[line_idx]
    if not lines.empty:
        line_gdf = _clip_line_poly(lines, poly)
    else:
        line_gdf = None

    order = pd.Series(range(len(gdf)), index=gdf.index)
    concat = pd.concat([point_gdf, line_gdf, poly_gdf])

    if (concat.geom_type == "GeometryCollection").any() and drop_slivers:
        concat = concat.explode()
        if not polys.empty:
            concat = concat.loc[concat.geom_type.isin(["Polygon", "MultiPolygon"])]
        elif not lines.empty:
            concat = concat.loc[
                concat.geom_type.isin(["LineString", "MultiLineString", "LinearRing"])
            ]
        elif not points.empty:
            concat = concat.loc[concat.geom_type.isin(["Point", "MultiPoint"])]
        else:
            raise TypeError("`drop_sliver` does not support {}.".format(type))
    elif (concat.geom_type == "GeometryCollection").any() and not drop_slivers:
        warnings.warn(
            "A geometry collection has been returned. Use .explode() to "
            "remove the collection object or drop_slivers=True to remove "
            "sliver geometries."
        )
    elif drop_slivers:
        warnings.warn("Drop slivers was called when no slivers existed.")
    concat["_order"] = order
    return concat.sort_values(by="_order").drop(columns="_order")
