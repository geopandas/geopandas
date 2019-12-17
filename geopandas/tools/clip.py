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
    """Clip point geometry to the polygon extent.

    Clip an input point GeoDataFrame to the polygon extent of the poly
    parameter. Points that intersect the poly geometry are extracted with
    associated attributes and returned.

    Parameters
    ----------
    gdf : GeoDataFrame, GeoSeries
        Composed of point geometry that will be clipped to the poly.

    poly : (Multi)Polygon
        Reference geometry used to spatially clip the data.

    Returns
    -------
    GeoDataFrame
        The returned GeoDataFrame is a subset of gdf that intersects
        with poly.
    """
    spatial_index = gdf.sindex
    bbox = poly.bounds
    sidx = list(spatial_index.intersection(bbox))
    gdf_sub = gdf.iloc[sidx]

    return gdf_sub[gdf_sub.geometry.intersects(poly)]


def _clip_line_poly(gdf, poly):
    """Clip line and polygon geometry to the polygon extent.

    Clip an input line or polygon to the polygon extent of the poly
    parameter. Parts of Lines or Polygons that intersect the poly geometry are
    extracted with associated attributes and returned.

    Parameters
    ----------
    gdf : GeoDataFrame, GeoSeries
        Line or polygon geometry that is clipped to poly.

    poly : (Multi)Polygon
        Reference polygon for clipping.

    Returns
    -------
    GeoDataFrame
        The returned GeoDataFrame is a clipped subset of gdf
        that intersects with poly.
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
    if isinstance(clipped, GeoDataFrame):
        clipped["geometry"] = gdf_sub.intersection(poly)

        # Return the clipped layer with no null geometry values or empty geometries
        return clipped[~clipped.geometry.is_empty & clipped.geometry.notnull()]
    clipped = gdf_sub.intersection(poly)
    return clipped[~clipped.is_empty & clipped.notnull()]


def clip(gdf, clip_obj, keep_geom_type=False):
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
    keep_geom_type : Boolean
          If a clip operation returns a GeometryCollection or more geometry
          types than the input, remove extra geometries from the output.

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
        raise ValueError("gdf and clip_obj extent do not overlap.")

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

    polys = ["Polygon", "MultiPolygon"]
    lines = ["LineString", "MultiLineString", "LinearRing"]
    points = ["Point", "MultiPoint"]

    # Check that the gdf for multiple geom types (points, lines and/or polys)
    orig_types_total = sum(
        [
            gdf.geom_type.isin(polys).any(),
            gdf.geom_type.isin(lines).any(),
            gdf.geom_type.isin(points).any(),
        ]
    )

    # Check how many geometry types are in the clipped GeoDataFrame
    clip_types_total = sum(
        [
            concat.geom_type.isin(polys).any(),
            concat.geom_type.isin(lines).any(),
            concat.geom_type.isin(points).any(),
        ]
    )

    # Check if the clipped geometry contains a geometry collection
    geometry_collection = (concat.geom_type == "GeometryCollection").any()

    # Check there aren't any new geom types in the clipped GeoDataFrame
    more_types = orig_types_total < clip_types_total

    if orig_types_total > 1 and keep_geom_type:
        warnings.warn("keep_geom_type can not be called on a mixed type GeoDataFrame.")
    elif keep_geom_type and not geometry_collection and not more_types:
        warnings.warn("keep_geom_type was called when no extra geometry types existed.")
    elif keep_geom_type and (geometry_collection or more_types):
        orig_type = gdf.geom_type.iloc[0]
        if geometry_collection:
            concat = concat.explode()
        if orig_type in polys:
            concat = concat.loc[concat.geom_type.isin(polys)]
        elif orig_type in lines:
            concat = concat.loc[concat.geom_type.isin(lines)]
    elif geometry_collection and not keep_geom_type:
        warnings.warn(
            "A geometry collection has been returned. Use .explode() to "
            "decompose the collection object or keep_geom_type=True to remove "
            "extra geometry types."
        )
    elif more_types and not keep_geom_type:
        warnings.warn(
            "More geometry types were returned than were in the original "
            "GeoDataFrame. This is likely due to an extra geometry type "
            "being created. To remove the extra geometry types set "
            "keep_geom_type=True."
        )
    if isinstance(concat, GeoDataFrame):
        concat["_order"] = order
        return concat.sort_values(by="_order").drop(columns="_order")
    concat = GeoDataFrame(geometry=concat)
    concat["_order"] = order
    return concat.sort_values(by="_order").geometry
