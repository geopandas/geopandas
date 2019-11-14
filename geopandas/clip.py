"""
geopandas.clip
============

A module to clip vector data using GeoPandas.

"""


import pandas as pd
import geopandas as gpd


def _clip_points(gdf, clip_obj):
    """Clip point geometry to the clip_obj GeoDataFrame extent.

    Clip an input point GeoDataFrame to the polygon extent of the clip_obj
    parameter. Points that intersect the clip_obj geometry are extracted with
    associated attributes and returned.

    Parameters
    ----------
    gdf : GeoDataFrame
        Composed of point geometry that will be clipped to the clip_obj.

    clip_obj : GeoDataFrame
        Reference geometry used to spatially clip the data.

    Returns
    -------
    GeoDataFrame
        The returned GeoDataFrame is a subset of gdf that intersects
        with clip_obj.
    """
    poly = clip_obj.geometry.unary_union

    spatial_index = gdf.sindex
    bbox = poly.bounds
    sidx = list(spatial_index.intersection(bbox))
    gdf_sub = gdf.iloc[sidx]

    return gdf_sub[gdf_sub.geometry.intersects(poly)]
    return gdf[gdf.geometry.intersects(poly)]


def _clip_multi_point(gdf, clip_obj):
    """Clip multi point features to the clip_obj GeoDataFrame extent.

    Clip an input multi point to the polygon extent of the clip_obj
    parameter. Points that intersect the clip_obj geometry are
    extracted with associated attributes returned.

    Parameters
    ----------
    gdf : GeoDataFrame
        Multipoint geometry that is clipped to clip_obj.

    clip_obj : GeoDataFrame
        Reference geometry used to spatially clip the data.

    Returns
    -------
    GeoDataFrame
        The returned GeoDataFrame is a clipped subset of gdf
        containing multi-point and point features.
    """

    # Explode multi-point features when clipping then recreate geom
    clipped = _clip_points(gdf.explode().reset_index(level=[1]), clip_obj)
    clipped = clipped.dissolve(by=[clipped.index]).drop(columns="level_1")

    return clipped


def _clip_line_poly(gdf, clip_obj):
    """Clip line and polygon geometry to the clip_obj GeoDataFrame extent.

    Clip an input line or polygon to the polygon extent of the clip_obj
    parameter. Lines or Polygons that intersect the clip_obj geometry are
    extracted with associated attributes and returned.

    Parameters
    ----------
    gdf : GeoDataFrame
        Line or polygon geometry that is clipped to clip_obj.

    clip_obj : GeoDataFrame
        Reference polygon for clipping.

    Returns
    -------
    GeoDataFrame
        The returned GeoDataFrame is a clipped subset of gdf
        that intersects with clip_obj.
    """
    # Create a single polygon object for clipping
    poly = clip_obj.geometry.unary_union
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


def _clip_multi_poly_line(gdf, clip_obj):
    """Clip multi lines and polygons to the clip_obj GeoDataFrame extent.

    Clip an input multi line or polygon to the polygon extent of the clip_obj
    parameter. Lines or Polygons that intersect the clip_obj geometry are
    extracted with associated attributes and returned.

    Parameters
    ----------
    gdf : GeoDataFrame
        multiLine or multipolygon geometry that is clipped to clip_obj.

    clip_obj : GeoDataFrame
        Reference polygon for clipping.

    Returns
    -------
    GeoDataFrame
        The returned GeoDataFrame is a clipped subset of gdf
        that intersects with clip_obj.
    """

    # Clip multi polygons
    # Explode multi polygons so that intersection works with all parts of the polygon
    # If this step isn't taken, intersection only gets the intersection of one part of the
    # multi part object. Also reset index to remove the column made by explode.
    clipped = _clip_line_poly(gdf.explode().reset_index(level=[1]), clip_obj)

    lines = clipped[
        (clipped.geometry.type == "MultiLineString")
        | (clipped.geometry.type == "LineString")
    ]
    line_diss = lines.dissolve(by=[lines.index]).drop(columns="level_1")

    polys = clipped[clipped.geometry.type == "Polygon"]
    poly_diss = polys.dissolve(by=[polys.index]).drop(columns="level_1")

    return gpd.GeoDataFrame(pd.concat([poly_diss, line_diss], ignore_index=True))


def clip(gdf, clip_obj):
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
    clip_obj : GeoDataFrame
          Polygon vector layer used to clip gdf.
          The clip_obj's geometry is dissolved into one geometric feature
          and intersected with gdf.

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
        >>> world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
        >>> south_america = world[world['continent'] == "South America"]
        >>> capitals = gpd.read_file(gpd.datasets.get_path('naturalearth_cities'))
        >>> capitals.shape
        (202, 2)
        >>> sa_capitals = cl.clip(capitals, south_america)
        >>> sa_capitals.shape
        (12, 2)
    """
    if not isinstance(gdf, (gpd.GeoDataFrame, gpd.GeoSeries)) and isinstance(clip_obj, (gpd.GeoDataFrame, gpd.GeoSeries)):
        raise AttributeError(
            "Please make sure that your input and clip GeoDataFrames have a"
            " valid geometry column"
        )

    if not any(gdf.intersects(clip_obj.unary_union)):
        raise ValueError("Shape and crop extent do not overlap.")

    if any(gdf.geometry.type == "MultiPoint"):
        return _clip_multi_point(gdf, clip_obj)
    elif any(gdf.geometry.type == "Point"):
        return _clip_points(gdf, clip_obj)
    elif any(gdf.geometry.type == "MultiPolygon") or any(
        gdf.geometry.type == "MultiLineString"
    ):
        return _clip_multi_poly_line(gdf, clip_obj)
    else:
        return _clip_line_poly(gdf, clip_obj)
