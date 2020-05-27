"""
geopandas.clip
==============

A module to clip vector data using GeoPandas.

"""
import warnings

import numpy as np
import pandas as pd

from shapely.geometry import Polygon, MultiPolygon

from geopandas import GeoDataFrame, GeoSeries
from geopandas.array import _check_crs, _crs_mismatch_warn


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
    return gdf.iloc[gdf.sindex.query(poly, predicate="intersects")]


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
    gdf_sub = gdf.iloc[gdf.sindex.query(poly, predicate="intersects")]

    # Clip the data with the polygon
    if isinstance(gdf_sub, GeoDataFrame):
        clipped = gdf_sub.copy()
        clipped["geometry"] = gdf_sub.intersection(poly)
    else:
        # GeoSeries
        clipped = gdf_sub.intersection(poly)

    return clipped


def clip(gdf, mask, keep_geom_type=False):
    """Clip points, lines, or polygon geometries to the mask extent.

    Both layers must be in the same Coordinate Reference System (CRS).
    The `gdf` will be clipped to the full extent of the clip object.

    If there are multiple polygons in mask, data from `gdf` will be
    clipped to the total boundary of all polygons in mask.

    Parameters
    ----------
    gdf : GeoDataFrame or GeoSeries
        Vector layer (point, line, polygon) to be clipped to mask.
    mask : GeoDataFrame, GeoSeries, (Multi)Polygon
        Polygon vector layer used to clip `gdf`.
        The mask's geometry is dissolved into one geometric feature
        and intersected with `gdf`.
    keep_geom_type : boolean, default False
        If True, return only geometries of original type in case of intersection
        resulting in multiple geometry types or GeometryCollections.
        If False, return all resulting geometries (potentially mixed-types).

    Returns
    -------
    GeoDataFrame or GeoSeries
         Vector data (points, lines, polygons) from `gdf` clipped to
         polygon boundary from mask.

    Examples
    --------
    Clip points (global cities) with a polygon (the South American continent):

    >>> import geopandas
    >>> path =
    >>> world = geopandas.read_file(
    ...     geopandas.datasets.get_path('naturalearth_lowres'))
    >>> south_america = world[world['continent'] == "South America"]
    >>> capitals = geopandas.read_file(
    ...     geopandas.datasets.get_path('naturalearth_cities'))
    >>> capitals.shape
    (202, 2)
    >>> sa_capitals = geopandas.clip(capitals, south_america)
    >>> sa_capitals.shape
    (12, 2)
    """
    if not isinstance(gdf, (GeoDataFrame, GeoSeries)):
        raise TypeError(
            "'gdf' should be GeoDataFrame or GeoSeries, got {}".format(type(gdf))
        )

    if not isinstance(mask, (GeoDataFrame, GeoSeries, Polygon, MultiPolygon)):
        raise TypeError(
            "'mask' should be GeoDataFrame, GeoSeries or"
            "(Multi)Polygon, got {}".format(type(gdf))
        )

    if isinstance(mask, (GeoDataFrame, GeoSeries)):
        if not _check_crs(gdf, mask):
            _crs_mismatch_warn(gdf, mask, stacklevel=3)

    if isinstance(mask, (GeoDataFrame, GeoSeries)):
        box_mask = mask.total_bounds
    else:
        box_mask = mask.bounds
    box_gdf = gdf.total_bounds
    if not (
        ((box_mask[0] <= box_gdf[2]) and (box_gdf[0] <= box_mask[2]))
        and ((box_mask[1] <= box_gdf[3]) and (box_gdf[1] <= box_mask[3]))
    ):
        return gdf.iloc[:0]

    if isinstance(mask, (GeoDataFrame, GeoSeries)):
        poly = mask.geometry.unary_union
    else:
        poly = mask

    geom_types = gdf.geometry.type
    poly_idx = np.asarray((geom_types == "Polygon") | (geom_types == "MultiPolygon"))
    line_idx = np.asarray(
        (geom_types == "LineString")
        | (geom_types == "LinearRing")
        | (geom_types == "MultiLineString")
    )
    point_idx = np.asarray((geom_types == "Point") | (geom_types == "MultiPoint"))
    geomcoll_idx = np.asarray((geom_types == "GeometryCollection"))

    if point_idx.any():
        point_gdf = _clip_points(gdf[point_idx], poly)
    else:
        point_gdf = None

    if poly_idx.any():
        poly_gdf = _clip_line_poly(gdf[poly_idx], poly)
    else:
        poly_gdf = None

    if line_idx.any():
        line_gdf = _clip_line_poly(gdf[line_idx], poly)
    else:
        line_gdf = None

    if geomcoll_idx.any():
        geomcoll_gdf = _clip_line_poly(gdf[geomcoll_idx], poly)
    else:
        geomcoll_gdf = None

    order = pd.Series(range(len(gdf)), index=gdf.index)
    concat = pd.concat([point_gdf, line_gdf, poly_gdf, geomcoll_gdf])

    if keep_geom_type:
        geomcoll_concat = (concat.geom_type == "GeometryCollection").any()
        geomcoll_orig = geomcoll_idx.any()

        new_collection = geomcoll_concat and not geomcoll_orig

        if geomcoll_orig:
            warnings.warn(
                "keep_geom_type can not be called on a "
                "GeoDataFrame with GeometryCollection."
            )
        else:
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

            # Check there aren't any new geom types in the clipped GeoDataFrame
            more_types = orig_types_total < clip_types_total

            if orig_types_total > 1:
                warnings.warn(
                    "keep_geom_type can not be called on a mixed type GeoDataFrame."
                )
            elif new_collection or more_types:
                orig_type = gdf.geom_type.iloc[0]
                if new_collection:
                    concat = concat.explode()
                if orig_type in polys:
                    concat = concat.loc[concat.geom_type.isin(polys)]
                elif orig_type in lines:
                    concat = concat.loc[concat.geom_type.isin(lines)]

    # Return empty GeoDataFrame or GeoSeries if no shapes remain
    if len(concat) == 0:
        return gdf.iloc[:0]

    # Preserve the original order of the input
    if isinstance(concat, GeoDataFrame):
        concat["_order"] = order
        return concat.sort_values(by="_order").drop(columns="_order")
    else:
        concat = GeoDataFrame(geometry=concat)
        concat["_order"] = order
        return concat.sort_values(by="_order").geometry
