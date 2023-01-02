"""
geopandas.clip
==============

A module to clip vector data using GeoPandas.

"""
import warnings

import numpy as np
import pandas.api.types
from shapely.geometry import Polygon, MultiPolygon, box

from geopandas import GeoDataFrame, GeoSeries
from geopandas.array import _check_crs, _crs_mismatch_warn


def _mask_is_list_like_rectangle(mask):
    return pandas.api.types.is_list_like(mask) and not isinstance(
        mask, (GeoDataFrame, GeoSeries, Polygon, MultiPolygon)
    )


def _clip_gdf_with_mask(gdf, mask):
    """Clip geometry to the polygon/rectangle extent.

    Clip an input GeoDataFrame to the polygon extent of the polygon
    parameter.

    Parameters
    ----------
    gdf : GeoDataFrame, GeoSeries
        Dataframe to clip.

    mask : (Multi)Polygon, list-like
        Reference polygon/rectangle for clipping.

    Returns
    -------
    GeoDataFrame
        The returned GeoDataFrame is a clipped subset of gdf
        that intersects with polygon/rectangle.
    """
    clipping_by_rectangle = _mask_is_list_like_rectangle(mask)
    if clipping_by_rectangle:
        intersection_polygon = box(*mask)
    else:
        intersection_polygon = mask

    gdf_sub = gdf.iloc[gdf.sindex.query(intersection_polygon, predicate="intersects")]

    # For performance reasons points don't need to be intersected with poly
    non_point_mask = gdf_sub.geom_type != "Point"

    if not non_point_mask.any():
        # only points, directly return
        return gdf_sub

    # Clip the data with the polygon
    if isinstance(gdf_sub, GeoDataFrame):
        clipped = gdf_sub.copy()
        if clipping_by_rectangle:
            clipped.loc[
                non_point_mask, clipped._geometry_column_name
            ] = gdf_sub.geometry.values[non_point_mask].clip_by_rect(*mask)
        else:
            clipped.loc[
                non_point_mask, clipped._geometry_column_name
            ] = gdf_sub.geometry.values[non_point_mask].intersection(mask)
    else:
        # GeoSeries
        clipped = gdf_sub.copy()
        if clipping_by_rectangle:
            clipped[non_point_mask] = gdf_sub.values[non_point_mask].clip_by_rect(*mask)
        else:
            clipped[non_point_mask] = gdf_sub.values[non_point_mask].intersection(mask)

    if clipping_by_rectangle:
        # clip_by_rect might return empty geometry collections in edge cases
        clipped = clipped[~clipped.is_empty]
    return clipped


def clip(gdf, mask, keep_geom_type=False):
    """Clip points, lines, or polygon geometries to the mask extent.

    Both layers must be in the same Coordinate Reference System (CRS).
    The ``gdf`` will be clipped to the full extent of the clip object.

    If there are multiple polygons in mask, data from ``gdf`` will be
    clipped to the total boundary of all polygons in mask.

    If the ``mask`` is list-like with four elements ``(minx, miny, maxx, maxy)``, a
    faster rectangle clipping algorithm will be used. Note that this can lead to
    slightly different results in edge cases, e.g. if a line would be reduced to a
    point, this point might not be returned.
    The geometry is clipped in a fast but possibly dirty way. The output is not
    guaranteed to be valid. No exceptions will be raised for topological errors.

    Parameters
    ----------
    gdf : GeoDataFrame or GeoSeries
        Vector layer (point, line, polygon) to be clipped to mask.
    mask : GeoDataFrame, GeoSeries, (Multi)Polygon, list-like
        Polygon vector layer used to clip ``gdf``.
        The mask's geometry is dissolved into one geometric feature
        and intersected with ``gdf``.
        If the mask is list-like with four elements ``(minx, miny, maxx, maxy)``,
        ``clip`` will use a faster rectangle clipping (:meth:`~GeoSeries.clip_by_rect`),
        possibly leading to slightly different results.
    keep_geom_type : boolean, default False
        If True, return only geometries of original type in case of intersection
        resulting in multiple geometry types or GeometryCollections.
        If False, return all resulting geometries (potentially mixed-types).

    Returns
    -------
    GeoDataFrame or GeoSeries
         Vector data (points, lines, polygons) from ``gdf`` clipped to
         polygon boundary from mask.

    See also
    --------
    GeoDataFrame.clip : equivalent GeoDataFrame method
    GeoSeries.clip : equivalent GeoSeries method

    Examples
    --------
    Clip points (global cities) with a polygon (the South American continent):

    >>> world = geopandas.read_file(
    ...     geopandas.datasets.get_path('naturalearth_lowres'))
    >>> south_america = world[world['continent'] == "South America"]
    >>> capitals = geopandas.read_file(
    ...     geopandas.datasets.get_path('naturalearth_cities'))
    >>> capitals.shape
    (243, 2)

    >>> sa_capitals = geopandas.clip(capitals, south_america)
    >>> sa_capitals.shape
    (15, 2)
    """
    if not isinstance(gdf, (GeoDataFrame, GeoSeries)):
        raise TypeError(
            "'gdf' should be GeoDataFrame or GeoSeries, got {}".format(type(gdf))
        )

    mask_is_list_like = _mask_is_list_like_rectangle(mask)
    if (
        not isinstance(mask, (GeoDataFrame, GeoSeries, Polygon, MultiPolygon))
        and not mask_is_list_like
    ):
        raise TypeError(
            "'mask' should be GeoDataFrame, GeoSeries,"
            f"(Multi)Polygon or list-like, got {type(mask)}"
        )

    if mask_is_list_like and len(mask) != 4:
        raise TypeError(
            "If 'mask' is list-like, it must have four values (minx, miny, maxx, maxy)"
        )

    if isinstance(mask, (GeoDataFrame, GeoSeries)):
        if not _check_crs(gdf, mask):
            _crs_mismatch_warn(gdf, mask, stacklevel=3)

    if isinstance(mask, (GeoDataFrame, GeoSeries)):
        box_mask = mask.total_bounds
    elif mask_is_list_like:
        box_mask = mask
    else:
        # Avoid empty tuple returned by .bounds when geometry is empty. A tuple of
        # all nan values is consistent with the behavior of
        # {GeoSeries, GeoDataFrame}.total_bounds for empty geometries.
        # TODO(shapely) can simpely use mask.bounds once relying on Shapely 2.0
        box_mask = mask.bounds if not mask.is_empty else (np.nan,) * 4
    box_gdf = gdf.total_bounds
    if not (
        ((box_mask[0] <= box_gdf[2]) and (box_gdf[0] <= box_mask[2]))
        and ((box_mask[1] <= box_gdf[3]) and (box_gdf[1] <= box_mask[3]))
    ):
        return gdf.iloc[:0]

    if isinstance(mask, (GeoDataFrame, GeoSeries)):
        combined_mask = mask.geometry.unary_union
    else:
        combined_mask = mask

    clipped = _clip_gdf_with_mask(gdf, combined_mask)

    if keep_geom_type:
        geomcoll_concat = (clipped.geom_type == "GeometryCollection").any()
        geomcoll_orig = (gdf.geom_type == "GeometryCollection").any()

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
                    clipped.geom_type.isin(polys).any(),
                    clipped.geom_type.isin(lines).any(),
                    clipped.geom_type.isin(points).any(),
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
                    clipped = clipped.explode(index_parts=False)
                if orig_type in polys:
                    clipped = clipped.loc[clipped.geom_type.isin(polys)]
                elif orig_type in lines:
                    clipped = clipped.loc[clipped.geom_type.isin(lines)]

    return clipped
