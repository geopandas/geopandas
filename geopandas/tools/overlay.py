from functools import reduce
import warnings

import numpy as np
import pandas as pd

from shapely.geometry import MultiLineString
from shapely.ops import polygonize, unary_union

from geopandas import GeoDataFrame, GeoSeries


def _uniquify(columns):
    ucols = []
    for col in columns:
        inc = 1
        newcol = col
        while newcol in ucols:
            inc += 1
            newcol = "{0}_{1}".format(col, inc)
        ucols.append(newcol)
    return ucols


def _extract_rings(df):
    """Collects all inner and outer linear rings from a GeoDataFrame
    with (multi)Polygon geometeries

    Parameters
    ----------
    df: GeoDataFrame with MultiPolygon or Polygon geometry column

    Returns
    -------
    rings: list of LinearRings
    """
    poly_msg = "overlay only takes GeoDataFrames with (multi)polygon geometries"
    rings = []
    geometry_column = df.geometry.name

    for i, feat in df.iterrows():
        geom = feat[geometry_column]

        if geom.type not in ["Polygon", "MultiPolygon"]:
            raise TypeError(poly_msg)

        if hasattr(geom, "geoms"):
            for poly in geom.geoms:  # if it's a multipolygon
                if not poly.is_valid:
                    # geom from layer is not valid attempting fix by buffer 0"
                    poly = poly.buffer(0)
                rings.append(poly.exterior)
                rings.extend(poly.interiors)
        else:
            if not geom.is_valid:
                # geom from layer is not valid attempting fix by buffer 0"
                geom = geom.buffer(0)
            rings.append(geom.exterior)
            rings.extend(geom.interiors)

    return rings


def _overlay_old(df1, df2, how, use_sindex=True, **kwargs):
    """Perform spatial overlay between two polygons.

    Currently only supports data GeoDataFrames with polygons.
    Implements several methods that are all effectively subsets of
    the union.

    Parameters
    ----------
    df1 : GeoDataFrame with MultiPolygon or Polygon geometry column
    df2 : GeoDataFrame with MultiPolygon or Polygon geometry column
    how : string
        Method of spatial overlay: 'intersection', 'union',
        'identity', 'symmetric_difference' or 'difference'.
    use_sindex : boolean, default True
        Use the spatial index to speed up operation if available.

    Returns
    -------
    df : GeoDataFrame
        GeoDataFrame with new set of polygons and attributes
        resulting from the overlay

    """
    allowed_hows = [
        "intersection",
        "union",
        "identity",
        "symmetric_difference",
        "difference",  # aka erase
    ]

    if how not in allowed_hows:
        raise ValueError(
            '`how` was "%s" but is expected to be in %s' % (how, allowed_hows)
        )

    if isinstance(df1, GeoSeries) or isinstance(df2, GeoSeries):
        raise NotImplementedError(
            "overlay currently only implemented for GeoDataFrames"
        )

    # Collect the interior and exterior rings
    rings1 = _extract_rings(df1)
    rings2 = _extract_rings(df2)
    mls1 = MultiLineString(rings1)
    mls2 = MultiLineString(rings2)

    # Union and polygonize
    mm = unary_union([mls1, mls2])
    newpolys = polygonize(mm)

    # determine spatial relationship
    collection = []
    for fid, newpoly in enumerate(newpolys):
        cent = newpoly.representative_point()

        # Test intersection with original polys
        # FIXME there should be a higher-level abstraction to search by bounds
        # and fall back in the case of no index?
        if use_sindex and df1.sindex is not None:
            candidates1 = [
                x.object for x in df1.sindex.intersection(newpoly.bounds, objects=True)
            ]
        else:
            candidates1 = [i for i, x in df1.iterrows()]

        if use_sindex and df2.sindex is not None:
            candidates2 = [
                x.object for x in df2.sindex.intersection(newpoly.bounds, objects=True)
            ]
        else:
            candidates2 = [i for i, x in df2.iterrows()]

        df1_hit = False
        df2_hit = False
        prop1 = None
        prop2 = None
        for cand_id in candidates1:
            cand = df1.loc[cand_id]
            if cent.intersects(cand[df1.geometry.name]):
                df1_hit = True
                prop1 = cand
                break  # Take the first hit
        for cand_id in candidates2:
            cand = df2.loc[cand_id]
            if cent.intersects(cand[df2.geometry.name]):
                df2_hit = True
                prop2 = cand
                break  # Take the first hit

        # determine spatial relationship based on type of overlay
        hit = False
        if how == "intersection" and (df1_hit and df2_hit):
            hit = True
        elif how == "union" and (df1_hit or df2_hit):
            hit = True
        elif how == "identity" and df1_hit:
            hit = True
        elif how == "symmetric_difference" and not (df1_hit and df2_hit):
            hit = True
        elif how == "difference" and (df1_hit and not df2_hit):
            hit = True

        if not hit:
            continue

        # gather properties
        if prop1 is None:
            prop1 = pd.Series(dict.fromkeys(df1.columns, None))
        if prop2 is None:
            prop2 = pd.Series(dict.fromkeys(df2.columns, None))

        # Concat but don't retain the original geometries
        out_series = pd.concat(
            [
                prop1.drop(df1._geometry_column_name),
                prop2.drop(df2._geometry_column_name),
            ]
        )

        out_series.index = _uniquify(out_series.index)

        # Create a geoseries and add it to the collection
        out_series["geometry"] = newpoly
        collection.append(out_series)

    # Return geodataframe with new indices
    return GeoDataFrame(collection, index=range(len(collection)))


def _ensure_geometry_column(df):
    """
    Helper function to ensure the geometry column is called 'geometry'.
    If another column with that name exists, it will be dropped.
    """
    if not df._geometry_column_name == "geometry":
        if "geometry" in df.columns:
            df.drop("geometry", axis=1, inplace=True)
        df.rename(
            columns={df._geometry_column_name: "geometry"}, copy=False, inplace=True
        )
        df.set_geometry("geometry", inplace=True)


def _overlay_intersection(df1, df2):
    """
    Overlay Intersection operation used in overlay function
    """
    # Spatial Index to create intersections
    spatial_index = df2.sindex
    bbox = df1.geometry.apply(lambda x: x.bounds)
    sidx = bbox.apply(lambda x: list(spatial_index.intersection(x)))
    # Create pairs of geometries in both dataframes to be intersected
    nei = []
    for i, j in enumerate(sidx):
        for k in j:
            nei.append([i, k])
    if nei != []:
        pairs = pd.DataFrame(nei, columns=["__idx1", "__idx2"])
        left = df1.geometry.take(pairs["__idx1"].values)
        left.reset_index(drop=True, inplace=True)
        right = df2.geometry.take(pairs["__idx2"].values)
        right.reset_index(drop=True, inplace=True)
        intersections = left.intersection(right).buffer(0)

        # only keep actual intersecting geometries
        pairs_intersect = pairs[~intersections.is_empty]
        geom_intersect = intersections[~intersections.is_empty]

        # merge data for intersecting geometries
        df1 = df1.reset_index(drop=True)
        df2 = df2.reset_index(drop=True)
        dfinter = pairs_intersect.merge(
            df1.drop(df1._geometry_column_name, axis=1),
            left_on="__idx1",
            right_index=True,
        )
        dfinter = dfinter.merge(
            df2.drop(df2._geometry_column_name, axis=1),
            left_on="__idx2",
            right_index=True,
            suffixes=["_1", "_2"],
        )

        return GeoDataFrame(dfinter, geometry=geom_intersect, crs=df1.crs)
    else:
        return GeoDataFrame(
            [],
            columns=list(set(df1.columns).union(df2.columns)) + ["__idx1", "__idx2"],
            crs=df1.crs,
        )


def _overlay_difference(df1, df2):
    """
    Overlay Difference operation used in overlay function
    """
    # Spatial Index to create intersections
    spatial_index = df2.sindex
    bbox = df1.geometry.apply(lambda x: x.bounds)
    sidx = bbox.apply(lambda x: list(spatial_index.intersection(x)))
    # Create differences
    new_g = []
    for geom, neighbours in zip(df1.geometry, sidx):
        new = reduce(
            lambda x, y: x.difference(y).buffer(0),
            [geom] + list(df2.geometry.iloc[neighbours]),
        )
        new_g.append(new)
    differences = GeoSeries(new_g, index=df1.index)
    geom_diff = differences[~differences.is_empty].copy()
    dfdiff = df1[~differences.is_empty].copy()
    dfdiff[dfdiff._geometry_column_name] = geom_diff
    return dfdiff


def _overlay_symmetric_diff(df1, df2):
    """
    Overlay Symmetric Difference operation used in overlay function
    """
    dfdiff1 = _overlay_difference(df1, df2)
    dfdiff2 = _overlay_difference(df2, df1)
    dfdiff1["__idx1"] = range(len(dfdiff1))
    dfdiff2["__idx2"] = range(len(dfdiff2))
    dfdiff1["__idx2"] = np.nan
    dfdiff2["__idx1"] = np.nan
    # ensure geometry name (otherwise merge goes wrong)
    _ensure_geometry_column(dfdiff1)
    _ensure_geometry_column(dfdiff2)
    # combine both 'difference' dataframes
    dfsym = dfdiff1.merge(
        dfdiff2, on=["__idx1", "__idx2"], how="outer", suffixes=["_1", "_2"]
    )
    geometry = dfsym.geometry_1.copy()
    geometry.name = "geometry"
    # https://github.com/pandas-dev/pandas/issues/26468 use loc for now
    geometry.loc[dfsym.geometry_1.isnull()] = dfsym.loc[
        dfsym.geometry_1.isnull(), "geometry_2"
    ]
    dfsym.drop(["geometry_1", "geometry_2"], axis=1, inplace=True)
    dfsym.reset_index(drop=True, inplace=True)
    dfsym = GeoDataFrame(dfsym, geometry=geometry, crs=df1.crs)
    return dfsym


def _overlay_union(df1, df2):
    """
    Overlay Union operation used in overlay function
    """
    dfinter = _overlay_intersection(df1, df2)
    dfsym = _overlay_symmetric_diff(df1, df2)
    dfunion = pd.concat([dfinter, dfsym], ignore_index=True, sort=False)
    # keep geometry column last
    columns = list(dfunion.columns)
    columns.remove("geometry")
    columns = columns + ["geometry"]
    return dfunion.reindex(columns=columns)


def overlay(df1, df2, how="intersection", make_valid=True, use_sindex=None):
    """Perform spatial overlay between two polygons.

    Currently only supports data GeoDataFrames with polygons.
    Implements several methods that are all effectively subsets of
    the union.

    Parameters
    ----------
    df1 : GeoDataFrame with MultiPolygon or Polygon geometry column
    df2 : GeoDataFrame with MultiPolygon or Polygon geometry column
    how : string
        Method of spatial overlay: 'intersection', 'union',
        'identity', 'symmetric_difference' or 'difference'.

    Returns
    -------
    df : GeoDataFrame
        GeoDataFrame with new set of polygons and attributes
        resulting from the overlay

    """
    if use_sindex is not None:
        warnings.warn(
            "'use_sindex' is deprecated. The overlay operation "
            "always requires a spatial index (rtree).",
            DeprecationWarning,
            stacklevel=2,
        )

    # Allowed operations
    allowed_hows = [
        "intersection",
        "union",
        "identity",
        "symmetric_difference",
        "difference",  # aka erase
    ]
    # Error Messages
    if how not in allowed_hows:
        raise ValueError(
            "`how` was '{0}' but is expected to be in %s".format(how, allowed_hows)
        )

    if isinstance(df1, GeoSeries) or isinstance(df2, GeoSeries):
        raise NotImplementedError(
            "overlay currently only implemented for GeoDataFrames"
        )

    accepted_types = ["Polygon", "MultiPolygon"]
    if (
        not df1.geom_type.isin(accepted_types).all()
        or not df2.geom_type.isin(accepted_types).all()
    ):
        raise TypeError(
            "overlay only takes GeoDataFrames with (multi)polygon  geometries."
        )

    # Computations
    df1 = df1.copy()
    df2 = df2.copy()
    df1[df1._geometry_column_name] = df1.geometry.buffer(0)
    df2[df2._geometry_column_name] = df2.geometry.buffer(0)

    if how == "difference":
        return _overlay_difference(df1, df2)
    elif how == "intersection":
        result = _overlay_intersection(df1, df2)
    elif how == "symmetric_difference":
        result = _overlay_symmetric_diff(df1, df2)
    elif how == "union":
        result = _overlay_union(df1, df2)
    elif how == "identity":
        dfunion = _overlay_union(df1, df2)
        result = dfunion[dfunion["__idx1"].notnull()].copy()
    result.reset_index(drop=True, inplace=True)
    result.drop(["__idx1", "__idx2"], axis=1, inplace=True)
    return result
