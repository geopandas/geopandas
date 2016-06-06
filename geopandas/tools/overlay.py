from shapely.ops import unary_union, polygonize
from shapely.geometry import MultiLineString
import pandas as pd
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

        if geom.type not in ['Polygon', 'MultiPolygon']:
            raise TypeError(poly_msg)

        if hasattr(geom, 'geoms'):
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

def overlay(df1, df2, how, use_sindex=True):
    """Perform spatial overlay between two polygons
    Currently only supports data GeoDataFrames with polygons

    Implements several methods (see `allowed_hows` list) that are
    all effectively subsets of the union.

    Parameters
    ----------
    df1 : GeoDataFrame with MultiPolygon or Polygon geometry column
    df2 : GeoDataFrame with MultiPolygon or Polygon geometry column
    how : method of spatial overlay
    use_sindex : Boolean; Use the spatial index to speed up operation. Default is True.

    Returns
    -------
    df : GeoDataFrame with new set of polygons and attributes resulting from the overlay
    """
    allowed_hows = [
        'intersection',
        'union',
        'identity',
        'symmetric_difference',
        'difference',  # aka erase
    ]

    if how not in allowed_hows:
        raise ValueError("`how` was \"%s\" but is expected to be in %s" % \
            (how, allowed_hows))

    if isinstance(df1, GeoSeries) or isinstance(df2, GeoSeries):
        raise NotImplementedError("overlay currently only implemented for GeoDataFrames")

    # Collect the interior and exterior rings
    rings1 = _extract_rings(df1)
    rings2 = _extract_rings(df2)
    mls1 = MultiLineString(rings1)
    mls2 = MultiLineString(rings2)

    # Union and polygonize
    try:
        # calculating union (try the fast unary_union)
        mm = unary_union([mls1, mls2])
    except:
        # unary_union FAILED
        # see https://github.com/Toblerity/Shapely/issues/47#issuecomment-18506767
        # calculating union again (using the slow a.union(b))
        mm = mls1.union(mls2)
    newpolys = polygonize(mm)

    # determine spatial relationship
    collect_df1 = []
    collect_df2 = []
    collect_geo = []
    for fid, newpoly in enumerate(newpolys):
        cent = newpoly.representative_point()

        # Test intersection with original polys
        # FIXME there should be a higher-level abstraction to search by bounds
        # and fall back in the case of no index?
        if use_sindex and df1.sindex is not None:
            candidates1 = df1.sindex.intersection(newpoly.bounds)
        else:
            candidates1 = [i for i, x in df1.iterrows()]

        if use_sindex and df2.sindex is not None:
            candidates2 = df2.sindex.intersection(newpoly.bounds)
        else:
            candidates2 = [i for i, x in df2.iterrows()]

        df1_hit = False
        df2_hit = False
        prop1 = None
        prop2 = None
        for cand_id1 in candidates1:
            if cent.intersects(df1.geometry.iat[cand_id1]):
                df1_hit = True
                break  # Take the first hit
        for cand_id2 in candidates2:
            if cent.intersects(df2.geometry.iat[cand_id2]):
                df2_hit = True
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

        # gather row ids and geometry for each new row
        collect_df1.append(cand_id1)
        collect_df2.append(cand_id2)
        collect_geo.append(newpoly)

    # take correct rows from df1 and df2 and combine in one dataframe
    df = pd.concat([df1.iloc[collect_df1].reset_index(drop=True),
                    df2.iloc[collect_df2].reset_index(drop=True)], axis=1)
    # remove the original geometries
    df = df.drop([df1._geometry_column_name, df2._geometry_column_name], axis=1)

    df.columns = _uniquify(df.columns)

    # Return geodataframe with new geometries
    return GeoDataFrame(df, geometry=collect_geo)
