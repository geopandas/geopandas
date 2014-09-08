import pandas as pd
from geopandas import GeoDataFrame
from .overlay import _uniquify

def sjoin(left_df, right_df, how="left", op="intersects", use_sindex=True, **kwargs):
    """Spatial join of two GeoDataFrames.

    left_df, right_df are GeoDataFrames
    how: type of join
        left -> use keys from left_df; retain only left_df geometry column
        right -> use keys from right_df; retain only right_df geometry column
        inner -> use intersection of keys from both dfs; retain only left_df geometry column
    op: binary predicate {'intersects', 'contains', 'within'}
        see http://toblerity.org/shapely/manual.html#binary-predicates
    use_sindex : Use the spatial index to speed up operation? Default is True
    kwargs: passed to op method
    """
    allowed_hows = ['left', 'right', 'inner']

    if how not in allowed_hows:
        raise ValueError("`how` was \"%s\" but is expected to be in %s" % \
            (how, allowed_hows))

    if how == "right":
        # right outer join just implemented as the inverse of left; swap names
        left_df, right_df = right_df, left_df

    collection = []
    for i, feat in left_df.iterrows():
        geom = feat.geometry

        if use_sindex and right_df.sindex:
            candidates = [x.object for x in
                           right_df.sindex.intersection(geom.bounds, objects=True)]
        else:
            candidates = [i for i, x in right_df.iterrows()]

        feature_hits = 0
        for cand_id in candidates:
            candidate = right_df.ix[cand_id]
            if getattr(geom, op)(candidate.geometry, **kwargs):
                newseries = candidate.drop(right_df._geometry_column_name) 
                newfeat = pd.concat([feat, newseries])
                newfeat.index = _uniquify(newfeat.index)
                collection.append(newfeat)
                feature_hits += 1

        # TODO Should we perform aggregation if feature_hit > 1?
        # Advantage: single step and possible performance improvement
        # Disadvantage: Pandas already has groupby so user can do this later

        # If left does not spatially join with any right features,
        # Fill in the right columns with NA
        if how != 'inner' and feature_hits == 0:
            empty = pd.Series(dict.fromkeys(right_df.columns, None))
            empty.drop(right_df._geometry_column_name, inplace=True) 

            newfeat = pd.concat([feat, empty])
            newfeat.index = _uniquify(newfeat.index)
            collection.append(newfeat)

    return GeoDataFrame(collection, index=range(len(collection)))
