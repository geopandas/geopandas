import geopandas as gpd
import numpy as np
import pandas as pd
import rtree
from shapely import prepared

def sjoin(left_df, right_df, how='left', op='intersects', crs_convert=True, lsuffix='left', rsuffix='right', **kwargs):
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

    # CHECK VALIDITY OF JOIN TYPE
    allowed_hows = ['left', 'right', 'inner']

    if how not in allowed_hows:
        raise ValueError("`how` was \"%s\" but is expected to be in %s" % \
            (how, allowed_hows))
    
    # CHECK VALIDITY OF PREDICATE OPERATION
    allowed_ops = ['contains', 'within', 'intersects']

    if op not in allowed_ops:
        raise ValueError("`op` was \"%s\" but is expected to be in %s" % \
            (op, allowed_ops))

    # IF WITHIN, SWAP NAMES
    if op == "within":
        # within implemented as the inverse of contains; swap names
        left_df, right_df = right_df, left_df

    # CONVERT CRS IF NOT EQUAL
    if left_df.crs != right_df.crs:
        print 'Warning: CRS does not match!'
        if crs_convert == True:
            print 'Converting CRS...'
            if left_df.values.nbytes >= right_df.values.nbytes:
                right_df = right_df.to_crs(left_df.crs)
            elif left_df.values.nbytes < right_df.values.nbytes:
                left_df = left_df.to_crs(right_df.crs)
	else:
	    pass

    # CONSTRUCT SPATIAL INDEX FOR RIGHT DATAFRAME
    tree_idx = rtree.index.Index()
    right_df_bounds = right_df['geometry'].apply(lambda x: x.bounds)
    for i in right_df_bounds.index:
        tree_idx.insert(i, right_df_bounds[i])

    # FIND INTERSECTION OF SPATIAL INDEX
    idxmatch = left_df['geometry'].apply(lambda x: x.bounds).apply(lambda x: list(tree_idx.intersection(x)))
    idxmatch = idxmatch[idxmatch.str.len() > 0]

    r_idx = np.concatenate(idxmatch.values)
    l_idx = np.concatenate((idxmatch.str.len()*pd.Series([[i] for i in idxmatch.index], index=idxmatch.index)).values)

    # VECTORIZE PREDICATE OPERATIONS
    def find_intersects(a1, a2):
        return a1.intersects(a2)

    def find_contains(a1, a2):
        return a1.contains(a2)

    predicate_d = {'intersects': find_intersects, 'contains': find_contains, 'within': find_contains}

    check_predicates = np.vectorize(predicate_d[op])

    # CHECK PREDICATES
    result = pd.DataFrame(np.column_stack([l_idx, r_idx, check_predicates(left_df['geometry'].apply(lambda x: prepared.prep(x)).values[l_idx], right_df['geometry'].values[r_idx])]))
    result.columns = ['index_%s' % lsuffix, 'index_%s' % rsuffix, 'match_bool']
    result = pd.DataFrame(result[result['match_bool']==1].set_index('index_%s' % lsuffix)['index_%s' % rsuffix])

    # IF 'WITHIN', SWAP NAMES AGAIN
    if op == "within":
        # within implemented as the inverse of contains; swap names
        left_df, right_df = right_df, left_df
        result = result.reset_index().rename(columns={'index_%s' % (lsuffix): 'index_%s' % (rsuffix), 'index_%s' % (rsuffix): 'index_%s' % (lsuffix)}).set_index('index_left').sort_index()    

    # APPLY JOIN
    return left_df.merge(result, left_index=True, right_index=True).merge(right_df, left_on='index_%s' % rsuffix, right_index=True, how=how, suffixes=('_%s' % lsuffix, '_%s' % rsuffix))
