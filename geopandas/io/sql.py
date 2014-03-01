from pandas import read_sql
import shapely.wkb


from geopandas import GeoSeries, GeoDataFrame

def read_postgis(sql, con, geom_col='geom', crs=None, index_col=None,
                 coerce_float=True, params=None):
    """
    Returns a GeoDataFrame corresponding to the result of the query 
    string, which must contain a geometry column.

    Examples:
    sql = "SELECT geom, kind FROM polygons;"
    df = geopandas.read_postgis(sql, con)

    Parameters
    ----------
    sql: string
    con: DB connection object
    geom_col: string, default 'geom'
        column name to convert to shapely geometries
    crs: optional
        CRS to use for the returned GeoDataFrame

    See the documentation for pandas.read_sql for further explanation 
    of the following parameters:
    index_col, coerce_float, params

    """
    df = read_sql(sql, con, index_col=index_col, coerce_float=coerce_float, 
                  params=params)
    if geom_col not in df:
        raise ValueError("Query missing geometry column '{}'".format(
            geom_col))

    wkb_geoms = df[geom_col]

    s = wkb_geoms.apply(lambda x: shapely.wkb.loads(x.decode('hex')))

    df[geom_col] = GeoSeries(s)

    return GeoDataFrame(df, crs=crs, geometry=geom_col)
