import binascii

from pandas import read_sql
import shapely.wkb

from geopandas import GeoSeries, GeoDataFrame


def read_postgis(sql, con, geom_col='geom', crs=None, index_col=None,
                 coerce_float=True, params=None):
    """
    Returns a GeoDataFrame corresponding to the result of the query
    string, which must contain a geometry column.

    Parameters
    ----------
    sql : string
        SQL query to execute in selecting entries from database.
    con : DB connection object or SQLAlchemy engine
        Active connection to the database to query.
    geom_col : string, default 'geom'
        column name to convert to shapely geometries
    crs : dict or str, optional
        CRS to use for the returned GeoDataFrame

    See the documentation for pandas.read_sql for further explanation
    of the following parameters:
    index_col, coerce_float, params

    Returns
    -------
    geodataframe : GeoDataFrame

    Example
    -------
    >>> sql = "SELECT geom, kind FROM polygons;"
    >>> df = geopandas.read_postgis(sql, con)
    """

    # add SRID call to sql str
    sql_srid = sql.lower().replace(" from ",
                                   ",st_srid({}) from ".format(geom_col))
    df = read_sql(sql_srid, con, index_col=index_col, coerce_float=coerce_float,
                  params=params)

    wkb_geoms = df[geom_col]

    s = wkb_geoms.apply(lambda x: shapely.wkb.loads(binascii.unhexlify(x.encode())))
    df[geom_col] = GeoSeries(s)

    srid_col = "st_srid"
    if crs is None:
        unique_srid = df[srid_col].unique()
        # only set a crs for frame if all polygons have the same one
        if len(unique_srid) == 1:
            srid = unique_srid[0]
            crs = {"init": "epsg:{}".format(srid)}
    # user didn't ask for the st_srid column, so get rid of it now
    del(df[srid_col])

    return GeoDataFrame(df, crs=crs, geometry=geom_col)
