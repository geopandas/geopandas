import pandas as pd
import shapely.wkb

from geopandas import GeoDataFrame


def read_postgis(sql, con, geom_col='geom', crs=None, hex_encoded=True,
                 index_col=None, coerce_float=True, params=None):
    """
    Returns a GeoDataFrame corresponding to the result of the query
    string, which must contain a geometry column.

    Parameters
    ----------
    sql : string
        SQL query to execute in selecting entries from database, or name
        of the table to read from the database.
    con : DB connection object or SQLAlchemy engine
        Active connection to the database to query.
    geom_col : string, default 'geom'
        column name to convert to shapely geometries
    crs : dict or str, optional
        CRS to use for the returned GeoDataFrame; if not set, tries to
        determine CRS from the SRID associated with the first geometry in
        the database, and assigns that to all geometries.
    hex_encoded : bool, optional
        Whether the geometry is in a hex-encoded string. Default is True,
        standard for postGIS. Use hex_encoded=False for sqlite databases.

    See the documentation for pandas.read_sql for further explanation
    of the following parameters:
    index_col, coerce_float, params

    Returns
    -------
    GeoDataFrame

    Example
    -------
    >>> sql = "SELECT geom, kind FROM polygons;"
    >>> df = geopandas.read_postgis(sql, con)
    """

    df = pd.read_sql(sql, con, index_col=index_col, coerce_float=coerce_float,
                     params=params)

    if geom_col not in df:
        raise ValueError("Query missing geometry column '{}'".format(geom_col))

    def load_geom(x):
        if isinstance(x, bytes):
            return shapely.wkb.loads(x, hex=hex_encoded)
        else:
            return shapely.wkb.loads(str(x), hex=hex_encoded)
    geoms = df[geom_col].apply(load_geom)
    df[geom_col] = geoms

    if crs is None:
        if len(geoms) > 0:
            srid = shapely.geos.lgeos.GEOSGetSRID(geoms[0]._geom)
            # if no defined SRID in geodatabase, returns SRID of 0
            if srid != 0:
                crs = {"init": "epsg:{}".format(srid)}

    return GeoDataFrame(df, crs=crs, geometry=geom_col)
