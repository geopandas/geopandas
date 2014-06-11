from binascii import unhexlify
from codecs import encode

from pandas import read_sql as pandas_sql
from shapely import wkb

from geopandas import GeoSeries, GeoDataFrame


def read_sql(sql, con, geom_col='geom', crs=None, index_col=None,
             coerce_float=True, params=None):
    """
    Returns a GeoDataFrame corresponding to the result of the query
    string, which must contain a geometry column.

    Examples:
    sql = "SELECT geom, kind FROM polygons;"
    df = geopandas.read_sql(sql, con)

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
    df = pandas_sql(sql, con, index_col=index_col, coerce_float=coerce_float,
                    params=params)
    if geom_col not in df:
        raise ValueError("Query missing geometry column '{0}'".format(
            geom_col))

    wkb_geoms = df[geom_col]

    # Inspect first entry for type
    if isinstance(wkb_geoms[0], bytes):  # SQLite WKB as bytes
        s = wkb_geoms.apply(lambda x: wkb.loads(unhexlify(encode(x, "hex"))))
    else:  # PostGIS WKB as string
        s = wkb_geoms.apply(lambda x: wkb.loads(unhexlify(x.encode())))

    df[geom_col] = GeoSeries(s)

    return GeoDataFrame(df, crs=crs, geometry=geom_col)


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
    return read_sql(sql, con, geom_col=geom_col, crs=crs, index_col=index_col,
                    coerce_float=coerce_float, params=params)


def read_sqlite(sql, con, geom_col='GEOMETRY', crs=None, index_col=None,
                coerce_float=True, params=None):
    """
    Returns a GeoDataFrame corresponding to the result of the query
    string, which must contain a geometry column.

    Examples:
    import sqlite3
    con = sqlite3.connect("path/to/polygons.sqlite")
    sql = "SELECT geom, kind FROM polygons;"
    df = geopandas.read_sqlite(sql, con)

    Note:
    This function expects OGC WKB geometries, which means that geometries
    stored in a spatialite database need to be returned in the standard WKB
    format, _not_ the internal BLOB format. A regular sqlite db will likely
    already be in WKB format. If using spatialite, you will have to enable the
    spatialite extension before querying the db:

    con.enable_load_extension(True)
    con.execute('SELECT load_extension("path/to/libspatialite.dll")') # Windows
    con.execute('SELECT load_extension("path/to/libspatialite.so")')  # Unix
    sql = "SELECT ST_AsBinary(geom), kind FROM polygons;"
    df = geopandas.read_sqlite(sql, con)

    Parameters
    ----------
    sql: string
    con: DB connection object
    geom_col: string, default 'geom'
        column name to convert to shapely geometries
    crs: optional
        CRS to use for the returned GeoDataFrame

    See the documentation for pandas.read_sql for further explanation
    of the `index_col` and `coerce_float` params.

    """
    return read_sql(sql, con, geom_col=geom_col, crs=crs, index_col=index_col,
                    coerce_float=coerce_float, params=params)
