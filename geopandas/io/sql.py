from pandas import read_sql
import shapely.wkb

from geopandas import GeoSeries, GeoDataFrame


def read_postgis(sql, con, geom_col='geom', crs=None, hex_encoded=True,
                 index_col=None, coerce_float=True, params=None):
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
        CRS to use for the returned GeoDataFrame; if not set, tries to
        determine CRS from the SRID associated with geometries in the
        database.
    hex_encoded : bool, optional
        Whether the geometry is in a hex-encoded string. Default is True,
        standard for postGIS.

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

    df = read_sql(sql, con, index_col=index_col, coerce_float=coerce_float,
                  params=params)

    if geom_col not in df:
        raise ValueError("Query missing geometry column '{}'".format(geom_col))

    def load_geom(x):
        return shapely.wkb.loads(str(x), hex=hex_encoded)
    wkb_geoms = df[geom_col]
    df[geom_col] = GeoSeries(wkb_geoms.apply(load_geom))

    if crs is None:
        def get_srid(x):
            return shapely.geos.lgeos.GEOSGetSRID(x._geom)
        unique_srids = df[geom_col].apply(get_srid).unique()
        # only set a srs for frame if all polygons have the same one
        if len(unique_srids) == 1:
            unique_srid = unique_srids[0]
            # if no defined SRID in geodatabase, returns SRID of 0
            if unique_srid != 0:
                crs = {"init": "epsg:{}".format(unique_srid)}

    return GeoDataFrame(df, crs=crs, geometry=geom_col)
