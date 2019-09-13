import sys

import pandas as pd

import shapely.wkb

from geopandas import GeoDataFrame


def read_postgis(
    sql,
    con,
    geom_col="geom",
    crs=None,
    index_col=None,
    coerce_float=True,
    parse_dates=None,
    params=None,
):
    """
    Returns a GeoDataFrame corresponding to the result of the query
    string, which must contain a geometry column in WKB representation.

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

    See the documentation for pandas.read_sql for further explanation
    of the following parameters:
    index_col, coerce_float, parse_dates, params

    Returns
    -------
    GeoDataFrame

    Example
    -------
    PostGIS
    >>> sql = "SELECT geom, kind FROM polygons"
    SpatiaLite
    >>> sql = "SELECT ST_AsBinary(geom) AS geom, kind FROM polygons"
    >>> df = geopandas.read_postgis(sql, con)
    """

    df = pd.read_sql(
        sql,
        con,
        index_col=index_col,
        coerce_float=coerce_float,
        parse_dates=parse_dates,
        params=params,
    )

    if geom_col not in df:
        raise ValueError("Query missing geometry column '{}'".format(geom_col))

    geoms = df[geom_col].dropna()

    if not geoms.empty:
        load_geom_bytes = shapely.wkb.loads
        """Load from Python 3 binary."""

        def load_geom_buffer(x):
            """Load from Python 2 binary."""
            return shapely.wkb.loads(str(x))

        def load_geom_text(x):
            """Load from binary encoded as text."""
            return shapely.wkb.loads(str(x), hex=True)

        if sys.version_info.major < 3:
            if isinstance(geoms.iat[0], buffer):
                load_geom = load_geom_buffer
            else:
                load_geom = load_geom_text
        elif isinstance(geoms.iat[0], bytes):
            load_geom = load_geom_bytes
        else:
            load_geom = load_geom_text

        df[geom_col] = geoms = geoms.apply(load_geom)
        if crs is None:
            srid = shapely.geos.lgeos.GEOSGetSRID(geoms.iat[0]._geom)
            # if no defined SRID in geodatabase, returns SRID of 0
            if srid != 0:
                crs = {"init": "epsg:{}".format(srid)}

    return GeoDataFrame(df, crs=crs, geometry=geom_col)
