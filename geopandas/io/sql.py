import fiona.crs
import pandas as pd
from pandas.io import sql as pdsql
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


def get_srid(crs):
    """
    Returns the SRID from a CRS dict or string, if available.

    Parameters
    ----------
    crs : dict or str

    Returns
    -------
    srid : str
        If unsuccessful, returns None.
    """

    if not crs:
        return

    if isinstance(crs, str):
        crs = fiona.crs.from_string(crs)

    if 'init' in crs.keys():
        if 'epsg' in crs['init']:
            return crs['init'].split('epsg:')[1]


def to_postgis(df, name, con, hex_encoded=True, **kwargs):
    """
    Writes a geodataframe to a postgis database.

    Parameters
    ----------
    df : geodataframe
    name : str
        Name of table in database
    engine : SQLAlchemy engine

    **kwargs are passed to pandas.to_sql; see documentation for available
    parameters
    """

    geom = df.geometry
    temp_df = df.copy()

    def convert_geometry(geom):
        return shapely.wkb.dumps(geom, hex=hex_encoded)

    temp_df[temp_df.geometry.name] = geom.apply(convert_geometry)
    temp_df.to_sql(name, con, **kwargs)
    alter_args = {"name": name, "geom_name": temp_df.geometry.name}
    alter_cmd = "ALTER TABLE {name} ALTER COLUMN {geom_name} TYPE geometry"
    srid = get_srid(df.crs)

    if srid is not None:
        alter_cmd += " USING ST_SetSRID({geom_name}, {srid})"
        alter_args["srid"] = srid

    alter_cmd += ";"
    pdsql.execute(alter_cmd.format(**alter_args), con)
