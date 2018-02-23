import binascii

import fiona.crs
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
    con: DB connection object or SQLAlchemy engine
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
        raise ValueError("Query missing geometry column '{0}'".format(
            geom_col))

    wkb_geoms = df[geom_col]

    def convert_geometry(binary_geom):
        return shapely.wkb.loads(binascii.unhexlify(binary_geom.encode()))

    s = wkb_geoms.apply(convert_geometry)

    df[geom_col] = GeoSeries(s)

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
    con : SQLAlchemy engine

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
    alter_cmd = "ALTER TABLE {name} ALTER COLUMN {geom_name} TYPE geometry;"
    con.execute(alter_cmd.format(**alter_args))

    srid = get_srid(df.crs)
    if srid is not None:
        srid_cmd = "SELECT UpdateGeometrySRID('{name}', '{geom_name}', {srid});"
        alter_args["srid"] = srid
        print(srid_cmd.format(**alter_args))
        con.execute(srid_cmd.format(**alter_args))
