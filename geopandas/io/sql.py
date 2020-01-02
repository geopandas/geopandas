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


def get_geometry_type(gdf):
    """
    Get basic geometry type of a GeoDataFrame,
    and information if the gdf contains Geometry Collections."""
    geom_types = list(gdf.geometry.geom_type.unique())
    geom_collection = False

    # Get the basic geometry type
    basic_types = []
    for gt in geom_types:
        if 'Multi' in gt:
            geom_collection = True
            basic_types.append(gt.replace('Multi', ''))
        else:
            basic_types.append(gt)
    geom_types = list(set(basic_types))

    # Check for mixed geometry types
    assert len(geom_types) < 2, "GeoDataFrame contains mixed geometry types."
    geom_type = geom_types[0]
    return (geom_type, geom_collection)


def get_srid_from_crs(gdf):
    """
    Get EPSG code from CRS if available. If not, return -1.
    """
    from pyproj import CRS

    if gdf.crs is not None:
        try:
            if isinstance(gdf.crs, dict):
                # If CRS is in dictionary format use only the value
                # to avoid pyproj Future warning
                if 'init' in gdf.crs.keys():
                    srid = CRS(gdf.crs['init']).to_epsg(min_confidence=25)
                else:
                    srid = CRS(gdf.crs).to_epsg(min_confidence=25)
            else:
                srid = CRS(gdf).to_epsg(min_confidence=25)
            if srid is None:
                srid = -1
        except Exception:
            srid = -1
            print("Warning: Could not parse coordinate reference system from GeoDataFrame.",
                  "Inserting data without defined CRS.")
    return srid


def convert_to_wkb(gdf, geom_name):
    """Convert geometries to wkb. Use pygeos if available, otherwise use shapely."""
    try:
        import pygeos
        use_pygeos = True
    except ModuleNotFoundError:
        use_pygeos = False
        from shapely.wkb import dumps

    if use_pygeos:
        # With pygeos
        gdf[geom_name] = pygeos.to_wkb(pygeos.from_shapely(gdf[geom_name].to_list()),
                                       hex=True)
    else:
        # With Shapely
        gdf[geom_name] = gdf[geom_name].apply(lambda x: dumps(x, hex=True))
    return gdf


def write_to_db(gdf, engine, index, tbl, srid, geom_name):
    import io
    import csv

    # Convert columns to lists and make a generator
    args = [list(gdf[i]) for i in gdf.columns]
    if index:
        args.insert(0, list(gdf.index))

    data_iter = zip(*args)

    # get list of columns using pandas
    keys = tbl.insert_data()[0]
    columns = ', '.join('"{}"'.format(k) for k in list(keys))

    s_buf = io.StringIO()
    writer = csv.writer(s_buf)
    writer.writerows(data_iter)
    s_buf.seek(0)

    conn = engine.raw_connection()
    cur = conn.cursor()

    try:
        sql = 'COPY {} ({}) FROM STDIN WITH CSV'.format(
            tbl.table.fullname, columns)
        cur.copy_expert(sql=sql, file=s_buf)

        sql = "SELECT UpdateGeometrySRID('{schema}','{tbl}','{geom}',{srid})".format(
            schema=tbl.table.schema, tbl=tbl.table.name, geom=geom_name, srid=srid)
        cur.execute(sql)
        conn.commit()

    except Exception as e:
        conn.connection.rollback()
        raise e
    conn.close()


def write_postgis(gdf, engine, table, if_exists='fail',
                  schema=None, dtype=None, index=False):
    """
    Upload GeoDataFrame into PostGIS database.

    Parameters
    ----------

    gdf : GeoDataFrame
        GeoDataFrame containing the data for upload.
    engine : SQLAclchemy engine.
        Connection.
    if_exists : str
        What to do if table exists already: 'replace' | 'append' | 'fail'.
    schema : db-schema
        Database schema where the data will be uploaded (default: 'public').
    dtype : dict of column name to SQL type, default None
        Optional specifying the datatype for columns. The SQL type should be a
        SQLAlchemy type, or a string for sqlite3 fallback connection.
    index : bool
        Store DataFrame index to the database as well.
    """
    from geoalchemy2 import Geometry
    from shapely.geometry import MultiLineString, MultiPoint, MultiPolygon

    gdf = gdf.copy()
    geom_name = gdf.geometry.name
    if schema is not None:
        schema_name = schema
    else:
        schema_name = 'public'

    # Get srid
    srid = get_srid_from_crs(gdf)

    # Check geometry types
    geometry_type, contains_multi_geoms = get_geometry_type(gdf)

    # Build target geometry type
    if contains_multi_geoms:
        target_geom_type = "Multi{geom_type}".format(geom_type=geometry_type)
    else:
        target_geom_type = geometry_type

    # Build dtype with Geometry (srid is updated afterwards)
    if dtype is not None:
        dtype[geom_name] = Geometry(geometry_type=target_geom_type)
    else:
        dtype = {geom_name: Geometry(geometry_type=target_geom_type)}

    # Get Pandas SQLTable object (ignore 'geometry')
    # If dtypes is used, update table schema accordingly.
    pandas_sql = pd.io.sql.SQLDatabase(engine)
    tbl = pd.io.sql.SQLTable(name=table, pandas_sql_engine=pandas_sql,
                             frame=gdf, dtype=dtype, index=index,
                             schema=schema_name)

    # Check if table exists
    if tbl.exists():
        # If it exists, check if should overwrite
        if if_exists == 'replace':
            pandas_sql.drop_table(table)
            tbl.create()
        elif if_exists == 'fail':
            raise Exception("Table '{table}' exists in the database.".format(
                table=table))
        elif if_exists == 'append':
            pass
    else:
        tbl.create()

    # Convert to MultiGeoms if needed
    if contains_multi_geoms:
        mask = gdf[geom_name].geom_type == geometry_type
        if geometry_type == 'Point':
            gdf.loc[mask, geom_name] = gdf.loc[mask, geom_name].apply(
                lambda geom: MultiPoint([geom]))
        elif geometry_type == 'LineString':
            gdf.loc[mask, geom_name] = gdf.loc[mask, geom_name].apply(
                lambda geom: MultiLineString([geom]))
        elif geometry_type == 'Polygon':
            gdf.loc[mask, geom_name] = gdf.loc[mask, geom_name].apply(
                lambda geom: MultiPolygon([geom]))

    # Convert geometries to WKB
    gdf = convert_to_wkb(gdf, geom_name)

    # Write to database
    write_to_db(gdf, engine, index, tbl, srid, geom_name)

    return
