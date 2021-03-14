import warnings
from contextlib import contextmanager

import pandas as pd

import shapely.wkb

from geopandas import GeoDataFrame

from .. import _compat as compat


@contextmanager
def _get_conn(conn_or_engine):
    """
    Yield a connection within a transaction context.

    Engine.begin() returns a Connection with an implicit Transaction while
    Connection.begin() returns the Transaction. This helper will always return a
    Connection with an implicit (possibly nested) Transaction.

    Parameters
    ----------
    conn_or_engine : Connection or Engine
        A sqlalchemy Connection or Engine instance
    Returns
    -------
    Connection
    """
    from sqlalchemy.engine.base import Engine, Connection

    if isinstance(conn_or_engine, Connection):
        with conn_or_engine.begin():
            yield conn_or_engine
    elif isinstance(conn_or_engine, Engine):
        with conn_or_engine.begin() as conn:
            yield conn
    else:
        raise ValueError(f"Unknown Connectable: {conn_or_engine}")


def _df_to_geodf(df, geom_col="geom", crs=None):
    """
    Transforms a pandas DataFrame into a GeoDataFrame.
    The column 'geom_col' must be a geometry column in WKB representation.
    To be used to convert df based on pd.read_sql to gdf.
    Parameters
    ----------
    df : DataFrame
        pandas DataFrame with geometry column in WKB representation.
    geom_col : string, default 'geom'
        column name to convert to shapely geometries
    crs : pyproj.CRS, optional
        CRS to use for the returned GeoDataFrame. The value can be anything accepted
        by :meth:`pyproj.CRS.from_user_input() <pyproj.crs.CRS.from_user_input>`,
        such as an authority string (eg "EPSG:4326") or a WKT string.
        If not set, tries to determine CRS from the SRID associated with the
        first geometry in the database, and assigns that to all geometries.
    Returns
    -------
    GeoDataFrame
    """

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

        if isinstance(geoms.iat[0], bytes):
            load_geom = load_geom_bytes
        else:
            load_geom = load_geom_text

        df[geom_col] = geoms = geoms.apply(load_geom)
        if crs is None:
            srid = shapely.geos.lgeos.GEOSGetSRID(geoms.iat[0]._geom)
            # if no defined SRID in geodatabase, returns SRID of 0
            if srid != 0:
                crs = "epsg:{}".format(srid)

    return GeoDataFrame(df, crs=crs, geometry=geom_col)


def _read_postgis(
    sql,
    con,
    geom_col="geom",
    crs=None,
    index_col=None,
    coerce_float=True,
    parse_dates=None,
    params=None,
    chunksize=None,
):
    """
    Returns a GeoDataFrame corresponding to the result of the query
    string, which must contain a geometry column in WKB representation.

    Parameters
    ----------
    sql : string
        SQL query to execute in selecting entries from database, or name
        of the table to read from the database.
    con : sqlalchemy.engine.Connection or sqlalchemy.engine.Engine
        Active connection to the database to query.
    geom_col : string, default 'geom'
        column name to convert to shapely geometries
    crs : dict or str, optional
        CRS to use for the returned GeoDataFrame; if not set, tries to
        determine CRS from the SRID associated with the first geometry in
        the database, and assigns that to all geometries.
    chunksize : int, default None
        If specified, return an iterator where chunksize is the number of rows to
        include in each chunk.

    See the documentation for pandas.read_sql for further explanation
    of the following parameters:
    index_col, coerce_float, parse_dates, params, chunksize

    Returns
    -------
    GeoDataFrame

    Examples
    --------
    PostGIS

    >>> from sqlalchemy import create_engine  # doctest: +SKIP
    >>> db_connection_url = "postgres://myusername:mypassword@myhost:5432/mydatabase"
    >>> con = create_engine(db_connection_url)  # doctest: +SKIP
    >>> sql = "SELECT geom, highway FROM roads"
    >>> df = geopandas.read_postgis(sql, con)  # doctest: +SKIP

    SpatiaLite

    >>> sql = "SELECT ST_Binary(geom) AS geom, highway FROM roads"
    >>> df = geopandas.read_postgis(sql, con)  # doctest: +SKIP
    """

    if chunksize is None:
        # read all in one chunk and return a single GeoDataFrame
        df = pd.read_sql(
            sql,
            con,
            index_col=index_col,
            coerce_float=coerce_float,
            parse_dates=parse_dates,
            params=params,
            chunksize=chunksize,
        )
        return _df_to_geodf(df, geom_col=geom_col, crs=crs)

    else:
        # read data in chunks and return a generator
        df_generator = pd.read_sql(
            sql,
            con,
            index_col=index_col,
            coerce_float=coerce_float,
            parse_dates=parse_dates,
            params=params,
            chunksize=chunksize,
        )
        return (_df_to_geodf(df, geom_col=geom_col, crs=crs) for df in df_generator)


def read_postgis(*args, **kwargs):
    import warnings

    warnings.warn(
        "geopandas.io.sql.read_postgis() is intended for internal "
        "use only, and will be deprecated. Use geopandas.read_postgis() instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    return _read_postgis(*args, **kwargs)


def _get_geometry_type(gdf):
    """
    Get basic geometry type of a GeoDataFrame. See more info from:
    https://geoalchemy-2.readthedocs.io/en/latest/types.html#geoalchemy2.types._GISType

    Following rules apply:
     - if geometries all share the same geometry-type,
       geometries are inserted with the given GeometryType with following types:
        - Point, LineString, Polygon, MultiPoint, MultiLineString, MultiPolygon,
          GeometryCollection.
        - LinearRing geometries will be converted into LineString -objects.
     - in all other cases, geometries will be inserted with type GEOMETRY:
        - a mix of Polygons and MultiPolygons in GeoSeries
        - a mix of Points and LineStrings in GeoSeries
        - geometry is of type GeometryCollection,
          such as GeometryCollection([Point, LineStrings])
     - if any of the geometries has Z-coordinate, all records will
       be written with 3D.
    """
    geom_types = list(gdf.geometry.geom_type.unique())
    has_curve = False

    for gt in geom_types:
        if gt is None:
            continue
        elif "LinearRing" in gt:
            has_curve = True

    if len(geom_types) == 1:
        if has_curve:
            target_geom_type = "LINESTRING"
        else:
            if geom_types[0] is None:
                raise ValueError("No valid geometries in the data.")
            else:
                target_geom_type = geom_types[0].upper()
    else:
        target_geom_type = "GEOMETRY"

    # Check for 3D-coordinates
    if any(gdf.geometry.has_z):
        target_geom_type = target_geom_type + "Z"

    return target_geom_type, has_curve


def _get_srid_from_crs(gdf):
    """
    Get EPSG code from CRS if available. If not, return -1.
    """

    # Use geoalchemy2 default for srid
    # Note: undefined srid in PostGIS is 0
    srid = -1
    warning_msg = (
        "Could not parse CRS from the GeoDataFrame. "
        + "Inserting data without defined CRS.",
    )
    if gdf.crs is not None:
        try:
            srid = gdf.crs.to_epsg(min_confidence=25)
            if srid is None:
                srid = -1
                warnings.warn(warning_msg, UserWarning, stacklevel=2)
        except Exception:
            warnings.warn(warning_msg, UserWarning, stacklevel=2)
    return srid


def _convert_linearring_to_linestring(gdf, geom_name):
    from shapely.geometry import LineString

    # Todo: Use Pygeos function once it's implemented:
    #  https://github.com/pygeos/pygeos/issues/76

    mask = gdf.geom_type == "LinearRing"
    gdf.loc[mask, geom_name] = gdf.loc[mask, geom_name].apply(
        lambda geom: LineString(geom)
    )
    return gdf


def _convert_to_ewkb(gdf, geom_name, srid):
    """Convert geometries to ewkb. """
    if compat.USE_PYGEOS:
        from pygeos import set_srid, to_wkb

        geoms = to_wkb(
            set_srid(gdf[geom_name].values.data, srid=srid), hex=True, include_srid=True
        )

    else:
        from shapely.wkb import dumps

        geoms = [dumps(geom, srid=srid, hex=True) for geom in gdf[geom_name]]

    # The gdf will warn that the geometry column doesn't hold in-memory geometries
    # now that they are EWKB, so convert back to a regular dataframe to avoid warning
    # the user that the dtypes are unexpected.
    df = pd.DataFrame(gdf, copy=False)
    df[geom_name] = geoms
    return df


def _psql_insert_copy(tbl, conn, keys, data_iter):
    import io
    import csv

    s_buf = io.StringIO()
    writer = csv.writer(s_buf)
    writer.writerows(data_iter)
    s_buf.seek(0)

    columns = ", ".join('"{}"'.format(k) for k in keys)

    dbapi_conn = conn.connection
    with dbapi_conn.cursor() as cur:
        sql = 'COPY "{}"."{}" ({}) FROM STDIN WITH CSV'.format(
            tbl.table.schema, tbl.table.name, columns
        )
        cur.copy_expert(sql=sql, file=s_buf)


def _write_postgis(
    gdf,
    name,
    con,
    schema=None,
    if_exists="fail",
    index=False,
    index_label=None,
    chunksize=None,
    dtype=None,
):
    """
    Upload GeoDataFrame into PostGIS database.

    This method requires SQLAlchemy and GeoAlchemy2, and a PostgreSQL
    Python driver (e.g. psycopg2) to be installed.

    Parameters
    ----------
    name : str
        Name of the target table.
    con : sqlalchemy.engine.Connection or sqlalchemy.engine.Engine
        Active connection to the PostGIS database.
    if_exists : {'fail', 'replace', 'append'}, default 'fail'
        How to behave if the table already exists:

        - fail: Raise a ValueError.
        - replace: Drop the table before inserting new values.
        - append: Insert new values to the existing table.
    schema : string, optional
        Specify the schema. If None, use default schema: 'public'.
    index : bool, default True
        Write DataFrame index as a column.
        Uses *index_label* as the column name in the table.
    index_label : string or sequence, default None
        Column label for index column(s).
        If None is given (default) and index is True,
        then the index names are used.
    chunksize : int, optional
        Rows will be written in batches of this size at a time.
        By default, all rows will be written at once.
    dtype : dict of column name to SQL type, default None
        Specifying the datatype for columns.
        The keys should be the column names and the values
        should be the SQLAlchemy types.

    Examples
    --------

    >>> from sqlalchemy import create_engine  # doctest: +SKIP
    >>> engine = create_engine("postgres://myusername:mypassword@myhost:5432\
/mydatabase";)  # doctest: +SKIP
    >>> gdf.to_postgis("my_table", engine)  # doctest: +SKIP
    """
    try:
        from geoalchemy2 import Geometry
    except ImportError:
        raise ImportError("'to_postgis()' requires geoalchemy2 package. ")

    if not compat.SHAPELY_GE_17:
        raise ImportError(
            "'to_postgis()' requires newer version of Shapely "
            "(>= '1.7.0').\nYou can update the library using "
            "'pip install shapely --upgrade' or using "
            "'conda update shapely' if using conda package manager."
        )

    gdf = gdf.copy()
    geom_name = gdf.geometry.name

    # Get srid
    srid = _get_srid_from_crs(gdf)

    # Get geometry type and info whether data contains LinearRing.
    geometry_type, has_curve = _get_geometry_type(gdf)

    # Build dtype with Geometry
    if dtype is not None:
        dtype[geom_name] = Geometry(geometry_type=geometry_type, srid=srid)
    else:
        dtype = {geom_name: Geometry(geometry_type=geometry_type, srid=srid)}

    # Convert LinearRing geometries to LineString
    if has_curve:
        gdf = _convert_linearring_to_linestring(gdf, geom_name)

    # Convert geometries to EWKB
    gdf = _convert_to_ewkb(gdf, geom_name, srid)

    if schema is not None:
        schema_name = schema
    else:
        schema_name = "public"

    if if_exists == "append":
        # Check that the geometry srid matches with the current GeoDataFrame
        with _get_conn(con) as connection:
            # Only check SRID if table exists
            if connection.dialect.has_table(connection, name, schema):
                target_srid = connection.execute(
                    "SELECT Find_SRID('{schema}', '{table}', '{geom_col}');".format(
                        schema=schema_name, table=name, geom_col=geom_name
                    )
                ).fetchone()[0]

                if target_srid != srid:
                    msg = (
                        "The CRS of the target table (EPSG:{epsg_t}) differs from the "
                        "CRS of current GeoDataFrame (EPSG:{epsg_src}).".format(
                            epsg_t=target_srid, epsg_src=srid
                        )
                    )
                    raise ValueError(msg)

    with _get_conn(con) as connection:

        gdf.to_sql(
            name,
            connection,
            schema=schema_name,
            if_exists=if_exists,
            index=index,
            index_label=index_label,
            chunksize=chunksize,
            dtype=dtype,
            method=_psql_insert_copy,
        )

    return
