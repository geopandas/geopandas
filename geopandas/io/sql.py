import warnings
from contextlib import contextmanager
from functools import lru_cache

import pandas as pd

import shapely
import shapely.wkb

from geopandas import GeoDataFrame


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
    from sqlalchemy.engine.base import Connection, Engine

    if isinstance(conn_or_engine, Connection):
        if not conn_or_engine.in_transaction():
            with conn_or_engine.begin():
                yield conn_or_engine
        else:
            yield conn_or_engine
    elif isinstance(conn_or_engine, Engine):
        with conn_or_engine.begin() as conn:
            yield conn
    else:
        raise ValueError(f"Unknown Connectable: {conn_or_engine}")


def _df_to_geodf(df, geom_col="geom", crs=None, con=None):
    """Transform a pandas DataFrame into a GeoDataFrame.

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
    con : sqlalchemy.engine.Connection or sqlalchemy.engine.Engine
        Active connection to the database to query.

    Returns
    -------
    GeoDataFrame
    """
    if geom_col not in df:
        raise ValueError(f"Query missing geometry column '{geom_col}'")

    if df.columns.to_list().count(geom_col) > 1:
        raise ValueError(
            f"Duplicate geometry column '{geom_col}' detected in SQL query output. Only"
            "one geometry column is allowed."
        )

    geoms = df[geom_col].dropna()

    if not geoms.empty:
        load_geom_bytes = shapely.wkb.loads
        """Load from Python 3 binary."""

        def load_geom_text(x):
            """Load from binary encoded as text."""
            return shapely.wkb.loads(str(x), hex=True)

        if isinstance(geoms.iat[0], bytes):
            load_geom = load_geom_bytes
        else:
            load_geom = load_geom_text

        df[geom_col] = geoms = geoms.apply(load_geom)
        if crs is None:
            srid = shapely.get_srid(geoms.iat[0])
            # if no defined SRID in geodatabase, returns SRID of 0
            if srid != 0:
                try:
                    spatial_ref_sys_df = _get_spatial_ref_sys_df(con, srid)
                except pd.errors.DatabaseError:
                    warning_msg = (
                        f"Could not find the spatial reference system table "
                        f"(spatial_ref_sys) in PostGIS."
                        f"Trying epsg:{srid} as a fallback."
                    )
                    warnings.warn(warning_msg, UserWarning, stacklevel=3)
                    crs = f"epsg:{srid}"
                else:
                    if not spatial_ref_sys_df.empty:
                        auth_name = spatial_ref_sys_df["auth_name"].item()
                        crs = f"{auth_name}:{srid}"
                    else:
                        warning_msg = (
                            f"Could not find srid {srid} in the "
                            f"spatial_ref_sys table. "
                            f"Trying epsg:{srid} as a fallback."
                        )
                        warnings.warn(warning_msg, UserWarning, stacklevel=3)
                        crs = f"epsg:{srid}"

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
    """Return a GeoDataFrame corresponding to the result of the query
    string, which must contain a geometry column in WKB representation.

    It is also possible to use :meth:`~GeoDataFrame.read_file` to read from a database.
    Especially for file geodatabases like GeoPackage or SpatiaLite this can be easier.

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
    >>> db_connection_url = "postgresql://myusername:mypassword@myhost:5432/mydatabase"
    >>> con = create_engine(db_connection_url)  # doctest: +SKIP
    >>> sql = "SELECT geom, highway FROM roads"
    >>> df = geopandas.read_postgis(sql, con)  # doctest: +SKIP

    SpatiaLite

    >>> sql = "SELECT ST_AsBinary(geom) AS geom, highway FROM roads"
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
        return _df_to_geodf(df, geom_col=geom_col, crs=crs, con=con)

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
        return (
            _df_to_geodf(df, geom_col=geom_col, crs=crs, con=con) for df in df_generator
        )


def _get_geometry_type(gdf):
    """Get basic geometry type of a GeoDataFrame.

    See more info from:
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
        target_geom_type += "Z"

    return target_geom_type, has_curve


def _get_srid_from_crs(gdf):
    """Get EPSG code from CRS if available. If not, return 0."""
    # Use geoalchemy2 default for srid
    # Note: undefined srid in PostGIS is 0
    srid = None
    warning_msg = (
        "Could not parse CRS from the GeoDataFrame. Inserting data without defined CRS."
    )
    if gdf.crs is not None:
        try:
            for confidence in (100, 70, 25):
                srid = gdf.crs.to_epsg(min_confidence=confidence)
                if srid is not None:
                    break
                auth_srid = gdf.crs.to_authority(
                    auth_name="ESRI", min_confidence=confidence
                )
                if auth_srid is not None:
                    srid = int(auth_srid[1])
                    break
        except Exception:
            warnings.warn(warning_msg, UserWarning, stacklevel=2)

    if srid is None:
        srid = 0
        warnings.warn(warning_msg, UserWarning, stacklevel=2)

    return srid


def _convert_linearring_to_linestring(gdf, geom_name):
    from shapely.geometry import LineString

    # Todo: Use shapely function once it's implemented:
    # https://github.com/shapely/shapely/issues/1617

    mask = gdf.geom_type == "LinearRing"
    gdf.loc[mask, geom_name] = gdf.loc[mask, geom_name].apply(
        lambda geom: LineString(geom)
    )
    return gdf


def _convert_to_ewkb(gdf, geom_name, srid):
    """Convert geometries to ewkb."""
    geoms = shapely.to_wkb(
        shapely.set_srid(gdf[geom_name].values._data, srid=srid),
        hex=True,
        include_srid=True,
    )

    # The gdf will warn that the geometry column doesn't hold in-memory geometries
    # now that they are EWKB, so convert back to a regular dataframe to avoid warning
    # the user that the dtypes are unexpected.
    df = pd.DataFrame(gdf, copy=False)
    df[geom_name] = geoms
    return df


def _psql_insert_copy(tbl, conn, keys, data_iter):
    import csv
    import io

    s_buf = io.StringIO()
    writer = csv.writer(s_buf)
    writer.writerows(data_iter)
    s_buf.seek(0)

    columns = ", ".join(f'"{k}"' for k in keys)

    dbapi_conn = conn.connection
    sql = (
        f'COPY "{tbl.table.schema}"."{tbl.table.name}" ({columns}) FROM STDIN WITH CSV'
    )
    with dbapi_conn.cursor() as cur:
        # Use psycopg method if it's available
        if hasattr(cur, "copy") and callable(cur.copy):
            with cur.copy(sql) as copy:
                copy.write(s_buf.read())
        else:  # otherwise use psycopg2 method
            cur.copy_expert(sql, s_buf)


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
    >>> engine = create_engine("postgresql://myusername:mypassword@myhost:5432\
/mydatabase";)  # doctest: +SKIP
    >>> gdf.to_postgis("my_table", engine)  # doctest: +SKIP
    """
    try:
        from geoalchemy2 import Geometry
        from sqlalchemy import text
    except ImportError:
        raise ImportError("'to_postgis()' requires geoalchemy2 package.")

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
                    text(
                        "SELECT Find_SRID(:schema_name, :name, :geom_name);"
                    ).bindparams(
                        schema_name=schema_name, name=name, geom_name=geom_name
                    )
                ).fetchone()[0]

                if target_srid != srid:
                    msg = (
                        f"The CRS of the target table (EPSG:{target_srid}) differs "
                        f"from the CRS of current GeoDataFrame (EPSG:{srid})."
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


@lru_cache
def _get_spatial_ref_sys_df(con, srid):
    spatial_ref_sys_sql = (
        f"SELECT srid, auth_name FROM spatial_ref_sys WHERE srid = {srid}"
    )
    return pd.read_sql(spatial_ref_sys_sql, con)
