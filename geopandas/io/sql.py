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
     """
    geom_types = list(gdf.geometry.geom_type.unique())
    has_single = False
    has_multi = False
    has_curve = False

    # Get the basic geometry type
    basic_types = []

    for gt in geom_types:
        if "Multi" in gt:
            has_multi = True
            # Keep track of the "root" geometry types
            basic_types.append(gt.replace("Multi", ""))
        elif "LinearRing" in gt:
            if "LineString" not in basic_types:
                basic_types.append("LineString")
                has_curve = True
        else:
            has_single = True
            basic_types.append(gt)

    geom_types = list(set(basic_types))

    # If there are mixed geometry types, use GEOMETRY.
    if len(geom_types) > 1:
        # TODO: Should we warn about mixed geometry types?
        # print("UserWarning: GeoDataFrame contains mixed geometry types.")
        target_geom_type = "GEOMETRY"
    # If there is a mix of single and multi-geometries, use GEOMETRY.
    elif has_single and has_multi:
        target_geom_type = "GEOMETRY"
    elif has_multi:
        target_geom_type = "MULTI{geom_type}".format(geom_type=geom_types[0].upper())
    else:
        target_geom_type = geom_types[0].upper()
    return target_geom_type, has_curve


def _get_srid_from_crs(gdf):
    """
    Get EPSG code from CRS if available. If not, return -1.
    """
    # TODO: Simplify this once new pyproj.CRS class has been
    #  integrated (#1101) -> https://github.com/geopandas/geopandas/pull/1101
    from pyproj import CRS

    # Use geoalchemy2 default for srid
    # Note: undefined srid in PostGIS is 0
    srid = -1
    if gdf.crs is not None:
        try:
            if isinstance(gdf.crs, dict):
                # If CRS is in dictionary format use only the value
                # to avoid pyproj Future warning
                if "init" in gdf.crs.keys():
                    srid = CRS(gdf.crs["init"]).to_epsg(min_confidence=25)
                else:
                    srid = CRS(gdf.crs).to_epsg(min_confidence=25)
            else:
                srid = CRS(gdf).to_epsg(min_confidence=25)
            if srid is None:
                srid = -1
        except Exception:
            print(
                "Warning: Could not parse CRS from the GeoDataFrame.",
                "Inserting data without defined CRS.",
            )
    return srid


def _convert_linearring_to_linestring(gdf, geom_name):
    from shapely.geometry import LineString

    mask = gdf.geom_type == "LinearRing"
    gdf.loc[mask, geom_name] = gdf.loc[mask, geom_name].apply(
        lambda geom: LineString(geom)
    )
    return gdf


def _convert_to_wkb(gdf, geom_name):
    """Convert geometries to wkb. """
    from shapely.wkb import dumps

    gdf[geom_name] = gdf[geom_name].apply(lambda geom: dumps(geom, hex=True))
    return gdf


def _populate_db(gdf, conn, cur, index, tbl):
    import io
    import csv

    # Convert columns to lists and make a generator
    args = [list(gdf[i]) for i in gdf.columns]
    if index:
        args.insert(0, list(gdf.index))

    data_iter = zip(*args)

    # get list of columns using pandas
    keys = tbl.insert_data()[0]
    columns = ", ".join('"{}"'.format(k) for k in list(keys))

    s_buf = io.StringIO()
    writer = csv.writer(s_buf)
    writer.writerows(data_iter)
    s_buf.seek(0)

    try:

        sql = "COPY {} ({}) FROM STDIN WITH CSV".format(tbl.table.fullname, columns)
        cur.copy_expert(sql=sql, file=s_buf)
        conn.commit()

    except Exception as e:
        raise e


def _get_chunks(gdf, chunksize):
    assert isinstance(
        chunksize, int
    ), "'chunksize' should be passed as an integer number."
    import numpy as np

    chunk_cnt = np.ceil(len(gdf) / chunksize)
    chunks = np.array_split(gdf, chunk_cnt)
    return chunks


def _write_to_db(gdf, engine, index, tbl, srid, geom_name, if_exists, chunksize):

    conn = engine.raw_connection()
    cur = conn.cursor()

    try:
        # If appending to an existing table, temporarily change
        # the srid to 0, and update the SRID afterwards
        if if_exists == "append":
            sql = "SELECT UpdateGeometrySRID('{schema}','{tbl}','{geom}',{crs})".format(
                schema=tbl.table.schema, tbl=tbl.table.name, geom=geom_name, crs=0
            )
            cur.execute(sql)

        if chunksize is None:
            _populate_db(gdf, conn, cur, index, tbl)
        else:
            # Insert in chunks
            chunks = _get_chunks(gdf, chunksize)

            for chunk in chunks:
                _populate_db(chunk, conn, cur, index, tbl)

        # SRID needs to be updated afterwards as Shapely does not support
        # EWKT/EWKB geometries, see:
        # https://community.gispython.narkive.com/qTVQCl3f/ewkt-ewkb-support-in-shapely
        sql = "SELECT UpdateGeometrySRID('{schema}','{tbl}','{geom}',{srid})".format(
            schema=tbl.table.schema, tbl=tbl.table.name, geom=geom_name, srid=srid
        )
        cur.execute(sql)
        conn.commit()

    except Exception as e:
        conn.connection.rollback()
        raise e
    finally:
        conn.close()


def write_postgis(
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

    Parameters
    ----------
    name : str
        Name of the target table.
    con : sqlalchemy.engine.Engine
        Active connection to the PostGIS database.
    if_exists : {‘fail’, ‘replace’, ‘append’}, default ‘fail’
        How to behave if the table already exists.
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
    """
    from geoalchemy2 import Geometry

    gdf = gdf.copy()
    geom_name = gdf.geometry.name
    if schema is not None:
        schema_name = schema
    else:
        schema_name = "public"

    # Get srid
    srid = _get_srid_from_crs(gdf)

    # Get geometry type and info whether data contains LinearRing
    geometry_type, has_curve = _get_geometry_type(gdf)

    # Build dtype with Geometry (srid is updated afterwards)
    if dtype is not None:
        dtype[geom_name] = Geometry(geometry_type=geometry_type)
    else:
        dtype = {geom_name: Geometry(geometry_type=geometry_type)}

    # Get Pandas SQLTable object (ignore 'geometry')
    # If dtypes is used, update table schema accordingly.
    pandas_sql = pd.io.sql.SQLDatabase(con)
    tbl = pd.io.sql.SQLTable(
        name=name,
        pandas_sql_engine=pandas_sql,
        frame=gdf,
        dtype=dtype,
        index=index,
        index_label=index_label,
        schema=schema_name,
    )

    # Check if table exists
    if tbl.exists():
        # If it exists, check if should overwrite
        if if_exists == "replace":
            pandas_sql.drop_table(name)
            tbl.create()
        elif if_exists == "fail":
            raise ValueError("Table '{table}' already exists.".format(table=name))
        elif if_exists == "append":
            # Check that the geometry srid matches with the current GeoDataFrame
            target_srid = con.execute(
                "SELECT Find_SRID('{schema}', '{table}', '{geom_col}');".format(
                    schema=schema_name, table=name, geom_col=geom_name
                )
            ).fetchone()[0]

            assert target_srid == srid, (
                "The CRS of the target table differs",
                "from the CRS of current GeoDataFrame.",
            )
    else:
        tbl.create()

    # Convert LinearRing geometries to LineString
    if has_curve:
        gdf = _convert_linearring_to_linestring(gdf, geom_name)

    # Convert geometries to WKB
    gdf = _convert_to_wkb(gdf, geom_name)

    # Write to database
    _write_to_db(gdf, con, index, tbl, srid, geom_name, if_exists, chunksize)

    return
