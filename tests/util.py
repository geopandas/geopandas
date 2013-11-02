import os.path
import urllib2

try:
    import psycopg2
    from psycopg2 import OperationalError
except ImportError:
    class OperationalError(Exception):
        pass

from geopandas import GeoDataFrame

def download_nybb():
    """ Returns the path to the NYC boroughs file. Downloads if necessary. """
    # Data from http://www.nyc.gov/html/dcp/download/bytes/nybb_13a.zip
    # saved as geopandas/examples/nybb_13a.zip.
    filename = os.path.join('examples', 'nybb_13a.zip')
    if not os.path.exists(filename):
        with open(filename, 'w') as f:
            response = urllib2.urlopen('http://www.nyc.gov/html/dcp/download/bytes/nybb_13a.zip')
            f.write(response.read())
    return filename
    
def validate_boro_df(test, df):
    """ Tests a GeoDataFrame that has been read in from the nybb dataset."""
    test.assertTrue(isinstance(df, GeoDataFrame))
    # Make sure all the columns are there and the geometries
    # were properly loaded as MultiPolygons
    test.assertEqual(len(df), 5)
    columns = ('borocode', 'boroname', 'shape_leng', 'shape_area')
    for col in columns:
        test.assertTrue(col in df.columns, 'Column {} missing'.format(col))
    test.assertTrue(all(df['geometry'].type == 'MultiPolygon'))

def connect(dbname):
    try:
        con = psycopg2.connect(dbname=dbname)
    except (NameError, OperationalError):
        return None

    return con

def create_db(df):
    """
    Create a nybb table in the test_geopandas PostGIS database.
    Returns a boolean indicating whether the database table was successfully 
    created
    
    """
    # Try to create the database, skip the db tests if something goes
    # wrong
    # If you'd like these tests to run, create a database called
    # 'test_geopandas' and enable postgis in it:
    # > createdb test_geopandas
    # > psql -c "CREATE EXTENSION postgis" -d test_geopandas
    con = connect('test_geopandas')
    if con is None:
        return False

    try:
        cursor = con.cursor()
        cursor.execute("DROP TABLE IF EXISTS nybb;")

        sql = """CREATE TABLE nybb (
            geom        geometry,
            borocode    integer,
            boroname    varchar(40),
            shape_leng  float,
            shape_area  float
        );"""
        cursor.execute(sql)

        for i, row in df.iterrows():
            sql = """INSERT INTO nybb VALUES (
                ST_GeometryFromText(%s), %s, %s, %s, %s 
            );"""
            cursor.execute(sql, (row['geometry'].wkt, 
                                 row['BoroCode'],
                                 row['BoroName'],
                                 row['Shape_Leng'],
                                 row['Shape_Area']))
    finally:
        cursor.close()
        con.commit()
        con.close()

    return True
