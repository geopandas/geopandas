import sqlalchemy as sa
import geopandas as gpd
import pandas as pd

def test_postgis_search_path():

    """
    Not a real test - feature WIP!
    """

    engine = sa.create_engine("postgresql://myuser:myuser@localhost:5432/postgres")
    engine.execute("DROP TABLE IF EXISTS public.pd_myschema_public;")
    engine.execute("DROP TABLE IF EXISTS myschema.pd_myschema_public;")
    engine.execute("DROP TABLE IF EXISTS myschema.pd_nonexistentschema_myschema_public;")
    engine.execute("DROP TABLE IF EXISTS myschema.pd_restrictedschema_myschema_public;")
    engine.execute("DROP TABLE IF EXISTS public.gpd_myschema_public;")
    engine.execute("DROP TABLE IF EXISTS myschema.gpd_myschema_public;")
    engine.execute("DROP TABLE IF EXISTS myschema.gpd_nonexistentschema_myschema_public;")
    engine.execute("DROP TABLE IF EXISTS myschema.gpd_restrictedschema_myschema_public;")

    engine_myschema_public = sa.create_engine("postgresql://myuser:myuser@localhost:5432/postgres",
                                              connect_args={'options': f'-csearch_path=myschema,public'},)
    engine_nonexistentschema_myschema_public = sa.create_engine("postgresql://myuser:myuser@localhost:5432/postgres",
                                                                connect_args={'options': f'-csearch_path=nonexistentschema,myschema,public'},)
    engine_restrictedschema_myschema_public = sa.create_engine("postgresql://myuser:myuser@localhost:5432/postgres",
                                                                connect_args={'options': f'-csearch_path=restrictedschema,myschema,public'},)
    engine_nonexistentschema = sa.create_engine("postgresql://myuser:myuser@localhost:5432/postgres",
                                                connect_args={'options': f'-csearch_path=nonexistentschema'},)
    engine_restrictedschema = sa.create_engine("postgresql://myuser:myuser@localhost:5432/postgres",
                                               connect_args={'options': f'-csearch_path=restrictedschema'},)

    # Pandas
    print("Testing pandas...")
    df = pd.DataFrame(data={'col1': [1, 2], 'col2': [3, 4]})

    # Create new table with custom-configured search_path and no explicit schema passed to .to_sql()
    #   -> Should create table in myschema
    print("Writing to myschema directly...")
    df.to_sql("pd_myschema_public", engine_myschema_public, schema=None)
    print("...Done")

    # Create new table with custom-configured search_path, but the first schema does not exist
    #   -> Should bypass nonexistentschema (because it doesn't exist) and create table in myschema
    print("Writing to myschema indirectly, bypassing nonexistent schema...")
    df.to_sql("pd_nonexistentschema_myschema_public", engine_nonexistentschema_myschema_public, schema=None)
    print("...Done")

    # Create new table with custom-configured search_path, but the first schema is restricted
    #   -> Should bypass restrictedschema (due to insufficient priveleges) and create table in myschema
    print("Writing to myschema indirectly, bypassing restricted schema...")
    df.to_sql("pd_restrictedschema_myschema_public", engine_restrictedschema_myschema_public, schema=None)
    print("...Done")

    # Write to existing table, where the same table exists in two schemas already
    df.to_sql("pd_myschema_public", engine_myschema_public, schema="public")
    # Try to append without specifying schema
    #   -> Should append to the one in myschema
    df.to_sql("pd_myschema_public", engine_myschema_public, schema=None, if_exists="append")


    # Geopandas
    print("Testing geopandas...")
    gdf = gpd.GeoDataFrame.from_file(gpd.datasets.get_path("naturalearth_lowres"))

    # Create new table with custom-configured search_path and no explicit schema passed to .to_postgis()
    #   -> Should create table in myschema
    print("Writing to myschema directly...")
    gdf.to_postgis("gpd_myschema_public", engine_myschema_public, schema=None)
    print("...Done")

    # Create new table with custom-configured search_path, but the first schema does not exist
    #   -> Should bypass nonexistentschema (because it doesn't exist) and create table in myschema
    print("Writing to myschema indirectly, bypassing nonexistent schema...")
    gdf.to_postgis("gpd_nonexistentschema_myschema_public", engine_nonexistentschema_myschema_public, schema=None)
    print("...Done")

    # Create new table with custom-configured search_path, but the first schema is restricted
    #   -> Should bypass restrictedschema (due to insufficient priveleges) and create table in myschema
    print("Writing to myschema indirectly, bypassing restricted schema...")
    gdf.to_postgis("gpd_restrictedschema_myschema_public", engine_restrictedschema_myschema_public, schema=None)
    print("...Done")

    # Write to existing table, where the same table exists in two schemas already
    print("Writing to table which exists in two schemas...")
    gdf.to_postgis("gpd_myschema_public", engine_myschema_public, schema="public")
    # Try to append without specifying schema
    #   -> Should append to the one in myschema
    gdf.to_postgis("gpd_myschema_public", engine_myschema_public, schema=None, if_exists="append")
    print("...Done")


    # Impossible situations

    # Try to create a table with a schema containing only an non-existent schema
    try:
        gdf.to_postgis("gpd_nonexistentschema", engine_nonexistentschema, schema=None)
    except RuntimeError as e:
        print("Failed with exception:")
        print(e.args[0])

    # Try to create a table with a schema containing only an non-existent schema
    try:
        gdf.to_postgis("gpd_restrictedschema", engine_restrictedschema, schema=None)
    except RuntimeError as e:
        print("Failed with exception:")
        print(e.args[0])

    assert True
