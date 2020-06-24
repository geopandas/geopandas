.. _io:

Reading and Writing Files
=========================================



Reading Spatial Data
---------------------

*geopandas* can read almost any vector-based spatial data format including ESRI shapefile, GeoJSON files and more using the command::

    geopandas.read_file()

which returns a GeoDataFrame object. (This is possible because *geopandas* makes use of the great `fiona <http://fiona.readthedocs.io/en/latest/manual.html>`_ library, which in turn makes use of a massive open-source program called `GDAL/OGR <http://www.gdal.org/>`_ designed to facilitate spatial data transformations).

Any arguments passed to :func:`geopandas.read_file` after the file name will be passed directly to ``fiona.open``, which does the actual data importation. In general, :func:`geopandas.read_file` is pretty smart and should do what you want without extra arguments, but for more help, type::

    import fiona; help(fiona.open)

Among other things, one can explicitly set the driver (shapefile, GeoJSON) with the ``driver`` keyword, or pick a single layer from a multi-layered file with the ``layer`` keyword::

    countries_gdf = geopandas.read_file("package.gpkg", layer='countries')

Where supported in ``fiona``, *geopandas* can also load resources directly from
a web URL, for example for GeoJSON files from `geojson.xyz <http://geojson.xyz/>`_::

    url = "http://d2ad6b4ur7yvpq.cloudfront.net/naturalearth-3.3.0/ne_110m_land.geojson"
    df = geopandas.read_file(url)

You can also load ZIP files that contain your data::

    zipfile = "zip:///Users/name/Downloads/cb_2017_us_state_500k.zip"
    states = geopandas.read_file(zipfile)

If the dataset is in a folder in the ZIP file, you have to append its name::

    zipfile = "zip:///Users/name/Downloads/gadm36_AFG_shp.zip!data"

If there are multiple datasets in a folder in the ZIP file, you also have to specify the filename::

    zipfile = "zip:///Users/name/Downloads/gadm36_AFG_shp.zip!data/gadm36_AFG_1.shp"

It is also possible to read any file-like objects with a ``read()`` method, such as a file handler (e.g. via built-in ``open`` function) or ``StringIO``::

    filename = "test.geojson"
    file = open(filename)
    df = geopandas.read_file(file)

You can also read path objects::

    import pathlib
    path_object = pathlib.path(filename)
    df = geopandas.read_file(path_object)

*geopandas* can also get data from a PostGIS database using the :func:`geopandas.read_postgis` command.


Reading subsets of the data
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Since geopandas is powered by Fiona, which is powered by GDAL, you can take advantage of
pre-filtering when loading in larger datasets. This can be done geospatially with a geometry
or bounding box. You can also filter rows loaded with a slice. Read more at :func:`geopandas.read_file`.

Geometry Filter
^^^^^^^^^^^^^^^

.. versionadded:: 0.7.0

The geometry filter only loads data that intersects with the geometry.

.. code-block:: python

    gdf_mask = geopandas.read_file(
        geopandas.datasets.get_path("naturalearth_lowres")
    )
    gdf = geopandas.read_file(
        geopandas.datasets.get_path("naturalearth_cities"),
        mask=gdf_mask[gdf_mask.continent=="Africa"],
    )

Bounding Box Filter
^^^^^^^^^^^^^^^^^^^

.. versionadded:: 0.1.0

The bounding box filter only loads data that intersects with the bounding box.

.. code-block:: python

    bbox = (
        1031051.7879884212, 224272.49231459625, 1047224.3104931959, 244317.30894023244
    )
    gdf = geopandas.read_file(
        geopandas.datasets.get_path("nybb"),
        bbox=bbox,
    )

Row Filter
^^^^^^^^^^

.. versionadded:: 0.7.0

Filter the rows loaded in from the file using an integer (for the first n rows)
or a slice object.

.. code-block:: python

    gdf = geopandas.read_file(
        geopandas.datasets.get_path("naturalearth_lowres"),
        rows=10,
    )
    gdf = geopandas.read_file(
        geopandas.datasets.get_path("naturalearth_lowres"),
        rows=slice(10, 20),
    )

Field/Column Filters
^^^^^^^^^^^^^^^^^^^^

Load in a subset of fields from the file:

.. note:: Requires Fiona 1.8+

.. code-block:: python

    gdf = geopandas.read_file(
        geopandas.datasets.get_path("naturalearth_lowres"),
        ignore_fields=["iso_a3", "gdp_md_est"],
    )

Skip loading geometry from the file:

.. note:: Requires Fiona 1.8+
.. note:: Returns :obj:`pandas.DataFrame`

.. code-block:: python

    pdf = geopandas.read_file(
        geopandas.datasets.get_path("naturalearth_lowres"),
        ignore_geometry=True,
    )


Writing Spatial Data
---------------------

GeoDataFrames can be exported to many different standard formats using the
:meth:`geopandas.GeoDataFrame.to_file` method.
For a full list of supported formats, type ``import fiona; fiona.supported_drivers``.

In addition, GeoDataFrames can be uploaded to `PostGIS <https://postgis.net/>`__ database (starting with GeoPandas 0.8)
by using the :meth:`geopandas.GeoDataFrame.to_postgis` method.

.. note::

    GeoDataFrame can contain more field types than supported by most of the file formats. For example tuples or lists
    can be easily stored in the GeoDataFrame, but saving them to e.g. GeoPackage or Shapefile will raise a ValueError.
    Before saving to a file, they need to be converted to a format supported by a selected driver.

**Writing to Shapefile**::

    countries_gdf.to_file("countries.shp")

**Writing to GeoJSON**::

    countries_gdf.to_file("countries.geojson", driver='GeoJSON')

**Writing to GeoPackage**::

    countries_gdf.to_file("package.gpkg", layer='countries', driver="GPKG")
    cities_gdf.to_file("package.gpkg", layer='cities', driver="GPKG")

**Writing to PostGIS**::

    from sqlalchemy import create_engine
    db_connection_url = "postgres://myusername:mypassword@myhost:5432/mydatabase";
    engine = create_engine(db_connection_url)
    countries_gdf.to_postgis(name="countries_table", con=engine)
