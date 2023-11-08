.. _io:

Reading and writing files
=========================

Reading spatial data
---------------------

GeoPandas can read almost any vector-based spatial data format including ESRI
shapefile, GeoJSON files and more using the :func:`geopandas.read_file` command::

    geopandas.read_file(...)

which returns a GeoDataFrame object. This is possible because GeoPandas makes
use of the massive open-source program called
`GDAL/OGR <http://www.gdal.org/>`_ designed to facilitate spatial data
transformations, through the Python packages `Fiona <http://fiona.readthedocs.io/en/latest/manual.html>`_
or `pyogrio <https://pyogrio.readthedocs.io/en/stable/>`_, which both provide bindings to GDAL.

.. note::

    GeoPandas currently defaults to use Fiona as the engine in ``read_file``. However,
    GeoPandas 1.0 will switch to use pyogrio as the default engine, since pyogrio can
    provide a significant speedup compared to Fiona. We recommend to already install
    pyogrio and specify the engine by using the ``engine`` keyword
    (``geopandas.read_file(..., engine="pyogrio")``), or by setting the default for the
    ``engine`` keyword globally with::

        geopandas.options.io_engine = "pyogrio"

Any arguments passed to :func:`geopandas.read_file` after the file name will be
passed directly to :func:`fiona.open` or :func:`pyogrio.read_dataframe`, which
does the actual data importation.
In general, :func:`geopandas.read_file` is pretty smart and should do what you want
without extra arguments, but for more help, type::

    import fiona; help(fiona.open)
    import pyogrio; help(pyogrio.read_dataframe)

Among other things, one can explicitly set the driver (shapefile, GeoJSON) with
the ``driver`` keyword, or pick a single layer from a multi-layered file with
the ``layer`` keyword::

    countries_gdf = geopandas.read_file("package.gpkg", layer='countries')

GeoPandas can also load resources directly from
a web URL, for example for GeoJSON files from `geojson.xyz <http://geojson.xyz/>`_::

    url = "http://d2ad6b4ur7yvpq.cloudfront.net/naturalearth-3.3.0/ne_110m_land.geojson"
    df = geopandas.read_file(url)

You can also load ZIP files that contain your data::

    zipfile = "zip:///Users/name/Downloads/cb_2017_us_state_500k.zip"
    states = geopandas.read_file(zipfile)

If the dataset is in a folder in the ZIP file, you have to append its name::

    zipfile = "zip:///Users/name/Downloads/gadm36_AFG_shp.zip!data"

If there are multiple datasets in a folder in the ZIP file, you also have to
specify the filename::

    zipfile = "zip:///Users/name/Downloads/gadm36_AFG_shp.zip!data/gadm36_AFG_1.shp"

It is also possible to read any file-like objects with a :func:`~os.read` method, such
as a file handler (e.g. via built-in :func:`open` function) or :class:`~io.StringIO`::

    filename = "test.geojson"
    file = open(filename)
    df = geopandas.read_file(file)

File-like objects from `fsspec <https://filesystem-spec.readthedocs.io/en/latest>`_
can also be used to read data, allowing for any combination of storage backends and caching
supported by that project::

    path = "simplecache::http://download.geofabrik.de/antarctica-latest-free.shp.zip"
    with fsspec.open(path) as file:
        df = geopandas.read_file(file)

You can also read path objects::

    import pathlib
    path_object = pathlib.path(filename)
    df = geopandas.read_file(path_object)

Reading subsets of the data
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Since geopandas is powered by GDAL, you can take advantage of pre-filtering when loading
in larger datasets. This can be done geospatially with a geometry or bounding box. You
can also filter rows loaded with a slice. Read more at :func:`geopandas.read_file`.

Geometry filter
^^^^^^^^^^^^^^^

The geometry filter only loads data that intersects with the geometry.

.. code-block:: python

    import geodatasets

    gdf_mask = geopandas.read_file(
        geodatasets.get_path("geoda.nyc")
    )
    gdf = geopandas.read_file(
        geodatasets.get_path("geoda.nyc education"),
        mask=gdf_mask[gdf_mask.name=="Coney Island"],
    )

Bounding box filter
^^^^^^^^^^^^^^^^^^^

The bounding box filter only loads data that intersects with the bounding box.

.. code-block:: python

    bbox = (
        1031051.7879884212, 224272.49231459625, 1047224.3104931959, 244317.30894023244
    )
    gdf = geopandas.read_file(
        geodatasets.get_path("nybb"),
        bbox=bbox,
    )

Row filter
^^^^^^^^^^

Filter the rows loaded in from the file using an integer (for the first n rows)
or a slice object.

.. code-block:: python

    gdf = geopandas.read_file(
        geodatasets.get_path("geoda.nyc"),
        rows=10,
    )
    gdf = geopandas.read_file(
        geodatasets.get_path("geoda.nyc"),
        rows=slice(10, 20),
    )

Field/column filters
^^^^^^^^^^^^^^^^^^^^

Load in a subset of fields from the file:

.. note:: Requires Fiona 1.9+

.. code-block:: python

    gdf = geopandas.read_file(
        geodatasets.get_path("geoda.nyc"),
        include_fields=["name", "rent2008", "kids2000"],
    )

.. note:: Requires Fiona 1.8+

.. code-block:: python

    gdf = geopandas.read_file(
        geodatasets.get_path("geoda.nyc"),
        ignore_fields=["rent2008", "kids2000"],
    )

Skip loading geometry from the file:

.. note:: Requires Fiona 1.8+
.. note:: Returns :obj:`pandas.DataFrame`

.. code-block:: python

    pdf = geopandas.read_file(
        geodatasets.get_path("geoda.nyc"),
        ignore_geometry=True,
    )


SQL WHERE filter
^^^^^^^^^^^^^^^^

.. versionadded:: 0.12

Load in a subset of data with a `SQL WHERE clause <https://gdal.org/user/ogr_sql_dialect.html#where>`__.

.. note:: Requires Fiona 1.9+ or the pyogrio engine.

.. code-block:: python

    gdf = geopandas.read_file(
        geodatasets.get_path("geoda.nyc"),
        where="subborough='Coney Island'",
    )

Supported drivers
~~~~~~~~~~~~~~~~~

Currently fiona only exposes the default drivers. To display those, type::

    import fiona; fiona.supported_drivers

There is a `list of available drivers <https://github.com/Toblerity/Fiona/blob/master/fiona/drvsupport.py>`_
which are unexposed but supported (depending on the GDAL-build). You can activate
these on runtime by updating the `supported_drivers` dictionary like::

    fiona.supported_drivers["NAS"] = "raw"

When using pyogrio, all drivers supported by the GDAL installation are enabled,
and you can check those with::

    import pyogrio; pyogrio.list_drivers()


Writing spatial data
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


Spatial databases
-----------------

GeoPandas can also get data from a PostGIS database using the
:func:`geopandas.read_postgis` command.

Writing to PostGIS::

    from sqlalchemy import create_engine
    db_connection_url = "postgresql://myusername:mypassword@myhost:5432/mydatabase";
    engine = create_engine(db_connection_url)
    countries_gdf.to_postgis("countries_table", con=engine)


Apache Parquet and Feather file formats
---------------------------------------

.. versionadded:: 0.8.0

GeoPandas supports writing and reading the Apache Parquet and Feather file
formats.

`Apache Parquet <https://parquet.apache.org/>`__ is an efficient, columnar
storage format (originating from the Hadoop ecosystem). It is a widely used
binary file format for tabular data. The Feather file format is the on-disk
representation of the `Apache Arrow <https://arrow.apache.org/>`__ memory
format, an open standard for in-memory columnar data.

The :func:`geopandas.read_parquet`, :func:`geopandas.read_feather`,
:meth:`GeoDataFrame.to_parquet` and :meth:`GeoDataFrame.to_feather` methods
enable fast roundtrip from GeoPandas to those binary file formats, preserving
the spatial information.

.. note::

    This is tracking version 1.0.0 of the GeoParquet specification at:
    https://github.com/opengeospatial/geoparquet.

    Previous versions are still supported as well. By default, the latest
    version is used when writing files (older versions can be specified using
    the ``schema_version`` keyword), and GeoPandas supports reading files
    of any version.
