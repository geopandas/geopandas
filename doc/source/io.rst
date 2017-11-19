.. _io:

Reading and Writing Files
=========================================



Reading Spatial Data
---------------------

*geopandas* can read almost any vector-based spatial data format including ESRI shapefile, GeoJSON files and more using the command::

    gpd.read_file()

which returns a GeoDataFrame object. (This is possible because *geopandas* makes use of the great `fiona <http://toblerity.org/fiona/manual.html>`_ library, which in turn makes use of a massive open-source program called `GDAL/OGR <http://www.gdal.org/>`_ designed to facilitate spatial data transformations).

Any arguments passed to ``read_file()`` after the file name will be passed directly to ``fiona.open``, which does the actual data importation. In general, ``read_file`` is pretty smart and should do what you want without extra arguments, but for more help, type::

    import fiona; help(fiona.open)

Among other things, one can explicitly set the driver (shapefile, GeoJSON) with the ``driver`` keyword, or pick a single layer from a multi-layered file with the ``layer`` keyword.

Where supported in ``fiona``, *geopandas* can also load resources directly from
a web URL, for example for GeoJSON files from `geojson.xyz <http://geojson.xyz/>`_::

    url = "http://d2ad6b4ur7yvpq.cloudfront.net/naturalearth-3.3.0/ne_110m_land.geojson"
    df = gpd.read_file(url)

*geopandas* can also get data from a PostGIS database using the ``read_postgis()`` command.


Writing Spatial Data
---------------------

GeoDataFrames can be exported to many different standard formats using the ``GeoDataFrame.to_file()`` method. For a full list of supported formats, type ``import fiona; fiona.supported_drivers``.
