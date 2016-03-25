
Reading and Writing Files
=========================================



Reading Spatial Data
---------------------

.. classmethod:: GeoDataFrame.from_file(filename, **kwargs)

  Load a ``GeoDataFrame`` from a file from any format recognized by
  `fiona`_.  See ``read_file()``.

.. classmethod:: GeoDataFrame.from_postgis(sql, con, geom_col='geom', crs=None, index_col=None, coerce_float=True, params=None)

  Load a ``GeoDataFrame`` from a file from a PostGIS database.
  See ``read_postgis()``.



Writing Spatial Data
---------------------

