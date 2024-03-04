.. currentmodule:: geopandas

.. ipython:: python
   :suppress:

   import geopandas

Migration from the Fiona to the Pyogrio IO engine
=================================================

Since version 0.11, GeoPandas supports two IO engines to read and write files:
`Fiona <https://fiona.readthedocs.io>`__ and
`Pyogrio <https://pyogrio.readthedocs.io>`__.

For Geopandas versions <1.0, GeoPandas defaults to use Fiona. It is possible to use
Pyogrio using the ``engine="pyogrio"`` parameter in :func:`geopandas.read_file` and
:func:`geopandas.GeoDataFrame.to_file`. You can also change the default engine globally
with:

.. code-block:: python

    geopandas.options.io_engine = "pyogrio"

Starting from GeoPandas 1.0, the global default will change from Fiona to Pyogrio. The
main reason for this change is performance. Pyogrio is optimized for the use case
relevant for GeoPandas: reading and writing in bulk. Because of this, in many cases
speedups >5-20x can be observed.

This guide outlines the (known) functional differences between both, so you can account
for them when switching to Pyogrio.


Write an attribute table to a file
----------------------------------

Using the Fiona IO engine, it was possible to write an attribute table (a table without
geometry column) to a file using the ``schema`` parameter to specify that the "geometry"
column should be ignored.

With Pyogrio you can write an attribute table by calling
:func:`geopandas.GeoDataFrame.to_file` on a GeoDataFrame without geometry column:

.. code-block:: python

    >>> gdf = gpd.GeoDataFrame({"data_column": [1, 2, 3]})
    >>> gdf.to_file("test_attribute_table.gpkg", engine="pyogrio")


No support for ``schema`` parameter
-----------------------------------

Pyogrio does not support overriding the schema/types of the data being read/written.


Writing EMPTY geometries
------------------------

Pyogrio writes EMPTY geometries to e.g. GPKG files, Fiona writes None.

.. ipython:: python

    import shapely
    
    gdf = geopandas.GeoDataFrame(geometry=[shapely.Polygon()], crs=31370)
    gdf.to_file("test_fiona.gpkg", engine="fiona")
    gdf.to_file("test_pyogrio.gpkg", engine="pyogrio")
    w_fiona_r_pyogrio = geopandas.read_file("test_fiona.gpkg", engine="pyogrio")
    w_fiona_r_pyogrio.head()
    w_pyogrio_r_pyogrio = geopandas.read_file("test_pyogrio.gpkg", engine="pyogrio")
    w_pyogrio_r_pyogrio.head()
    
    w_fiona_r_fiona = geopandas.read_file("test_fiona.gpkg", engine="fiona")
    w_fiona_r_fiona.head()
    w_pyogrio_r_fiona = geopandas.read_file("test_pyogrio.gpkg", engine="fiona")
    w_pyogrio_r_fiona.head()
