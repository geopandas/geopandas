.. currentmodule:: geopandas

.. ipython:: python
   :suppress:

   import geopandas

Migration from the Fiona to the Pyogrio read/write engine
=========================================================

Since version 0.11, GeoPandas started supporting two engines to read and write files:
`Fiona <https://fiona.readthedocs.io>`__ and
`Pyogrio <https://pyogrio.readthedocs.io>`__.

It became possible to choose the engine using the ``engine=`` parameter in
:func:`geopandas.read_file` and :func:`geopandas.GeoDataFrame.to_file`. It became also
possible to change the default engine globally with:

.. code-block:: python

    geopandas.options.io_engine = "pyogrio"

For Geopandas versions <1.0, GeoPandas defaulted to use Fiona. Starting from GeoPandas
version 1.0, the global default has changed from Fiona to Pyogrio.

The main reason for this change is performance. Pyogrio is optimized for the use case
relevant for GeoPandas: reading and writing in bulk. Because of this, in many cases
speedups >5-20x can be observed.

This guide outlines the (known) functional differences between both, so you can account
for them when switching to Pyogrio.


Write an attribute table to a file
----------------------------------

Using the Fiona engine, it was possible to write an attribute table (a table without
geometry column) to a file using the ``schema`` parameter to specify that the "geometry"
column of a GeoDataFrame should be ignored.

With Pyogrio you can write an attribute table by using :func:`pyogrio.write_dataframe`
and passing a pandas DataFrame to it:

.. code-block:: python

    >>> import pyogrio
    >>> df = pd.DataFrame({"data_column": [1, 2, 3]})
    >>> pyogrio.write_dataframe(df, "test_attribute_table.gpkg")


No support for ``schema`` parameter to write files
--------------------------------------------------

Pyogrio does not support specifying the `schema` parameter to write files. This means
it is not possible to specify the types of attributes being written explicitly.


Writing EMPTY geometries
------------------------

Pyogrio writes EMPTY and None geometries as such to e.g. GPKG files, Fiona writes both
as None.

.. ipython:: python
    :okwarning:

    import shapely
    
    gdf = geopandas.GeoDataFrame(geometry=[shapely.Polygon(), None], crs=31370)
    gdf.to_file("test_fiona.gpkg", engine="fiona")
    gdf.to_file("test_pyogrio.gpkg", engine="pyogrio")
    geopandas.read_file("test_fiona.gpkg").head()
    geopandas.read_file("test_pyogrio.gpkg").head()
