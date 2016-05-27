.. ipython:: python
   :suppress:

   import geopandas as gpd


Set-Operations with Overlay
============================

When working with multiple spatial datasets -- especially multiple *polygon* or *line* datasets -- users often wish to create new shapes based on places where those datasets overlap (or don't overlap). These manipulations are often referred using the language of sets -- intersections, unions, and differences. These types of operations are made available in the *geopandas* library through the ``overlay`` function.

The basic idea is demonstrated by the graphic below but keep in mind that overlays operate at the DataFrame level, not on individual geometries, and the properties from both are retained. In effect, for every shape in the first GeoDataFrame, this operation is executed against every other shape in the other GeoDataFrame:

.. image:: _static/overlay_operations.png

**Source: QGIS Documentation**

(Note to users familiar with the *shapely* library: ``overlay`` can be thought of as offering versions of the standard *shapely* set-operations that deal with the complexities of applying set operations to two *GeoSeries*. The standard *shapely* set-operations are also available as ``GeoSeries`` methods.)


Overlay Example
-----------------

First, we load some example data:

.. ipython:: python

    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    capitals = gpd.read_file(gpd.datasets.get_path('naturalearth_cities'))

    # Select some columns
    countries = world[['geometry', 'name']]

    # Project to crs that uses meters as distance measure
    countries = countries.to_crs('+init=epsg:3395')[countries.name!="Antarctica"]
    capitals = capitals.to_crs('+init=epsg:3395')

To illustrate the ``overlay`` function, consider the following case in which one wishes to identify the "core" portion of each country -- defined as areas within 500km of a capital -- using a ``GeoDataFrame`` of countries and a ``GeoDataFrame`` of capitals.

.. ipython:: python

    # Look at countries:
    @savefig world_basic.png width=5in
    countries.plot();

    # Now buffer cities to find area within 500km.
    # Check CRS -- World Mercator, units of meters.
    capitals.crs

    # make 500km buffer
    capitals['geometry']= capitals.buffer(500000)
    @savefig capital_buffers.png width=5in
    capitals.plot();


To select only the portion of countries within 500km of a capital, we specify the ``how`` option to be "intersect", which creates a new set of polygons where these two layers overlap:

.. ipython:: python

   country_cores = gpd.overlay(countries, capitals, how='intersection')
   @savefig country_cores.png width=5in
   country_cores.plot();

Changing the "how" option allows for different types of overlay operations. For example, if we were interested in the portions of countries *far* from capitals (the peripheries), we would compute the difference of the two.

.. ipython:: python

   country_peripheries = gpd.overlay(countries, capitals, how='difference')
   @savefig country_peripheries.png width=5in
   country_peripheries.plot();

More Examples
-----------------

A larger set of examples of the use of ``overlay`` can be found `here <http://nbviewer.jupyter.org/github/geopandas/geopandas/blob/master/examples/overlays.ipynb>`_



.. toctree::
   :maxdepth: 2
