.. currentmodule:: geopandas

.. ipython:: python
   :suppress:

   import geopandas as gpd


Merging Data
=========================================

There are two ways to combine datasets in *geopandas* -- attribute joins and spatial joins.

In an attribute join, a ``GeoSeries`` or ``GeoDataFrame`` is combined with a regular *pandas* ``Series`` or ``DataFrame`` based on a common variable. This is analogous to normal merging or joining in *pandas*.

In a Spatial Join, observations from to ``GeoSeries`` or ``GeoDataFrames`` are combined based on their spatial relationship to one another.

In the following examples, we use these datasets:

.. ipython:: python

   world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
   cities = gpd.read_file(gpd.datasets.get_path('naturalearth_cities'))

   # For attribute join
   country_shapes = world[['geometry', 'iso_a3']]
   country_names = world[['name', 'iso_a3']]

   # For spatial join
   countries = world[['geometry', 'name']]
   countries = countries.rename(columns={'name':'country'})


Attribute Joins
----------------

Attribute joins are accomplished using the ``merge`` method. In general, it is recommended to use the ``merge`` method called from the spatial dataset. With that said, the stand-alone ``merge`` function will work if the GeoDataFrame is in the ``left`` argument; if a DataFrame is in the ``left`` argument and a GeoDataFrame is in the ``right`` position, the result will no longer be a GeoDataFrame.


For example, consider the following merge that adds full names to a ``GeoDataFrame`` that initially has only ISO codes for each country by merging it with a *pandas* ``DataFrame``.

.. ipython:: python

   # `country_shapes` is GeoDataFrame with country shapes and iso codes
   country_shapes.head()

   # `country_names` is DataFrame with country names and iso codes
   country_names.head()

   # Merge with `merge` method on shared variable (iso codes):
   country_shapes = country_shapes.merge(country_names, on='iso_a3')
   country_shapes.head()



Spatial Joins
----------------

In a Spatial Join, two geometry objects are merged based on their spatial relationship to one another.

.. ipython:: python


   # One GeoDataFrame of countries, one of Cities.
   # Want to merge so we can get each city's country.
   countries.head()
   cities.head()

   # Execute spatial join

   cities_with_country = gpd.sjoin(cities, countries, how="inner", op='intersects')
   cities_with_country.head()


The ``op`` options determines the type of join operation to apply. ``op`` can be set to "intersects", "within" or "contains" (these are all equivalent when joining points to polygons, but differ when joining polygons to other polygons or lines).

Note more complicated spatial relationships can be studied by combining geometric operations with spatial join. To find all polygons within a given distance of a point, for example, one can first use the ``buffer`` method to expand each point into a circle of appropriate radius, then intersect those buffered circles with the polygons in question.
