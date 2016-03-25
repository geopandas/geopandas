
Geocoding
==========

[TO BE COMPLETED]


.. function:: geopandas.geocode.geocode(strings, provider='googlev3', **kwargs)

  Geocode a list of strings and return a GeoDataFrame containing the
  resulting points in its ``geometry`` column.  Available
  ``provider``s include ``googlev3``, ``bing``, ``google``, ``yahoo``,
  ``mapquest``, and ``openmapquest``.  ``**kwargs`` will be passed as
  parameters to the appropriate geocoder.

  Requires `geopy`_.  Please consult the Terms of Service for the
  chosen provider.
