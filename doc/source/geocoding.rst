.. currentmodule:: geopandas

.. ipython:: python
   :suppress:

   import geopandas as gpd


Geocoding
==========

``geopandas`` supports geocoding (i.e., converting place names to
location on Earth) through `geopy`_, an optional dependency of ``geopandas``.
The following example shows how to use the `Google geocoding API
<https://developers.google.com/maps/documentation/geocoding/start>`_ to get the
locations of boroughs in New York City, and plots those locations along
with the detailed borough boundary file included within ``geopandas``.

.. _geopy: http://geopy.readthedocs.io/

.. ipython:: python

    boros = gpd.read_file(gpd.datasets.get_path("nybb"))
    boros.BoroName
    boro_locations = gpd.tools.geocode(boros.BoroName, provider="google")
    boro_locations

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()

    boros.to_crs({"init": "epsg:4326"}).plot(ax=ax, color="white", edgecolor="black");
    @savefig boro_centers_over_bounds.png
    boro_locations.plot(ax=ax, color="red");


The argument to ``provider`` can either be a string referencing geocoding
services, such as ``'google'``, ``'bing'``, ``'yahoo'``, and
``'openmapquest'``, or an instance of a ``Geocoder`` from ``geopy``. See
``geopy.geocoders.SERVICE_TO_GEOCODER`` for the full list.
For many providers, parameters such as API keys need to be passed as
``**kwargs`` in the ``geocode`` call.

Please consult the Terms of Service for the chosen provider.
