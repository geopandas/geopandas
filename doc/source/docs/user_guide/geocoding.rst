.. currentmodule:: geopandas

.. ipython:: python
   :suppress:

   import geopandas


Geocoding
==========

GeoPandas supports geocoding (i.e., converting place names to
location on Earth) through `geopy`_, an optional dependency of GeoPandas.
The following example shows how to get the
locations of boroughs in New York City, and plots those locations along
with the detailed borough boundary file included within GeoPandas.

.. _geopy: http://geopy.readthedocs.io/

.. ipython:: python

    import geodatasets

    boros = geopandas.read_file(geodatasets.get_path("nybb"))
    boros.BoroName
    boro_locations = geopandas.tools.geocode(boros.BoroName)
    boro_locations

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()

    boros.to_crs("EPSG:4326").plot(ax=ax, color="white", edgecolor="black");
    @savefig boro_centers_over_bounds.png
    boro_locations.plot(ax=ax, color="red");


By default, the :func:`~geopandas.tools.geocode` function uses the
`Photon geocoding API <https://photon.komoot.io>`__.
But a different geocoding service can be specified with the
``provider`` keyword.

The argument to ``provider`` can either be a string referencing geocoding
services, such as ``'google'``, ``'bing'``, ``'yahoo'``, and
``'openmapquest'``, or an instance of a :mod:`Geocoder <geopy.geocoders>` from :mod:`geopy`. See
``geopy.geocoders.SERVICE_TO_GEOCODER`` for the full list.
For many providers, parameters such as API keys need to be passed as
``**kwargs`` in the :func:`~geopandas.tools.geocode` call.

For example, to use the OpenStreetMap Nominatim geocoder, you need to specify
a user agent:

.. code-block:: python

    geopandas.tools.geocode(boros.BoroName, provider='nominatim', user_agent="my-application")

.. attention::

    Please consult the Terms of Service for the chosen provider. The example
    above uses ``'photon'`` (the default), which expects fair usage
    - extensive usage will be throttled.
    (`Photon's Terms of Use <https://photon.komoot.io>`_).
