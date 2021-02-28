.. currentmodule:: geopandas

.. ipython:: python
   :suppress:

   import geopandas


Geocoding
==========

``geopandas`` supports geocoding (i.e., converting place names to
location on Earth) through `geopy`_, an optional dependency of ``geopandas``.
The following example shows how to get the
locations of boroughs in New York City, and plots those locations along
with the detailed borough boundary file included within ``geopandas``.

.. _geopy: http://geopy.readthedocs.io/

.. ipython:: python

    boros = geopandas.read_file(geopandas.datasets.get_path("nybb"))
    boros.BoroName
    boro_locations = geopandas.tools.geocode(boros.BoroName)
    boro_locations

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()

    boros.to_crs("EPSG:4326").plot(ax=ax, color="white", edgecolor="black");
    @savefig boro_centers_over_bounds.png
    boro_locations.plot(ax=ax, color="red");


By default, the ``geocode`` function uses the
`GeoCode.Farm geocoding API <https://geocode.farm/>`__ with a rate limitation
applied. But a different geocoding service can be specified with the
``provider`` keyword.

The argument to ``provider`` can either be a string referencing geocoding
services, such as ``'google'``, ``'bing'``, ``'yahoo'``, and
``'openmapquest'``, or an instance of a ``Geocoder`` from ``geopy``. See
``geopy.geocoders.SERVICE_TO_GEOCODER`` for the full list.
For many providers, parameters such as API keys need to be passed as
``**kwargs`` in the ``geocode`` call.

For example, to use the OpenStreetMap Nominatim geocoder, you need to specify
a user agent:

.. code-block:: python

    geopandas.tools.geocode(boros.BoroName, provider='nominatim', user_agent="my-application")

.. attention::

    Please consult the Terms of Service for the chosen provider. The example
    above uses ``'geocodefarm'`` (the default), for which free users are
    limited to 250 calls per day and 4 requests per second
    (`geocodefarm ToS <https://geocode.farm/geocoding/free-api-documentation/>`_).
