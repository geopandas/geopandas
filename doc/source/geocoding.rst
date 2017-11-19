.. currentmodule:: geopandas

.. ipython:: python
   :suppress:

   import geopandas as gpd



Geocoding
==========

``geopandas`` supports geocoding (i.e., converting place names to
location on Earth) through `geopy`_, an optional dependency.
The following example shows how to use the Google geocoding API to get the
locations of boroughs in New York City, and plots those centers along
with the detailed borough boundary file included within ``geopandas``.

.. _geopy: http://geopy.readthedocs.io/

.. ipython:: python

    boros = gpd.read_file(gpd.datasets.get_path("nybb"))
    print(boros.BoroName)
    boro_centers = gpd.tools.geocode(boros.BoroName, provider="google")

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()

    boros.to_crs({"init": "epsg:4326"}).plot(ax=ax, color="white", edgecolor="black");
    boro_centers.plot(ax=ax, color="red");
    @savefig boro_centers_over_bounds.png
    plt.show();

Available ``provider`` arguments include ``'google'``, ``'bing'``,
``'yahoo'``, and ``'openmapquest'``. See
``geopy.geocoders.SERVICE_TO_GEOCODER`` for the full list.
For many providers, parameters such as API keys need to be passed as
**kwargs in the ``geocode`` call.

Please consult the Terms of Service for the chosen provider.
