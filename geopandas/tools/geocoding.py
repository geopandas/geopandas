from collections import defaultdict
import time

import pandas as pd

from shapely.geometry import Point

import geopandas


def _get_throttle_time(provider):
    """
    Amount of time to wait between requests to a geocoding API, for providers
    that specify rate limits in their terms of service.
    """
    import geopy.geocoders

    # https://operations.osmfoundation.org/policies/nominatim/
    if provider == geopy.geocoders.Nominatim:
        return 1
    else:
        return 0


def geocode(strings, provider=None, **kwargs):
    """
    Geocode a set of strings and get a GeoDataFrame of the resulting points.

    Parameters
    ----------
    strings : list or Series of addresses to geocode
    provider : str or geopy.geocoder
        Specifies geocoding service to use. If none is provided,
        will use 'photon' (see the Photon's terms of service at:
        https://photon.komoot.io).

        Either the string name used by geopy (as specified in
        geopy.geocoders.SERVICE_TO_GEOCODER) or a geopy Geocoder instance
        (e.g., geopy.geocoders.Photon) may be used.

        Some providers require additional arguments such as access keys
        See each geocoder's specific parameters in geopy.geocoders

    Notes
    -----
    Ensure proper use of the results by consulting the Terms of Service for
    your provider.

    Geocoding requires geopy. Install it using 'pip install geopy'. See also
    https://github.com/geopy/geopy

    Examples
    --------
    >>> df = geopandas.tools.geocode(  # doctest: +SKIP
    ...         ["boston, ma", "1600 pennsylvania ave. washington, dc"]
    ...     )
    >>> df  # doctest: +SKIP
                        geometry                                            address
    0  POINT (-71.05863 42.35899)                          Boston, MA, United States
    1  POINT (-77.03651 38.89766)  1600 Pennsylvania Ave NW, Washington, DC 20006...
    """

    if provider is None:
        provider = "photon"
    throttle_time = _get_throttle_time(provider)

    return _query(strings, True, provider, throttle_time, **kwargs)


def reverse_geocode(points, provider=None, **kwargs):
    """
    Reverse geocode a set of points and get a GeoDataFrame of the resulting
    addresses.

    The points

    Parameters
    ----------
    points : list or Series of Shapely Point objects.
        x coordinate is longitude
        y coordinate is latitude
    provider : str or geopy.geocoder (opt)
        Specifies geocoding service to use. If none is provided,
        will use 'photon' (see the Photon's terms of service at:
        https://photon.komoot.io).

        Either the string name used by geopy (as specified in
        geopy.geocoders.SERVICE_TO_GEOCODER) or a geopy Geocoder instance
        (e.g., geopy.geocoders.Photon) may be used.

        Some providers require additional arguments such as access keys
        See each geocoder's specific parameters in geopy.geocoders

    Notes
    -----
    Ensure proper use of the results by consulting the Terms of Service for
    your provider.

    Reverse geocoding requires geopy. Install it using 'pip install geopy'.
    See also https://github.com/geopy/geopy

    Examples
    --------
    >>> from shapely.geometry import Point
    >>> df = geopandas.tools.reverse_geocode(  # doctest: +SKIP
    ...     [Point(-71.0594869, 42.3584697), Point(-77.0365305, 38.8977332)]
    ... )
    >>> df  # doctest: +SKIP
                         geometry                                            address
    0  POINT (-71.05941 42.35837)       29 Court Sq, Boston, MA 02108, United States
    1  POINT (-77.03641 38.89766)  1600 Pennsylvania Ave NW, Washington, DC 20006...
    """

    if provider is None:
        provider = "photon"
    throttle_time = _get_throttle_time(provider)

    return _query(points, False, provider, throttle_time, **kwargs)


def _query(data, forward, provider, throttle_time, **kwargs):
    # generic wrapper for calls over lists to geopy Geocoders
    from geopy.geocoders.base import GeocoderQueryError
    from geopy.geocoders import get_geocoder_for_service

    if forward:
        if not isinstance(data, pd.Series):
            data = pd.Series(data)
    else:
        if not isinstance(data, geopandas.GeoSeries):
            data = geopandas.GeoSeries(data)

    if isinstance(provider, str):
        provider = get_geocoder_for_service(provider)

    coder = provider(**kwargs)
    results = {}
    for i, s in data.items():
        try:
            if forward:
                results[i] = coder.geocode(s)
            else:
                results[i] = coder.reverse((s.y, s.x), exactly_one=True)
        except (GeocoderQueryError, ValueError):
            results[i] = (None, None)
        time.sleep(throttle_time)

    df = _prepare_geocode_result(results)
    return df


def _prepare_geocode_result(results):
    """
    Helper function for the geocode function

    Takes a dict where keys are index entries, values are tuples containing:
    (address, (lat, lon))

    """
    # Prepare the data for the DataFrame as a dict of lists
    d = defaultdict(list)
    index = []

    for i, s in results.items():
        if s is None:
            p = Point()
            address = None

        else:
            address, loc = s

            # loc is lat, lon and we want lon, lat
            if loc is None:
                p = Point()
            else:
                p = Point(loc[1], loc[0])

        d["geometry"].append(p)
        d["address"].append(address)
        index.append(i)

    df = geopandas.GeoDataFrame(d, index=index, crs="EPSG:4326")

    return df
