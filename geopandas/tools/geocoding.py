from collections import defaultdict
import time

from fiona.crs import from_epsg
import numpy as np
import pandas as pd
from shapely.geometry import Point
from six import iteritems, string_types

import geopandas


def _throttle_time(provider):
    """ Amount of time to wait between requests to a geocoding API.

    Currently implemented for Nominatim, as their terms of service
    require a maximum of 1 request per second.
    https://wiki.openstreetmap.org/wiki/Nominatim_usage_policy
    """
    import geopy.geocoders
    if provider == geopy.geocoders.Nominatim:
        return 1
    else:
        return 0


def geocode(strings, provider='googlev3', **kwargs):
    """
    Geocode a set of strings and get a GeoDataFrame of the resulting points.

    Parameters
    ----------
    strings : list or Series of addresses to geocode
    provider : str or geopy.geocoder
        Specifies geocoding service to use, default is 'googlev3'.
        Either the string name used by geopy (as specified in
        geopy.geocoders.SERVICE_TO_GEOCODER) or a geopy Geocoder instance
        (e.g., geopy.geocoders.GoogleV3) may be used.

        Some providers require additional arguments such as access keys
        See each geocoder's specific parameters in geopy.geocoders

    Ensure proper use of the results by consulting the Terms of Service for
    your provider.

    Geocoding requires geopy. Install it using 'pip install geopy'. See also
    https://github.com/geopy/geopy

    Example
    -------
    >>> df = geocode(['boston, ma', '1600 pennsylvania ave. washington, dc'])

                                                 address  \
    0                                    Boston, MA, USA
    1  1600 Pennsylvania Avenue Northwest, President'...

                             geometry
    0  POINT (-71.0597732 42.3584308)
    1  POINT (-77.0365305 38.8977332)

    """
    return _query(strings, True, provider, **kwargs)


def reverse_geocode(points, provider='googlev3', **kwargs):
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
        Specifies geocoding service to use, default is 'googlev3'.
        Either the string name used by geopy (as specified in
        geopy.geocoders.SERVICE_TO_GEOCODER) or a geopy Geocoder instance
        (e.g., geopy.geocoders.GoogleV3) may be used.

        Some providers require additional arguments such as access keys
        See each geocoder's specific parameters in geopy.geocoders

    Ensure proper use of the results by consulting the Terms of Service for
    your provider.

    Reverse geocoding requires geopy. Install it using 'pip install geopy'.
    See also https://github.com/geopy/geopy

    Example
    -------
    >>> df = reverse_geocode([Point(-71.0594869, 42.3584697),
                              Point(-77.0365305, 38.8977332)])

                                             address  \
    0             29 Court Square, Boston, MA 02108, USA
    1  1600 Pennsylvania Avenue Northwest, President'...

                             geometry
    0  POINT (-71.0594869 42.3584697)
    1  POINT (-77.0365305 38.8977332)

    """
    return _query(points, False, provider, **kwargs)


def _query(data, forward, provider, **kwargs):
    # generic wrapper for calls over lists to geopy Geocoders
    import geopy
    from geopy.geocoders.base import GeocoderQueryError
    from geopy.geocoders import get_geocoder_for_service

    if not isinstance(data, pd.Series):
        data = pd.Series(data)

    if isinstance(provider, string_types):
        provider = get_geocoder_for_service(provider)

    coder = provider(**kwargs)
    results = {}
    for i, s in iteritems(data):
        try:
            if forward:
                results[i] = coder.geocode(s)
            else:
                results[i] = coder.reverse((s.y, s.x), exactly_one=True)
        except (GeocoderQueryError, ValueError):
            results[i] = (None, None)
        time.sleep(_throttle_time(provider))

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

    for i, s in iteritems(results):
        address, loc = s

        # loc is lat, lon and we want lon, lat
        if loc is None:
            p = Point()
        else:
            p = Point(loc[1], loc[0])

        if address is None:
            address = np.nan

        d['geometry'].append(p)
        d['address'].append(address)
        index.append(i)

    df = geopandas.GeoDataFrame(d, index=index)
    df.crs = from_epsg(4326)

    return df
