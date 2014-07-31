from collections import defaultdict
import time

from fiona.crs import from_epsg
import numpy as np
import pandas as pd
from shapely.geometry import Point
from six import iteritems

try:

    from geopy.geocoders import get_geocoder_for_service
    from geopy.exc import GeocoderNotFound, GeocoderServiceError
except ImportError:
    get_geocoder_for_service = GeocoderNotFound = GeocoderServiceError = None

import geopandas as gpd


def _throttle_time(provider):
    """ Amount of time to wait between requests to a geocoding API.

    Currently implemented for Nominatim, as their terms of service
    require a maximum of 1 request per second.
    https://wiki.openstreetmap.org/wiki/Nominatim_usage_policy
    """
    if provider == 'nominatim':
        return 1
    else:
        return 0


def geocode(strings, provider='googlev3', **kwargs):
    """
    Geocode a set of strings and get a GeoDataFrame of the resulting points.

    Parameters
    ----------
    strings : list or Series of addresses to geocode
    provider : geopy geocoder to use, default 'googlev3'.
        Some providers require additional arguments such as access keys.
        See each geocoder's specific parameters in geopy.geocoders.
        Any geocoder available in the installed version of geopy is available
        here; see geopy's documentation for more.

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
    provider : geopy geocoder to use, default 'googlev3'.
        These are the same options as the geocode() function.
        Some providers require additional arguments such as access keys.
        See each geocoder's specific parameters in geopy.geocoders.
        Any geocoder available in the installed version of geopy is available
        here; see geopy's documentation for more.

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
    if get_geocoder_for_service is None:
        raise ImportError("`geopy` package must be installed to geocode.")

    if not isinstance(data, pd.Series):
        data = pd.Series(data)

    try:
        geocoder_cls = get_geocoder_for_service(provider)
    except GeocoderNotFound:
        raise ValueError('Unknown geocoding provider: {0}'.format(provider))

    coder = geocoder_cls(**kwargs)
    results = {}
    for i, s in iteritems(data):
        try:
            if forward:
                results[i] = coder.geocode(s, exactly_one=True)
            else:
                results[i] = coder.reverse((s.y, s.x), exactly_one=True)
        except GeocoderServiceError:
            results[i] = None
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

    for i, loc in iteritems(results):
        if loc is not None:
            d['geometry'].append(
                Point(loc.longitude, loc.latitude, loc.altitude)
            )
            d['address'].append(loc.address)
        else:
            d['geometry'].append(Point())
            d['address'].append(np.nan)
        index.append(i)

    df = gpd.GeoDataFrame(d, index=index)
    df.crs = from_epsg(4326)

    return df
