from collections import defaultdict

import fiona
import numpy as np
import pandas as pd
from shapely.geometry import Point
from six import iteritems

import geopandas as gpd

def geocode(strings, provider='googlev3', **kwargs):
    """
    Geocode a set of strings and get a GeoDataFrame of the resulting points.

    Parameters
    ----------
    strings : list or Series of addresses to geocode
    provider : geopy geocoder to use, default 'googlev3'
        Some providers require additional arguments such as access keys
        See each geocoder's specific parameters in geopy.geocoders
        * googlev3, default
        * bing
        * google
        * yahoo
        * mapquest
        * openmapquest
    
    Ensure proper use of the results by consulting the Terms of Service for
    your provider.

    Geocoding requires geopy. Install it using 'pip install geopy'. See also
    https://github.com/geopy/geopy

    Example
    -------
    >>> df = geocode(['boston, ma', '1600 pennsylvania ave. washington, dc'])
    address                                               geometry
    0                                    Boston, MA, USA  POINT (-71.0597731999999951 42.3584308000000007)
    1  1600 Pennsylvania Avenue Northwest, President'...  POINT (-77.0365122999999983 38.8978377999999978)

    """
    import geopy
    from geopy.geocoders.base import GeocoderResultError

    if not isinstance(strings, pd.Series):
        strings = pd.Series(strings)

    # workaround changed name in 0.96
    try:
        Yahoo = geopy.geocoders.YahooPlaceFinder
    except AttributeError:
        Yahoo = geopy.geocoders.Yahoo

    coders = {'googlev3': geopy.geocoders.GoogleV3,
              'bing': geopy.geocoders.Bing,
              'yahoo': Yahoo,
              'mapquest': geopy.geocoders.MapQuest,
              'openmapquest': geopy.geocoders.OpenMapQuest}

    if provider not in coders:
        raise ValueError('Unknown geocoding provider: {}'.format(provider))

    coder = coders[provider](**kwargs)
    results = {}
    for i, s in iteritems(strings):
        try:
            results[i] = coder.geocode(s)
        except (GeocoderResultError, ValueError):
            results[i] = (None, None)

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
            address = pd.np.nan

        d['geometry'].append(p)
        d['address'].append(address)
        index.append(i)

    df = gpd.GeoDataFrame(d, index=index)
    df.crs = fiona.crs.from_epsg(4326)

    return df
