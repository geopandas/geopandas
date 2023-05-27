from collections import defaultdict

import pandas as pd

from shapely.geometry import Point

import geopandas


def geocode(
    strings,
    provider="photon",
    min_delay_seconds=0,
    max_retries=2,
    error_wait_seconds=5,
    **kwargs
):
    """
    Geocode a set of strings and get a GeoDataFrame of the resulting points.

    Parameters
    ----------
    strings : list or Series(string) of addresses to geocode

    provider : str or geopy.geocoder, default "photon"
        Specifies geocoding service to use. Default will use "photon", see the
        Photon's terms of service at: https://photon.komoot.io. Either the string
        name used by geopy (as specified in ``geopy.geocoders.SERVICE_TO_GEOCODER``)
        or a geopy Geocoder instance (e.g., :obj:`~geopy.geocoders.Photon`) may be
        used. Some providers require additional arguments such as access keys, please
        see each geocoder's specific parameters in :mod:`geopy.geocoders`.

    min_delay_seconds, max_retries, error_wait_seconds
        See the documentation for :func:`~geopy.extra.rate_limiter.RateLimiter` for
        complete details on these arguments.

    **kwargs
        Additional keyword arguments to pass to the geocoder.

    Returns
    -------
    GeoDataFrame

    Notes
    -----
    Ensure proper use of the results by consulting the Terms of Service for your
    provider.

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

    return _query(strings, True, provider, throttle_time, **kwargs)


def reverse_geocode(
    points,
    provider="photon",
    min_delay_seconds=0,
    max_retries=2,
    error_wait_seconds=5,
    **kwargs
):
    """
    Reverse geocode a set of points and get a GeoDataFrame of the resulting addresses.

    Parameters
    ----------
    points : list or Series of Shapely Point objects.
        x coordinate is longitude, y coordinate is latitude

    provider : str or geopy.geocoder, default "photon"
        Specifies geocoding service to use. Default will use "photon", see the
        Photon's terms of service at: https://photon.komoot.io. Either the string
        name used by geopy (as specified in ``geopy.geocoders.SERVICE_TO_GEOCODER``)
        or a geopy Geocoder instance (e.g., :obj:`~geopy.geocoders.Photon`) may be
        used. Some providers require additional arguments such as access keys, please
        see each geocoder's specific parameters in :mod:`geopy.geocoders`.

    min_delay_seconds, max_retries, error_wait_seconds
        See the documentation for :func:`~geopy.extra.rate_limiter.RateLimiter` for
        complete details on these arguments.

    **kwargs
        Additional keyword arguments to pass to the geocoder.

    Returns
    -------
    GeoDataFrame

    Notes
    -----
    Ensure proper use of the results by consulting the Terms of Service for your
    provider.

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

    return _query(points, False, provider, throttle_time, **kwargs)


def _query(data, forward, provider, throttle_time, **kwargs):
    # generic wrapper for calls over lists to geopy Geocoders
    from functools import partial

    from geopy.extra.rate_limiter import RateLimiter
    from geopy.geocoders import get_geocoder_for_service
    from geopy.geocoders.base import GeocoderQueryError

    # Get the actual 'geocoder' from the provider name
    if provider is None:
        provider = "photon"
    if isinstance(provider, str):
        provider = get_geocoder_for_service(provider)

    # Set default throttle time if not provided
    if throttle_time is None:
        throttle_time = _get_throttle_time(provider)

    # Transform data into Series
    if forward and not isinstance(data, pd.Series):
        data = pd.Series(data)
    elif not isinstance(data, geopandas.GeoSeries):
        data = geopandas.GeoSeries(data)

    # Init geocoder
    coder = provider(**kwargs)
    transform = coder.geocode if forward else coder.reverse
    transform = partial(transform, exactly_one=True)
    transform = RateLimiter(transform, min_delay_seconds=throttle_time)

    results = {}
    for i, s in data.items():
        try:
            results[i] = transform(s) if forward else transform((s.y, s.x))
        except (GeocoderQueryError, ValueError):
            results[i] = (None, None)

    return _prepare_geocode_result(results)


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
