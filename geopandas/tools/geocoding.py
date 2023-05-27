import pandas as pd

from shapely.geometry import Point

import geopandas


def geocode(
    strings,
    provider="photon",
    min_delay_seconds=0,
    max_retries=2,
    error_wait_seconds=5,
    **kwargs,
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
                                                 address                    geometry
    0                          Boston, MA, United States  POINT (-71.05863 42.35899)
    1  1600 Pennsylvania Ave NW, Washington, DC 20006...  POINT (-77.03651 38.89766)
    """

    if not isinstance(strings, pd.Series):
        strings = pd.Series(strings)

    geolocate = geolocator(
        provider, True, min_delay_seconds, max_retries, error_wait_seconds, **kwargs
    )
    geometry = strings.apply(query_address, geolocate=geolocate)
    print(geometry)
    return geopandas.GeoDataFrame(
        strings.rename(strings.name or "address"),
        geometry=geometry,
        crs=4326,
    )


def reverse_geocode(
    points,
    provider="photon",
    min_delay_seconds=0,
    max_retries=2,
    error_wait_seconds=5,
    **kwargs,
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
                                                 address                    geometry
    0       29 Court Sq, Boston, MA 02108, United States  POINT (-71.05941 42.35837)
    1  1600 Pennsylvania Ave NW, Washington, DC 20006...  POINT (-77.03641 38.89766)
    """

    if not isinstance(data, geopandas.GeoSeries):
        points = geopandas.GeoSeries(points, crs=4326)

    geolocate = geolocator(
        provider, False, min_delay_seconds, max_retries, error_wait_seconds, **kwargs
    )
    address = (
        points.get_coordinates()
        .apply(lambda s: (s.y, s.x), axis=1)
        .apply(query_address, geolocate=geolocate)
        .rename("address")
    )

    return geopandas.GeoDataFrame(address, geometry=points, crs=4326)


def geolocator(
    provider,
    forward,
    min_delay_seconds,
    max_retries,
    error_wait_seconds,
    **kwargs,
):
    from geopy.extra.rate_limiter import RateLimiter
    from geopy.geocoders import get_geocoder_for_service

    # Get the actual 'geocoder' from the provider name
    if isinstance(provider, str):
        provider = get_geocoder_for_service(provider)

    return RateLimiter(
        getattr(provider(**kwargs), "geocode" if forward else "reverse"),
        min_delay_seconds=min_delay_seconds,
        max_retries=max_retries,
        error_wait_seconds=error_wait_seconds,
    )


def query_point(address, geolocate):
    from geopy.geocoders.base import GeocoderQueryError

    try:
        loc = geolocate(address)
        return Point(loc.longitude, loc.latitude)

    except (GeocoderQueryError, ValueError, AttributeError):
        return None


def query_address(point, geolocate):
    from geopy.geocoders.base import GeocoderQueryError

    try:
        return geolocate(point).address

    except (GeocoderQueryError, ValueError, AttributeError):
        return None
