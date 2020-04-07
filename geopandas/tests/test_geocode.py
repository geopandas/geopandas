import numpy as np
import pandas as pd

from shapely.geometry import Point

from geopandas import GeoDataFrame, GeoSeries
from geopandas.tools import geocode, reverse_geocode
from geopandas.tools.geocoding import _prepare_geocode_result

from geopandas.tests.util import assert_geoseries_equal, mock
from pandas.testing import assert_series_equal
import pytest

geopy = pytest.importorskip("geopy")


class ForwardMock(mock.MagicMock):
    """
    Mock the forward geocoding function.
    Returns the passed in address and (p, p+.5) where p increases
    at each call

    """

    def __init__(self, *args, **kwargs):
        super(ForwardMock, self).__init__(*args, **kwargs)
        self._n = 0.0

    def __call__(self, *args, **kwargs):
        self.return_value = args[0], (self._n, self._n + 0.5)
        self._n += 1
        return super(ForwardMock, self).__call__(*args, **kwargs)


class ReverseMock(mock.MagicMock):
    """
    Mock the reverse geocoding function.
    Returns the passed in point and 'address{p}' where p increases
    at each call

    """

    def __init__(self, *args, **kwargs):
        super(ReverseMock, self).__init__(*args, **kwargs)
        self._n = 0

    def __call__(self, *args, **kwargs):
        self.return_value = "address{0}".format(self._n), args[0]
        self._n += 1
        return super(ReverseMock, self).__call__(*args, **kwargs)


@pytest.fixture
def locations():
    locations = ["260 Broadway, New York, NY", "77 Massachusetts Ave, Cambridge, MA"]
    return locations


@pytest.fixture
def points():
    points = [Point(-71.0597732, 42.3584308), Point(-77.0365305, 38.8977332)]
    return points


def test_prepare_result():
    # Calls _prepare_result with sample results from the geocoder call
    # loop
    p0 = Point(12.3, -45.6)  # Treat these as lat/lon
    p1 = Point(-23.4, 56.7)
    d = {"a": ("address0", p0.coords[0]), "b": ("address1", p1.coords[0])}

    df = _prepare_geocode_result(d)
    assert type(df) is GeoDataFrame
    assert df.crs == "EPSG:4326"
    assert len(df) == 2
    assert "address" in df

    coords = df.loc["a"]["geometry"].coords[0]
    test = p0.coords[0]
    # Output from the df should be lon/lat
    assert coords[0] == pytest.approx(test[1])
    assert coords[1] == pytest.approx(test[0])

    coords = df.loc["b"]["geometry"].coords[0]
    test = p1.coords[0]
    assert coords[0] == pytest.approx(test[1])
    assert coords[1] == pytest.approx(test[0])


def test_prepare_result_none():
    p0 = Point(12.3, -45.6)  # Treat these as lat/lon
    d = {"a": ("address0", p0.coords[0]), "b": (None, None)}

    df = _prepare_geocode_result(d)
    assert type(df) is GeoDataFrame
    assert df.crs == "EPSG:4326"
    assert len(df) == 2
    assert "address" in df

    row = df.loc["b"]
    # The shapely.geometry.Point() is actually a GeometryCollection, and thus
    # gets converted to that in conversion to pygeos. When converting back
    # on access, you now get a GeometryCollection object instead of Point,
    # which has no coords
    # see https://github.com/Toblerity/Shapely/issues/742/#issuecomment-545296708
    # TODO we should probably replace this with a missing value instead of point?
    # assert len(row["geometry"].coords) == 0
    assert row["geometry"].is_empty
    assert np.isnan(row["address"])


def test_bad_provider_forward():
    from geopy.exc import GeocoderNotFound

    with pytest.raises(GeocoderNotFound):
        geocode(["cambridge, ma"], "badprovider")


def test_bad_provider_reverse():
    from geopy.exc import GeocoderNotFound

    with pytest.raises(GeocoderNotFound):
        reverse_geocode(["cambridge, ma"], "badprovider")


def test_forward(locations, points):
    from geopy.geocoders import GeocodeFarm

    for provider in ["geocodefarm", GeocodeFarm]:
        with mock.patch("geopy.geocoders.GeocodeFarm.geocode", ForwardMock()) as m:
            g = geocode(locations, provider=provider, timeout=2)
            assert len(locations) == m.call_count

        n = len(locations)
        assert isinstance(g, GeoDataFrame)
        expected = GeoSeries(
            [Point(float(x) + 0.5, float(x)) for x in range(n)], crs="EPSG:4326"
        )
        assert_geoseries_equal(expected, g["geometry"])
        assert_series_equal(g["address"], pd.Series(locations, name="address"))


def test_reverse(locations, points):
    from geopy.geocoders import GeocodeFarm

    for provider in ["geocodefarm", GeocodeFarm]:
        with mock.patch("geopy.geocoders.GeocodeFarm.reverse", ReverseMock()) as m:
            g = reverse_geocode(points, provider=provider, timeout=2)
            assert len(points) == m.call_count

        assert isinstance(g, GeoDataFrame)

        expected = GeoSeries(points, crs="EPSG:4326")
        assert_geoseries_equal(expected, g["geometry"])
        address = pd.Series(
            ["address" + str(x) for x in range(len(points))], name="address"
        )
        assert_series_equal(g["address"], address)
