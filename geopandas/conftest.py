import pytest
import geopandas
from geopandas.tests.util import _NYBB, _NATURALEARTH_LOWRES, _NATURALEARTH_CITIES


@pytest.fixture(autouse=True)
def add_geopandas(doctest_namespace):
    doctest_namespace["geopandas"] = geopandas


# Datasets used in our tests


@pytest.fixture(scope="session")
def naturalearth_lowres() -> str:
    return _NATURALEARTH_LOWRES


@pytest.fixture(scope="session")
def naturalearth_cities() -> str:
    return _NATURALEARTH_CITIES


@pytest.fixture(scope="session")
def nybb_filename() -> str:
    return _NYBB


@pytest.fixture(scope="class")
def _setup_class_nybb_filename(nybb_filename, request):
    """Attach nybb_filename class attribute for unittest style setup_method"""
    request.cls.nybb_filename = nybb_filename


@pytest.fixture(scope="class")
def _setup_class_naturalearth_lowres(naturalearth_lowres, request):
    """Attach nybb_filename class attribute for unittest style setup_method"""
    request.cls.naturalearth_lowres = naturalearth_lowres
