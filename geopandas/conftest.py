import pytest
import geopandas


@pytest.fixture(autouse=True)
def add_geopandas(doctest_namespace):
    doctest_namespace["geopandas"] = geopandas


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "skip_no_sindex: skips the tests if there is no spatial index backend",
    )


try:
    geopandas.sindex._get_sindex_class()
    has_sindex_backend = True
except ImportError:
    has_sindex_backend = False


def pytest_runtest_setup(item):
    skip_no_sindex = any(mark for mark in item.iter_markers(name="skip_no_sindex"))
    if skip_no_sindex and not has_sindex_backend:
        pytest.skip("Skipped because there is no spatial index backend available")
