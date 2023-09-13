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
