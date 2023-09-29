import geopandas

import pytest


def test_options():
    assert "display_precision: " in repr(geopandas.options)

    assert set(dir(geopandas.options)) == {
        "display_precision",
        "use_pygeos",
        "io_engine",
    }

    with pytest.raises(AttributeError):
        geopandas.options.non_existing_option

    with pytest.raises(AttributeError):
        geopandas.options.non_existing_option = 10


def test_options_display_precision():
    assert geopandas.options.display_precision is None
    geopandas.options.display_precision = 5
    assert geopandas.options.display_precision == 5

    with pytest.raises(ValueError):
        geopandas.options.display_precision = "abc"

    with pytest.raises(ValueError):
        geopandas.options.display_precision = -1

    geopandas.options.display_precision = None


def test_options_io_engine():
    assert geopandas.options.io_engine is None
    geopandas.options.io_engine = "pyogrio"
    assert geopandas.options.io_engine == "pyogrio"

    with pytest.raises(ValueError):
        geopandas.options.io_engine = "abc"

    with pytest.raises(ValueError):
        geopandas.options.io_engine = -1

    geopandas.options.io_engine = None
