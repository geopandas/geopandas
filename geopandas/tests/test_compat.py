from geopandas._compat import import_optional_dependency

import pytest


def test_import_optional_dependency_present():
    # pandas is not optional, but we know it is present
    pandas = import_optional_dependency("pandas")
    assert pandas is not None

    # module imported normally must be same
    import pandas as pd

    assert pandas == pd


def test_import_optional_dependency_absent():
    with pytest.raises(ImportError, match="Missing optional dependency 'foo'"):
        import_optional_dependency("foo")

    with pytest.raises(ImportError, match="foo is required"):
        import_optional_dependency("foo", extra="foo is required")


@pytest.mark.parametrize(
    "bad_import", [["foo"], 0, False, True, {}, {"foo"}, {"foo": "bar"}]
)
def test_import_optional_dependency_invalid(bad_import):
    with pytest.raises(ValueError, match="Invalid module name"):
        import_optional_dependency(bad_import)
