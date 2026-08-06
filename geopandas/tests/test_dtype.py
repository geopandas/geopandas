from geopandas.array import GeometryDtype


def test_string_construction():
    assert GeometryDtype.construct_from_string("geometry") == GeometryDtype(
        engine="planar"
    )
    assert GeometryDtype.construct_from_string("geometry[spherical]") == GeometryDtype(
        engine="spherical"
    )
