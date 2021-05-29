"""
Public API for extending geopandas objects.
"""


from geopandas.accessor import (
    register_geodataframe_accessor,
    register_geoseries_accessor,
)

__all__ = [
    "register_geodataframe_accessor",
    "register_geoseries_accessor"
]
