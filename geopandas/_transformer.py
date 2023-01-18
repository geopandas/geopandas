from pyproj import Transformer as PyprojTransformer
from pyproj import __version__ as pyproj_version

# Setup cached Transformer for pyproj 2.x and 3.x
if pyproj_version.startswith(("2", "3")):
    from functools import lru_cache

    TransformerFromCRS = lru_cache(PyprojTransformer.from_crs)
else:
    TransformerFromCRS = PyprojTransformer.from_crs


__all__ = ["TransformerFromCRS"]
