from geopandas._transformer import TransformerFromCRS
from pyproj import Transformer
from shapely.geometry import Point
from shapely.ops import transform


class TestTransformer:
    def test_transformer(self):
        pyproj_tf = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
        tf = TransformerFromCRS("EPSG:4326", "EPSG:3857", always_xy=True)

        geom = Point(0, 0)
        expect = transform(pyproj_tf.transform, geom)
        result = transform(tf.transform, geom)

        assert result.almost_equals(expect)
