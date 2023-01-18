from geopandas._transformer import TransformerFromCRS
from pyproj import Transformer
from shapely.geometry import Point
from shapely.ops import transform


class TestTransformer:
    def test_transformer(self):
        pyproj_tf = Transformer.from_crs("4326", "3857", always_xy=True)
        tf = TransformerFromCRS("4326", "3857", always_xy=True)

        geom = Point(0, 0)
        expect = transform(pyproj_tf.transform, geom)
        result = transform(tf.transform, geom)

        assert result.equals(expect)
