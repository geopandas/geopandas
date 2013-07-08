import json

from geopandas import GeoDataFrame

def test_from_file_():
    # Data from http://www.nyc.gov/html/dcp/download/bytes/nybb_13a.zip
    # saved as geopandas/examples/nybb_13a.zip.
    df = GeoDataFrame.from_file(
        '/nybb_13a/nybb.shp', vfs='zip://examples/nybb_13a.zip')
    assert 'geometry' in df
    assert len(df) == 5

def test_to_json():
    df = GeoDataFrame.from_file(
        '/nybb_13a/nybb.shp', vfs='zip://examples/nybb_13a.zip')
    text = df.to_json()
    data = json.loads(text)
    assert data['type'] == 'FeatureCollection'
    assert len(data['features']) == 5
