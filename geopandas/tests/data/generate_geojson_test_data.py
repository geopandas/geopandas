"""
Util to convert test data in traditional GIS formats into geojson.

It is possible to install geopandas without any IO (fiona, pyogrio) or pyproj,
however some of the geopandas tests depend on read_file working (with crs).
It is possible to avoid this dependence by converting the test data into
geojson which can be read natively (along with crs and dtypes explicitly passed
through).
"""

import json
from pathlib import Path

import geopandas as gpd
from geopandas.testing import assert_geodataframe_equal
from geopandas.tests.util import (
    _GEOJSON_CONVERTED_DATA_DIR,
    _TEST_DATA_DIR,
    get_test_df_from_geojson,
)


def convert_dataset_to_geojson(input_path: Path) -> None:
    gdf_io_engine = gpd.read_file(input_path)
    input_filename = input_path.stem
    geojson_path = _GEOJSON_CONVERTED_DATA_DIR / f"{input_filename}.geojson"
    additional_info_path = (
        _GEOJSON_CONVERTED_DATA_DIR / f"{input_filename}_additional.json"
    )
    geojson_path.write_text(gdf_io_engine.to_json(drop_id=True))
    additional_info_path.write_text(
        json.dumps(
            {
                "crs": str(gdf_io_engine.crs),
                "dtypes": gdf_io_engine.dtypes.astype("string").to_dict(),
            }
        )
    )

    # Read written info and make sure we can round trip it
    gdf = get_test_df_from_geojson(
        input_filename
    )  # function also expected to read from _GEOJSON_CONVERTED_DATA_DIR
    assert_geodataframe_equal(gdf, gdf_io_engine)


if __name__ == "__main__":
    _GEOJSON_CONVERTED_DATA_DIR.mkdir(exist_ok=True)
    datasets = [
        Path(_TEST_DATA_DIR) / "nybb_16a.zip",
        Path(_TEST_DATA_DIR) / "naturalearth_lowres/naturalearth_lowres.shp",
        Path(_TEST_DATA_DIR) / "naturalearth_cities/naturalearth_cities.shp",
    ]
    for dataset in datasets:
        convert_dataset_to_geojson(dataset)
