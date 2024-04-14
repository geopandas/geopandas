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

TEST_DATA_DIR = Path(__file__).parent


def convert_dataset_to_geojson(input_path: Path, output_folder: Path) -> None:
    df = gpd.read_file(input_path)
    input_filename = input_path.stem
    geojson_path = output_folder / f"{input_filename}.geojson"
    additional_info_path = output_path / f"{input_filename}_additional.json"
    geojson_path.write_text(df.to_json(drop_id=True))
    additional_info_path.write_text(
        json.dumps({"crs": str(df.crs), "dtypes": df.dtypes.astype("string").to_dict()})
    )

    # Read written info and make sure we can round trip it
    with open(geojson_path, "r") as f:
        raw_geojson = json.load(f)
    with open(additional_info_path, "r") as f2:
        addtional_info = json.load(f2)
    crs = addtional_info["crs"]
    dtypes = addtional_info["dtypes"]
    df = gpd.GeoDataFrame.from_features(raw_geojson, crs=crs)
    # fix the column order, geometry should always be last
    df = df[df.columns[df.columns != "geometry"].to_list() + ["geometry"]]
    # as geojson isn't precise with dtypes, pass these through explicitly
    df = df.astype(dtypes)
    # check data is equivalent
    assert_geodataframe_equal(df, df)


if __name__ == "__main__":
    output_path = TEST_DATA_DIR / "as_geojson"
    output_path.mkdir(exist_ok=True)
    datasets = [
        Path(TEST_DATA_DIR / "nybb_16a.zip"),
        Path(TEST_DATA_DIR / "naturalearth_lowres/naturalearth_lowres.shp"),
        Path(TEST_DATA_DIR / "naturalearth_cities/naturalearth_cities.shp"),
    ]
    for dataset in datasets:
        convert_dataset_to_geojson(dataset, output_path)
