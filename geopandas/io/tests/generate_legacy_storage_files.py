"""
Script to create the data and write legacy storage (pickle) files.

Based on pandas' generate_legacy_storage_files.py script.

To use this script, create an environment for which you want to
generate pickles, activate the environment, and run this script as:

$ python geopandas/geopandas/io/tests/generate_legacy_storage_files.py \
    geopandas/geopandas/io/tests/data/pickle/ pickle

This script generates a storage file for the current arch, system,

The idea here is you are using the *current* version of the
generate_legacy_storage_files with an *older* version of geopandas to
generate a pickle file. We will then check this file into a current
branch, and test using test_pickle.py. This will load the *older*
pickles and test versus the current data that is generated
(with master). These are then compared.

"""
import os
import pickle
import platform
import sys

import pandas as pd

import geopandas
from shapely.geometry import Point


def create_pickle_data():
    """create the pickle data"""

    # custom geometry column name
    gdf_the_geom = geopandas.GeoDataFrame(
        {"a": [1, 2, 3], "the_geom": [Point(1, 1), Point(2, 2), Point(3, 3)]},
        geometry="the_geom",
    )

    # with crs
    gdf_crs = geopandas.GeoDataFrame(
        {"a": [0.1, 0.2, 0.3], "geometry": [Point(1, 1), Point(2, 2), Point(3, 3)]},
        crs="EPSG:4326",
    )

    return {"gdf_the_geom": gdf_the_geom, "gdf_crs": gdf_crs}


def platform_name():
    return "_".join(
        [
            str(geopandas.__version__),
            "pd-" + str(pd.__version__),
            "py-" + str(platform.python_version()),
            str(platform.machine()),
            str(platform.system().lower()),
        ]
    )


def write_legacy_pickles(output_dir):
    print(
        "This script generates a storage file for the current arch, system, "
        "and python version"
    )
    print("geopandas version: {}").format(geopandas.__version__)
    print("   output dir    : {}".format(output_dir))
    print("   storage format: pickle")

    pth = "{}.pickle".format(platform_name())

    fh = open(os.path.join(output_dir, pth), "wb")
    pickle.dump(create_pickle_data(), fh, pickle.DEFAULT_PROTOCOL)
    fh.close()

    print("created pickle file: {}".format(pth))


def main():
    if len(sys.argv) != 3:
        sys.exit(
            "Specify output directory and storage type: generate_legacy_"
            "storage_files.py <output_dir> <storage_type> "
        )

    output_dir = str(sys.argv[1])
    storage_type = str(sys.argv[2])

    if storage_type == "pickle":
        write_legacy_pickles(output_dir=output_dir)
    else:
        sys.exit("storage_type must be one of {'pickle'}")


if __name__ == "__main__":
    main()
