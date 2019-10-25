import json
import os
import warnings

from pandas.io.feather_format import (
    to_feather as _base_to_feather,
    read_feather as _base_read_feather,
)
from shapely.wkb import loads
from geopandas import GeoDataFrame


# TODO: CRS that is not dict or proj4, test w/ WKT


def to_feather(df, path):
    df = df.copy()

    geom_col = df._geometry_column_name

    # write the crs to an associated file
    if df.crs:
        with open("{}.crs".format(path), "w") as crsfile:
            crs = df.crs
            if isinstance(crs, str):
                crs = {"proj4": crs}
            crsfile.write(json.dumps(crs))

    df["_wkb"] = df[geom_col].apply(lambda g: g.wkb)
    _base_to_feather(df.drop(columns=[geom_col]), path)


def read_feather(path):
    crs = None
    crsfilename = "{}.crs".format(path)
    if os.path.exists(crsfilename):
        crs = json.loads(open(crsfilename).read())
        if "proj4" in crs:
            crs = crs["proj4"]
    else:
        warnings.warn(
            """{} coordinate reference system file is missing;
            no crs will be set for this GeoDataFrame.""".format(
                crsfilename
            )
        )

    df = _base_read_feather(path)
    df["geometry"] = df._wkb.apply(lambda wkb: loads(wkb))
    return GeoDataFrame(df.drop(columns=["_wkb"]), geometry="geometry", crs=crs)
