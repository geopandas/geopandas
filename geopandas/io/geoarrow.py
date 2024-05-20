import json

import numpy as np
import pyarrow as pa

import shapely
from shapely import GeometryType

from geopandas import GeoDataFrame
from geopandas.array import from_shapely, from_wkb

GEOARROW_ENCODINGS = [
    "point",
    "linestring",
    "polygon",
    "multipoint",
    "multilinestring",
    "multipolygon",
]


def get_arrow_geometry_field(field):
    if (meta := field.metadata) is not None:
        if (ext_name := meta.get(b"ARROW:extension:name", None)) is not None:
            if ext_name.startswith(b"geoarrow."):
                if (
                    ext_meta := meta.get(b"ARROW:extension:metadata", None)
                ) is not None:
                    ext_meta = json.loads(ext_meta.decode())
                return ext_name.decode(), ext_meta
    return None


def arrow_to_geopandas(table):
    """
    Convert pyarrow.Table to a GeoDataFrame based on GeoArrow extension types.
    """
    geom_fields = []

    for field in table.schema:
        geom = get_arrow_geometry_field(field)
        if geom is not None:
            geom_fields.append((field.name, *geom))

    df = table.to_pandas()

    if len(geom_fields) == 0:
        raise ValueError("""Missing geometry columns.""")

    for col, ext_name, ext_meta in geom_fields:
        crs = None
        if ext_meta is not None and "crs" in ext_meta:
            crs = ext_meta["crs"]

        if ext_name == "geoarrow.wkb":
            df[col] = from_wkb(df[col].values, crs=crs)
        elif ext_name.split(".")[1] in GEOARROW_ENCODINGS:

            df[col] = from_shapely(
                construct_shapely_array(table[col].combine_chunks(), ext_name), crs=crs
            )
        else:
            raise ValueError(f"Unknown GeoArrow extension type: {ext_name}")

    return GeoDataFrame(df, geometry=geom_fields[0][0])


def _get_inner_coords(arr):
    if pa.types.is_struct(arr.type):
        if arr.type.num_fields == 2:
            coords = np.column_stack(
                [np.asarray(arr.field("x")), np.asarray(arr.field("y"))]
            )
        else:
            coords = np.column_stack(
                [
                    np.asarray(arr.field("x")),
                    np.asarray(arr.field("y")),
                    np.asarray(arr.field("z")),
                ]
            )
        return coords
    else:
        # fixed size list
        return np.asarray(arr.values).reshape(len(arr), -1)


def construct_shapely_array(arr: pa.Array, extension_name: str):
    """
    Construct a NumPy array of shapely geometries from a pyarrow.Array
    with GeoArrow extension type.

    """
    if extension_name == "geoarrow.point":
        coords = _get_inner_coords(arr)
        result = shapely.from_ragged_array(GeometryType.POINT, coords, None)

    elif extension_name == "geoarrow.linestring":
        coords = _get_inner_coords(arr.values)
        offsets1 = np.asarray(arr.offsets)
        offsets = (offsets1,)
        result = shapely.from_ragged_array(GeometryType.LINESTRING, coords, offsets)

    elif extension_name == "geoarrow.polygon":
        coords = _get_inner_coords(arr.values.values)
        offsets2 = np.asarray(arr.offsets)
        offsets1 = np.asarray(arr.values.offsets)
        offsets = (offsets1, offsets2)
        result = shapely.from_ragged_array(GeometryType.POLYGON, coords, offsets)

    elif extension_name == "geoarrow.multipoint":
        coords = _get_inner_coords(arr.values)
        offsets1 = np.asarray(arr.offsets)
        offsets = (offsets1,)
        result = shapely.from_ragged_array(GeometryType.MULTIPOINT, coords, offsets)

    elif extension_name == "geoarrow.multilinestring":
        coords = _get_inner_coords(arr.values.values)
        offsets2 = np.asarray(arr.offsets)
        offsets1 = np.asarray(arr.values.offsets)
        offsets = (offsets1, offsets2)
        result = shapely.from_ragged_array(
            GeometryType.MULTILINESTRING, coords, offsets
        )

    elif extension_name == "geoarrow.multipolygon":
        coords = _get_inner_coords(arr.values.values.values)
        offsets3 = np.asarray(arr.offsets)
        offsets2 = np.asarray(arr.values.offsets)
        offsets1 = np.asarray(arr.values.values.offsets)
        offsets = (offsets1, offsets2, offsets3)
        result = shapely.from_ragged_array(GeometryType.MULTIPOLYGON, coords, offsets)

    else:
        raise ValueError(extension_name)

    # apply validity mask
    if arr.null_count:
        mask = np.asarray(arr.is_null())
        result = np.where(mask, None, result)

    return result
