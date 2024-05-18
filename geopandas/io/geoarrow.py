import numpy as np
import pyarrow as pa

import shapely
from shapely import GeometryType


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
