import json
from enum import Enum
from typing import Dict, Optional, Tuple

import numpy as np
import pyarrow as pa
import shapely
from numpy.typing import NDArray
from shapely import GeometryType


class CoordinateDimension(str, Enum):
    XY = "xy"
    XYZ = "xyz"
    XYM = "xym"
    XYZM = "xyzm"


def construct_geometry_array(
    shapely_arr: NDArray[np.object_],
    include_z: Optional[bool] = None,
    *,
    field_name: str = "geometry",
    crs_str: Optional[str] = None,
) -> Tuple[pa.Field, pa.Array]:
    # NOTE: this implementation returns a (field, array) pair so that it can set the
    # extension metadata on the field without instantiating extension types into the
    # global pyarrow registry
    geom_type, coords, offsets = shapely.to_ragged_array(
        shapely_arr, include_z=include_z
    )

    if coords.shape[-1] == 2:
        dims = CoordinateDimension.XY
    elif coords.shape[-1] == 3:
        dims = CoordinateDimension.XYZ
    else:
        raise ValueError(f"Unexpected coords dimensions: {coords.shape}")

    extension_metadata: Dict[str, str] = {}
    if crs_str is not None:
        extension_metadata["ARROW:extension:metadata"] = json.dumps({"crs": crs_str})

    if geom_type == GeometryType.POINT:
        parr = pa.FixedSizeListArray.from_arrays(coords.flatten(), len(dims))
        extension_metadata["ARROW:extension:name"] = "geoarrow.point"
        field = pa.field(
            field_name,
            parr.type,
            nullable=True,
            metadata=extension_metadata,
        )
        return field, parr

    elif geom_type == GeometryType.LINESTRING:
        assert len(offsets) == 1, "Expected one offsets array"
        (geom_offsets,) = offsets
        _parr = pa.FixedSizeListArray.from_arrays(coords.flatten(), len(dims))
        parr = pa.ListArray.from_arrays(pa.array(geom_offsets), _parr)
        extension_metadata["ARROW:extension:name"] = "geoarrow.linestring"
        field = pa.field(
            field_name,
            parr.type,
            nullable=True,
            metadata=extension_metadata,
        )
        return field, parr

    elif geom_type == GeometryType.POLYGON:
        assert len(offsets) == 2, "Expected two offsets arrays"
        ring_offsets, geom_offsets = offsets
        _parr = pa.FixedSizeListArray.from_arrays(coords.flatten(), len(dims))
        _parr1 = pa.ListArray.from_arrays(pa.array(ring_offsets), _parr)
        parr = pa.ListArray.from_arrays(pa.array(geom_offsets), _parr1)
        extension_metadata["ARROW:extension:name"] = "geoarrow.polygon"
        field = pa.field(
            field_name,
            parr.type,
            nullable=True,
            metadata=extension_metadata,
        )
        return field, parr

    elif geom_type == GeometryType.MULTIPOINT:
        assert len(offsets) == 1, "Expected one offsets array"
        (geom_offsets,) = offsets
        _parr = pa.FixedSizeListArray.from_arrays(coords.flatten(), len(dims))
        parr = pa.ListArray.from_arrays(pa.array(geom_offsets), _parr)
        extension_metadata["ARROW:extension:name"] = "geoarrow.multipoint"
        field = pa.field(
            field_name,
            parr.type,
            nullable=True,
            metadata=extension_metadata,
        )
        return field, parr

    elif geom_type == GeometryType.MULTILINESTRING:
        assert len(offsets) == 2, "Expected two offsets arrays"
        ring_offsets, geom_offsets = offsets
        _parr = pa.FixedSizeListArray.from_arrays(coords.flatten(), len(dims))
        _parr1 = pa.ListArray.from_arrays(pa.array(ring_offsets), _parr)
        parr = pa.ListArray.from_arrays(pa.array(geom_offsets), _parr1)
        extension_metadata["ARROW:extension:name"] = "geoarrow.multilinestring"
        field = pa.field(
            field_name,
            parr.type,
            nullable=True,
            metadata=extension_metadata,
        )
        return field, parr

    elif geom_type == GeometryType.MULTIPOLYGON:
        assert len(offsets) == 3, "Expected three offsets arrays"
        ring_offsets, polygon_offsets, geom_offsets = offsets
        _parr = pa.FixedSizeListArray.from_arrays(coords.flatten(), len(dims))
        _parr1 = pa.ListArray.from_arrays(pa.array(ring_offsets), _parr)
        _parr2 = pa.ListArray.from_arrays(pa.array(polygon_offsets), _parr1)
        parr = pa.ListArray.from_arrays(pa.array(geom_offsets), _parr2)
        extension_metadata["ARROW:extension:name"] = "geoarrow.multipolygon"
        field = pa.field(
            field_name,
            parr.type,
            nullable=True,
            metadata=extension_metadata,
        )
        return field, parr

    else:
        raise ValueError(f"Unsupported type for geoarrow: {geom_type}")
