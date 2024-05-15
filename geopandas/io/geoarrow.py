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


def _convert_inner_coords(coords, interleaved, dims, mask=None):
    if interleaved:
        coords_field = pa.field(
            "xy" if len(dims) == 2 else "xyz", pa.float64()
        )  # , nullable=False)
        typ = pa.list_(coords_field, len(dims))
        parr = pa.FixedSizeListArray.from_arrays(coords.flatten(), type=typ, mask=mask)
    else:
        if dims == CoordinateDimension.XY:
            parr = pa.StructArray.from_arrays(
                [coords[:, 0].copy(), coords[:, 1].copy()], names=["x", "y"], mask=mask
            )
        else:
            parr = pa.StructArray.from_arrays(
                [coords[:, 0].copy(), coords[:, 1].copy(), coords[:, 2].copy()],
                names=["x", "y", "z"],
                mask=mask,
            )
    return parr


def _linestring_type(point_type):
    return pa.list_(pa.field("vertices", point_type))  # , nullable=False))


def _polygon_type(point_type):
    return pa.list_(
        pa.field(
            "rings", pa.list_(pa.field("vertices", point_type))  # , nullable=False)
        )  # , nullable=False)
    )


def _multipoint_type(point_type):
    return pa.list_(pa.field("points", point_type))  # , nullable=False))


def _multilinestring_type(point_type):
    return pa.list_(
        pa.field("linestrings", _linestring_type(point_type))  # , nullable=False)
    )


def _multipolygon_type(point_type):
    return pa.list_(
        pa.field("polygons", _polygon_type(point_type))  # , nullable=False)
    )


def construct_geometry_array(
    shapely_arr: NDArray[np.object_],
    include_z: Optional[bool] = None,
    *,
    field_name: str = "geometry",
    crs_str: Optional[str] = None,
    interleaved: bool = True,
) -> Tuple[pa.Field, pa.Array]:
    # NOTE: this implementation returns a (field, array) pair so that it can set the
    # extension metadata on the field without instantiating extension types into the
    # global pyarrow registry
    geom_type, coords, offsets = shapely.to_ragged_array(
        shapely_arr, include_z=include_z
    )

    mask = shapely.is_missing(shapely_arr)
    if mask.any():
        mask = pa.array(mask, type=pa.bool_())
    else:
        mask = None

    if coords.shape[-1] == 2:
        dims = CoordinateDimension.XY
    elif coords.shape[-1] == 3:
        dims = CoordinateDimension.XYZ
    else:
        raise ValueError(f"Unexpected coords dimensions: {coords.shape}")

    extension_metadata: Dict[str, str] = {}
    if crs_str is not None:
        extension_metadata["ARROW:extension:metadata"] = json.dumps({"crs": crs_str})
    else:
        # TODO we shouldn't do this, just to get testing passed for now
        extension_metadata["ARROW:extension:metadata"] = "{}"

    if geom_type == GeometryType.POINT:
        # TODO support shapely < 2.0.4 for missing values
        parr = _convert_inner_coords(coords, interleaved, dims, mask=mask)
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
        _parr = _convert_inner_coords(coords, interleaved, dims)
        parr = pa.ListArray.from_arrays(
            pa.array(geom_offsets), _parr, _linestring_type(_parr.type), mask=mask
        )
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
        _parr = _convert_inner_coords(coords, interleaved, dims)
        _parr1 = pa.ListArray.from_arrays(pa.array(ring_offsets), _parr)
        parr = pa.ListArray.from_arrays(
            pa.array(geom_offsets), _parr1, type=_polygon_type(_parr.type), mask=mask
        )
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
        _parr = _convert_inner_coords(coords, interleaved, dims)
        parr = pa.ListArray.from_arrays(
            pa.array(geom_offsets), _parr, type=_multipoint_type(_parr.type), mask=mask
        )
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
        _parr = _convert_inner_coords(coords, interleaved, dims)
        _parr1 = pa.ListArray.from_arrays(pa.array(ring_offsets), _parr)
        parr = pa.ListArray.from_arrays(pa.array(geom_offsets), _parr1, mask=mask)
        parr = parr.cast(_multilinestring_type(_parr.type))
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
        _parr = _convert_inner_coords(coords, interleaved, dims)
        _parr1 = pa.ListArray.from_arrays(pa.array(ring_offsets), _parr)
        _parr2 = pa.ListArray.from_arrays(pa.array(polygon_offsets), _parr1)
        parr = pa.ListArray.from_arrays(pa.array(geom_offsets), _parr2, mask=mask)
        parr = parr.cast(_multipolygon_type(_parr.type))
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
