import json
from packaging.version import Version
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
from numpy.typing import NDArray

import shapely
from shapely import GeometryType

from geopandas import GeoDataFrame
from geopandas._compat import SHAPELY_GE_204
from geopandas.array import from_shapely, from_wkb

GEOARROW_ENCODINGS = [
    "point",
    "linestring",
    "polygon",
    "multipoint",
    "multilinestring",
    "multipolygon",
]


## GeoPandas -> GeoArrow


class ArrowTable:
    """
    Wrapper class for Arrow data.

    This class implements the `Arrow PyCapsule Protocol`_ (i.e. having an
    ``__arrow_c_stream__`` method). This object can then be consumed by
    your Arrow implementation of choice that supports this protocol.

    .. _Arrow PyCapsule Protocol: https://arrow.apache.org/docs/format/CDataInterface/PyCapsuleInterface.html

    Example
    -------
    >>> import pyarrow as pa
    >>> pa.table(gdf.to_arrow())  # doctest: +SKIP

    """

    def __init__(self, pa_table):
        self._pa_table = pa_table

    def __arrow_c_stream__(self, requested_schema=None):
        return self._pa_table.__arrow_c_stream__(requested_schema=requested_schema)


class GeoArrowArray:
    """
    Wrapper class for a geometry array as Arrow data.

    This class implements the `Arrow PyCapsule Protocol`_ (i.e. having an
    ``__arrow_c_array/stream__`` method). This object can then be consumed by
    your Arrow implementation of choice that supports this protocol.

    .. _Arrow PyCapsule Protocol: https://arrow.apache.org/docs/format/CDataInterface/PyCapsuleInterface.html

    Example
    -------
    >>> import pyarrow as pa
    >>> pa.array(ser.to_arrow())  # doctest: +SKIP

    """

    def __init__(self, pa_field, pa_array):
        self._pa_array = pa_array
        self._pa_field = pa_field

    def __arrow_c_array__(self, requested_schema=None):
        if requested_schema is not None:
            raise NotImplementedError(
                "Requested schema is not supported for geometry arrays"
            )
        return (
            self._pa_field.__arrow_c_schema__(),
            self._pa_array.__arrow_c_array__()[1],
        )


def geopandas_to_arrow(
    df,
    index=None,
    geometry_encoding="WKB",
    interleaved=True,
    include_z=None,
):
    """
    Convert GeoDataFrame to a pyarrow.Table.

    Parameters
    ----------
    df : GeoDataFrame
        The GeoDataFrame to convert.
    index : bool, default None
        If ``True``, always include the dataframe's index(es) as columns
        in the file output.
        If ``False``, the index(es) will not be written to the file.
        If ``None``, the index(ex) will be included as columns in the file
        output except `RangeIndex` which is stored as metadata only.
    geometry_encoding : {'WKB', 'geoarrow' }, default 'WKB'
        The GeoArrow encoding to use for the data conversion.
    interleaved : bool, default True
        Only relevant for 'geoarrow' encoding. If True, the geometries'
        coordinates are interleaved in a single fixed size list array.
        If False, the coordinates are stored as separate arrays in a
        struct type.
    include_z : bool, default None
        Only relevant for 'geoarrow' encoding (for WKB, the dimensionality
        of the individial geometries is preserved).
        If False, return 2D geometries. If True, include the third dimension
        in the output (if a geometry has no third dimension, the z-coordinates
        will be NaN). By default, will infer the dimensionality from the
        input geometries. Note that this inference can be unreliable with
        empty geometries (for a guaranteed result, it is recommended to
        specify the keyword).

    """
    mask = df.dtypes == "geometry"
    geometry_columns = df.columns[mask]
    geometry_indices = np.asarray(mask).nonzero()[0]

    df_attr = pd.DataFrame(df.copy(deep=False))

    # replace geometry columns with dummy values -> will get converted to
    # Arrow null column (not holding any memory), so we can afterwards
    # fill the resulting table with the correct geometry fields
    for col in geometry_columns:
        df_attr[col] = None

    table = pa.Table.from_pandas(df_attr, preserve_index=index)

    geometry_encoding_dict = {}

    if geometry_encoding.lower() == "geoarrow":
        if Version(pa.__version__) < Version("10.0.0"):
            raise ValueError("Converting to 'geoarrow' requires pyarrow >= 10.0.")

        # Encode all geometry columns to GeoArrow
        for i, col in zip(geometry_indices, geometry_columns):
            field, geom_arr = construct_geometry_array(
                np.array(df[col].array),
                include_z=include_z,
                field_name=col,
                crs=df[col].crs,
                interleaved=interleaved,
            )
            table = table.set_column(i, field, geom_arr)
            geometry_encoding_dict[col] = (
                field.metadata[b"ARROW:extension:name"]
                .decode()
                .removeprefix("geoarrow.")
            )

    elif geometry_encoding.lower() == "wkb":
        # Encode all geometry columns to WKB
        for i, col in zip(geometry_indices, geometry_columns):
            field, wkb_arr = construct_wkb_array(
                np.asarray(df[col].array), field_name=col, crs=df[col].crs
            )
            table = table.set_column(i, field, wkb_arr)
            geometry_encoding_dict[col] = "WKB"

    else:
        raise ValueError(
            f"Expected geometry encoding 'WKB' or 'geoarrow' got {geometry_encoding}"
        )
    return table, geometry_encoding_dict


def construct_wkb_array(
    shapely_arr: NDArray[np.object_],
    *,
    field_name: str = "geometry",
    crs: Optional[str] = None,
) -> Tuple[pa.Field, pa.Array]:

    if shapely.geos_version > (3, 10, 0):
        kwargs = {"flavor": "iso"}
    else:
        if shapely.has_z(shapely_arr).any():
            raise ValueError("Cannot write 3D geometries with GEOS<3.10")
        kwargs = {}

    wkb_arr = shapely.to_wkb(shapely_arr, **kwargs)
    extension_metadata = {"ARROW:extension:name": "geoarrow.wkb"}
    if crs is not None:
        extension_metadata["ARROW:extension:metadata"] = json.dumps(
            {"crs": crs.to_json()}
        )
    else:
        # In theory this should not be needed, but otherwise pyarrow < 17
        # crashes on receiving such data through C Data Interface
        # https://github.com/apache/arrow/issues/41741
        extension_metadata["ARROW:extension:metadata"] = "{}"

    field = pa.field(
        field_name, type=pa.binary(), nullable=True, metadata=extension_metadata
    )
    parr = pa.array(np.asarray(wkb_arr), pa.binary())
    return field, parr


def _convert_inner_coords(coords, interleaved, dims, mask=None):
    if interleaved:
        coords_field = pa.field(dims, pa.float64(), nullable=False)
        typ = pa.list_(coords_field, len(dims))
        if mask is None:
            # mask keyword only added in pyarrow 15.0.0
            parr = pa.FixedSizeListArray.from_arrays(coords.ravel(), type=typ)
        else:
            parr = pa.FixedSizeListArray.from_arrays(
                coords.ravel(), type=typ, mask=mask
            )
    else:
        if dims == "xy":
            fields = [
                pa.field("x", pa.float64(), nullable=False),
                pa.field("y", pa.float64(), nullable=False),
            ]
            parr = pa.StructArray.from_arrays(
                [coords[:, 0].copy(), coords[:, 1].copy()], fields=fields, mask=mask
            )
        else:
            fields = [
                pa.field("x", pa.float64(), nullable=False),
                pa.field("y", pa.float64(), nullable=False),
                pa.field("z", pa.float64(), nullable=False),
            ]
            parr = pa.StructArray.from_arrays(
                [coords[:, 0].copy(), coords[:, 1].copy(), coords[:, 2].copy()],
                fields=fields,
                mask=mask,
            )
    return parr


def _linestring_type(point_type):
    return pa.list_(pa.field("vertices", point_type, nullable=False))


def _polygon_type(point_type):
    return pa.list_(
        pa.field(
            "rings",
            pa.list_(pa.field("vertices", point_type, nullable=False)),
            nullable=False,
        )
    )


def _multipoint_type(point_type):
    return pa.list_(pa.field("points", point_type, nullable=False))


def _multilinestring_type(point_type):
    return pa.list_(
        pa.field("linestrings", _linestring_type(point_type), nullable=False)
    )


def _multipolygon_type(point_type):
    return pa.list_(pa.field("polygons", _polygon_type(point_type), nullable=False))


def construct_geometry_array(
    shapely_arr: NDArray[np.object_],
    include_z: Optional[bool] = None,
    *,
    field_name: str = "geometry",
    crs: Optional[str] = None,
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
        if (
            geom_type == GeometryType.POINT
            and interleaved
            and Version(pa.__version__) < Version("15.0.0")
        ):
            raise ValueError(
                "Converting point geometries with missing values is not supported "
                "for interleaved coordinates with pyarrow < 15.0.0. Please "
                "upgrade to a newer version of pyarrow."
            )
        mask = pa.array(mask, type=pa.bool_())

        if geom_type == GeometryType.POINT and not SHAPELY_GE_204:
            # bug in shapely < 2.0.4, see https://github.com/shapely/shapely/pull/2034
            # this workaround only works if there are no empty points
            indices = np.nonzero(mask)[0]
            indices = indices - np.arange(len(indices))
            coords = np.insert(coords, indices, np.nan, axis=0)

    else:
        mask = None

    if coords.shape[-1] == 2:
        dims = "xy"
    elif coords.shape[-1] == 3:
        dims = "xyz"
    else:
        raise ValueError(f"Unexpected coords dimensions: {coords.shape}")

    extension_metadata: Dict[str, str] = {}
    if crs is not None:
        extension_metadata["ARROW:extension:metadata"] = json.dumps(
            {"crs": crs.to_json()}
        )
    else:
        # In theory this should not be needed, but otherwise pyarrow < 17
        # crashes on receiving such data through C Data Interface
        # https://github.com/apache/arrow/issues/41741
        extension_metadata["ARROW:extension:metadata"] = "{}"

    if geom_type == GeometryType.POINT:
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
        parr = pa.ListArray.from_arrays(pa.array(geom_offsets), _parr1, mask=mask)
        parr = parr.cast(_polygon_type(_parr.type))
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


## GeoArrow -> GeoPandas


def _get_arrow_geometry_field(field):
    if (meta := field.metadata) is not None:
        if (ext_name := meta.get(b"ARROW:extension:name", None)) is not None:
            if ext_name.startswith(b"geoarrow."):
                if (
                    ext_meta := meta.get(b"ARROW:extension:metadata", None)
                ) is not None:
                    ext_meta = json.loads(ext_meta.decode())
                return ext_name.decode(), ext_meta

    if isinstance(field.type, pa.ExtensionType):
        ext_name = field.type.extension_name
        if ext_name.startswith("geoarrow."):
            ext_meta_ser = field.type.__arrow_ext_serialize__()
            if ext_meta_ser:
                ext_meta = json.loads(ext_meta_ser.decode())
            else:
                ext_meta = None
            return ext_name, ext_meta

    return None


def arrow_to_geopandas(table, geometry=None):
    """
    Convert Arrow table object to a GeoDataFrame based on GeoArrow extension types.

    Parameters
    ----------
    table : pyarrow.Table
        The Arrow table to convert.
    geometry : str, default None
        The name of the geometry column to set as the active geometry
        column. If None, the first geometry column found will be used.

    Returns
    -------
    GeoDataFrame

    """
    if not isinstance(table, pa.Table):
        table = pa.table(table)

    geom_fields = []

    for i, field in enumerate(table.schema):
        geom = _get_arrow_geometry_field(field)
        if geom is not None:
            geom_fields.append((i, field.name, *geom))

    if len(geom_fields) == 0:
        raise ValueError("No geometry column found in the Arrow table.")

    table_attr = table.drop([f[1] for f in geom_fields])
    df = table_attr.to_pandas()

    for i, col, ext_name, ext_meta in geom_fields:
        crs = None
        if ext_meta is not None and "crs" in ext_meta:
            crs = ext_meta["crs"]

        if ext_name == "geoarrow.wkb":
            geom_arr = from_wkb(np.array(table[col]), crs=crs)
        elif ext_name.split(".")[1] in GEOARROW_ENCODINGS:

            geom_arr = from_shapely(
                construct_shapely_array(table[col].combine_chunks(), ext_name), crs=crs
            )
        else:
            raise TypeError(f"Unknown GeoArrow extension type: {ext_name}")

        df.insert(i, col, geom_arr)

    return GeoDataFrame(df, geometry=geometry or geom_fields[0][1])


def arrow_to_geometry_array(arr):
    """
    Convert Arrow array object (representing single GeoArrow array) to a
    geopandas GeometryArray.

    Specifically for GeoSeries.from_arrow.
    """
    if Version(pa.__version__) < Version("14.0.0"):
        raise ValueError("Importing from Arrow requires pyarrow >= 14.0.")

    schema_capsule, array_capsule = arr.__arrow_c_array__()
    field = pa.Field._import_from_c_capsule(schema_capsule)
    pa_arr = pa.Array._import_from_c_capsule(field.__arrow_c_schema__(), array_capsule)

    geom_info = _get_arrow_geometry_field(field)
    if geom_info is None:
        raise ValueError("No GeoArrow geometry field found.")
    ext_name, ext_meta = geom_info

    crs = None
    if ext_meta is not None and "crs" in ext_meta:
        crs = ext_meta["crs"]

    if ext_name == "geoarrow.wkb":
        geom_arr = from_wkb(np.array(pa_arr), crs=crs)
    elif ext_name.split(".")[1] in GEOARROW_ENCODINGS:

        geom_arr = from_shapely(construct_shapely_array(pa_arr, ext_name), crs=crs)
    else:
        raise ValueError(f"Unknown GeoArrow extension type: {ext_name}")

    return geom_arr


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
    if isinstance(arr, pa.ExtensionArray):
        arr = arr.storage

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
