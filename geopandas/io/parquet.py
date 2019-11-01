import json
import warnings

from pandas.io.common import get_filepath_or_buffer
from pandas.io.parquet import PyArrowImpl as PandasPyArrowImpl
from pandas import DataFrame

from geopandas.array import from_wkb, to_wkb
from geopandas import GeoDataFrame
import geopandas


PARQUET_METADATA_VERSION = "1.0.0"


def _encode_crs(crs):
    if isinstance(crs, str):
        return {"proj4": crs}

    return crs


def _decode_crs(crs):
    if crs and "proj4" in crs:
        return crs["proj4"]

    return crs


class PyArrowImpl(PandasPyArrowImpl):
    """Extension of Pandas PyArrowImpl to handle serializing geometries to WKB.
    """

    def write(
        self,
        df,
        path,
        compression="snappy",
        coerce_timestamps="ms",
        index=None,
        **kwargs
    ):
        """Write data frame to a parquet file, serializing any geometry
        columns to WKB format.

        See docstring for ``to_parquet``.
        """

        self.validate_dataframe(df)

        geometry_columns = df.columns[df.dtypes == "geometry"].tolist()
        geometry_column = df._geometry_column_name
        crs = _encode_crs(df.crs)

        # Convert to a DataFrame so we can convert geometries to WKB while
        # retaining original column names
        df = DataFrame(df.copy())

        path, _, _, _ = get_filepath_or_buffer(path, mode="wb")

        # Encode all geometry columns to WKB
        for col in geometry_columns:
            df[col] = to_wkb(df[col].values)

        if index is None:
            from_pandas_kwargs = {}
        else:
            from_pandas_kwargs = {"preserve_index": index}

        table = self.api.Table.from_pandas(df, **from_pandas_kwargs)

        metadata = table.schema.metadata

        # Store geopandas specific metadata
        metadata.update(
            {
                "geo": json.dumps(
                    {
                        "crs": crs,
                        "primary_column": geometry_column,
                        "columns": geometry_columns,
                        "schema_version": PARQUET_METADATA_VERSION,
                        "creator": {
                            "library": "geopandas",
                            "version": geopandas.__version__,
                        },
                    }
                ).encode("utf-8")
            }
        )
        table = table.replace_schema_metadata(metadata)

        self.api.parquet.write_table(
            table,
            path,
            compression=compression,
            coerce_timestamps=coerce_timestamps,
            **kwargs
        )

    def read(
        self,
        path,
        columns=None,
        geometry_columns=None,
        geometry=None,
        crs=None,
        **kwargs
    ):
        """Read from a parquet file into a GeoDataFrame, converting any geometry
        columns present from WKB to GeoSeries.

        See docstring for ``from_parquet``.
        """

        path, _, _, should_close = get_filepath_or_buffer(path)

        kwargs["use_pandas_metadata"] = True
        table = self.api.parquet.read_table(path, columns=columns, **kwargs)

        df = table.to_pandas()

        if should_close:
            try:
                path.close()
            except:  # noqa: flake8
                pass

        metadata = table.schema.metadata

        try:
            geo_metadata = json.loads(metadata.get(b"geo", b"").decode("utf-8"))

        except (TypeError, json.decoder.JSONDecodeError):
            warnings.warn(
                "Missing or malformed geo metadata in parquet file", UserWarning
            )
            # raise ValueError("Missing or malformed geo metadata in parquet file")
            geo_metadata = None

        if geo_metadata:
            # Validate that required keys are present
            required_keys = ("crs", "primary_column", "columns")
            for key in required_keys:
                if key not in geo_metadata:
                    raise ValueError(
                        "Geo metadata missing required key: {}".format(key)
                    )

            if geometry_columns is None:
                geometry_columns = df.columns.intersection(geo_metadata["columns"])

            if geometry is None:
                geometry = geo_metadata["primary_column"]

                # Missing geometry likely indicates a subset of columns was read;
                # promote the first available geometry to the primary geometry.
                if len(geometry_columns) and geometry not in geometry_columns:
                    geometry = geometry_columns[0]

            if crs is None:
                crs = _decode_crs(geo_metadata["crs"])

        elif geometry_columns is None:
            raise ValueError(
                """geometry_columns must be specified;
                these are not available from the metadata of the parquet file."""
            )

        missing_cols = set(geometry_columns).difference(df.columns)
        if len(missing_cols):
            raise ValueError(
                """The parquet file does not include the specified
                geometry_columns: {}""".format(
                    missing_cols
                )
            )

        if not len(geometry_columns):
            raise ValueError(
                """No geometry columns are included in the columns read from
                the parquet file."""
            )

            if geometry is None:
                # promote first available geometry column
                geometry = geometry_columns[0]

            elif geometry not in geometry_columns:
                # The user asked for a geometry that wasn't available;
                # it would be a problem to promote a different column
                # in this case.
                raise ValueError(
                    """The specified geometry column was not read from the
                    parquet file: {}""".format(
                        geometry
                    )
                )

        # Convert the WKB columns that are present back to geometry
        for col in geometry_columns:
            df[col] = from_wkb(df[col].values)

        return GeoDataFrame(df, geometry=geometry, crs=crs)


def to_parquet(df, path, compression="snappy", index=None, **kwargs):
    """
    Write a GeoDataFrame to the parquet format.

    Any geometry columns present are serialized to WKB format in the file.

    Requires 'pyarrow'.

    Parameters
    ----------
    path : str
        File path or Root Directory path. Will be used as Root Directory path
        while writing a partitioned dataset.
    compression : {'snappy', 'gzip', 'brotli', None}, default 'snappy'
        Name of the compression to use. Use ``None`` for no compression.
    index : bool, default None
        If ``True``, include the dataframe's index(es) in the file output. If
        ``False``, they will not be written to the file. If ``None``, the
        engine's default behavior will be used.
    kwargs
        Additional keyword arguments passed to the engine
    """

    impl = PyArrowImpl()
    return impl.write(df, path, compression=compression, index=index, **kwargs)


def read_parquet(
    path, columns=None, geometry_columns=None, geometry=None, crs=None, **kwargs
):
    """
    Load a parquet object from the file path, returning a GeoDataFrame.

    Either the list of geometry columns must be available in the metadata of the parquet
    file or it must be passed in as a parameter.

    You can read a subset of columns in the file using the ``columns`` parameter.
    However, the structure of the returned GeoDataFrame will depend on which
    columns you read:
    * if no geometry columns are read, this will raise a ``ValueError`` - you
      should use the pandas `read_parquet` method instead.
    * if the primary geometry column saved to this file is not read, the first
      available geometry column will be set as the geometry column of the
      returned GeoDataFrame.

    Requires 'pyarrow'.

    Parameters
    ----------
    path : str, path object or file-like object
        Any valid string path is acceptable. The string could be a URL. Valid
        URL schemes include http, ftp, s3, and file. For file URLs, a host is
        expected. A local file could be:
        ``file://localhost/path/to/table.parquet``.

        If you want to pass in a path object, geopandas accepts any
        ``os.PathLike``.

        By file-like object, we refer to objects with a ``read()`` method,
        such as a file handler (e.g. via builtin ``open`` function)
        or ``StringIO``.
    columns : list-like of strings, default=None
        If not None, only these columns will be read from the file.  If
        the primary geometry column is not included, the first secondary
        geometry read from the file will be set as the geometry column
        of the returned GeoDataFrame.  If no geometry columns are present,
        a ``ValueError`` will be raised.
    geometry_columns : list-like of strings, default=None
        If not None, these columns will be extracted as geometries;
        this will take precedence over the list stored in the metadata
        of the parquet file.
        By default, the list of geometry columns is read from the
        metadata of the parquet file.
    geometry : string, default=None
        If not None, is the name of the primary geometry column;
        this will take precedence over the name stored in the metadata
        of the parquet file.
        By default, this column name is read from the metadata of the
        parquet file.
    crs : valid GeoPandas CRS object, default=None
        If not None, will be set as the crs of the returned GeoDataFrame;
        this will take precedence over the value stored in the metadata
        of the parquet file.
        By default, the crs is read from the metadata of the parquet file.
    **kwargs
        Any additional kwargs are passed to the engine.

    Returns
    -------
    GeoDataFrame
    """

    impl = PyArrowImpl()
    return impl.read(
        path,
        columns=columns,
        geometry_columns=geometry_columns,
        geometry=geometry,
        crs=crs,
        **kwargs
    )
