import json

from pandas.io.parquet import PyArrowImpl as PandasPyArrowImpl
from pandas.io.common import get_filepath_or_buffer
from pandas import get_option
from shapely.wkb import loads

from geopandas import GeoDataFrame, GeoSeries


def _wkb_name(column):
    return "__wkb_{}".format(column)


def _encode_crs(crs):
    if isinstance(crs, str):
        return {"proj4": crs}

    return crs


def _decode_crs(crs):
    return crs.get("proj4", crs)


def get_engine(engine):
    """ return our implementation """

    if engine == "auto":
        engine = get_option("io.parquet.engine")

    if engine == "auto":
        # try engines in this order
        try:
            return PyArrowImpl()
        except ImportError:
            pass

        # TODO:
        # try:
        #     return FastParquetImpl()
        # except ImportError:
        #     pass

        raise ImportError(
            "Unable to find a usable engine; "
            "tried using: 'pyarrow', 'fastparquet'.\n"
            "pyarrow or fastparquet is required for parquet "
            "support"
        )

    if engine not in ["pyarrow", "fastparquet"]:
        raise ValueError("engine must be one of 'pyarrow', 'fastparquet'")

    if engine == "pyarrow":
        return PyArrowImpl()
    elif engine == "fastparquet":
        raise NotImplementedError("TODO:")
        # return FastParquetImpl()


class PyArrowImpl(PandasPyArrowImpl):
    def write(
        self,
        df,
        path,
        compression="snappy",
        coerce_timestamps="ms",
        index=None,
        **kwargs
    ):
        self.validate_dataframe(df)

        df = df.copy()

        path, _, _, _ = get_filepath_or_buffer(path, mode="wb")

        # find all geometry columns and encode them to WKB
        geom_cols = df.dtypes.loc[df.dtypes == "geometry"].index.to_list()
        for col in geom_cols:
            df[_wkb_name(col)] = df[col].apply(lambda g: g.wkb)

        # drop geometry columns
        df = df.drop(columns=geom_cols)

        if index is None:
            from_pandas_kwargs = {}
        else:
            from_pandas_kwargs = {"preserve_index": index}

        table = self.api.Table.from_pandas(df, **from_pandas_kwargs)

        metadata = table.schema.metadata

        # New info needs to be JSON encoded UTF-8 string, same as pandas
        metadata.update(
            {
                "geo": json.dumps(
                    {
                        # capture CRS and store string value in an attribute
                        "crs": _encode_crs(df.crs),
                        "primary": df._geometry_column_name,
                        "columns": geom_cols,
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

    def read(self, path, columns=None, **kwargs):
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
            geo_metadata = json.loads(metadata.get(b"geo", ""))

        except (TypeError, json.decoder.JSONDecodeError):
            raise ValueError("Missing or malformed geo metadata in parquet file")

        # Validate that required keys are present
        required_keys = ("crs", "primary", "columns")
        for key in required_keys:
            if key not in geo_metadata:
                raise ValueError("Geo metadata missing required key: {}".format(key))

        # Convert the WKB columns back to geometry
        wkb_cols = []
        for col in geo_metadata["columns"]:
            wkb_col = _wkb_name(col)
            if wkb_col not in df.columns:
                raise ValueError(
                    "Missing geometry column from parquet file: {}".format(wkb_col)
                )

            df[col] = GeoSeries(df[wkb_col].apply(lambda wkb: loads(wkb)))
            wkb_cols.append(wkb_col)

        df = df.drop(columns=wkb_cols)

        return GeoDataFrame(
            df, geometry=geo_metadata["primary"], crs=_decode_crs(geo_metadata["crs"])
        )


def to_parquet(df, path, engine="auto", compression="snappy", index=None, **kwargs):
    """
    Write a DataFrame to the parquet format.

    Parameters
    ----------
    path : str
        File path or Root Directory path. Will be used as Root Directory path
        while writing a partitioned dataset.

        .. versionchanged:: 0.24.0

    engine : {'auto', 'pyarrow', 'fastparquet'}, default 'auto'
        Parquet library to use. If 'auto', then the option
        ``io.parquet.engine`` is used. The default ``io.parquet.engine``
        behavior is to try 'pyarrow', falling back to 'fastparquet' if
        'pyarrow' is unavailable.

    TODO: support fastparquet
    compression : {'snappy', 'gzip', 'brotli', None}, default 'snappy'
        Name of the compression to use. Use ``None`` for no compression.
    index : bool, default None
        If ``True``, include the dataframe's index(es) in the file output. If
        ``False``, they will not be written to the file. If ``None``, the
        engine's default behavior will be used.
    kwargs
        Additional keyword arguments passed to the engine
    """
    impl = get_engine(engine)
    return impl.write(df, path, compression=compression, index=index, **kwargs)


def read_parquet(path, engine="auto", columns=None, **kwargs):
    """
    Load a parquet object from the file path, returning a GeoDataFrame.

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
    engine : {'auto', 'pyarrow', 'fastparquet'}, default 'auto'
        Parquet library to use. If 'auto', then the option
        ``io.parquet.engine`` is used. The default ``io.parquet.engine``
        behavior is to try 'pyarrow', falling back to 'fastparquet' if
        'pyarrow' is unavailable.
    columns : list, default=None
        If not None, only these columns will be read from the file.

        TODO: can we support this?
    **kwargs
        Any additional kwargs are passed to the engine.

    Returns
    -------
    GeoDataFrame
    """

    impl = get_engine(engine)
    return impl.read(path, columns=columns, **kwargs)
