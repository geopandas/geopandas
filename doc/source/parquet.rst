.. _parquet:

Parquet Metadata Schema - Version 1.0
=====================================

*geopandas* can read and write *parquet* files, a binary columnar file format
that supports fast file-based I/O. Geometry columns are encoded to a serialized
format such as Well-Known Binary (WKB) within the *parquet* file for storage on
disk.

In order to save a *GeoDataFrame* to a *parquet* file, *geopandas* needs to
store additional metadata that describes the geometry columns.  This information
includes:

- spatial properties of each geometry column
- name of the primary geometry column

Metadata is stored as a JSON-encoded UTF-8 string under the "geo" key within the
file-level metadata of a *parquet* file. All elements below are required unless
otherwise noted.

For clarity, the following shows the unencoded JSON (``dict``) structure:

.. code-block:: none

    "geo": {
        "columns": [...],  # JSON array of geometry columns, see below
        "creator": {
            "library": "geopandas",
            "version": "0.7.0",
        },
        "primary_column": "<primary geometry column name>",
        "version": "1.0.0"  # version of the metadata schema
    }


Each of the column entries in the ``"columns"`` field above has the following content:

.. code-block:: none

    {
        "bounds": [<xmin>, <ymin>, <xmax>, <ymax>],  # OPTIONAL: total bounds of all geometries in column, in CRS of column
        "crs": "<WKT representation of CRS>",
        "encoding: "WKB",  # encoding identifier, see below
        "name": "<column name>",
    }


CRS encoding
------------

While verbose, the current guidance is to use
`WKT <https://proj.org/faq.html#what-is-the-best-format-for-describing-coordinate-reference-systems>`_ for
serializing CRS information.

Multiple dialects of WKT exist, including:
* GDAL WKT
* ESRI WKT
* WKT2:2015 (ISO 19162:2015)
* WKT2:2018 (ISO 19162:2018)

Any WKT encoding used in *parquet* must be supported by `Proj <https://proj.org/index.html>`_.


Geometry encoding
-----------------

Geometries are encoded to a binary format for storage within *parquet* files.

Well-Known Binary (WKB) is currently the only supported geometry encoding and
is recommended for widest cross-language / cross-library portability.

Other encodings may be introduced in future versions of this schema.