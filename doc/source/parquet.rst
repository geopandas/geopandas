.. _parquet:

Parquet Metadata Schema - Version 1.0
=====================================

*geopandas* can read and write *parquet* files, a binary columnar file format that supports fast file-based I/O.
Geometry columns are encoded to Well-Known Binary (WKB) within the *parquet* file.

In order to save a *GeoDataFrame* to a *parquet* file, *geopandas* needs to store additional metadata that
describes the geometry columns.  This information includes:

- name of the primary geometry column
- list of geometry columns
- Coordinate Reference System (CRS) of each geometry column

This information is stored using a combination of file-level and column level metadata within the *parquet* file.


File-Level Metadata
-------------------

The file-level metadata is stored as a JSON-encoded UTF-8 string under the "geo" key.
For clarity, the following shows the unencoded JSON / `dict` structure:

.. code-block:: none

    "geo": {
        "columns": [...],  # JSON array of geometry columns, see below
        "creator": {
            "library": "geopandas",
            "version": "0.7.0",
        },
        "primary_column": "...",  # name of primary geometry column
        "version": "1.0.0"
    }


Column-Level Metadata
---------------------

Each of the column entries in the ``"columns"`` field of the file-level metadata (see above)
is another key-value object with following content:

.. code-block:: none

    {
        "name": "<geometry column name>",
        "crs": "...",  # CRS description, see below
    }

The CRS information is stored within a JSON-encoded UTF-8 string for each geometry column, under the "crs" key.

Multiple options exist for storing CRS information. The current guidance is to use
`WKT <https://proj.org/faq.html#what-is-the-best-format-for-describing-coordinate-reference-systems>`_.
or SRIDs (e.g., EPSG codes), whereas `PROJJSON <https://proj.org/usage/projjson.html#projjson>`_.
is a promising new portable representation that may take some time to fully adopt across other languages and toolkits.

The following approach requires that at least one representation of CRS be present in the file:
- "srid"
- "json" (PROJJSON)
- "wkt"

If multiple are present, they would follow this order of precedence.


For clarity, the following shows the unencoded JSON / `dict` structure:

.. ipython:: python

    "crs": {
        "srid": {  # optional
            "authority": "EPSG",
            "code": 4326
        },
        "json": {  # optional
            "$schema": "https://proj.org/schemas/v0.1/projjson.schema.json",
            "type": "GeographicCRS",
            "name": "WGS 84",
            "datum": {
                "type": "GeodeticReferenceFrame",
                "name": "World Geodetic System 1984",
                "ellipsoid": {
                    "name": "WGS 84",
                    "semi_major_axis": 6378137,
                    "inverse_flattening": 298.257223563
                }
            },
            "coordinate_system": {
                "subtype": "ellipsoidal",
                "axis": [
                {
                    "name": "Geodetic latitude",
                    "abbreviation": "Lat",
                    "direction": "north",
                    "unit": "degree"
                },
                {
                    "name": "Geodetic longitude",
                    "abbreviation": "Lon",
                    "direction": "east",
                    "unit": "degree"
                }
                ]
            },
            "area": "World",
            "bbox": {
                "south_latitude": -90,
                "west_longitude": -180,
                "north_latitude": 90,
                "east_longitude": 180
            },
            "id": {
                "authority": "EPSG",
                "code": 4326
            }
        },
        "wkt": {  # optional
            "version": "WKT2_2018",
            "value": "..."  # omitted for brevity
        }
    }
