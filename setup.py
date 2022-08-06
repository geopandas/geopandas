#!/usr/bin/env/python
"""Installation script

"""

import os
import sys

from setuptools import setup

# ensure the current directory is on sys.path so versioneer can be imported
# when pip uses PEP 517/518 build rules.
# https://github.com/python-versioneer/python-versioneer/issues/193
sys.path.append(os.path.dirname(__file__))

import versioneer  # noqa: E402

LONG_DESCRIPTION = """GeoPandas is a project to add support for geographic data to
`pandas`_ objects.

The goal of GeoPandas is to make working with geospatial data in
python easier. It combines the capabilities of `pandas`_ and `shapely`_,
providing geospatial operations in pandas and a high-level interface
to multiple geometries to shapely. GeoPandas enables you to easily do
operations in python that would otherwise require a spatial database
such as PostGIS.

.. _pandas: http://pandas.pydata.org
.. _shapely: http://shapely.readthedocs.io/en/latest/
"""

if os.environ.get("READTHEDOCS", False) == "True":
    INSTALL_REQUIRES = []
else:
    INSTALL_REQUIRES = [
        "pandas >= 1.0.0",
        "shapely >= 1.7, < 2",
        "fiona >= 1.8",
        "pyproj >= 2.6.1.post1",
        "packaging",
    ]

# get all data dirs in the datasets module
data_files = []

for item in os.listdir("geopandas/datasets"):
    if not item.startswith("__"):
        if os.path.isdir(os.path.join("geopandas/datasets/", item)):
            data_files.append(os.path.join("datasets", item, "*"))
        elif item.endswith(".zip"):
            data_files.append(os.path.join("datasets", item))

data_files.append("tests/data/*")


setup(
    name="geopandas",
    version=versioneer.get_version(),
    description="Geographic pandas extensions",
    license="BSD",
    author="GeoPandas contributors",
    author_email="kjordahl@alum.mit.edu",
    url="http://geopandas.org",
    project_urls={
        "Source": "https://github.com/geopandas/geopandas",
    },
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/x-rst",
    packages=[
        "geopandas",
        "geopandas.io",
        "geopandas.tools",
        "geopandas.datasets",
        "geopandas.tests",
        "geopandas.tools.tests",
    ],
    package_data={"geopandas": data_files},
    python_requires=">=3.8",
    install_requires=INSTALL_REQUIRES,
    cmdclass=versioneer.get_cmdclass(),
)
