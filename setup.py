#!/usr/bin/env/python
"""Installation script

"""

import os

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

import versioneer

LONG_DESCRIPTION = """GeoPandas is a project to add support for geographic data to
`pandas`_ objects.

The goal of GeoPandas is to make working with geospatial data in
python easier. It combines the capabilities of `pandas`_ and `shapely`_,
providing geospatial operations in pandas and a high-level interface
to multiple geometries to shapely. GeoPandas enables you to easily do
operations in python that would otherwise require a spatial database
such as PostGIS.

.. _pandas: http://pandas.pydata.org
.. _shapely: http://toblerity.github.io/shapely
"""

if os.environ.get('READTHEDOCS', False) == 'True':
    INSTALL_REQUIRES = []
else:
    INSTALL_REQUIRES = ['pandas', 'shapely', 'fiona', 'descartes', 'pyproj']

# get all data dirs in the datasets module
data_files = []

for item in os.listdir("geopandas/datasets"):
    if os.path.isdir(os.path.join("geopandas/datasets/", item)) \
            and not item.startswith('__'):
        data_files.append(os.path.join("datasets", item, '*'))

setup(name='geopandas',
      version=versioneer.get_version(),
      description='Geographic pandas extensions',
      license='BSD',
      author='GeoPandas contributors',
      author_email='kjordahl@alum.mit.edu',
      url='http://geopandas.org',
      long_description=LONG_DESCRIPTION,
      packages=['geopandas', 'geopandas.io', 'geopandas.tools',
                'geopandas.datasets'],
      package_data={'geopandas': data_files},
      install_requires=INSTALL_REQUIRES,
      cmdclass=versioneer.get_cmdclass())
