#!/usr/bin/env/python
"""Installation script

Version handling borrowed from pandas project.
"""

import os
import warnings

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

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

MAJOR = 0
MINOR = 1
MICRO = 0
ISRELEASED = False
VERSION = '%d.%d.%d' % (MAJOR, MINOR, MICRO)
QUALIFIER = ''

FULLVERSION = VERSION
if not ISRELEASED:
    FULLVERSION += '.dev'
    try:
        import subprocess
        try:
            pipe = subprocess.Popen(["git", "rev-parse", "--short", "HEAD"],
                                    stdout=subprocess.PIPE).stdout
        except OSError:
            # msysgit compatibility
            pipe = subprocess.Popen(
                ["git.cmd", "describe", "HEAD"],
                stdout=subprocess.PIPE).stdout
        rev = pipe.read().strip()

        FULLVERSION = '%d.%d.%d.dev-%s' % (MAJOR, MINOR, MICRO, rev)

    except:
        warnings.warn("WARNING: Couldn't get git revision")
else:
    FULLVERSION += QUALIFIER


def write_version_py(filename=None):
    cnt = """\
version = '%s'
short_version = '%s'
"""
    if not filename:
        filename = os.path.join(
            os.path.dirname(__file__), 'geopandas', 'version.py')

    a = open(filename, 'w')
    try:
        a.write(cnt % (FULLVERSION, VERSION))
    finally:
        a.close()

write_version_py()

setup(name='geopandas',
      version=FULLVERSION,
      description='Geographic pandas extensions',
      license='BSD',
      author='Kelsey Jordahl',
      author_email='kjordahl@enthought.com',
      url='http://geopandas.org',
      long_description=LONG_DESCRIPTION,
      packages=['geopandas'],
      install_requires=['pandas', 'shapely', 'fiona', 'descartes', 'pyproj'],
)
