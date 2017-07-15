#!/usr/bin/env/python
"""Build and install ``geopandas`` with or without Cython or C compiler

Building with Cython must be specified with the "--with-cython" option such as:

    $ python setup.py build_ext --inplace --with-cython

Not using Cython by default makes contributing to ``geopandas`` easy,
because Cython and a C compiler are not required for development or usage.

During installation, C extension modules will be automatically built with a
C compiler if possible, but will fail gracefully if there is an error during
compilation.  Use of C extensions significantly improves the performance of
``geopandas``, but a pure Python implementation will be used if the
extension modules are unavailable.

"""

import os
import sys
import warnings

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

from distutils.core import Extension
from distutils.command.build_ext import build_ext
from distutils.errors import (CCompilerError, DistutilsExecError,
                              DistutilsPlatformError)

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
.. _shapely: http://shapely.readthedocs.io/en/latest/
"""

if os.environ.get('READTHEDOCS', False) == 'True':
    INSTALL_REQUIRES = []
else:
    INSTALL_REQUIRES = ['pandas >= 0.23.0', 'shapely', 'fiona', 'pyproj']

try:
    from Cython.Build import cythonize
    has_cython = True
except ImportError:
    has_cython = False

use_cython = False
if '--without-cython' in sys.argv:
    sys.argv.remove('--without-cython')

if '--with-cython' in sys.argv:
    use_cython = True
    sys.argv.remove('--with-cython')
    if use_cython and not has_cython:
        raise RuntimeError('ERROR: Cython not found.  Exiting.\n       '
                           'Install Cython or don\'t use "--with-cython"')

suffix = '.pyx' if use_cython else '.c'
ext_modules = []
for modname in ['vectorized']:
    ext_modules.append(Extension('geopandas.' + modname,
                                 ['geopandas/' + modname + suffix]))
if use_cython:
    # Set global Cython options
    # http://docs.cython.org/en/latest/src/reference/compilation.html#compiler-directives
    from Cython.Compiler.Options import get_directive_defaults
    directive_defaults = get_directive_defaults()
    # directive_defaults['...'] = True

    # Cythonize all the things!
    ext_modules = cythonize(ext_modules)

build_exceptions = (CCompilerError, DistutilsExecError, DistutilsPlatformError,
                    IOError, SystemError)


class build_ext_may_fail(build_ext):
    """ Allow compilation of extensions modules to fail, but warn if they do"""

    warning_message = """
*********************************************************************
WARNING: %s
         could not be compiled.  See the output above for details.

Compiled C extension modules are not required for `geopandas`
to run, but they do result in significant speed improvements.
Proceeding to build `geopandas` as a pure Python package.

If you are using Linux, you probably need to install GCC or the
Python development package.

Debian and Ubuntu users should issue the following command:

    $ sudo apt-get install build-essential python-dev

RedHat, CentOS, and Fedora users should issue the following command:

    $ sudo yum install gcc python-devel

*********************************************************************
"""

    def run(self):
        try:
            build_ext.run(self)
        except build_exceptions:
            self.warn_failed()

    def build_extension(self, ext):
        try:
            build_ext.build_extension(self, ext)
        except build_exceptions:
            self.warn_failed(name=ext.name)

    def warn_failed(self, name=None):
        if name is None:
            name = 'Extension modules'
        else:
            name = 'The "%s" extension module' % name
        exc = sys.exc_info()[1]
        sys.stdout.write('%s\n' % str(exc))
        warnings.warn(self.warning_message % name)


cmdclass = versioneer.get_cmdclass()
if use_cython:
    cmdclass['build_ext'] = build_ext_may_fail

# get all data dirs in the datasets module
data_files = ['*.pyx', '*.pxd']

for item in os.listdir("geopandas/datasets"):
    if not item.startswith('__'):
        if os.path.isdir(os.path.join("geopandas/datasets/", item)):
            data_files.append(os.path.join("datasets", item, '*'))
        elif item.endswith('.zip'):
            data_files.append(os.path.join("datasets", item))

data_files.append('tests/data/*')


setup(name='geopandas',
      version=versioneer.get_version(),
      description='Geographic pandas extensions',
      license='BSD',
      author='GeoPandas contributors',
      author_email='kjordahl@alum.mit.edu',
      url='http://geopandas.org',
      long_description=LONG_DESCRIPTION,
      packages=['geopandas', 'geopandas.io', 'geopandas.tools',
                'geopandas.datasets',
                'geopandas.tests', 'geopandas.tools.tests'],
      package_data={'geopandas': data_files},
      install_requires=INSTALL_REQUIRES,
      ext_modules=ext_modules,
      cmdclass=cmdclass)
