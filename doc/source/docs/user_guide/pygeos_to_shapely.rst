Migration from PyGEOS geometry backend to Shapely 2.0
=====================================================

Since the 0.8 version, GeoPandas includes an experimental support of PyGEOS as an
alternative geometry backend to Shapely. Recently, PyGEOS codebase was merged into the
Shapely project and released as part of Shapely 2.0. GeoPandas will therefore
deprecate support of the PyGEOS backend and will go forward with Shapely 2.0 as the
only geometry engine exposing GEOS functionality.

Given that historically the PyGEOS engine was automatically used if the package is installed (this behaviour will changed in GeoPandas 0.14 where Shapely 2.0 is used by default if installed), some downstream code may depend on
PyGEOS geometries being available as underlying data of a ``GeometryArray``.

This guide outlines the migration from the PyGEOS-based code to the Shapely-based code.

Migration period
----------------

The migration is planned for three releases spanning approximately one year, starting
with 0.13 released in the second quarter of 2023.

GeoPandas 0.13
^^^^^^^^^^^^^^

- PyGEOS is still used as a default backend over Shapely (1.8 or 2.0) if installed,
  with a ``FutureWarning`` warning about upcoming changes.

GeoPandas 0.14
^^^^^^^^^^^^^^

- The default backend is Shapely 2.0 and the PyGEOS is used only
  if Shapely 1.8 is installed instead of 2.0 or newer. The PyGEOS backend is still
  supported, but a user needs to opt in using the environment variable
  ``USE_PYGEOS`` as explained in the
  `installation instructions <../../getting_started/install.rst>`__.

GeoPandas 1.0
^^^^^^^^^^^^^

- GeoPandas will remove support of both PyGEOS and Shapely<2.

How to prepare your code for transition
---------------------------------------

If you don't use PyGEOS explicitly, there nothing to be done as GeoPandas internals will
take care of the transition. If you use PyGEOS directly and access an array of PyGEOS
geometries using ``GeoSeries.values.data``, you will need to make some changes to avoid
code breakage.

The recommended way is using Shapely vectorized operations on the ``GeometryArray``
instead of accessing the NumPy array of geometries and using PyGEOS/Shapely operations
on the array.

This is a common pattern used with GeoPandas 0.12 (or earlier), that should now be avoided in new code:

.. code-block:: python

    >>> import pygeos
    >>> geometries = gdf.geometry.values.data
    >>> mrr = pygeos.minimum_rotated_rectangle(geometries)

The recommended way of refactoring this code would look like this (with Geopandas 0.12 or later):

.. code-block:: python

    >>> import shapely  # shapely 2.0
    >>> mrr = shapely.minimum_rotated_rectangle(gdf.geometry.array)

This code will work no matter which geometry backend GeoPandas actually uses, because on
the ``GeometryArray`` level, it always returns Shapely geometry. Although keep in mind, that
it may involve additional overhead cost of converting PyGEOS geometry to Shapely
geometry.

Note that while in most cases, a simple replacement of ``pygeos`` with ``shapely``
together with a change of ``gdf.geometry.values.data`` to ``gdf.geometry.values`` or
analogous ``gdf.geometry.array``  should work, there are some differences between the
API of PyGEOS and that of Shapely. Please consult the
`Migrating from PyGEOS <https://shapely.readthedocs.io/en/stable/migration_pygeos.html>`__
document for details.
