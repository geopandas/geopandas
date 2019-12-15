.. currentmodule:: geopandas

.. ipython:: python
   :suppress:

   import geopandas
   import matplotlib
   orig = matplotlib.rcParams['figure.figsize']
   matplotlib.rcParams['figure.figsize'] = [orig[0] * 1.5, orig[1]]



Data Structures
=========================================

GeoPandas implements two main data structures, a :class:`GeoSeries` and a
:class:`GeoDataFrame`.  These are subclasses of pandas ``Series`` and
``DataFrame``, respectively.

GeoSeries
---------

A :class:`GeoSeries` is essentially a vector where each entry in the vector
is a set of shapes corresponding to one observation. An entry may consist
of only one shape (like a single polygon) or multiple shapes that are
meant to be thought of as one observation (like the many polygons that
make up the State of Hawaii or a country like Indonesia).

*geopandas* has three basic classes of geometric objects (which are actually *shapely* objects):

* Points / Multi-Points
* Lines / Multi-Lines
* Polygons / Multi-Polygons

Note that all entries in  a :class:`GeoSeries` need not be of the same geometric type, although certain export operations will fail if this is not the case.

Overview of Attributes and Methods
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The :class:`GeoSeries` class implements nearly all of the attributes and
methods of Shapely objects.  When applied to a :class:`GeoSeries`, they
will apply elementwise to all geometries in the series.  Binary
operations can be applied between two :class:`GeoSeries`, in which case the
operation is carried out elementwise.  The two series will be aligned
by matching indices.  Binary operations can also be applied to a
single geometry, in which case the operation is carried out for each
element of the series with that geometry.  In either case, a
``Series`` or a :class:`GeoSeries` will be returned, as appropriate.

A short summary of a few attributes and methods for GeoSeries is
presented here, and a full list can be found in the :doc:`all attributes and methods page <reference>`.
There is also a family of methods for creating new shapes by expanding
existing shapes or applying set-theoretic operations like "union" described
in :doc:`geometric manipulations <geometric_manipulations>`.

Attributes
^^^^^^^^^^^^^^^
* :attr:`~GeoSeries.area`: shape area (units of projection -- see :doc:`projections <projections>`)
* :attr:`~GeoSeries.bounds`: tuple of max and min coordinates on each axis for each shape
* :attr:`~GeoSeries.total_bounds`: tuple of max and min coordinates on each axis for entire GeoSeries
* :attr:`~GeoSeries.geom_type`: type of geometry.
* :attr:`~GeoSeries.is_valid`: tests if coordinates make a shape that is reasonable geometric shape (`according to this <http://www.opengeospatial.org/standards/sfa>`_).

Basic Methods
^^^^^^^^^^^^^^

* :meth:`~GeoSeries.distance`: returns ``Series`` with minimum distance from each entry to ``other``
* :attr:`~GeoSeries.centroid`: returns :class:`GeoSeries` of centroids
* :meth:`~GeoSeries.representative_point`:  returns :class:`GeoSeries` of points that are guaranteed to be within each geometry. It does **NOT** return centroids.
* :meth:`~GeoSeries.to_crs`: change coordinate reference system. See :doc:`projections <projections>`
* :meth:`~GeoSeries.plot`: plot :class:`GeoSeries`. See :doc:`mapping <mapping>`.

Relationship Tests
^^^^^^^^^^^^^^^^^^^

* :meth:`~GeoSeries.geom_almost_equals`: is shape almost the same as ``other`` (good when floating point precision issues make shapes slightly different)
* :meth:`~GeoSeries.contains`: is shape contained within ``other``
* :meth:`~GeoSeries.intersects`: does shape intersect ``other``


GeoDataFrame
------------

A :class:`GeoDataFrame` is a tabular data structure that contains a :class:`GeoSeries`.

The most important property of a :class:`GeoDataFrame` is that it always has one :class:`GeoSeries` column that holds a special status. This :class:`GeoSeries` is referred to as the :class:`GeoDataFrame`'s "geometry". When a spatial method is applied to a :class:`GeoDataFrame` (or a spatial attribute like ``area`` is called), this commands will always act on the "geometry" column.

The "geometry" column -- no matter its name -- can be accessed through the :attr:`~GeoDataFrame.geometry` attribute (``gdf.geometry``), and the name of the ``geometry`` column can be found by typing ``gdf.geometry.name``.

A :class:`GeoDataFrame` may also contain other columns with geometrical (shapely) objects, but only one column can be the active geometry at a time. To change which column is the active geometry column, use the :meth:`GeoDataFrame.set_geometry` method.

An example using the ``worlds`` GeoDataFrame:

.. ipython:: python

    world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))

    world.head()
    #Plot countries
    @savefig world_borders.png
    world.plot();

Currently, the column named "geometry" with country borders is the active
geometry column:

.. ipython:: python

    world.geometry.name

We can also rename this column to "borders":

.. ipython:: python

    world = world.rename(columns={'geometry': 'borders'}).set_geometry('borders')
    world.geometry.name

Now, we create centroids and make it the geometry:

.. ipython:: python

    world['centroid_column'] = world.centroid
    world = world.set_geometry('centroid_column')

    @savefig world_centroids.png
    world.plot();


**Note:** A :class:`GeoDataFrame` keeps track of the active column by name, so if you rename the active geometry column, you must also reset the geometry::

    gdf = gdf.rename(columns={'old_name': 'new_name'}).set_geometry('new_name')

**Note 2:** Somewhat confusingly, by default when you use the ``read_file`` command, the column containing spatial objects from the file is named "geometry" by default, and will be set as the active geometry column. However, despite using the same term for the name of the column and the name of the special attribute that keeps track of the active column, they are distinct. You can easily shift the active geometry column to a different :class:`GeoSeries` with the :meth:`~GeoDataFrame.set_geometry` command. Further, ``gdf.geometry`` will always return the active geometry column, *not* the column named ``geometry``. If you wish to call a column named "geometry", and a different column is the active geometry column, use ``gdf['geometry']``, not ``gdf.geometry``.

Attributes and Methods
~~~~~~~~~~~~~~~~~~~~~~

Any of the attributes calls or methods described for a :class:`GeoSeries` will work on a :class:`GeoDataFrame` -- effectively, they are just applied to the "geometry" :class:`GeoSeries`.

However, ``GeoDataFrames`` also have a few extra methods for input and output which are described on the :doc:`Input and Output <io>` page and for geocoding with are described in :doc:`Geocoding <geocoding>`.


.. ipython:: python
    :suppress:

    matplotlib.rcParams['figure.figsize'] = orig


Display options
---------------

GeoPandas has an ``options`` attribute with currently a single configuration
option to control:

.. ipython:: python

    import geopandas
    geopandas.options

The ``geopandas.options.display_precision`` option can control the number of
decimals to show in the display of coordinates in the geometry column. 
In the ``world`` example of above, the default is to show 5 decimals for
geographic coordinates:

.. ipython:: python

    world['centroid_column'].head()

If you want to change this, for example to see more decimals, you can do:

.. ipython:: python

    geopandas.options.display_precision = 9
    world['centroid_column'].head()
