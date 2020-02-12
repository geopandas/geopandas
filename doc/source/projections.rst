.. currentmodule:: geopandas

.. ipython:: python
   :suppress:

   import geopandas


Managing Projections
=========================================


Coordinate Reference Systems
-----------------------------

The Coordinate Reference System (CRS) is important because the geometric shapes
in a GeoSeries or GeoDataFrame object are simply a collection of coordinates in
an arbitrary space. A CRS tells Python how those coordinates relate to places on
the Earth.

You can find the codes for most commonly used projections from
`www.spatialreference.org <https://spatialreference.org/>`_.

The same CRS can often be referred to in many ways. For example, one of the most
commonly used CRS is the WGS84 latitude-longitude projection. This can be
referred to using the authority code ``"EPSG:4326"``.

*geopandas* can accept anything accepted by `pyproj.CRS.from_user_input() <https://pyproj4.github.io/pyproj/stable/api/crs.html#pyproj.crs.CRS.from_user_input>`_:

- CRS WKT string
- An authority string (i.e. "epsg:4326")
- An EPSG integer code (i.e. 4326)
- A ``pyproj.CRS``
- An object with a to_wkt method.
- PROJ string
- Dictionary of PROJ parameters
- PROJ keyword arguments for parameters
- JSON string with PROJ parameters

For reference, a few very common projections and their EPSG codes:

* WGS84 Latitude/Longitude: ``"EPSG:4326"``
* UTM Zones (North): ``"EPSG:32633"``
* UTM Zones (South): ``"EPSG:32733"``


What is the best format to store the CRS information?
-----------------------------------------------------

Generally, WKT or SRID's are preferred over PROJ strings as they can contain more information about a given CRS.
Conversions between WKT and PROJ strings will in most cases cause a loss of information, potentially leading to erroneous transformations. If possible WKT2 should be used.

For more details, see https://proj.org/faq.html#what-is-the-best-format-for-describing-coordinate-reference-systems


Setting a Projection
----------------------

There are two relevant operations for projections: setting a projection and re-projecting.

Setting a projection may be necessary when for some reason *geopandas* has coordinate data (x-y values), but no information about how those coordinates refer to locations in the real world. Setting a projection is how one tells *geopandas* how to interpret coordinates. If no CRS is set, *geopandas* geometry operations will still work, but coordinate transformations will not be possible and exported files may not be interpreted correctly by other software.

Be aware that **most of the time** you don't have to set a projection. Data loaded from a reputable source (using the :func:`geopandas.read_file()` command) *should* always include projection information. You can see an objects current CRS through the :attr:`GeoSeries.crs` attribute.

From time to time, however, you may get data that does not include a projection. In this situation, you have to set the CRS so *geopandas* knows how to interpret the coordinates.

For example, if you convert a spreadsheet of latitudes and longitudes into a GeoSeries by hand, you would set the projection by assigning the WGS84 latitude-longitude CRS to the :attr:`GeoSeries.crs` attribute:

.. sourcecode:: python

   my_geoseries.crs = "EPSG:4326"


Re-Projecting
----------------

Re-projecting is the process of changing the representation of locations from one coordinate system to another. All projections of locations on the Earth into a two-dimensional plane `are distortions <https://en.wikipedia.org/wiki/Map_projection#Which_projection_is_best.3F>`_, the projection that is best for your application may be different from the projection associated with the data you import. In these cases, data can be re-projected using the :meth:`GeoDataFrame.to_crs` command:

.. ipython:: python

    # load example data
    world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))

    # Check original projection
    # (it's Platte Carre! x-y are long and lat)
    world.crs

    # Visualize
    ax = world.plot()
    @savefig world_starting.png
    ax.set_title("WGS84 (lat/lon)");

    # Reproject to Mercator (after dropping Antartica)
    world = world[(world.name != "Antarctica") & (world.name != "Fr. S. Antarctic Lands")]
    world = world.to_crs("EPSG:3395") # world.to_crs(epsg=3395) would also work
    ax = world.plot()
    @savefig world_reproj.png
    ax.set_title("Mercator");

Upgrading to GeoPandas 0.7 with pyproj > 2.2 and PROJ > 6
---------------------------------------------------------

Starting with GeoPandas 0.7, the `.crs` attribute of a GeoSeries or GeoDataFrame
stores the CRS information as a ``pyproj.CRS``, and no longer as a proj4 string
or dict.

Before, you might have seen this:

.. code-block:: python

   >>> gdf.crs
   {'init': 'epsg:4326'}

while now you will see something like this:

.. code-block:: python

   >>> gdf.crs
   <Geographic 2D CRS: EPSG:4326>
   Name: WGS 84
   Axis Info [ellipsoidal]:
   - Lat[north]: Geodetic latitude (degree)
   - Lon[east]: Geodetic longitude (degree)
   ...
   >>> type(gdf.crs)
   pyproj.crs.CRS

This gives a better user interface and integrates improvements from pyproj and
PROJ 6, but might also require some changes in your code. See `this blogpost
<https://jorisvandenbossche.github.io/blog/2020/02/11/geopandas-pyproj-crs/>`__
for some more background, and the subsections below cover different possible
migration issues.

See the `pyproj docs <https://pyproj4.github.io/pyproj/stable/>`__ for more on
the ``pyproj.CRS`` object.

Importing data from files
^^^^^^^^^^^^^^^^^^^^^^^^^

When reading geospatial files with :func:`geopandas.read_file`, things should
mostly work out of the box. For example, reading the example countries dataset
yields a proper CRS:

.. ipython:: python

   df = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
   df.crs


Manually specifying the CRS
^^^^^^^^^^^^^^^^^^^^^^^^^^^

When specifying the CRS manually in your code (e.g., because your data has not
yet a CRS, or when converting to another CRS), this might require a change in
your code.

**"init" proj4 strings/dicts**

Currently, a lot of people (and also the GeoPandas docs showed that before)
specify the EPSG code using the "init" proj4 string:

.. code-block:: python

   ## OLD
   GeoDataFrame(..., crs={'init': 'epsg:4326'})
   # or
   gdf.crs = {'init': 'epsg:4326'}
   # or
   gdf.to_crs({'init': 'epsg:4326'})

The above will now raise a deprecation warning from pyproj, and instead of the
"init" proj4 string, you should use only the EPSG code itself as follows:

.. code-block:: python

   ## NEW
   GeoDataFrame(..., crs="EPSG:4326")
   # or
   gdf.crs = "EPSG:4326"
   # or
   gdf.to_crs("EPSG:4326")


**proj4 strings/dicts**

Although a full proj4 string is not deprecated (as opposed to the "init" string
above), it is still recommended to change it with an EPSG code if possible.

For example, instead of:

.. code-block:: python

   gdf.crs = "+proj=lcc +lat_0=90 +lon_0=4.36748666666667 +lat_1=51.1666672333333 +lat_2=49.8333339 +x_0=150000.013 +y_0=5400088.438 +ellps=intl +units=m +no_defs +type=crs"

we recommenend to do:

.. code-block:: python

   gdf.crs = "EPSG:31370"

*if* you know the EPSG code for the projection you are using.

One possible way to find out the EPSG code is using pyproj for this:

.. code-block:: python

   >>> crs = pyproj.CRS("+proj=lcc +lat_0=90 +lon_0=4.36748666666667 +lat_1=51.1666672333333 +lat_2=49.8333339 +x_0=150000.013 +y_0=5400088.438 +ellps=intl +units=m +no_defs +type=crs")
   >>> crs.to_epsg()
   31370

(you might need to set the ``min_confidence`` keyword of ``to_epsg`` to a lower
value if the match is not perfect)

Further, on websites such as `spatialreference.org <https://spatialreference.org/>`__
and `epsg.io <https://epsg.io/>`__ the descriptions of many CRS can be found
including their EPSG codes and proj4 string definitions.


What is it with the axis order?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Briefly explain that certain CRS can have different axis order (eg EPSG:4326),
  but in GeoPandas always x, y (or lon, lat)


I get a "BoundCRS"?
^^^^^^^^^^^^^^^^^^^

- If you have a crs definition with towgs84 term, you will get a BoundCRS. Can
  use ``.source_crs`` attribute to get the actual CRS.


The ``.crs`` attribute is no longer a dict or string
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you relied on the ``.crs`` object being a dict or a string, such code can
be broken given it is now a ``pyproj.CRS`` object. But this object actually
provides a more robust interface to get information about the CRS.

For example, if you used the following code to get the EPSG code:

.. code-block:: python

   gdf.crs['init']

This will no longer work. To get the EPSG code from a ``crs`` object, you can use
the ``to_epsg()`` method.

Or to check if a CRS was a certain UTM zone:

.. code-block:: python

   '+proj=utm ' in gdf.crs

could be replaced with the longer but more robust check:

.. code-block:: python

   gdf.crs.is_projected and gdf.crs.coordinate_operation.name.upper().startswith('UTM')

And there are many other methods available on the ``pyproj.CRS`` class to get
information about the CRS.
