.. currentmodule:: geopandas

.. ipython:: python
   :suppress:

   import geopandas


Managing Projections
=========================================


Coordinate Reference Systems
-----------------------------

CRS are important because the geometric shapes in a GeoSeries or GeoDataFrame object are simply a collection of coordinates in an arbitrary space. A CRS tells Python how those coordinates related to places on the Earth.

You can find the codes for most commonly used projections from
`www.spatialreference.org <http://spatialreference.org/>`_.

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
