# Ecosystem

## GeoPandas dependencies

GeoPandas brings together the full capability of `pandas` and the open-source geospatial
tools `Shapely`, which brings manipulation and analysis of geometric objects backed by
[`GEOS`](https://trac.osgeo.org/geos) library, `pyogrio`, allowing us to read and write
geographic data files using [`GDAL`](https://gdal.org), and `pyproj`, a library for
cartographic projections and coordinate transformations, which is a Python interface to
[`PROJ`](https://proj.org).

Furthermore, GeoPandas has several optional dependencies as
`mapclassify`, or `geopy`.

### Required dependencies

#### [pandas](https://github.com/pandas-dev/pandas)
`pandas` is a Python package that provides fast, flexible, and expressive data
structures designed to make working with structured (tabular, multidimensional,
potentially heterogeneous) and time series data both easy and intuitive. It aims to be
the fundamental high-level building block for doing practical, real world data analysis
in Python. Additionally, it has the broader goal of becoming the most powerful and
flexible open source data analysis / manipulation tool available in any language. It is
already well on its way toward this goal.

#### [Shapely](https://github.com/Toblerity/Shapely)
`Shapely` is a BSD-licensed Python package for manipulation and analysis of planar
geometric objects. It is based on the widely deployed `GEOS` (the engine of PostGIS) and
`JTS` (from which `GEOS` is ported) libraries. `Shapely` is not concerned with data
formats or coordinate systems, but can be readily integrated with packages that are.

#### [pyogrio](https://github.com/geopandas/pyogrio)
Pyogrio provides a GeoPandas-oriented API to OGR vector data sources, such as ESRI
Shapefile, GeoPackage, and GeoJSON. Vector data sources have geometries, such as points,
lines, or polygons, and associated records with potentially many columns worth of data.

#### [pyproj](https://github.com/pyproj4/pyproj)
`pyproj` is a Python interface to `PROJ` (cartographic projections and coordinate
transformations library). GeoPandas uses a `pyproj.crs.CRS` object to keep track of the
projection of each `GeoSeries` and its `Transformer` object to manage re-projections.

### Optional dependencies

#### [mapclassify](https://github.com/pysal/mapclassify)
`mapclassify` provides functionality for Choropleth map classification. Currently,
fifteen different classification schemes are available, including a highly-optimized
implementation of Fisher-Jenks optimal classification. Each scheme inherits a common
structure that ensures computations are scalable and supports applications in streaming
contexts.

#### [geopy](https://github.com/geopy/geopy)
`geopy` is a Python client for several popular geocoding web services. `geopy` makes it
easy for Python developers to locate the coordinates of addresses, cities, countries,
and landmarks across the globe using third-party geocoders and other data sources.

#### [matplotlib](https://github.com/matplotlib/matplotlib)
`Matplotlib` is a comprehensive library for creating static, animated, and interactive
visualizations in Python. Matplotlib produces publication-quality figures in a variety
of hardcopy formats and interactive environments across platforms. Matplotlib can be
used in Python scripts, the Python and IPython shell, web application servers, and
various graphical user interface toolkits.

#### [Fiona](https://github.com/Toblerity/Fiona)
`Fiona` is `GDAL’s` neat and nimble vector API for Python programmers. Fiona is designed
to be simple and dependable. It focuses on reading and writing data in standard Python
IO style and relies upon familiar Python types and protocols such as files,
dictionaries, mappings, and iterators instead of classes specific to `OGR`. Fiona can
read and write real-world data using multi-layered GIS formats and zipped virtual file
systems and integrates readily with other Python GIS packages such as `pyproj`, `Rtree`,
and `Shapely`.

## GeoPandas ecosystem

Various packages are built on top of GeoPandas addressing specific geospatial data
processing needs, analysis, and visualization. Below is an incomplete list (in no
particular order) of tools which form the GeoPandas-related Python ecosystem.

### Spatial analysis and Machine Learning

#### [PySAL](https://github.com/pysal/pysal)
`PySAL`, the Python spatial analysis library, is an open source cross-platform library
for geospatial data science with an emphasis on geospatial vector data written in
Python. `PySAL` is a family of packages, some of which are listed below.

##### [libpysal](https://github.com/pysal/libpysal)
`libpysal` provides foundational algorithms and data structures that support the rest of
the library. This currently includes the following modules: input/output (`io`), which
provides readers and writers for common geospatial file formats; weights (`weights`),
which provides the main class to store spatial weights matrices, as well as several
utilities to manipulate and operate on them; computational geometry (`cg`), with several
algorithms, such as Voronoi tessellations or alpha shapes that efficiently process
geometric shapes; and an additional module with example data sets (`examples`).

##### [esda](https://github.com/pysal/esda)
`esda` implements methods for the analysis of both global (map-wide) and local (focal)
spatial autocorrelation, for both continuous and binary data. In addition, the package
increasingly offers cutting-edge statistics about boundary strength and measures of
aggregation error in statistical analyses.

##### [segregation](https://github.com/pysal/segregation)
`segregation` package calculates over 40 different segregation indices and provides a
suite of additional features for measurement, visualization, and hypothesis testing that
together represent the state of the art in quantitative segregation analysis.

##### [mgwr](https://github.com/pysal/mgwr)
`mgwr` provides scalable algorithms for estimation, inference, and prediction using
single- and multi-scale geographically weighted regression models in a variety of
generalized linear model frameworks, as well as model diagnostics tools.

##### [tobler](https://github.com/pysal/tobler)
`tobler` provides functionality for areal interpolation and dasymetric mapping.
`tobler` includes functionality for interpolating data using area-weighted approaches,
regression model-based approaches that leverage remotely-sensed raster data as auxiliary
information, and hybrid approaches.

#### [movingpandas](https://github.com/anitagraser/movingpandas)
`MovingPandas` is a package for dealing with movement data. `MovingPandas` implements a
`Trajectory` class and corresponding methods based on GeoPandas. A trajectory has a
time-ordered series of point geometries. These points and associated attributes are
stored in a `GeoDataFrame`. `MovingPandas` implements spatial and temporal data access
and analysis functions as well as plotting functions.

#### [momepy](https://github.com/martinfleis/momepy)
`momepy` is a library for quantitative analysis of urban form - urban morphometrics. It
is built on top of `GeoPandas`, `PySAL` and `networkX`. `momepy` aims to provide a wide
range of tools for a systematic and exhaustive analysis of urban form. It can work with
a wide range of elements, while focused on building footprints and street networks.

#### [geosnap](https://github.com/spatialucr/geosnap)
`geosnap` makes it easier to explore, model, analyze, and visualize the social and
spatial dynamics of neighborhoods. `geosnap` provides a suite of tools for creating
socio-spatial datasets, harmonizing those datasets into consistent set of time-static
boundaries, modeling bespoke neighborhoods and prototypical neighborhood types, and
modeling neighborhood change using classic and spatial statistical methods. It also
provides a set of static and interactive visualization tools to help you display and
understand the critical information at each step of the process.

#### [mesa-geo](https://github.com/Corvince/mesa-geo)
`mesa-geo` implements a GeoSpace that can host GIS-based GeoAgents, which are like
normal Agents, except they have a shape attribute that is a `Shapely` object. You can
use `Shapely` directly to create arbitrary shapes, but in most cases you will want to
import your shapes from a file. Mesa-geo allows you to create GeoAgents from any vector
data file (e.g. shapefiles), valid GeoJSON objects or a GeoPandas `GeoDataFrame`.

#### [Pyspatialml](https://github.com/stevenpawley/Pyspatialml)
`Pyspatialml` is a Python module for applying `scikit-learn` machine learning models to
'stacks' of raster datasets. Pyspatialml includes functions and classes for working with
multiple raster datasets and performing a typical machine learning workflow consisting
of extracting training data and applying the predict or `predict_proba` methods of
`scikit-learn` estimators to a stack of raster datasets. Pyspatialml is built upon the
`rasterio` Python module for all of the heavy lifting, and is also designed for working
with vector data using the `geopandas` module.

#### [PyGMI](https://github.com/Patrick-Cole/pygmi)
`PyGMI` stands for Python Geoscience Modelling and Interpretation. It is a modelling and
interpretation suite aimed at magnetic, gravity and other datasets.

### Visualization

#### [hvPlot](https://hvplot.holoviz.org/user_guide/Geographic_Data.html#Geopandas)
`hvPlot` provides interactive Bokeh-based plotting for GeoPandas
dataframes and series using the same API as the Matplotlib `.plot()`
support that comes with GeoPandas. hvPlot makes it simple to pan and zoom into
your plots, use widgets to explore multidimensional data, and render even the
largest datasets in web browsers using [Datashader](https://datashader.org).

#### [contextily](https://github.com/geopandas/contextily)
`contextily` is a small Python 3 (3.6 and above) package to retrieve tile maps from the
internet. It can add those tiles as basemap to `matplotlib` figures or write tile maps
to disk into geospatial raster files. Bounding boxes can be passed in both WGS84
(EPSG:4326) and Spheric Mercator (EPSG:3857).

#### [cartopy](https://github.com/SciTools/cartopy)
`Cartopy` is a Python package designed to make drawing maps for data analysis and
visualisation easy. It features: object oriented projection definitions; point, line,
polygon and image transformations between projections; integration to expose advanced
mapping in `Matplotlib` with a simple and intuitive interface; powerful vector data
handling by integrating shapefile reading with `Shapely` capabilities.

#### [bokeh](https://github.com/bokeh/bokeh)
`Bokeh` is an interactive visualization library for modern web browsers. It provides
elegant, concise construction of versatile graphics, and affords high-performance
interactivity over large or streaming datasets. `Bokeh` can help anyone who would like
to quickly and easily make interactive plots, dashboards, and data applications.

#### [folium](https://github.com/python-visualization/folium)
`folium` builds on the data wrangling strengths of the Python ecosystem and the mapping
strengths of the `Leaflet.js` library. Manipulate your data in Python, then visualize it
in a `Leaflet` map via `folium`.

#### [kepler.gl](https://github.com/keplergl/kepler.gl)
`Kepler.gl` is a data-agnostic, high-performance web-based application for visual
exploration of large-scale geolocation data sets. Built on top of Mapbox GL and
`deck.gl`, `kepler.gl` can render millions of points representing thousands of trips and
perform spatial aggregations on the fly.

#### [geoplot](https://github.com/ResidentMario/geoplot)
`geoplot` is a high-level Python geospatial plotting library. It's an extension to
`cartopy` and `matplotlib` which makes mapping easy: like `seaborn` for geospatial. It
comes with the high-level plotting API, native projection support and compatibility with
`matplotlib`.

#### [GeoViews](https://github.com/holoviz/geoviews)
`GeoViews` is a Python library that makes it easy to explore and
visualize any data that includes geographic locations, with native
support for GeoPandas dataframes and series objects.  It has
particularly powerful support for multidimensional meteorological and
oceanographic datasets, such as those used in weather, climate, and
remote sensing research, but is useful for almost anything that you
would want to plot on a map!

#### [EarthPy](https://github.com/earthlab/earthpy)
`EarthPy` is a python package that makes it easier to plot and work with spatial raster
and vector data using open source tools. `Earthpy` depends upon `geopandas` which has a
focus on vector data and `rasterio` with facilitates input and output of raster data
files. It also requires `matplotlib` for plotting operations. `EarthPy’s` goal is to
make working with spatial data easier for scientists.

#### [splot](https://github.com/pysal/splot)
`splot` provides statistical visualizations for spatial analysis. It methods for
visualizing global and local spatial autocorrelation (through Moran scatterplots and
cluster maps), temporal analysis of cluster dynamics (through heatmaps and rose
diagrams), and multivariate choropleth mapping (through value-by-alpha maps). A high
level API supports the creation of publication-ready visualizations

#### [legendgram](https://github.com/pysal/legendgram)
`legendgram` is a small package that provides "legendgrams" legends that visualize the
distribution of observations by color in a given map. These distributional
visualizations for map classification schemes assist in analytical cartography and
spatial data visualization.

#### [buckaroo](https://github.com/paddymul/buckaroo)
`buckaroo` is a modern data table for Jupyter that expedites the most
common exploratory data analysis tasks. It provides scrollable tables,
histograms, and summary stats. Buckaroo supports many DataFrame
libraries including `geopandas`. It can display `GeoDataFrame`s as
tables, it also supports rendering the Geometry as an SVG in the
table.

### Geometry manipulation

#### [TopoJSON](https://github.com/mattijn/topojson)
`topojson` is a library for creating a TopoJSON encoding of nearly any
geographical object in Python. With topojson it is possible to reduce the size of
your geographical data, typically by orders of magnitude. It is able to do so through
eliminating redundancy through computation of a topology, fixed-precision integer
encoding of coordinates, and simplification and quantization of arcs.

#### [geocube](https://github.com/corteva/geocube)
Tool to convert geopandas vector data into rasterized `xarray` data.

### Data retrieval

#### [OSMnx](https://github.com/gboeing/osmnx)
`OSMnx` is a Python package that lets you download spatial data from OpenStreetMap and
model, project, visualize, and analyze real-world street networks. You can download and
model walkable, drivable, or bikeable urban networks with a single line of Python code
and then easily analyze and visualize them. You can just as easily download and work with
other infrastructure types, amenities/points of interest, building footprints, elevation
data, street bearings/orientations, and speed/travel time.

#### [pyrosm](https://github.com/HTenkanen/pyrosm)
`Pyrosm` is a Python library for reading OpenStreetMap data from Protocolbuffer Binary
Format -files (`*.osm.pbf`) into Geopandas `GeoDataFrames`. `Pyrosm` makes it easy to
extract various datasets from OpenStreetMap pbf-dumps including e.g. road networks,
buildings, Points of Interest (POI), landuse and natural elements. Also fully customized
queries are supported which makes it possible to parse the data from OSM with more
specific filters.

#### [geobr](https://github.com/ipeaGIT/geobr)
`geobr` is a computational package to download official spatial data sets of Brazil. The
package includes a wide range of geospatial data in geopackage format (like shapefiles
but better), available at various geographic scales and for various years with
harmonized attributes, projection and topology.

#### [cenpy](https://github.com/cenpy-devs/cenpy)
An interface to explore and query the US Census API and return Pandas `Dataframes`. This
package is intended for exploratory data analysis and draws inspiration from
sqlalchemy-like interfaces and `acs.R`. With separate APIs for application developers
and folks who only want to get their data quickly & painlessly, `cenpy` should meet the
needs of most who aim to get US Census Data into Python.

#### [pygadm](https://github.com/12rambau/pygadm)
`pygadm` is a Python package that lets you request spatial data from [GADM](https://gadm.org/)
without manually downloading any file. This package aims at simplifying the requests
of the data using few parameters such as the name and the subdivision levels.
Outputs are served as `GeoDataFrame` in `epsg:4326`.

```{admonition} Expand this page
Do know a package which should be here? [Let us
know](https://github.com/geopandas/geopandas/issues) or [add it by
yourself](contributing.rst)!
```
