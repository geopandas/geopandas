# Getting Started

```{toctree}
---
maxdepth: 2
caption: Getting Started
hidden:
---

Installation <getting_started/install>
Introduction to GeoPandas <getting_started/introduction>
Examples Gallery <gallery/index>
```

## Installation

Developers write GeoPandas in pure Python, but it has several dependencies written in C
([GEOS](https://geos.osgeo.org), [GDAL](https://www.gdal.org/), [PROJ](https://proj.org/)).  Those base C libraries can sometimes be a challenge to
install. Therefore, we advise you to closely follow the recommendations below to avoid
installation problems.

### Easy way

The best way to install GeoPandas is using ``conda`` and ``conda-forge`` channel:

```
conda install -c conda-forge geopandas
```

### Detailed instructions

Do you prefer ``pip install`` or installation from source? Or want to install a specific version? See
{doc}`detailed instructions <getting_started/install>`.

### What now?

- If you don't have GeoPandas yet, check {doc}`Installation <getting_started/install>`.
- If you have never used GeoPandas and want to get familiar with it and its core
  functionality quickly, see {doc}`Getting Started Tutorial <getting_started/introduction>`.
- Detailed illustration of how to work with different parts of GeoPandas, make maps,
  manage projections, spatially merge data or geocode are part of our
  {doc}`User Guide <docs/user_guide>`.
- And if you are interested in the complete
  documentation of all classes, functions, method and attributes GeoPandas offers,
  {doc}`API Reference <docs/reference>` is here for you.

```{container} button

{doc}`Installation <getting_started/install>` {doc}`Introduction <getting_started/introduction>`
{doc}`User Guide <docs/user_guide>` {doc}`API Reference <docs/reference>`
```

## Get in touch

Haven't found what you were looking for?

- Ask usage questions ("How do I?") on [StackOverflow](https://stackoverflow.com/questions/tagged/geopandas) or [GIS StackExchange](https://gis.stackexchange.com/questions/tagged/geopandas).
- Get involved in [discussions on GitHub](https://github.com/geopandas/geopandas/discussions)
- Report bugs, suggest features or view the source code on [GitHub](https://github.com/geopandas/geopandas).
- For a quick question about a bug report or feature request, or Pull Request,
  head over to the [gitter channel](https://gitter.im/geopandas/geopandas).
- For less well defined questions or ideas, or to announce other projects of
  interest to GeoPandas users, ... use the [mailing list](https://groups.google.com/forum/#!forum/geopandas).
