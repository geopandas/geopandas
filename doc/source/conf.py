# -*- coding: utf-8 -*-
#
# GeoPandas documentation build configuration file, created by
# sphinx-quickstart on Tue Oct 15 08:08:14 2013.
#
# This file is execfile()d with the current directory set to its containing dir.
#
# Note that not all possible configuration values are present in this
# autogenerated file.
#
# All configuration values have a default; values that are commented out
# serve to show the default.

import sys, os
import warnings

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
# sys.path.insert(0, os.path.abspath('.'))

# -- General configuration -----------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
# needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be extensions
# coming with Sphinx (named 'sphinx.ext.*') or your custom ones.
extensions = [
    "IPython.sphinxext.ipython_console_highlighting",
    "IPython.sphinxext.ipython_directive",
    "sphinx_gallery.load_style",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autodoc",
    "myst_parser",
    "nbsphinx",
    "numpydoc",
    "sphinx_toggleprompt",
    "matplotlib.sphinxext.plot_directive",
]

# continue doc build and only print warnings/errors in examples
ipython_warning_is_error = False
ipython_exec_lines = [
    # ensure that dataframes are not truncated in the IPython code blocks
    "import pandas as _pd",
    '_pd.set_option("display.max_columns", 20)',
    '_pd.set_option("display.width", 100)',
]

# Fix issue with warnings from numpydoc (see discussion in PR #534)
numpydoc_show_class_members = False


def setup(app):
    app.add_css_file("custom.css")  # may also be an URL


# Add any paths that contain templates here, relative to this directory.

templates_path = ["_templates"]

autosummary_generate = True

nbsphinx_execute = "always"
nbsphinx_allow_errors = True
nbsphinx_kernel_name = 'python3'

# suppress matplotlib warning in examples
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="Matplotlib is currently using agg, which is a"
    " non-GUI backend, so cannot show the figure.",
)

# The suffix of source filenames.
source_suffix = [".rst", ".md"]

# The encoding of source files.
# source_encoding = 'utf-8-sig'

# The master toctree document.
master_doc = "index"

# General information about the project.
project = u"GeoPandas"
copyright = u"2013–2021, GeoPandas developers"

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
import geopandas

version = release = geopandas.__version__

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
# language = None

# There are two options for replacing |today|: either, you set today to some
# non-false value, then it is used:
# today = ''
# Else, today_fmt is used as the format for a strftime call.
# today_fmt = '%B %d, %Y'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = []

# The reST default role (used for this markup: `text`) to use for all documents.
# default_role = None

# If true, '()' will be appended to :func: etc. cross-reference text.
# add_function_parentheses = True

# If true, the current module name will be prepended to all description
# unit titles (such as .. function::).
# add_module_names = True

# If true, sectionauthor and moduleauthor directives will be shown in the
# output. They are ignored by default.
# show_authors = False

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# A list of ignored prefixes for module index sorting.
# modindex_common_prefix = []


# -- Options for HTML output ---------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = "pydata_sphinx_theme"

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
html_theme_options = {
    "search_bar_position": "sidebar",
    "github_url": "https://github.com/geopandas/geopandas",
    "twitter_url": "https://twitter.com/geopandas",
}

# Add any paths that contain custom themes here, relative to this directory.
# html_theme_path = []

# The name for this set of Sphinx documents.  If None, it defaults to
# "<project> v<release> documentation".
# html_title = None

# A shorter title for the navigation bar.  Default is the same as html_title.
# html_short_title = None

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
html_logo = "_static/logo/geopandas_logo_web.svg"

# The name of an image file (within the static path) to use as favicon of the
# docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large.
html_favicon = "_static/logo/favicon.png"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# If not '', a 'Last updated on:' timestamp is inserted at every page bottom,
# using the given strftime format.
# html_last_updated_fmt = '%b %d, %Y'

# If true, SmartyPants will be used to convert quotes and dashes to
# typographically correct entities.
# html_use_smartypants = True

# Custom sidebar templates, maps document names to template names.
# html_sidebars = {}

# Additional templates that should be rendered to pages, maps page names to
# template names.
# html_additional_pages = {}

# Add redirect for previously existing pages, each item is like `(from_old, to_new)`

moved_pages = [
    # user guide
    ("aggregation_with_dissolve", "docs/user_guide/aggregation_with_dissolve"),
    ("data_structures", "docs/user_guide/data_structures"),
    ("geocoding", "docs/user_guide/geocoding"),
    ("geometric_manipulations", "docs/user_guide/geometric_manipulations"),
    ("indexing", "docs/user_guide/indexing"),
    ("io", "docs/user_guide/io"),
    ("mapping", "docs/user_guide/mapping"),
    ("mergingdata", "docs/user_guide/mergingdata"),
    ("missing_empty", "docs/user_guide/missing_empty"),
    ("projections", "docs/user_guide/projections"),
    ("set_operations", "docs/user_guide/set_operations"),
    # other
    ("install", "getting_started/install"),
    ("reference", "docs/reference"),
    ("changelog", "docs/changelog"),
    ("code_of_conduct", "community/code_of_conduct"),
    ("contributing", "community/contributing"),
]

html_additional_pages = {page[0]: "redirect.html" for page in moved_pages}

html_context = {"redirects": {old: new for old, new in moved_pages}}

# If false, no module index is generated.
# html_domain_indices = True

# If false, no index is generated.
# html_use_index = True

# If true, the index is split into individual pages for each letter.
# html_split_index = False

# If true, links to the reST sources are added to the pages.
# html_show_sourcelink = True

# If true, "Created using Sphinx" is shown in the HTML footer. Default is True.
# html_show_sphinx = True

# If true, "(C) Copyright ..." is shown in the HTML footer. Default is True.
# html_show_copyright = True

# If true, an OpenSearch description file will be output, and all pages will
# contain a <link> tag referring to it.  The value of this option must be the
# base URL from which the finished HTML is served.
# html_use_opensearch = ''

# This is the file name suffix for HTML files (e.g. ".xhtml").
# html_file_suffix = None

# Output file base name for HTML help builder.
htmlhelp_basename = "GeoPandasdoc"


# -- Options for LaTeX output --------------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #'papersize': 'letterpaper',
    # The font size ('10pt', '11pt' or '12pt').
    #'pointsize': '10pt',
    # Additional stuff for the LaTeX preamble.
    #'preamble': '',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title, author, documentclass [howto/manual]).
latex_documents = [
    ("index", "GeoPandas.tex", u"GeoPandas Documentation", u"Kelsey Jordahl", "manual")
]

# The name of an image file (relative to this directory) to place at the top of
# the title page.
# latex_logo = None

# For "manual" documents, if this is true, then toplevel headings are parts,
# not chapters.
# latex_use_parts = False

# If true, show page references after internal links.
# latex_show_pagerefs = False

# If true, show URL addresses after external links.
# latex_show_urls = False

# Documents to append as an appendix to all manuals.
# latex_appendices = []

# If false, no module index is generated.
# latex_domain_indices = True


# -- Options for manual page output --------------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [("index", "geopandas", u"GeoPandas Documentation", [u"Kelsey Jordahl"], 1)]

# If true, show URL addresses after external links.
# man_show_urls = False


# -- Options for Texinfo output ------------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (
        "index",
        "GeoPandas",
        u"GeoPandas Documentation",
        u"Kelsey Jordahl",
        "GeoPandas",
        "One line description of project.",
        "Miscellaneous",
    )
]

# Documents to append as an appendix to all manuals.
# texinfo_appendices = []

# If false, no module index is generated.
# texinfo_domain_indices = True

# How to display URL addresses: 'footnote', 'no', or 'inline'.
# texinfo_show_urls = 'footnote'

nbsphinx_prolog = r"""
{% set docname = env.doc2path(env.docname, base=None) %}

.. only:: html

    .. role:: raw-html(raw)
        :format: html

    .. note::

        | This page was generated from `{{ docname }}`__.
        | Interactive online version: :raw-html:`<a href="https://mybinder.org/v2/gh/geopandas/geopandas/master?urlpath=lab/tree/doc/source/{{ docname }}"><img alt="Binder badge" src="https://mybinder.org/badge_logo.svg" style="vertical-align:text-bottom"></a>`

        __ https://github.com/geopandas/geopandas/blob/master/doc/source/{{ docname }}
"""

#  --Options for sphinx extensions -----------------------------------------------

# connect docs in other projects
intersphinx_mapping = {
    "cartopy": (
        "https://scitools.org.uk/cartopy/docs/latest/",
        "https://scitools.org.uk/cartopy/docs/latest/objects.inv",
    ),
    "contextily": (
        "https://contextily.readthedocs.io/en/stable/",
        "https://contextily.readthedocs.io/en/stable/objects.inv",
    ),
    "fiona": (
        "https://fiona.readthedocs.io/en/stable/",
        "https://fiona.readthedocs.io/en/stable/objects.inv",
    ),
    "folium": (
        "https://python-visualization.github.io/folium/",
        "https://python-visualization.github.io/folium/objects.inv",
    ),
    "geoplot": (
        "https://residentmario.github.io/geoplot/index.html",
        "https://residentmario.github.io/geoplot/objects.inv",
    ),
    "geopy": (
        "https://geopy.readthedocs.io/en/stable/",
        "https://geopy.readthedocs.io/en/stable/objects.inv",
    ),
    "libpysal": (
        "https://pysal.org/libpysal/",
        "https://pysal.org/libpysal/objects.inv",
    ),
    "mapclassify": (
        "https://pysal.org/mapclassify/",
        "https://pysal.org/mapclassify/objects.inv",
    ),
    "matplotlib": (
        "https://matplotlib.org/stable/",
        "https://matplotlib.org/stable/objects.inv",
    ),
    "pandas": (
        "https://pandas.pydata.org/pandas-docs/stable/",
        "https://pandas.pydata.org/pandas-docs/stable/objects.inv",
    ),
    "pyarrow": ("https://arrow.apache.org/docs/", None),
    "pyepsg": (
        "https://pyepsg.readthedocs.io/en/stable/",
        "https://pyepsg.readthedocs.io/en/stable/objects.inv",
    ),
    "pygeos": (
        "https://pygeos.readthedocs.io/en/latest/",
        "https://pygeos.readthedocs.io/en/latest/objects.inv",
    ),
    "pyproj": (
        "https://pyproj4.github.io/pyproj/stable/",
        "https://pyproj4.github.io/pyproj/stable/objects.inv",
    ),
    "python": ("https://docs.python.org/3", "https://docs.python.org/3/objects.inv"),
    "rtree": (
        "https://rtree.readthedocs.io/en/stable/",
        "https://rtree.readthedocs.io/en/stable/objects.inv",
    ),
    "rasterio": (
        "https://rasterio.readthedocs.io/en/stable/",
        "https://rasterio.readthedocs.io/en/stable/objects.inv",
    ),
    "shapely": (
        "https://shapely.readthedocs.io/en/stable/",
        "https://shapely.readthedocs.io/en/stable/objects.inv",
    ),
}
