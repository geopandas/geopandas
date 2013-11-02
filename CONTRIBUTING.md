Guidelines
==========

Contributions to GeoPandas are very welcome.  They are likely to
be accepted more quickly if they follow these guidelines.

At this stage of GeoPandas development, the priorities are to define a
simple, usable, and stable API and to have clean, maintainable,
readable code.  Performance matters, but not at the expense of those
goals.

In general, GeoPandas follows the conventions of the pandas project
where applicable.  Please read [pandas contributing
guidelines](https://github.com/pydata/pandas/blob/master/CONTRIBUTING.md).

In particular, when submitting a pull request:

- All existing tests should pass.  Please make sure that the test
  suite passes, both locally and on
  [Travis CI](https://travis-ci.org/kjordahl/geopandas).  Status on
  Travis will be visible on a pull request.  If you want to enable
  Travis CI on your own fork, please read the pandas guidelines link
  above or the
  [getting started docs](http://about.travis-ci.org/docs/user/getting-started/).

- New functionality should include tests.  Please write reasonable
  tests for your code and make sure that they pass on your pull request.

- Classes, methods, functions, etc. should have docstrings.  The first
  line of a docstring should be a standalone summary.  Parameters and
  return values should be ducumented explicitly.

Improving the documentation and testing for code already in GeoPandas
is a great way to get started if you'd like to make a contribution.

Style
-----

- Follow PEP 8 when possible.

- Imports should be grouped with standard library imports first,
  3rd-party libraries next, and geopandas imports third.  Within each
  grouping, imports should be alphabetized.  Always use absolute
  imports rather than relative imports.
