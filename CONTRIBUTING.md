Guidelines
==========

Contributions to GeoPandas are very welcome. They are likely to
be accepted more quickly if they follow these guidelines.

At this stage of GeoPandas development, the priorities are to define a
simple, usable, and stable API and to have clean, maintainable,
readable code. Performance matters, but not at the expense of those
goals.

In general, GeoPandas follows the conventions of the pandas project
where applicable. Please read the [contributing
guidelines](https://geopandas.readthedocs.io/en/latest/community/contributing.html).

In particular, when submitting a pull request:

- Install the requirements for the development environment (one can do this
  with either conda, and the environment.yml file, or pip, and the
  requirements-dev.txt file, and can use the pandas contributing guidelines
  as a guide).
- All existing tests should pass. Please make sure that the test
  suite passes, both locally and on
  [GitHub Actions](https://github.com/geopandas/geopandas/actions). Status on
  GHA will be visible on a pull request. GHA are automatically enabled
  on your own fork as well. To trigger a check, make a PR to your own fork.

- New functionality should include tests. Please write reasonable
  tests for your code and make sure that they pass on your pull request.

- Classes, methods, functions, etc. should have docstrings. The first
  line of a docstring should be a standalone summary. Parameters and
  return values should be documented explicitly.

- Unless your PR implements minor changes or internal work only, make sure
  it contains a note describing the changes in the `CHANGELOG.md` file.

Improving the documentation and testing for code already in GeoPandas
is a great way to get started if you'd like to make a contribution.

Style
-----

- GeoPandas supports Python 3.10+ only.

- GeoPandas follows [the PEP 8
  standard](http://www.python.org/dev/peps/pep-0008/) and uses
  [ruff](https://docs.astral.sh/ruff/) to ensure a consistent
  code format throughout the project.

- Imports should be grouped with standard library imports first,
  third-party libraries next, and GeoPandas imports third. Within each
  grouping, imports should be alphabetized. Always use absolute
  imports when possible, and explicit relative imports for local
  imports when necessary in tests.

- You can set up [pre-commit hooks](https://pre-commit.com/) to
  automatically run `ruff` when you make a git
  commit. This can be done by installing `pre-commit`:

    $ python -m pip install pre-commit

  From the root of the geopandas repository, you should then install
  `pre-commit`:

    $ pre-commit install

  Then `ruff` will be run automatically each time you
  commit changes. You can skip these checks with `git commit
  --no-verify`. You can also configure your local git clone to have
  `git blame` ignore the commits that introduced large formatting-only
  changes with:

    $ git config blame.ignoreRevsFile .git-blame-ignore-revs
