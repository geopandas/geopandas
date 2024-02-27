from textwrap import dedent

from geopandas._decorator import doc


@doc(method="cumsum", operation="sum")
def cumsum(whatever):
    """
    This is the {method} method.

    It computes the cumulative {operation}.
    """
    ...


@doc(
    cumsum,
    dedent(
        """
        Examples
        --------

        >>> cumavg([1, 2, 3])
        2
        """
    ),
    method="cumavg",
    operation="average",
)
def cumavg(whatever): ...


@doc(cumsum, method="cummax", operation="maximum")
def cummax(whatever): ...


@doc(cummax, method="cummin", operation="minimum")
def cummin(whatever): ...


def test_docstring_formatting():
    docstr = dedent(
        """
        This is the cumsum method.

        It computes the cumulative sum.
        """
    )
    assert cumsum.__doc__ == docstr


def test_docstring_appending():
    docstr = dedent(
        """
        This is the cumavg method.

        It computes the cumulative average.

        Examples
        --------

        >>> cumavg([1, 2, 3])
        2
        """
    )
    assert cumavg.__doc__ == docstr


def test_doc_template_from_func():
    docstr = dedent(
        """
        This is the cummax method.

        It computes the cumulative maximum.
        """
    )
    assert cummax.__doc__ == docstr


def test_inherit_doc_template():
    docstr = dedent(
        """
        This is the cummin method.

        It computes the cumulative minimum.
        """
    )
    assert cummin.__doc__ == docstr
