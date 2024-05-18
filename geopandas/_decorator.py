from textwrap import dedent
from typing import Callable, Union

# doc decorator function ported with modifications from Pandas
# https://github.com/pandas-dev/pandas/blob/master/pandas/util/_decorators.py


def doc(*docstrings: Union[str, Callable], **params) -> Callable:
    """
    A decorator take docstring templates, concatenate them and perform string
    substitution on it.
    This decorator will add a variable "_docstring_components" to the wrapped
    callable to keep track the original docstring template for potential usage.
    If it should be consider as a template, it will be saved as a string.
    Otherwise, it will be saved as callable, and later user __doc__ and dedent
    to get docstring.

    Parameters
    ----------
    *docstrings : str or callable
        The string / docstring / docstring template to be appended in order
        after default docstring under callable.
    **params
        The string which would be used to format docstring template.
    """

    def decorator(decorated: Callable) -> Callable:
        # collecting docstring and docstring templates
        docstring_components: list[Union[str, Callable]] = []
        if decorated.__doc__:
            docstring_components.append(dedent(decorated.__doc__))

        for docstring in docstrings:
            if hasattr(docstring, "_docstring_components"):
                docstring_components.extend(docstring._docstring_components)
            elif isinstance(docstring, str) or docstring.__doc__:
                docstring_components.append(docstring)

        # formatting templates and concatenating docstring
        decorated.__doc__ = "".join(
            (
                component.format(**params)
                if isinstance(component, str)
                else dedent(component.__doc__ or "")
            )
            for component in docstring_components
        )

        decorated._docstring_components = docstring_components
        return decorated

    return decorator
