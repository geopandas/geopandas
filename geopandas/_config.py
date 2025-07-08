"""
Lightweight options machinery.

Based on https://github.com/topper-123/optioneer, but simplified (don't deal
with nested options, deprecated options, ..), just the attribute-style dict
like holding the options and giving a nice repr.
"""
from collections import namedtuple
import textwrap
from typing import Optional, Callable


Option = namedtuple("Option", "key default_value doc validator callback")


class Options:
    """Provide attribute-style access to configuration dict."""

    def __init__(self, options):
        super().__setattr__("_options", options)
        # populate with default values
        config = {}
        for key, option in options.items():
            config[key] = option.default_value

        super().__setattr__("_config", config)

    def __setattr__(self, key, value):
        # you can't set new keys
        if key in self._config:
            option = self._options[key]
            if option.validator:
                option.validator(value)
            self._config[key] = value
            if option.callback:
                option.callback(key, value)
        else:
            msg = "You can only set the value of existing options"
            raise AttributeError(msg)

    def __getattr__(self, key):
        try:
            return self._config[key]
        except KeyError:
            raise AttributeError("No such option")

    def __dir__(self):
        return list(self._config.keys())

    def __repr__(self):
        cls = self.__class__.__name__
        description = ""
        for key, option in self._options.items():
            descr = f"{key}: {self._config[key]!r} [default: {option.default_value!r}]\n"
            description += descr

            if option.doc:
                doc_text = "\n".join(textwrap.wrap(option.doc, width=70))
            else:
                doc_text = "No description available."
            doc_text = indent(doc_text, prefix="    ")
            description += doc_text + "\n"
        space = "\n  "
        description = description.replace("\n", space)
        return f"{cls}({space}{description})"


def indent(text, prefix, predicate: Optional[Callable] = None) -> str:
    """
    This is the python 3 textwrap.indent function, which is not available in
    python 2.
    """
    if predicate is None:

        def predicate(line):
            return line.strip()

    def prefixed_lines():
        for line in text.splitlines(True):
            yield (prefix + line if predicate(line) else line)

    return "".join(prefixed_lines())


def _validate_display_precision(value) -> None:
    if value is not None:
        if not isinstance(value, int) or not (0 <= value <= 16):
            raise ValueError("Invalid value, needs to be an integer [0-16]")


display_precision = Option(
    key="display_precision",
    default_value=None,
    doc=(
        "The precision (maximum number of decimals) of the coordinates in "
        "the WKT representation in the Series/DataFrame display. "
        "By default (None), it tries to infer and use 3 decimals for projected "
        "coordinates and 5 decimals for geographic coordinates."
    ),
    validator=_validate_display_precision,
    callback=None,
)


def _validate_bool(value) -> None:
    if not isinstance(value, bool):
        raise TypeError(f"Expected bool value, got {type(value)}")


def _default_use_pygeos():
    import geopandas._compat as compat

    return compat.USE_PYGEOS


def _callback_use_pygeos(key, value) -> None:
    assert key == "use_pygeos"
    import geopandas._compat as compat

    compat.set_use_pygeos(value)


use_pygeos = Option(
    key="use_pygeos",
    default_value=_default_use_pygeos(),
    doc=(
        "Whether to use PyGEOS to speed up spatial operations. The default is True "
        "if PyGEOS is installed, and follows the USE_PYGEOS environment variable "
        "if set."
    ),
    validator=_validate_bool,
    callback=_callback_use_pygeos,
)


options = Options({"display_precision": display_precision, "use_pygeos": use_pygeos})
