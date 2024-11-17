"""
Lightweight options machinery.

Based on https://github.com/topper-123/optioneer, but simplified (don't deal
with nested options, deprecated options, ..), just the attribute-style dict
like holding the options and giving a nice repr.
"""

import textwrap
import warnings
from collections import namedtuple

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
            descr = (
                f"{key}: {self._config[key]!r} [default: {option.default_value!r}]\n"
            )
            description += descr

            if option.doc:
                doc_text = "\n".join(textwrap.wrap(option.doc, width=70))
            else:
                doc_text = "No description available."
            doc_text = textwrap.indent(doc_text, prefix="    ")
            description += doc_text + "\n"
        space = "\n  "
        description = description.replace("\n", space)
        return f"{cls}({space}{description})"


def _validate_display_precision(value):
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


def _warn_use_pygeos_deprecated(_value):
    warnings.warn(
        "pygeos support was removed in 1.0. "
        "geopandas.use_pygeos is a no-op and will be removed in geopandas 1.1.",
        stacklevel=3,
    )


def _validate_io_engine(value):
    if value is not None:
        if value not in ("pyogrio", "fiona"):
            raise ValueError(f"Expected 'pyogrio' or 'fiona', got '{value}'")


io_engine = Option(
    key="io_engine",
    default_value=None,
    doc=(
        "The default engine for ``read_file`` and ``to_file``. "
        "Options are 'pyogrio' and 'fiona'."
    ),
    validator=_validate_io_engine,
    callback=None,
)

# TODO: deprecate this
use_pygeos = Option(
    key="use_pygeos",
    default_value=False,
    doc=(
        "Deprecated option previously used to enable PyGEOS. "
        "It will be removed in GeoPandas 1.1."
    ),
    validator=_warn_use_pygeos_deprecated,
    callback=None,
)

options = Options(
    {
        "display_precision": display_precision,
        "use_pygeos": use_pygeos,
        "io_engine": io_engine,
    }
)
