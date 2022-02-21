"""
Lightweight options machinery.

Based on https://github.com/topper-123/optioneer, but simplified (don't deal
with nested options, deprecated options, ..), just the attribute-style dict
like holding the options and giving a nice repr.
"""
from collections import namedtuple
import textwrap


Option = namedtuple("Option", "key default_value doc validator callback")


class Options(object):
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
            descr = u"{key}: {cur!r} [default: {default!r}]\n".format(
                key=key, cur=self._config[key], default=option.default_value
            )
            description += descr

            if option.doc:
                doc_text = "\n".join(textwrap.wrap(option.doc, width=70))
            else:
                doc_text = u"No description available."
            doc_text = textwrap.indent(doc_text, prefix="    ")
            description += doc_text + "\n"
        space = "\n  "
        description = description.replace("\n", space)
        return "{}({}{})".format(cls, space, description)


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


def _validate_bool(value):
    if not isinstance(value, bool):
        raise TypeError("Expected bool value, got {0}".format(type(value)))


def _default_use_pygeos():
    import geopandas._compat as compat

    return compat.USE_PYGEOS


def _callback_use_pygeos(key, value):
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
