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
        super(Options, self).__setattr__("_options", options)
        # populate with default values
        config = {}
        for key, option in options.items():
            config[key] = option.default_value

        super(Options, self).__setattr__("_config", config)

    def __setattr__(self, key, value):
        # you can't set new keys
        if key in self._config:
            option = self._options[key]
            if option.validator:
                option.validator(value)
            self._config[key] = value
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
            doc_text = indent(doc_text, prefix="    ")
            description += doc_text
        space = "\n  "
        description = description.replace("\n", space)
        return "{}({}{})".format(cls, space, description)


def indent(text, prefix, predicate=None):
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

options = Options({"display_precision": display_precision})
