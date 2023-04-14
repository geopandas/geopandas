import os

from warnings import warn

__all__ = ["available", "get_path"]

_module_path = os.path.dirname(__file__)
_available_dir = [p for p in next(os.walk(_module_path))[1] if not p.startswith("__")]
_available_zip = {"nybb": "nybb_16a.zip"}
available = _available_dir + list(_available_zip.keys())


def get_path(dataset):
    """
    Get the path to the data file.

    Parameters
    ----------
    dataset : str
        The name of the dataset. See ``geopandas.datasets.available`` for
        all options.

    Examples
    --------
    >>> geopandas.datasets.get_path("naturalearth_lowres")  # doctest: +SKIP
    '.../python3.8/site-packages/geopandas/datasets/\
naturalearth_lowres/naturalearth_lowres.shp'

    """
    ne_message = "https://www.naturalearthdata.com/downloads/110m-cultural-vectors/."
    nybb_message = (
        "the geodatasets package.\n\nfrom geodatasets import get_path\n"
        "path_to_file = get_path('nybb')\n"
    )
    depr_warning = (
        "The geopandas.dataset module is deprecated and will be removed in GeoPandas "
        f"1.0. You can get the original '{dataset}' data from "
        f"{ne_message if 'natural' in dataset else nybb_message}"
    )

    if dataset in _available_dir:
        warn(
            depr_warning,
            FutureWarning,
            stacklevel=2,
        )
        return os.path.abspath(os.path.join(_module_path, dataset, dataset + ".shp"))
    elif dataset in _available_zip:
        warn(
            depr_warning,
            FutureWarning,
            stacklevel=2,
        )
        fpath = os.path.abspath(os.path.join(_module_path, _available_zip[dataset]))
        return "zip://" + fpath
    else:
        msg = "The dataset '{data}' is not available. ".format(data=dataset)
        msg += "Available datasets are {}".format(", ".join(available))
        raise ValueError(msg)
