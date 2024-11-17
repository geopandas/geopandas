__all__ = []
available = []  # previously part of __all__
_prev_available = ["naturalearth_cities", "naturalearth_lowres", "nybb"]


def get_path(dataset):
    ne_message = "https://www.naturalearthdata.com/downloads/110m-cultural-vectors/."
    nybb_message = (
        "the geodatasets package.\n\nfrom geodatasets import get_path\n"
        "path_to_file = get_path('nybb')\n"
    )
    error_msg = (
        "The geopandas.dataset has been deprecated and was removed in GeoPandas "
        f"1.0. You can get the original '{dataset}' data from "
        f"{ne_message if 'natural' in dataset else nybb_message}"
    )
    if dataset in _prev_available:
        raise AttributeError(error_msg)
    else:
        error_msg = (
            "The geopandas.dataset has been deprecated and "
            "was removed in GeoPandas 1.0. New sample datasets are now available "
            "in the geodatasets package (https://geodatasets.readthedocs.io/en/latest/)"
        )
        raise AttributeError(error_msg)
