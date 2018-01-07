import os


__all__ = ['available', 'get_path']

_module_path = os.path.dirname(__file__)
_available_dir = [p for p in next(os.walk(_module_path))[1]
                  if not p.startswith('__')]
_available_zip = {'nybb': 'nybb_16a.zip'}
available = _available_dir + list(_available_zip.keys())


def get_path(dataset):
    """
    Get the path to the data file.

    Parameters
    ----------
    dataset : str
        The name of the dataset. See ``geopandas.datasets.available`` for
        all options.

    """
    if dataset in _available_dir:
        return os.path.abspath(
            os.path.join(_module_path, dataset, dataset + '.shp'))
    elif dataset in _available_zip:
        fpath = os.path.abspath(
            os.path.join(_module_path, _available_zip[dataset]))
        return 'zip://' + fpath
    else:
        msg = "The dataset '{data}' is not available".format(data=dataset)
        raise ValueError(msg)
