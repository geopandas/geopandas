import os


__all__ = ['available', 'get_path']

module_path = os.path.dirname(__file__)
available = [p for p in next(os.walk(module_path))[1]
             if not p.startswith('__')]


def get_path(dataset):
    """
    Get the path to the data file.

    Parameters
    ----------
    dataset : str
        The name of the dataset. See ``geopandas.datasets.available`` for
        all options.

    """
    if dataset in available:
        return os.path.abspath(
            os.path.join(module_path, dataset, dataset + '.shp'))
    else:
        msg = "The dataset '{data}' is not available".format(data=dataset)
        raise ValueError(msg)
