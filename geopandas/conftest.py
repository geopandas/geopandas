def pytest_ignore_collect(path):
    """
    Pytest doctest configuration to skip test of docstrings in certain files.
    """
    if "datasets" in str(path):
        return True
    if "test_api" in str(path):
        return True
    return False
