def _try_import():
    # need to import topojson.py on first use

    try:
        import topojson
    except ImportError:

        # give a nice error message
        raise ImportError("the topojson.py library is not installed\n"
                          "you can install via conda\n"
                          "conda install feather-format -c conda-forge\n"
                          "or via pip\n"
                          "pip install -U feather-format\n")