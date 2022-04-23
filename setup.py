#!/usr/bin/env/python
"""Installation script

See setup.cfg for static metadata.
"""

import os
import sys

from setuptools import setup

# ensure the current directory is on sys.path so versioneer can be imported
# when pip uses PEP 517/518 build rules.
# https://github.com/python-versioneer/python-versioneer/issues/193
sys.path.append(os.path.dirname(__file__))

import versioneer  # noqa: E402

setup(
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
)
