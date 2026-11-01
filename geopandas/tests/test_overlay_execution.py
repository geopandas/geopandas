import os
from typing import Optional, Type

import pytest

from ._execution_test.generators import (
    RandomCenterTargetsGenerator,
    RandomPolyGenerator,
    RandomRadiusTargetsGenerator,
    RandomSpotsGenerator,
)
from ._execution_test.runner import run_execution_and_timing_test

generator_types = [
    RandomCenterTargetsGenerator,
    RandomRadiusTargetsGenerator,
    RandomSpotsGenerator,
]


@pytest.mark.skipif(
    os.getenv("OVERLAY_EXECUTION_TEST_ENABLED") != "1",
    reason="Only run if switched on explicitly",
)
@pytest.mark.parametrize("precision", [None, 1])
@pytest.mark.parametrize("generator_t", generator_types)
def test_execution(generator_t: Type[RandomPolyGenerator], precision: Optional[float]):
    generator = generator_t(precision=precision)
    n_procs = int(os.getenv("OVERLAY_EXECUTION_TEST_NPROC", os.cpu_count()))
    n_tests = int(os.getenv("OVERLAY_EXECUTION_TEST_NTESTS", str(10 * n_procs)))
    assert n_procs > 0
    assert n_tests >= n_procs
    df = run_execution_and_timing_test(generator, n_procs, n_tests)
    error_df = df.loc[~df.error.isna()]
    assert len(error_df) == 0, f"Failed seeds: {error_df.seed.unique()=}"
