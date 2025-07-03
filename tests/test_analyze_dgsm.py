# -*- coding: utf-8 -*-

import numpy as np
from numpy.testing import assert_allclose, assert_equal

from SALib.analyze import dgsm
from SALib.sample import finite_diff
from SALib.test_functions import Ishigami
from SALib import ProblemSpec


def test_dgsm_analyze_parallel():
    """
    Tests Derivative-based Global Sensitivity Measure (DGSM) with parallel computation.
    Compares results against a serial run.
    """
    problem_dict = {
        "num_vars": 3,
        "names": ["x1", "x2", "x3"],
        "bounds": [[-np.pi, np.pi]] * 3,
    }
    problem = ProblemSpec(problem_dict)

    # Generate samples - Finite Differences
    # N (number of points) results in N * (D + 1) samples
    param_values = finite_diff.sample(problem, N=100, seed=101)

    # Run model (Ishigami)
    Y = Ishigami.evaluate(param_values)

    num_resamples_conf = 50 # Fewer resamples for test speed

    # Serial analysis
    Si_serial = dgsm.analyze(
        problem,
        param_values, # X input for dgsm is the sample matrix
        Y,
        num_resamples=num_resamples_conf,
        conf_level=0.95,
        print_to_console=False,
        seed=1234
    )

    # Parallel analysis
    Si_parallel = dgsm.analyze(
        problem,
        param_values, # X input
        Y,
        num_resamples=num_resamples_conf,
        conf_level=0.95,
        print_to_console=False,
        seed=1234,
        parallel=True,
        n_processors=2
    )

    # Compare key results
    assert_allclose(Si_parallel["dgsm"], Si_serial["dgsm"], rtol=1e-5, err_msg="Parallel 'dgsm' differs from serial")
    assert_allclose(Si_parallel["vi"], Si_serial["vi"], rtol=1e-5, err_msg="Parallel 'vi' differs from serial")
    assert_allclose(Si_parallel["vi_std"], Si_serial["vi_std"], rtol=1e-5, err_msg="Parallel 'vi_std' differs from serial")

    # Confidence intervals
    assert_allclose(Si_parallel["dgsm_conf"], Si_serial["dgsm_conf"], rtol=1e-1, err_msg="Parallel 'dgsm_conf' differs significantly from serial")

    # Check names
    assert_equal(Si_parallel["names"], problem["names"])


    # Test with ProblemSpec interface as well
    sp_serial = ProblemSpec(problem_dict)
    sp_serial.sample_finite_diff(N=100, seed=101)
    sp_serial.evaluate(Ishigami.evaluate)
    sp_serial.analyze_dgsm(num_resamples=num_resamples_conf, seed=1234, parallel=False)

    sp_parallel = ProblemSpec(problem_dict)
    sp_parallel.sample_finite_diff(N=100, seed=101)
    sp_parallel.evaluate(Ishigami.evaluate)
    sp_parallel.analyze_dgsm(num_resamples=num_resamples_conf, seed=1234, parallel=True, n_processors=2)

    assert_allclose(sp_parallel.analysis['dgsm'], sp_serial.analysis['dgsm'], rtol=1e-5)
    assert_allclose(sp_parallel.analysis['vi'], sp_serial.analysis['vi'], rtol=1e-5)
    assert_allclose(sp_parallel.analysis['vi_std'], sp_serial.analysis['vi_std'], rtol=1e-5)
    assert_allclose(sp_parallel.analysis['dgsm_conf'], sp_serial.analysis['dgsm_conf'], rtol=1e-1)
