# -*- coding: utf-8 -*-

import numpy as np
from numpy.testing import assert_allclose, assert_equal

from SALib.analyze import delta
from SALib.sample import latin
from SALib.test_functions import Ishigami
from SALib import ProblemSpec


def test_delta_analyze_parallel():
    """
    Tests Delta Moment-Independent Analysis with parallel computation.
    Compares results against a serial run.
    """
    problem_dict = {
        "num_vars": 3,
        "names": ["x1", "x2", "x3"],
        "bounds": [[-np.pi, np.pi]] * 3,
    }
    problem = ProblemSpec(problem_dict)

    # Generate samples - Latin Hypercube
    param_values = latin.sample(problem, N=1000, seed=101) # N matches example in delta.py

    # Run model (Ishigami)
    Y = Ishigami.evaluate(param_values)

    num_resamples_conf = 50 # Fewer resamples for test speed

    # Serial analysis
    Si_serial = delta.analyze(
        problem,
        param_values,
        Y,
        num_resamples=num_resamples_conf,
        conf_level=0.95,
        print_to_console=False,
        seed=1234
    )

    # Parallel analysis
    Si_parallel = delta.analyze(
        problem,
        param_values,
        Y,
        num_resamples=num_resamples_conf,
        conf_level=0.95,
        print_to_console=False,
        seed=1234,
        parallel=True,
        n_processors=2
    )

    # Compare key results
    assert_allclose(Si_parallel["delta"], Si_serial["delta"], rtol=1e-5, err_msg="Parallel 'delta' differs from serial")
    assert_allclose(Si_parallel["S1"], Si_serial["S1"], rtol=1e-5, err_msg="Parallel 'S1' differs from serial")

    # Confidence intervals
    assert_allclose(Si_parallel["delta_conf"], Si_serial["delta_conf"], rtol=1e-1, err_msg="Parallel 'delta_conf' differs significantly from serial")
    assert_allclose(Si_parallel["S1_conf"], Si_serial["S1_conf"], rtol=1e-1, err_msg="Parallel 'S1_conf' differs significantly from serial")

    # Check names
    assert_equal(Si_parallel["names"], problem["names"])

    # Test with ProblemSpec interface as well
    sp_serial = ProblemSpec(problem_dict)
    sp_serial.sample_latin(N=1000, seed=101)
    sp_serial.evaluate(Ishigami.evaluate)
    sp_serial.analyze_delta(num_resamples=num_resamples_conf, seed=1234, parallel=False)

    sp_parallel = ProblemSpec(problem_dict)
    sp_parallel.sample_latin(N=1000, seed=101)
    sp_parallel.evaluate(Ishigami.evaluate)
    sp_parallel.analyze_delta(num_resamples=num_resamples_conf, seed=1234, parallel=True, n_processors=2)

    assert_allclose(sp_parallel.analysis['delta'], sp_serial.analysis['delta'], rtol=1e-5)
    assert_allclose(sp_parallel.analysis['S1'], sp_serial.analysis['S1'], rtol=1e-5)
    assert_allclose(sp_parallel.analysis['delta_conf'], sp_serial.analysis['delta_conf'], rtol=1e-1)
    assert_allclose(sp_parallel.analysis['S1_conf'], sp_serial.analysis['S1_conf'], rtol=1e-1)
