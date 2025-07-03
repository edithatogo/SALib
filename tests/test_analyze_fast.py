from SALib import ProblemSpec
from SALib.test_functions import lake_problem
import random


def test_odd_sample_size():
    """Specific regression test to ensure odd number of samples are handled."""
    sp = ProblemSpec(
        {
            "names": ["a", "q", "b", "mean", "stdev", "delta", "alpha"],
            "bounds": [
                [0.0, 0.1],
                [2.0, 4.5],
                [0.1, 0.45],
                [0.01, 0.05],
                [0.001, 0.005],
                [0.93, 0.99],
                [0.2, 0.5],
            ],
            "outputs": ["max_P", "Utility", "Inertia", "Reliability"],
        }
    )

    # eFAST analysis with a series of odd sample sizes
    # These should all pass and not raise IndexError.
    (sp.sample_fast(511).evaluate(lake_problem.evaluate).analyze_fast())

    (sp.sample_fast(1313).evaluate(lake_problem.evaluate).analyze_fast())

    odds = [random.randrange(5099, 6500, 2) for _ in range(0, 3)]

    for No in odds:
        (sp.sample_fast(No).evaluate(lake_problem.evaluate).analyze_fast())


def test_even_sample_size():
    """Specific regression test to ensure odd number of samples are handled."""
    sp = ProblemSpec(
        {
            "names": ["a", "q", "b", "mean", "stdev", "delta", "alpha"],
            "bounds": [
                [0.0, 0.1],
                [2.0, 4.5],
                [0.1, 0.45],
                [0.01, 0.05],
                [0.001, 0.005],
                [0.93, 0.99],
                [0.2, 0.5],
            ],
            "outputs": ["max_P", "Utility", "Inertia", "Reliability"],
        }
    )

    # Test specific even number
    (sp.sample_fast(1024).evaluate(lake_problem.evaluate).analyze_fast())

    evens = [random.randrange(5026, 6500, 2) for _ in range(0, 3)]

    for Ne in evens:
        (sp.sample_fast(Ne).evaluate(lake_problem.evaluate).analyze_fast())


from SALib.analyze import fast
from SALib.sample import fast_sampler
from SALib.test_functions import Ishigami
import numpy as np
from numpy.testing import assert_allclose


def test_fast_analyze_parallel():
    """
    Tests FAST analysis with parallel computation using ProblemSpec.
    Compares results against a serial run.
    """
    problem = {
        "num_vars": 3,
        "names": ["x1", "x2", "x3"],
        "bounds": [[-np.pi, np.pi]] * 3,
    }

    N_samples = 1025  # Keep N odd as per FAST typical usage, and large enough for some work
    num_resamples_conf = 50 # Fewer resamples for test speed

    # Generate samples (once for both tests)
    param_values = fast_sampler.sample(problem, N_samples, seed=1234)
    Y = Ishigami.evaluate(param_values)

    # Serial analysis
    Si_serial = fast.analyze(
        problem,
        Y,
        M=4,
        num_resamples=num_resamples_conf,
        conf_level=0.95,
        print_to_console=False,
        seed=1234
    )

    # Parallel analysis
    Si_parallel = fast.analyze(
        problem,
        Y,
        M=4,
        num_resamples=num_resamples_conf,
        conf_level=0.95,
        print_to_console=False,
        seed=1234,
        parallel=True,
        n_processors=2
    )

    # Compare key results
    assert_allclose(Si_parallel["S1"], Si_serial["S1"], rtol=1e-5, err_msg="Parallel 'S1' differs from serial")
    assert_allclose(Si_parallel["ST"], Si_serial["ST"], rtol=1e-5, err_msg="Parallel 'ST' differs from serial")

    # Confidence intervals for FAST can also be compared, expecting them to be close
    assert_allclose(Si_parallel["S1_conf"], Si_serial["S1_conf"], rtol=1e-1, err_msg="Parallel 'S1_conf' differs significantly from serial")
    assert_allclose(Si_parallel["ST_conf"], Si_serial["ST_conf"], rtol=1e-1, err_msg="Parallel 'ST_conf' differs significantly from serial")

    # Check names
    assert np.array_equal(Si_parallel["names"], problem["names"])

    # Example using ProblemSpec interface (optional, but good to ensure it also works)
    sp = ProblemSpec(problem)
    # Evaluate is done above with Y

    # Store Y in ProblemSpec - Note: evaluate() normally does this.
    # We need to set it manually if we are not calling sp.evaluate()
    # However, fast.analyze takes Y directly, not from ProblemSpec.results.
    # The ProblemSpec analyze_fast method would internally pass Y.

    # To test ProblemSpec chaining:
    # Create a new Y for this specific test to avoid state issues if sp.evaluate modified Y
    param_values_ps = fast_sampler.sample(sp.problem, N_samples, seed=1234)
    Y_ps = Ishigami.evaluate(param_values_ps)
    sp.samples = param_values_ps # Manually set samples for sp
    sp.results = Y_ps # Manually set results for sp

    sp_serial = ProblemSpec(problem)
    sp_serial.samples = param_values_ps
    sp_serial.results = Y_ps
    sp_serial.analyze_fast(M=4, num_resamples=num_resamples_conf, seed=1234, parallel=False)

    sp_parallel = ProblemSpec(problem)
    sp_parallel.samples = param_values_ps
    sp_parallel.results = Y_ps
    sp_parallel.analyze_fast(M=4, num_resamples=num_resamples_conf, seed=1234, parallel=True, n_processors=2)

    assert_allclose(sp_parallel.analysis['S1'], sp_serial.analysis['S1'], rtol=1e-5)
    assert_allclose(sp_parallel.analysis['ST'], sp_serial.analysis['ST'], rtol=1e-5)
    assert_allclose(sp_parallel.analysis['S1_conf'], sp_serial.analysis['S1_conf'], rtol=1e-1)
    assert_allclose(sp_parallel.analysis['ST_conf'], sp_serial.analysis['ST_conf'], rtol=1e-1)
