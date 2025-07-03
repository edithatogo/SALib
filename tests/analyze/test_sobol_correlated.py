# -*- coding: utf-8 -*-
"""
Tests for the correlation-aware Sobol analyzer.
"""
import numpy as np
from numpy.testing import assert_allclose, assert_equal
from pytest import raises

from SALib.analyze.sobol_correlated import analyze as analyze_sobol_correlated
from SALib.sample.sobol_correlated import sample as sample_sobol_correlated # To generate samples for tests
from SALib.test_functions import Ishigami # Example test function
from SALib.util.problem import ProblemSpec
from SALib.test_functions import linear_model


# Helper to define a standard correlated problem for testing
def get_test_problem_correlated_for_analyze(num_vars=3):
    if num_vars == 3:
        problem_dict = {
            "num_vars": 3,
            "names": ["x1", "x2", "x3"],
            "bounds": [[-np.pi, np.pi]] * num_vars, # Ishigami bounds
            "corr_matrix": np.array([[1.0, 0.6, 0.2],
                                     [0.6, 1.0, 0.4],
                                     [0.2, 0.4, 1.0]]),
            # No dists, defaults to uniform for Ishigami
        }
    else: # Default to 2 vars for simplicity in some tests
        problem_dict = {
            "num_vars": 2,
            "names": ["x1", "x2"],
            "bounds": [[-1.0, 1.0]] * num_vars,
            "corr_matrix": np.array([[1.0, 0.8],
                                     [0.8, 1.0]]),
        }
    return ProblemSpec(problem_dict)


class TestSobolCorrelatedAnalyze:

    def test_analyze_ishigami_runs(self):
        """Test that analysis runs with Ishigami function (smoke test)."""
        N_test = 100  # Base samples, total will be N*(2+2D)
        problem = get_test_problem_correlated_for_analyze(num_vars=3)

        param_values = sample_sobol_correlated(problem, N=N_test, seed=101)
        Y = Ishigami.evaluate(param_values)

        Si = analyze_sobol_correlated(problem, Y, num_resamples=10, seed=123) # Low resamples for speed

        assert "S1_full" in Si
        assert "ST_full" in Si
        assert "S1_full_conf" in Si
        assert "ST_full_conf" in Si
        assert len(Si["S1_full"]) == problem["num_vars"]

        # Basic plausibility checks
        assert np.all(Si["S1_full"] > -0.1) # Allow for small negative due to estimator variance
        assert np.all(Si["ST_full"] > -0.1)
        # For Ishigami, x1 and x2 are important, x3 is less so (though correlations complicate)
        # No strict check on values due to correlation, just that it runs.

    def test_analyze_linear_model_basic_properties(self):
        """Test with a simple linear model."""
        N_test = 500 # More samples for better estimates
        problem_dict = {
            "num_vars": 2,
            "names": ["x1", "x2"],
            "bounds": [[0, 1], [0, 1]], # Uniform marginals
            "corr_matrix": np.array([[1.0, 0.5],
                                     [0.5, 1.0]]),
            "dists": ['unif', 'unif']
        }
        problem = ProblemSpec(problem_dict)

        # Model: Y = 2*X1 + 0*X2 (X2 is non-influential directly)
        coeffs = np.array([2.0, 0.0])

        param_values = sample_sobol_correlated(problem, N=N_test, seed=201)
        Y = linear_model.evaluate(param_values, coeffs=coeffs)

        Si = analyze_sobol_correlated(problem, Y, num_resamples=50, seed=202)

        # X2 has no direct effect, but is correlated with X1.
        # Its S1_full might not be exactly zero due to correlation estimator properties.
        # Its ST_full might also not be zero.
        # This test mainly checks if it runs and if X1 is more important.
        assert Si["S1_full"][0] > Si["S1_full"][1] # X1 should be more important
        assert Si["ST_full"][0] > Si["ST_full"][1]

        # S1_full for X2 should be small, ST_full for X2 might pick up correlation effects
        # For this conceptual estimator, Cov(Y_A, Y_C2) where Y_C2 = 2*X_A1 (since X_B2 has coeff 0)
        # So Cov(2*X_A1, 2*X_A1) / V(Y) = V(2*X_A1)/V(Y) which is S1 for X1.
        # This implies S1_full for X2 might be zero if estimator is Cov(Y_A, Y_C2).
        # Let's check if S1_full[1] (for X2) is small.
        assert abs(Si["S1_full"][1]) < 0.1 # Should be close to 0

        # ST_full for X2: 0.5 * E[(Y_A - Y_D2)^2] / V(Y)
        # Y_A = 2*X_A1
        # Y_D2 = f(X_A1, X_B2) = 2*X_A1. So Y_A - Y_D2 = 0. ST_full[1] should be 0.
        assert abs(Si["ST_full"][1]) < 0.05 # Should be very close to 0


    def test_constant_output(self):
        """Test behavior when model output Y is constant."""
        N_test = 50
        problem = get_test_problem_correlated_for_analyze(num_vars=2)
        param_values = sample_sobol_correlated(problem, N=N_test, seed=301)
        Y = np.ones(param_values.shape[0]) * 5.0 # Constant output

        with raises(UserWarning, match="Total variance of model output is zero"):
             Si = analyze_sobol_correlated(problem, Y, num_resamples=10, seed=302)

        # In new implementation, it warns and returns zeros/nans
        Si_const = analyze_sobol_correlated(problem, Y, num_resamples=10, seed=303, print_to_console=False) # suppress print for clean test
        assert_allclose(Si_const["S1_full"], 0.0)
        assert_allclose(Si_const["ST_full"], 0.0)
        assert np.all(np.isnan(Si_const["S1_full_conf"]))
        assert np.all(np.isnan(Si_const["ST_full_conf"]))


    def test_incorrect_y_size(self):
        problem = get_test_problem_correlated_for_analyze(num_vars=2)
        Y_wrong_size = np.random.rand(10) # Clearly wrong size
        with raises(ValueError, match="Incorrect number of samples in Y"):
            analyze_sobol_correlated(problem, Y_wrong_size)

    def test_parallel_vs_serial_bootstrap(self):
        """Compare confidence intervals from serial and parallel bootstrap."""
        N_test = 100
        problem = get_test_problem_correlated_for_analyze(num_vars=3)
        param_values = sample_sobol_correlated(problem, N=N_test, seed=401)
        Y = Ishigami.evaluate(param_values) # Use a non-trivial function

        Si_serial = analyze_sobol_correlated(problem, Y, num_resamples=50, seed=402, parallel=False)
        Si_parallel = analyze_sobol_correlated(problem, Y, num_resamples=50, seed=402, parallel=True, n_processors=2)

        # Main indices should be identical as they don't depend on bootstrap seed path (if main seed is same)
        assert_allclose(Si_parallel["S1_full"], Si_serial["S1_full"], rtol=1e-9)
        assert_allclose(Si_parallel["ST_full"], Si_serial["ST_full"], rtol=1e-9)

        # Confidence intervals should be close (minor diffs due to RNG in bootstrap workers)
        assert_allclose(Si_parallel["S1_full_conf"], Si_serial["S1_full_conf"], rtol=0.2, atol=0.01) # Looser tolerance
        assert_allclose(Si_parallel["ST_full_conf"], Si_serial["ST_full_conf"], rtol=0.2, atol=0.01)

    def test_print_to_console(self, capsys):
        N_test = 20
        problem = get_test_problem_correlated_for_analyze(num_vars=2)
        param_values = sample_sobol_correlated(problem, N=N_test, seed=501)
        Y = linear_model.evaluate(param_values, coeffs=np.array([1.0, 0.5]))

        analyze_sobol_correlated(problem, Y, num_resamples=5, seed=502, print_to_console=True)
        captured = capsys.readouterr()
        assert "S1_full" in captured.out
        assert "ST_full" in captured.out

    def test_s2_full_calculation_and_shape(self):
        """Test S2_full calculation, shape, and calc_second_order flag."""
        N_test = 100
        D_test = 3
        problem = get_test_problem_correlated_for_analyze(num_vars=D_test)
        param_values = sample_sobol_correlated(problem, N=N_test, seed=601, calc_second_order=True) # Ensure sampler provides enough for S2

        # Y = X1 + X2 + X3 (no interactions, S2_full should be small)
        coeffs = np.array([1.0, 1.0, 1.0])
        Y = linear_model.evaluate(param_values, coeffs=coeffs)

        # Test with calc_second_order = True (default)
        Si_s2_true = analyze_sobol_correlated(problem, Y, num_resamples=10, seed=602, calc_second_order=True)

        assert "S2_full" in Si_s2_true
        assert "S2_full_conf" in Si_s2_true
        assert Si_s2_true["S2_full"].shape == (D_test, D_test)
        assert Si_s2_true["S2_full_conf"].shape == (D_test, D_test)

        # For a purely additive model, S2_full terms should be close to zero
        # Allow some noise due to estimators and correlation
        assert np.allclose(Si_s2_true["S2_full"][np.triu_indices(D_test, k=1)], 0.0, atol=0.1)

        # Test symmetry S2_full[i,j] == S2_full[j,i]
        assert np.allclose(Si_s2_true["S2_full"], Si_s2_true["S2_full"].T, atol=1e-9)
        assert np.allclose(Si_s2_true["S2_full_conf"], Si_s2_true["S2_full_conf"].T, atol=1e-9, nan_ok=True)


        # Test with calc_second_order = False
        Si_s2_false = analyze_sobol_correlated(problem, Y, num_resamples=10, seed=603, calc_second_order=False)

        # Option 1: Keys are not present (if not added to ResultDict)
        # assert "S2_full" not in Si_s2_false
        # assert "S2_full_conf" not in Si_s2_false
        # Option 2: Keys are present but values are NaN arrays (current implementation)
        assert "S2_full" in Si_s2_false # It's initialized with NaNs
        assert "S2_full_conf" in Si_s2_false # It's initialized with NaNs
        assert np.all(np.isnan(Si_s2_false["S2_full"]))
        assert np.all(np.isnan(Si_s2_false["S2_full_conf"]))

    def test_s2_full_with_interaction_term(self):
        """Test S2_full with a model having a clear interaction."""
        N_test = 1000 # More samples for interaction terms
        problem_dict_interact = {
            "num_vars": 3,
            "names": ["x1", "x2", "x3"],
            "bounds": [[-1, 1], [-1, 1], [-1, 1]], # Using [-1,1] for simplicity
            "corr_matrix": np.array([[1.0, 0.0, 0.0],  # No correlation for simplicity of this test
                                     [0.0, 1.0, 0.0],
                                     [0.0, 0.0, 1.0]]),
            "dists": ['unif'] * 3
        }
        problem_interact = ProblemSpec(problem_dict_interact)

        # Model: Y = X1 + X2 + 5*X1*X2 + X3
        # (X3 is additive, X1,X2 interact)
        def model_with_interaction(X):
            return X[:,0] + X[:,1] + 5*X[:,0]*X[:,1] + X[:,2]

        param_values = sample_sobol_correlated(problem_interact, N=N_test, seed=701)
        Y = model_with_interaction(param_values)

        Si = analyze_sobol_correlated(problem_interact, Y, num_resamples=20, seed=702, calc_second_order=True)

        # S2_full for (X1,X2) should be significant and larger than for other pairs
        s2_x1_x2 = Si["S2_full"][0,1]
        s2_x1_x3 = Si["S2_full"][0,2]
        s2_x2_x3 = Si["S2_full"][1,2]

        assert s2_x1_x2 > 0.1 # Expecting a reasonably large positive interaction term
        assert abs(s2_x1_x3) < abs(s2_x1_x2) # Interaction with X3 should be smaller
        assert abs(s2_x2_x3) < abs(s2_x1_x2) # Interaction with X3 should be smaller
        assert abs(s2_x1_x3) < 0.1 # Should be small if no true interaction and inputs are independent
        assert abs(s2_x2_x3) < 0.1 # Should be small
