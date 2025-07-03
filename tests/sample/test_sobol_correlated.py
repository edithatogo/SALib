# -*- coding: utf-8 -*-
"""
Tests for the correlated Sobol sampler.
"""
import numpy as np
from numpy.testing import assert_allclose, assert_equal
from pytest import raises

from SALib.sample.sobol_correlated import sample as sample_sobol_correlated
from SALib.util.problem import ProblemSpec
from scipy.stats import spearmanr


# Common problem definition for tests
def get_test_problem_correlated(num_vars=3):
    if num_vars == 2:
        problem_dict = {
            "num_vars": 2,
            "names": ["x1", "x2"],
            "bounds": [[0.0, 1.0], [0.0, 1.0]], # Keep marginals U[0,1] for easy rank corr check
            "corr_matrix": np.array([[1.0, 0.7],
                                     [0.7, 1.0]]),
            # No dists needed if bounds are [0,1] for U[0,1] output from latin sampler
        }
    elif num_vars == 3:
        problem_dict = {
            "num_vars": 3,
            "names": ["x1", "x2", "x3"],
            "bounds": [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
            "corr_matrix": np.array([[1.0, 0.7, 0.3],
                                     [0.7, 1.0, 0.5],
                                     [0.3, 0.5, 1.0]]),
        }
    else:
        raise ValueError("num_vars must be 2 or 3 for this test problem setup")
    return ProblemSpec(problem_dict)


class TestSobolCorrelatedSample:
    def test_missing_corr_matrix_error(self):
        problem_dict_no_corr = {
            "num_vars": 2,
            "names": ["x1", "x2"],
            "bounds": [[0, 1], [0, 1]],
        }
        problem = ProblemSpec(problem_dict_no_corr)
        with raises(ValueError, match="A `corr_matrix` must be defined"):
            sample_sobol_correlated(problem, N=10)

    def test_groups_and_corr_matrix_error(self):
        problem_dict_groups_corr = {
            "num_vars": 2,
            "names": ["x1", "x2"],
            "bounds": [[0, 1], [0, 1]],
            "groups": ["g1", "g1"],
            "corr_matrix": np.array([[1.0, 0.5], [0.5, 1.0]])
        }
        problem = ProblemSpec(problem_dict_groups_corr)
        with raises(ValueError, match="does not support grouped parameters"):
            sample_sobol_correlated(problem, N=10)

    def test_output_shape(self):
        N_test = 50
        D_test = 3
        problem = get_test_problem_correlated(num_vars=D_test)
        samples = sample_sobol_correlated(problem, N=N_test, seed=123)

        expected_rows = N_test * (2 + 2 * D_test)
        expected_cols = D_test
        assert samples.shape == (expected_rows, expected_cols), \
            f"Expected shape ({expected_rows}, {expected_cols}), got {samples.shape}"

    def test_base_samples_correlation_and_structure(self):
        N_test = 1000 # Reasonably large N for better correlation estimate
        D_test = 2
        problem = get_test_problem_correlated(num_vars=D_test)

        # We set bounds to [0,1] and no dists, so latin_sample output is U[0,1]
        # This makes checking rank correlation straightforward.

        all_samples = sample_sobol_correlated(problem, N=N_test, seed=456)

        X_A = all_samples[0:N_test, :]
        X_B = all_samples[N_test:2*N_test, :]

        # Check X_A correlations
        spearman_A, _ = spearmanr(X_A)
        # If D_test=2, spearmanr returns a scalar if applied to 2xN, or matrix if Nx2
        # Ensure spearmanr output is a matrix if D_test > 1
        if D_test == 2 :
             assert spearman_A[0,1] == approx(problem['corr_matrix'][0,1], abs=0.1) # Looser tolerance for sampling
        else : # spearman_A is already the matrix
             assert np.allclose(spearman_A, problem['corr_matrix'], atol=0.1)


        # Check X_B correlations (should also match, due to independent generation with same spec)
        spearman_B, _ = spearmanr(X_B)
        if D_test == 2 :
            assert spearman_B[0,1] == approx(problem['corr_matrix'][0,1], abs=0.1)
        else:
            assert np.allclose(spearman_B, problem['corr_matrix'], atol=0.1)

        # Check structure of one X_Ci and one X_Di
        # X_C0: col 0 from X_A, other cols from X_B
        X_C0_expected_col0 = X_A[:, 0]
        X_C0_actual_col0 = all_samples[2*N_test : 3*N_test, 0] # First C matrix (C0)
        assert_allclose(X_C0_actual_col0, X_C0_expected_col0)

        if D_test > 1:
            X_C0_expected_col1 = X_B[:, 1]
            X_C0_actual_col1 = all_samples[2*N_test : 3*N_test, 1]
            assert_allclose(X_C0_actual_col1, X_C0_expected_col1)

        # X_D0: col 0 from X_B, other cols from X_A
        # This is the (2+D)*N_test to (2+D+1)*N_test block
        idx_start_XD0 = N_test * (2 + D_test)
        idx_end_XD0 = N_test * (2 + D_test + 1)

        X_D0_expected_col0 = X_B[:, 0]
        X_D0_actual_col0 = all_samples[idx_start_XD0:idx_end_XD0, 0]
        assert_allclose(X_D0_actual_col0, X_D0_expected_col0)

        if D_test > 1:
            X_D0_expected_col1 = X_A[:, 1]
            X_D0_actual_col1 = all_samples[idx_start_XD0:idx_end_XD0, 1]
            assert_allclose(X_D0_actual_col1, X_D0_expected_col1)

    def test_sampler_with_3vars(self):
        N_test = 1000
        D_test = 3
        problem = get_test_problem_correlated(num_vars=D_test)
        all_samples = sample_sobol_correlated(problem, N=N_test, seed=789)

        X_A = all_samples[0:N_test, :]
        X_B = all_samples[N_test:2*N_test, :]

        expected_rows = N_test * (2 + 2 * D_test)
        assert all_samples.shape == (expected_rows, D_test)

        spearman_A, _ = spearmanr(X_A)
        assert np.allclose(spearman_A, problem['corr_matrix'], atol=0.1) # Check entire matrix

        spearman_B, _ = spearmanr(X_B)
        assert np.allclose(spearman_B, problem['corr_matrix'], atol=0.1)

        # Check structure of X_C1 (second C matrix)
        # X_C1: col 1 from X_A, cols 0, 2 from X_B
        idx_start_XC1 = N_test * (2 + 1) # C0 is at 2N, C1 is at (2+1)N
        idx_end_XC1 = N_test * (2 + 1 + 1)
        X_C1_sampled = all_samples[idx_start_XC1:idx_end_XC1, :]

        assert_allclose(X_C1_sampled[:,0], X_B[:,0]) # col 0 from B
        assert_allclose(X_C1_sampled[:,1], X_A[:,1]) # col 1 from A
        assert_allclose(X_C1_sampled[:,2], X_B[:,2]) # col 2 from B
