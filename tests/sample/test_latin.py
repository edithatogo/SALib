from SALib.sample.latin import sample
from SALib.util.problem import ProblemSpec # For easier problem definition
from pytest import approx, raises
import numpy as np
from scipy.stats import spearmanr


class TestLatinSample:
    def test_latin_sample_trivial(self):
        problem = {"num_vars": 1, "bounds": [[0, 1]], "names": ["var1"]}

        actual = sample(problem, 10, seed=42)
        expected = np.array(
            [
                [0.8601115],
                [0.27319939],
                [0.03745401],
                [0.60580836],
                [0.78661761],
                [0.97080726],
                [0.35986585],
                [0.19507143],
                [0.41560186],
                [0.51559945],
            ]
        )
        np.testing.assert_allclose(actual, expected)

    def test_latin_sample_trivial_group(self):
        problem = {
            "num_vars": 1,
            "bounds": [[0, 1]],
            "names": ["var1"],
            "groups": ["group1"],
        }

        actual = sample(problem, 10, seed=42)
        expected = np.array(
            [
                [0.8601115],
                [0.27319939],
                [0.03745401],
                [0.60580836],
                [0.78661761],
                [0.97080726],
                [0.35986585],
                [0.19507143],
                [0.41560186],
                [0.51559945],
            ]
        )
        np.testing.assert_allclose(actual, expected)

    def test_latin_sample_one_group(self):
        problem = {
            "num_vars": 2,
            "bounds": [[0, 1], [0, 1]],
            "names": ["var1", "var2"],
            "groups": ["group1", "group1"],
        }

        actual = sample(problem, 10, seed=42)
        expected = np.array(
            [
                [0.8601115, 0.8601115],
                [0.27319939, 0.27319939],
                [0.03745401, 0.03745401],
                [0.60580836, 0.60580836],
                [0.78661761, 0.78661761],
                [0.97080726, 0.97080726],
                [0.35986585, 0.35986585],
                [0.19507143, 0.19507143],
                [0.41560186, 0.41560186],
                [0.51559945, 0.51559945],
            ]
        )
        np.testing.assert_allclose(actual, expected)

    def test_latin_sample_no_groups(self):
        problem = {
            "num_vars": 2,
            "bounds": [[0, 1], [0, 1]],
            "names": ["var1", "var2"],
            "groups": None,
        }

        actual = sample(problem, 10, seed=42)
        expected = np.array(
            [
                [0.86011150, 0.15247564],
                [0.27319939, 0.84560700],
                [0.03745401, 0.73663618],
                [0.60580836, 0.51394939],
                [0.78661761, 0.46118529],
                [0.97080726, 0.97851760],
                [0.35986585, 0.03042422],
                [0.19507143, 0.32912291],
                [0.41560186, 0.62921446],
                [0.51559945, 0.24319450],
            ]
        )
        approx(actual, expected)

    def test_latin_sample_two_groups(self):
        problem = {
            "num_vars": 2,
            "bounds": [[0, 1], [0, 1]],
            "names": ["var1", "var2"],
            "groups": ["group1", "group2"],
        }

        actual = sample(problem, 10, seed=42)
        expected = np.array(
            [
                [0.17319939, 0.85247564],
                [0.50205845, 0.21559945],
                [0.4601115, 0.59699099],
                [0.83042422, 0.71834045],
                [0.03745401, 0.38661761],
                [0.7181825, 0.15986585],
                [0.68324426, 0.92912291],
                [0.30580836, 0.09507143],
                [0.21560186, 0.47080726],
                [0.9431945, 0.62123391],
            ]
        )
        np.testing.assert_allclose(actual, expected)

    def test_latin_group_constant(self):
        """Ensure grouped parameters have identical values."""
        problem = {
            "num_vars": 6,
            "names": ["P1", "P2", "P3", "P4", "P5", "P6"],
            "bounds": [[-100.0, 100.0] * 6],
            "groups": ["A", "B"] * 3,
        }
        samples = sample(problem, 10, seed=42)

        # Group samples should have the same values
        # Get (max - min) with the `ptp()` method, the result of which should be
        # an array of zeros
        diff = np.ptp(samples[:, ::2], axis=1)
        assert np.all(diff == 0), "Grouped samples do not have the same values"

        diff = np.ptp(samples[:, 1::2], axis=1)
        assert np.all(diff == 0), "Grouped samples do not have the same values"

    def test_latin_correlation_and_groups_error(self):
        problem_dict = {
            "num_vars": 2,
            "names": ["x1", "x2"],
            "bounds": [[0, 1], [0, 1]],
            "groups": ["g1", "g1"], # Define groups
            "corr_matrix": np.array([[1.0, 0.5], [0.5, 1.0]]) # Also define corr_matrix
        }
        problem = ProblemSpec(problem_dict)
        with raises(ValueError, match="Groups and corr_matrix cannot be used simultaneously"):
            sample(problem, 10)

    def test_latin_sample_correlated_generation_2vars(self):
        N_test = 2000 # Number of samples for testing correlation
        problem_dict = {
            "num_vars": 2,
            "names": ["x1", "x2"],
            "bounds": [[0, 1], [0, 1]], # Output will be U[0,1], easy to check rank corr
            "corr_matrix": np.array([[1.0, 0.8],
                                     [0.8, 1.0]])
        }
        problem = ProblemSpec(problem_dict)

        # Using ProblemSpec to ensure corr_matrix is validated
        # No need to catch problem spec validation error here, it's tested elsewhere.

        samples_correlated = sample(problem, N_test, seed=123)

        assert samples_correlated.shape == (N_test, 2)
        # Check marginals are still U[0,1] (basic check)
        assert np.all(samples_correlated >= 0.0) and np.all(samples_correlated <= 1.0)
        # Check min/max are close to 0 and 1 for U[0,1]
        # LHS specific: check stratification if possible (harder for correlated)
        # For now, focus on correlation and basic bounds.
        assert np.min(samples_correlated[:,0]) < 0.1 # Roughly, should be small
        assert np.max(samples_correlated[:,0]) > 0.9 # Roughly, should be large
        assert np.min(samples_correlated[:,1]) < 0.1
        assert np.max(samples_correlated[:,1]) > 0.9


        # Calculate Spearman rank correlation from the generated samples
        # These samples are already in the [0,1] range, reflecting the rank correlation
        spearman_corr_matrix, _ = spearmanr(samples_correlated)

        # Check if the empirical correlation is close to the target
        # For rank correlation from Iman & Conover method, it should be quite close
        # to the target Pearson correlation of the normal variables.
        target_corr = problem["corr_matrix"][0, 1]
        empirical_corr = spearman_corr_matrix[0, 1]

        # Tolerance can be relatively loose due to sampling variability
        assert empirical_corr == approx(target_corr, abs=0.05), \
            f"Empirical rank correlation {empirical_corr:.3f} not close to target {target_corr:.3f}"

    def test_latin_sample_correlated_generation_3vars(self):
        N_test = 3000 # More samples for 3 vars
        problem_dict = {
            "num_vars": 3,
            "names": ["x1", "x2", "x3"],
            "bounds": [[0, 1], [0, 1], [0,1]], # U[0,1] marginals
            "corr_matrix": np.array([[1.0, 0.7, 0.3],
                                     [0.7, 1.0, 0.5],
                                     [0.3, 0.5, 1.0]])
        }
        problem = ProblemSpec(problem_dict)
        samples_correlated = sample(problem, N_test, seed=456)

        assert samples_correlated.shape == (N_test, 3)
        assert np.all(samples_correlated >= 0.0) and np.all(samples_correlated <= 1.0)

        spearman_corr_matrix, _ = spearmanr(samples_correlated)

        # Check all off-diagonal elements
        assert spearman_corr_matrix[0, 1] == approx(problem["corr_matrix"][0, 1], abs=0.05)
        assert spearman_corr_matrix[0, 2] == approx(problem["corr_matrix"][0, 2], abs=0.05)
        assert spearman_corr_matrix[1, 2] == approx(problem["corr_matrix"][1, 2], abs=0.05)

        # Check diagonal elements are 1 (or very close)
        assert np.allclose(np.diag(spearman_corr_matrix), 1.0, atol=1e-3)
