.. _correlated-sobol-analysis:

Sobol' Sensitivity Analysis with Correlated Inputs
===================================================

The standard Sobol' method for variance-based sensitivity analysis assumes that input parameters are independent. When input parameters are correlated, the interpretation of classical Sobol' indices (S1, ST, S2) can be misleading, as the variance decomposition no longer uniquely assigns portions of output variance to individual inputs or their interactions in a straightforward manner.

SALib provides an experimental implementation of a Sobol-like sensitivity analysis method designed to handle correlated input parameters, available through `SALib.sample.sobol_correlated.sample` and `SALib.analyze.sobol_correlated.analyze`.

Conceptual Basis
----------------

The method implemented is based on the concept of estimating "full" first-order and total-order sensitivity indices. These indices attempt to quantify the importance of each input variable while accounting for the existing correlation structure among all inputs. The estimators are inspired by pick-freeze type schemes adapted for correlated inputs, requiring :math:`N \\times (2 + 2D)` model evaluations, where :math:`N` is the base sample size and :math:`D` is the number of input variables.

The key idea is to evaluate how the output variance changes when an input :math:`X_i` is fixed (for first-order effects) or when all other inputs :math:`X_{\\sim i}` are fixed (for total-order effects), under the true joint distribution of the correlated inputs.

**Important Note:** The precise mathematical formulation and estimators used are based on common conceptual approaches found in literature (e.g., related to work by Janon et al., 2013; Mara & Tarantola, 2012; Kucherenko et al.). For rigorous application, users should refer to specific academic papers detailing these types of "full" indices for correlated systems. This implementation should be considered experimental.

Indices Calculated
------------------

The `analyze_sobol_correlated` function returns the following "full" indices:

*   **S1_full (Full First-Order Index):**
    This index measures the main effect of :math:`X_i` on the output :math:`Y`, including effects that are shared due to its correlation with other input variables.
    The specific estimator used is:
    :math:`S1\_full_i = ( \\frac{1}{N} \\sum_{k=1}^{N} Y_A^{(k)} Y_{C_i}^{(k)} - E[Y_A] E[Y_{C_i}] ) / V(Y)`
    where :math:`Y_A = f(X_A)`, :math:`Y_{C_i} = f(X_{A,i}, X_{B,\\sim i})`, :math:`E[Y_A]` and :math:`E[Y_{C_i}]` are the sample means of :math:`Y_A` and :math:`Y_{C_i}` respectively, and :math:`V(Y)` is the total variance of the output (estimated from :math:`Y_A` and :math:`Y_B`). This estimator approximates :math:`Cov(Y_A, Y_{C_i}) / V(Y)`.
    The sum of `S1_full` indices may not be equal to 1 and can even exceed 1, especially with strong correlations.
    *[Further citation to specific literature, e.g., Janon et al. (2013), needed here after full literature review for this exact estimator form and its properties.]*

*   **ST_full (Full Total-Order Index):**
    This index measures the total effect of :math:`X_i` on :math:`Y`, including its main effect, all interaction effects involving :math:`X_i`, and all effects shared due to its correlations with other input variables.
    The specific estimator used is:
    :math:`ST\_full_i = ( \\frac{1}{2N} \\sum_{k=1}^{N} (Y_A^{(k)} - Y_{D_i}^{(k)})^2 ) / V(Y)`
    where :math:`Y_{D_i} = f(X_{A,\\sim i}, X_{B,i})`. This estimator approximates :math:`(0.5 \\times E[(Y_A - Y_{D_i})^2]) / V(Y)`.
    *[Further citation to specific literature, e.g., Saltelli et al. (2010) for the form, adapted as in Janon et al. (2013) or Mara & Tarantola (2012) for dependent inputs, needed here after full literature review.]*

Confidence intervals for these indices are estimated using bootstrapping.

Usage
-----

**1. Define the Problem with Correlation:**
Your problem dictionary must include a `corr_matrix` key, which is a NumPy array representing the target rank correlation matrix (Spearman's rho) for your input parameters.

.. code-block:: python

    import numpy as np

    problem = {
        'num_vars': 3,
        'names': ['x1', 'x2', 'x3'],
        'bounds': [[-np.pi, np.pi]] * 3, # Example bounds
        'corr_matrix': np.array([[1.0, 0.7, 0.3],
                                 [0.7, 1.0, 0.5],
                                 [0.3, 0.5, 1.0]]),
        # 'dists': ['norm', 'unif', 'norm'] # Optional: define marginal distributions
    }

**2. Generate Samples:**
Use the `SALib.sample.sobol_correlated.sample` function. This sampler uses correlated Latin Hypercube Sampling for the base matrices :math:`X_A` and :math:`X_B`.

.. code-block:: python

    from SALib.sample.sobol_correlated import sample as sample_sobol_correlated

    N = 1024  # Base number of samples
    # Total samples will be N * (2 + 2 * D)
    param_values = sample_sobol_correlated(problem, N, seed=101)

**3. Run Your Model:**
Evaluate your model using the generated `param_values`.

.. code-block:: python

    # Assuming Y is your model evaluation function
    # Y_output = my_model_evaluate_function(param_values)
    from SALib.test_functions import Ishigami # Example
    Y_output = Ishigami.evaluate(param_values)


**4. Perform Analysis:**
Use the `SALib.analyze.sobol_correlated.analyze` function.

.. code-block:: python

    from SALib.analyze.sobol_correlated import analyze as analyze_sobol_correlated

    Si_correlated = analyze_sobol_correlated(problem, Y_output,
                                             num_resamples=100, # For CIs
                                             seed=101,
                                             print_to_console=True)

    # Access results:
    # Si_correlated['S1_full']
    # Si_correlated['S1_full_conf']
    # Si_correlated['ST_full']
    # Si_correlated['ST_full_conf']

Interpretation
--------------

*   `S1_full` for :math:`X_i` indicates the expected reduction in output variance if :math:`X_i` were fixed, considering its correlations with other inputs. It represents the "total main effect" of :math:`X_i` in the correlated system.
*   `ST_full` for :math:`X_i` indicates the expected remaining variance if all other variables :math:`X_{\\sim i}` were fixed, again, considering the full correlation structure. It represents the "total overall effect" of :math:`X_i`, including all interactions it's involved in, magnified or diminished by correlations.
*   Unlike standard Sobol' indices for independent inputs, these "full" indices do not neatly sum to 1 (for S1_full) or provide a simple decomposition of variance into disjoint parts.
*   A high `S1_full` suggests :math:`X_i` is important on its own, even accounting for its correlations.
*   A high `ST_full` suggests :math:`X_i` is involved in the model's behavior, either directly or through interactions, considering correlations.
*   The difference `ST_full - S1_full` can give an indication of the importance of :math:`X_i` due to interactions, but this also includes correlation effects.

**Limitations & Cautions:**
*   **Experimental:** This method should be considered experimental. The exact interpretation and properties of these "full" indices can be complex and depend on the specific mathematical definitions chosen from literature.
*   **Estimator Choice:** The specific estimators used here are common but might differ from other proposed estimators for correlated inputs. Always refer to the source literature for precise definitions if making critical decisions based on these indices.
*   **No Standard Second-Order:** Currently, "full" second-order indices are not calculated by `analyze_sobol_correlated`.
*   **Alternative Approaches:** Other approaches for SA with correlated inputs exist, such as transforming inputs to an uncorrelated space (though this can make interpretation difficult) or using regression-based measures.

Always complement these quantitative indices with qualitative understanding of your model and the nature of the input correlations.
