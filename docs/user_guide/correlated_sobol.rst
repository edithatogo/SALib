.. _correlated-sobol-analysis:

Sobol' Sensitivity Analysis with Correlated Inputs
===================================================

The standard Sobol' method for variance-based sensitivity analysis assumes that input parameters are independent. When input parameters are correlated, the interpretation of classical Sobol' indices (S1, ST, S2) can be misleading, as the variance decomposition no longer uniquely assigns portions of output variance to individual inputs or their interactions in a straightforward manner.

SALib provides an experimental implementation of a Sobol-like sensitivity analysis method designed to handle correlated input parameters, available through `SALib.sample.sobol_correlated.sample` and `SALib.analyze.sobol_correlated.analyze`.

Conceptual Basis
----------------

The method implemented is based on the concept of estimating "full" first-order and total-order sensitivity indices. These indices attempt to quantify the importance of each input variable while accounting for the existing correlation structure among all inputs. The estimators are inspired by pick-freeze type schemes adapted for correlated inputs, requiring :math:`N \\times (2 + 2D)` model evaluations, where :math:`N` is the base sample size and :math:`D` is the number of input variables.

The key idea is to evaluate how the output variance changes when an input :math:`X_i` is fixed (for first-order effects) or when all other inputs :math:`X_{\\sim i}` are fixed (for total-order effects), under the true joint distribution of the correlated inputs.

**Important Note:** This implementation is **experimental**. The estimators used are designed to calculate "full" sensitivity indices, which account for input correlations. These estimators are consistent with principles and forms discussed in literature addressing sensitivity analysis with dependent inputs. Key inspirational references include:

*   **Janon, A., Klein, T., Lagnoux, A., Nodet, M., & Prieur, C. (2013).** Variance-based sensitivity analysis for model output with dependent inputs. *Communications in Applied and Industrial Mathematics, 5*(1), e-413. (Provides context for first and total order full effects).
*   **Saltelli, A. (2002).** Making best use of model evaluations to compute sensitivity indices. *Computer Physics Communications, 145*(2), 280-297. (The structural form of the second-order estimator is analogous to estimators for independent inputs presented here, adapted for the correlated sampling scheme).
*   **Mara, T. A., & Tarantola, S. (2012).** Variance-based sensitivity indices for models with dependent inputs. *Reliability Engineering & System Safety, 107*, 115-121. (General context on SA with dependent inputs).

Users **must** consult this and other relevant academic literature for a detailed understanding of the theoretical background, precise definitions, properties, and interpretation of such "full" indices in correlated systems.

Indices Calculated
------------------

The `analyze_sobol_correlated` function estimates the following "full" indices, based on the methodologies referenced above, using specific Monte Carlo estimators:

*   **S1_full (Full First-Order Index):**
    This index quantifies the main effect of :math:`X_i` on the output :math:`Y`, inclusive of effects shared due to its correlation with other input variables.
    The implemented estimator is:
    :math:`\hat{S1}_{full,i} = \frac{\frac{1}{N} \sum_{k=1}^{N} Y_A^{(k)} Y_{C_i}^{(k)} - \left(\frac{1}{N}\sum_{k=1}^{N} Y_A^{(k)}\right) \left(\frac{1}{N}\sum_{k=1}^{N} Y_{C_i}^{(k)}\right)}{\hat{V}(Y)}`
    where :math:`Y_A = f(X_A)`, :math:`Y_{C_i} = f(X_{A,i}, X_{B,\sim i})`, and :math:`\hat{V}(Y)` is the estimated total variance of :math:`Y` (from :math:`Y_A, Y_B`). This estimates :math:`Cov(Y_A, Y_{C_i}) / V(Y)`.
    The sum of `S1_full` indices may not be equal to 1 and can exceed 1 or be negative.

*   **ST_full (Full Total-Order Index):**
    This index quantifies the total effect of :math:`X_i` on :math:`Y`, including its main effect, all interaction effects involving :math:`X_i`, and all effects shared due to its correlations with other input variables.
    The implemented estimator is:
    :math:`\hat{ST}_{full,i} = \frac{\frac{1}{2N} \sum_{k=1}^{N} (Y_A^{(k)} - Y_{D_i}^{(k)})^2}{\hat{V}(Y)}`
    where :math:`Y_{D_i} = f(X_{A,\sim i}, X_{B,i})`. This estimates :math:`\frac{E[(Y_A - Y_{D_i})^2]}{2V(Y)}`.

*   **S2_full (Full Second-Order Index):**
    This index quantifies the interaction effect between :math:`X_i` and :math:`X_j` on :math:`Y`, inclusive of effects shared due to their joint correlations with other input variables and their own mutual correlation.
    The implemented estimator, inspired by adaptations of Saltelli (2002) for dependent inputs, is:
    :math:`\hat{S2}_{full,ij} = \frac{ \left( \frac{1}{N} \sum_{k=1}^{N} Y_{C_i}^{(k)} Y_{D_j}^{(k)} \right) - E[Y_A]E[Y_B] }{\hat{V}(Y)} - \hat{S1}_{full,i} - \hat{S1}_{full,j}`
    where :math:`Y_{C_i} = f(X_{A,i}, X_{B,\sim i})` and :math:`Y_{D_j} = f(X_{A,\sim j}, X_{B,j})`.
    This index can also be negative due to estimator variance or complex correlation effects.

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
    # If calc_second_order=True:
    # Si_correlated['S2_full']
    # Si_correlated['S2_full_conf']

Interpretation
--------------

*   `S1_full` for :math:`X_i` indicates the expected reduction in output variance if :math:`X_i` were fixed, considering its correlations with other inputs. It represents the "total main effect" of :math:`X_i` in the correlated system.
*   `ST_full` for :math:`X_i` indicates the expected remaining variance if all other variables :math:`X_{\\sim i}` were fixed, again, considering the full correlation structure. It represents the "total overall effect" of :math:`X_i`, including all interactions it's involved in, magnified or diminished by correlations.
*   Unlike standard Sobol' indices for independent inputs, these "full" indices do not neatly sum to 1 (for S1_full) or provide a simple decomposition of variance into disjoint parts.
*   A high `S1_full` suggests :math:`X_i` is important on its own, even accounting for its correlations.
*   A high `ST_full` suggests :math:`X_i` is involved in the model's behavior, either directly or through interactions, considering correlations.
*   The difference `ST_full - S1_full` can give an indication of the importance of :math:`X_i` due to interactions, but this also includes correlation effects.
*   `S2_full_ij` provides a measure of the joint importance of :math:`X_i` and :math:`X_j` beyond their first-order effects, within the correlated system. Its interpretation requires care, especially if it's negative.

**Limitations & Cautions:**
*   **Experimental:** This method, including S2_full, should be considered experimental. The exact interpretation and properties of these "full" indices can be complex and depend on the specific mathematical definitions chosen from literature.
*   **Estimator Choice:** The specific estimators used here are inspired by common approaches but might differ from other proposed estimators for correlated inputs. Always refer to the source literature for precise definitions if making critical decisions based on these indices.
*   **Higher-Order Interactions:** Interactions beyond second-order are not explicitly quantified by this method.
*   **Alternative Approaches:** Other approaches for SA with correlated inputs exist, such as transforming inputs to an uncorrelated space (though this can make interpretation difficult) or using regression-based measures.

Always complement these quantitative indices with qualitative understanding of your model and the nature of the input correlations.

Example
-------

For a practical demonstration of how to use this method and interpret its results with the Ishigami function, please see the example script:
:ref:`ishigami_correlated_example` (TODO: Add a proper Sphinx reference or link if sphinx-gallery is used, for now, path below)

The script can be found in the SALib examples directory:
`examples/sobol_correlated_experimental/ishigami_correlated_example.py`
