# -*- coding: utf-8 -*-
"""
This module implements a correlation-aware Sobol sensitivity analysis method,
calculating 'full' first-order and total-order indices.
"""
import logging
import numpy as np
from scipy.stats import norm
from multiprocess import Pool  # For parallel bootstrapping
from SALib.util import ResultDict  # For structuring results
import warnings

logger = logging.getLogger(__name__)

# Define the worker function for bootstrapping (must be top-level for pickling)
def _bootstrap_sobol_correlated_worker(args):
    """
    Worker function for parallel bootstrapping of correlated Sobol' indices.
    Recalculates S1_full and ST_full for a single bootstrap sample of Y values.
    """
    (   Y_A_boot, Y_B_boot, Y_C_i_all_boot, Y_D_i_all_boot,
        D_vars, N_base, V_Y_total_variance_boot, # Pass total variance of this bootstrap sample
        calc_second_order_flag # New argument
    ) = args

    s1_full_boot = np.empty(D_vars)
    st_full_boot = np.empty(D_vars)

    # Calculate S1_full for this bootstrap sample
    for i in range(D_vars):
        Y_Ci_boot = Y_C_i_all_boot[:, i] # Y_C_i for this bootstrap sample
        # Numerator: Cov(Y_A_boot, Y_Ci_boot)
        s1_numerator = np.mean(Y_A_boot * Y_Ci_boot) - (np.mean(Y_A_boot) * np.mean(Y_Ci_boot))
        s1_full_boot[i] = s1_numerator / V_Y_total_variance_boot if V_Y_total_variance_boot != 0 else 0.0


    # Calculate ST_full for this bootstrap sample
    for i in range(D_vars):
        Y_Di_boot = Y_D_i_all_boot[:, i] # Y_D_i for this bootstrap sample
        st_full_boot[i] = (0.5 * np.mean((Y_A_boot - Y_Di_boot)**2)) / V_Y_total_variance_boot if V_Y_total_variance_boot != 0 else 0.0

    s2_full_boot = None # Default if not calculated
    if calc_second_order_flag:
        s2_full_boot = np.full((D_vars, D_vars), np.nan)
        mean_YA_boot = np.mean(Y_A_boot)
        mean_YB_boot = np.mean(Y_B_boot)
        for i in range(D_vars):
            for j in range(i + 1, D_vars):
                Y_Ci_boot = Y_C_i_all_boot[:, i]
                Y_Dj_boot = Y_D_i_all_boot[:, j] # Y_Dj from the D_i_all matrix

                numerator_S2_ij_boot = np.mean(Y_Ci_boot * Y_Dj_boot) - mean_YA_boot * mean_YB_boot
                s2_val_boot = (numerator_S2_ij_boot / V_Y_total_variance_boot if V_Y_total_variance_boot != 0 else 0.0) \
                              - s1_full_boot[i] - s1_full_boot[j]

                s2_full_boot[i, j] = s2_val_boot
                s2_full_boot[j, i] = s2_val_boot

    return s1_full_boot, st_full_boot, s2_full_boot


def analyze(problem: dict, Y: np.ndarray, calc_second_order: bool = True,
            num_resamples: int = 100, conf_level: float = 0.95,
            print_to_console: bool = False, seed: int = None,
            parallel: bool = False, n_processors: int = None) -> ResultDict:
    """
    Performs a correlation-aware Sobol sensitivity analysis on model outputs.

    This function estimates "full" first-order (S1_full) and total-order (ST_full)
    sensitivity indices. These indices account for correlations between input
    parameters and their interpretation differs from standard Sobol' indices
    (which assume input independence).

    The implemented estimators are:
    - For S1_full_i (:math:`S_i^F`):
      :math:`\hat{S1}_{full,i} = \frac{ \text{Cov}(Y_A, Y_{C_i}) }{ \hat{V}(Y) } = \frac{ \frac{1}{N} \sum_{k=1}^{N} Y_A^{(k)} Y_{C_i}^{(k)} - E[Y_A] E[Y_{C_i}] }{ \hat{V}(Y) }`
      where :math:`Y_A = f(X_A)`, :math:`Y_{C_i} = f(X_{A,i}, X_{B,\sim i})`.
    - For ST_full_i (:math:`ST_i^F`):
      :math:`\hat{ST}_{full,i} = \frac{ \frac{1}{2N} \sum_{k=1}^{N} (Y_A^{(k)} - Y_{D_i}^{(k)})^2 }{ \hat{V}(Y) }`
      where :math:`Y_{D_i} = f(X_{A,\sim i}, X_{B,i})`.
    - If `calc_second_order` is True, for S2_full_ij (:math:`S_{ij}^F`):
      :math:`\hat{S2}_{full,ij} = \frac{ \text{Cov}(Y_{C_i}, Y_{D_j}) - \text{Cov}(Y_A, Y_B) }{ \hat{V}(Y) } - \hat{S1}_{full,i} - \hat{S1}_{full,j}`
      A common estimator form is used:
      :math:`\hat{S2}_{full,ij} = \frac{ (\frac{1}{N} \sum_{k=1}^{N} Y_{C_i}^{(k)} Y_{D_j}^{(k)}) - E[Y_A]E[Y_B] }{\hat{V}(Y)} - \hat{S1}_{full,i} - \hat{S1}_{full,j}`
      where :math:`Y_{C_i} = f(X_{A,i}, X_{B,\sim i})` and :math:`Y_{D_j} = f(X_{A,\sim j}, X_{B,j})`.
      Note: The precise definition and properties of S2_full can vary in literature.

    These estimators are inspired by approaches for estimating "full"
    sensitivity indices with correlated inputs (e.g., Janon et al., 2013;
    Mara & Tarantola, 2012; Saltelli, 2002 for S2 structure). Users should
    consult these references for detailed theoretical background.
    This implementation is **experimental**.

    It expects model outputs `Y` generated from samples produced by
    `SALib.sample.sobol_correlated.sample()`.

    Parameters
    ----------
    problem : dict
        The problem definition, including:
        - `num_vars` (int): Number of input variables.
        - `names` (list): List of variable names.
        - `corr_matrix` (np.ndarray): The correlation matrix used for sampling.
    Y : np.ndarray
        A NumPy array containing the model outputs. Must correspond to samples
        generated by `sample_sobol_correlated`. Expected length is N * (2 + 2*D).
    calc_second_order : bool, optional
        If True, this flag is noted, but current implementation only provides
        S1_full and ST_full. Second-order "full" effects are not yet implemented.
        Default is True.
    num_resamples : int, optional
        Number of bootstrap resamples for confidence interval estimation (default 100).
    conf_level : float, optional
        Confidence level for bootstrap intervals (default 0.95).
    print_to_console : bool, optional
        If True, prints results to console (default False).
    seed : int, optional
        Seed for random number generation (for bootstrapping).
    parallel : bool, optional
        If True, parallelizes bootstrap computations (default False).
    n_processors : int, optional
        Number of processors for parallel bootstrap (default None).

    Returns
    -------
    ResultDict
        A dictionary containing the sensitivity indices:
        - `S1_full` (np.array): Full first-order indices (D,).
        - `S1_full_conf` (np.array): Confidence intervals for S1_full (D,).
        - `ST_full` (np.array): Full total-order indices (D,).
        - `ST_full_conf` (np.array): Confidence intervals for ST_full (D,).
        - `S2_full` (np.array, optional): Full second-order indices (D,D).
          Present if `calc_second_order=True`.
        - `S2_full_conf` (np.array, optional): Confidence intervals for S2_full (D,D).
          Present if `calc_second_order=True`.
    """
    if seed is not None:
        np.random.seed(seed)

    D = problem['num_vars']

    if problem.get('corr_matrix') is None:
        # This should ideally be caught by sampler, but good to double check
        raise ValueError("`analyze_sobol_correlated` requires a `corr_matrix` in the problem definition.")

    if problem.get('groups') is not None:
        warnings.warn("Warning: `analyze_sobol_correlated` received a problem with 'groups' defined. "
                      "This method is not designed or validated for use with grouped sampling "
                      "in combination with its correlation-specific estimators. "
                      "Results may be unreliable.", UserWarning)

    # Infer N_base from Y and D
    # Total samples = N_base * (2 + 2*D)
    expected_block_size = 2 + 2 * D
    if Y.size % expected_block_size != 0:
        raise ValueError(f"Incorrect number of samples in Y. Expected a multiple of (2 + 2*D) = {expected_block_size}, "
                         f"but got Y.size = {Y.size}")
    N_base = Y.size // expected_block_size

    if N_base == 0:
        raise ValueError("Not enough samples in Y to perform analysis.")

    # Separate Y into Y_A, Y_B, Y_Ci_all, Y_Di_all
    Y_A = Y[0:N_base]
    Y_B = Y[N_base : 2 * N_base]

    Y_C_matrices_flat = Y[2 * N_base : N_base * (2 + D)]
    Y_D_matrices_flat = Y[N_base * (2 + D) : N_base * (2 + 2 * D)]

    # Reshape Y_C_i and Y_D_i into (N_base, D) arrays
    # Column j of Y_C_i_all corresponds to Y_Cj (output from X_Cj)
    Y_C_i_all = np.reshape(Y_C_matrices_flat, (N_base, D), order='F') # Fortran order for column major fill
    Y_D_i_all = np.reshape(Y_D_matrices_flat, (N_base, D), order='F')


    # Calculate Total Variance V_Y
    # Using Y_A and Y_B for variance calculation is a common choice.
    # Or use just Y_A. Let's use Y_A and Y_B.
    V_Y = np.var(np.concatenate((Y_A, Y_B)), ddof=1)
    if V_Y == 0: # Handle constant output case
        warnings.warn("Total variance of model output is zero. All sensitivity indices will be zero.", UserWarning)
        # Return zeros or NaNs
        S1_full = np.zeros(D)
        ST_full = np.zeros(D)
        S1_full_conf = np.full(D, np.nan)
        ST_full_conf = np.full(D, np.nan)

        Si = ResultDict([('S1_full', S1_full), ('S1_full_conf', S1_full_conf),
                         ('ST_full', ST_full), ('ST_full_conf', ST_full_conf)])
        Si['names'] = problem['names']
        if print_to_console:
            logger.info(Si)
        return Si


    # Calculate Full First-Order Indices (S1_full)
    S1_full = np.empty(D)
    for i in range(D):
        Y_Ci = Y_C_i_all[:, i]
        # Numerator: Cov(Y_A, Y_Ci)
        s1_numerator = np.mean(Y_A * Y_Ci) - (np.mean(Y_A) * np.mean(Y_Ci))
        S1_full[i] = s1_numerator / V_Y

    # Calculate Full Total-Order Indices (ST_full)
    ST_full = np.empty(D)
    for i in range(D):
        Y_Di = Y_D_i_all[:, i]
        ST_full[i] = (0.5 * np.mean((Y_A - Y_Di)**2)) / V_Y

    # Calculate Full Second-Order Indices (S2_full) if requested
    S2_full = np.full((D, D), np.nan)
    if calc_second_order:
        mean_YA = np.mean(Y_A)
        mean_YB = np.mean(Y_B)
        for i in range(D):
            for j in range(i + 1, D):
                Y_Ci = Y_C_i_all[:, i]
                Y_Dj = Y_D_i_all[:, j] # Note: Y_Dj uses column j of Y_D_i_all
                                       # Y_D_i_all[:,j] is Y_Dj = f(X_A_~j, X_B_j)

                # Estimator for S2_full_ij based on conceptual literature alignment:
                # ( np.mean(Y_Ci * Y_Dj) - np.mean(Y_A) * np.mean(Y_B) ) / V_Y - S1_full[i] - S1_full[j]
                # This uses Y_Ci and Y_Dj.
                # Saltelli (2002) for independent inputs uses Y_ABj (my Y_Dj) and Y_BAi (my Y_Ci)
                # So the product term is np.mean(Y_Ci * Y_Dj)

                numerator_S2_ij = np.mean(Y_Ci * Y_Dj) - mean_YA * mean_YB
                s2_val = (numerator_S2_ij / V_Y) - S1_full[i] - S1_full[j]

                S2_full[i, j] = s2_val
                S2_full[j, i] = s2_val # Symmetric

    # Confidence Intervals via Bootstrapping
    S1_full_conf_values = np.full(D, np.nan)
    ST_full_conf_values = np.full(D, np.nan)

    if num_resamples > 0:
        if parallel:
            # Prepare tasks for parallel execution
            # Each task needs bootstrapped Y_A, Y_B, Y_C_i_all, Y_D_i_all and the total variance of that bootstrap sample

            bootstrap_tasks = []
            rng_indices = np.random.randint(0, N_base, size=(num_resamples, N_base))

            for k in range(num_resamples):
                sample_indices = rng_indices[k, :]
                Y_A_boot = Y_A[sample_indices]
                Y_B_boot = Y_B[sample_indices]
                Y_C_i_all_boot = Y_C_i_all[sample_indices, :]
                Y_D_i_all_boot = Y_D_i_all[sample_indices, :]

                V_Y_boot = np.var(np.concatenate((Y_A_boot, Y_B_boot)), ddof=1)
                if V_Y_boot == 0: V_Y_boot = 1e-12 # Avoid division by zero in worker if output is constant for a bootstrap sample

                bootstrap_tasks.append((Y_A_boot, Y_B_boot, Y_C_i_all_boot, Y_D_i_all_boot, D, N_base, V_Y_boot, calc_second_order)) # Pass calc_second_order

            pool_processors = n_processors
            with Pool(processes=pool_processors) as pool:
                bootstrap_results = pool.map(_bootstrap_sobol_correlated_worker, bootstrap_tasks)

            s1_full_resamples = np.array([res[0] for res in bootstrap_results])
            st_full_resamples = np.array([res[1] for res in bootstrap_results])
            if calc_second_order:
                s2_full_resamples = np.array([res[2] for res in bootstrap_results]) # Shape (num_resamples, D, D)


        else: # Serial bootstrapping
            s1_full_resamples = np.empty((num_resamples, D))
            st_full_resamples = np.empty((num_resamples, D))
            if calc_second_order:
                s2_full_resamples = np.empty((num_resamples, D, D))

            for k in range(num_resamples):
                sample_indices = np.random.randint(0, N_base, size=N_base)
                Y_A_boot = Y_A[sample_indices]
                Y_B_boot = Y_B[sample_indices]
                Y_C_i_all_boot = Y_C_i_all[sample_indices, :]
                Y_D_i_all_boot = Y_D_i_all[sample_indices, :]

                V_Y_boot = np.var(np.concatenate((Y_A_boot, Y_B_boot)), ddof=1)
                if V_Y_boot == 0: V_Y_boot = 1e-12

                args_for_worker = (Y_A_boot, Y_B_boot, Y_C_i_all_boot, Y_D_i_all_boot, D, N_base, V_Y_boot, calc_second_order) # Pass calc_second_order
                s1_boot_k, st_boot_k, s2_boot_k = _bootstrap_sobol_correlated_worker(args_for_worker)

                s1_full_resamples[k, :] = s1_boot_k
                st_full_resamples[k, :] = st_boot_k
                if calc_second_order:
                    s2_full_resamples[k, :, :] = s2_boot_k

        # Calculate confidence intervals from bootstrap resamples
        z_norm = norm.ppf(0.5 + conf_level / 2.0)
        S1_full_conf_values = z_norm * np.std(s1_full_resamples, axis=0, ddof=1)
        ST_full_conf_values = z_norm * np.std(st_full_resamples, axis=0, ddof=1)

        S2_full_conf_values = np.full((D,D), np.nan)
        if calc_second_order:
            S2_full_conf_values = z_norm * np.std(s2_full_resamples, axis=0, ddof=1)


    # Store results
    results_list = [
        ('S1_full', S1_full),
        ('S1_full_conf', S1_full_conf_values),
        ('ST_full', ST_full),
        ('ST_full_conf', ST_full_conf_values)
    ]
    if calc_second_order:
        results_list.append(('S2_full', S2_full))
        results_list.append(('S2_full_conf', S2_full_conf_values))

    Si = ResultDict(results_list)
    Si['names'] = problem['names'] # Store parameter names

    if print_to_console:
        # Basic print, ProblemSpec.to_df() might need adaptation for these new keys
        logger.info(Si)

    return Si
