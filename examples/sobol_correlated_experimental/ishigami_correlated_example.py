# -*- coding: utf-8 -*-
"""
Example script for using the experimental `sobol_correlated` sampler and analyzer
for sensitivity analysis with correlated inputs, using the Ishigami function.

As this method is experimental, users are strongly advised to consult relevant
academic literature on Sobol' GSA with dependent inputs (e.g., Janon et al., 2013;
Mara & Tarantola, 2012; Kucherenko et al., 2009) to fully understand the
interpretation and limitations of the "full" sensitivity indices produced.
"""

# 0. Imports
import sys
import numpy as np

# Ensure SALib is importable (if running from examples directory)
# This might need adjustment based on how SALib is installed or if this script
# is moved into the main SALib examples structure that handles paths.
# For now, assuming SALib is in the Python path.
# sys.path.append('../..') # Example if running from a subdir of SALib root

from SALib.sample.sobol_correlated import sample as sample_sobol_correlated
from SALib.analyze.sobol_correlated import analyze as analyze_sobol_correlated
from SALib.test_functions import Ishigami
from SALib.util.problem import ProblemSpec # For validating problem dict if needed, though not strictly necessary for this script


# 1. Introduction
# This script demonstrates the use of an experimental feature in SALib for
# performing a Sobol-like sensitivity analysis when input parameters are correlated.
# The method estimates "full" first-order (S1_full), total-order (ST_full),
# and second-order (S2_full) sensitivity indices. These indices attempt to
# account for the correlation structure among inputs.
#
# !!! WARNING: This is an EXPERIMENTAL feature. !!!
# The interpretation of these "full" indices is more complex than standard
# Sobol' indices for independent inputs. Users must exercise caution and consult
# relevant literature.

def main():
    # 2. Problem Definition
    # We use the Ishigami function, which has 3 input variables (x1, x2, x3).
    # Standard bounds are [-pi, pi] for each.
    # We will define a correlation structure between these inputs.
    problem = {
        'num_vars': 3,
        'names': ['x1', 'x2', 'x3'],
        'bounds': [[-np.pi, np.pi]] * 3,
        'corr_matrix': np.array([[1.0,  0.6,  0.2],
                                 [0.6,  1.0, -0.4],
                                 [0.2, -0.4,  1.0]]),
        # 'dists' is not specified, so inputs will be sampled from uniform marginal
        # distributions within their bounds, then correlation will be induced.
    }

    # (Optional: Validate problem dictionary - ProblemSpec does this internally if used)
    # spec = ProblemSpec(problem) # This would validate corr_matrix etc.

    print("Problem Definition:")
    print(problem)
    print("-" * 60)

    # 3. Generate Samples
    # N is the base number of samples.
    # The sobol_correlated sampler will generate N * (2 + 2*D) samples.
    N_base = 1024  # Base sample size (e.g., 2^10)
    # For D=3, total samples = N_base * (2 + 2*3) = N_base * 8

    print(f"Generating N_base = {N_base} samples using `sample_sobol_correlated`...")
    param_values = sample_sobol_correlated(problem, N=N_base, seed=101)
    print(f"Shape of generated parameter samples: {param_values.shape}")
    print(f"Expected shape: ({N_base * (2 + 2 * problem['num_vars'])}, {problem['num_vars']})")
    print("-" * 60)

    # 4. Run Model (Ishigami Function)
    print("Running Ishigami function with the generated samples...")
    Y = Ishigami.evaluate(param_values)
    print(f"Shape of model outputs Y: {Y.shape}")
    print("-" * 60)

    # 5. Perform Analysis
    # We will calculate full first-order, total-order, and second-order indices.
    print("Performing correlated Sobol analysis using `analyze_sobol_correlated`...")
    Si_correlated = analyze_sobol_correlated(problem, Y,
                                             calc_second_order=True,
                                             num_resamples=100, # Number of bootstrap resamples for CI
                                             conf_level=0.95,
                                             seed=101,
                                             print_to_console=False) # We will print manually

    print("-" * 60)
    print("Analysis Complete.")
    print("-" * 60)

    # 6. Results & Interpretation
    print("\n--- Sensitivity Indices (Experimental - Correlated Inputs) ---")

    print("\nFull First-Order Indices (S1_full):")
    for i, name in enumerate(problem['names']):
        print(f"{name}: {Si_correlated['S1_full'][i]:.3f} (CI: {Si_correlated['S1_full_conf'][i]:.3f})")

    print("\nFull Total-Order Indices (ST_full):")
    for i, name in enumerate(problem['names']):
        print(f"{name}: {Si_correlated['ST_full'][i]:.3f} (CI: {Si_correlated['ST_full_conf'][i]:.3f})")

    if 'S2_full' in Si_correlated:
        print("\nFull Second-Order Indices (S2_full):")
        for i in range(problem['num_vars']):
            for j in range(i + 1, problem['num_vars']):
                name_i = problem['names'][i]
                name_j = problem['names'][j]
                print(f"({name_i}, {name_j}): {Si_correlated['S2_full'][i,j]:.3f} (CI: {Si_correlated['S2_full_conf'][i,j]:.3f})")

    print("\n--- Interpretation Notes ---")
    print("The Ishigami function is defined as: Y = sin(x1) + a * sin(x2)^2 + b * x3^4 * sin(x1)")
    print(f"With a=7, b=0.1. Standard (independent) Sobol indices are roughly:")
    print("  S1(x1) approx 0.31, S1(x2) approx 0.44, S1(x3) = 0.0")
    print("  ST(x1) approx 0.56, ST(x2) approx 0.44, ST(x3) = 0.24 (due to x1*x3 interaction)")
    print("  S2(x1,x3) is the main interaction term.")
    print("\nCorrelation Structure Defined:")
    print(f"  Corr(x1,x2) = {problem['corr_matrix'][0,1]}, Corr(x1,x3) = {problem['corr_matrix'][0,2]}, Corr(x2,x3) = {problem['corr_matrix'][1,2]}")

    print("\nInterpreting 'Full' Indices (S_full):")
    print("S1_full (Full First-Order Index):")
    print("  - Measures the main effect of a variable Xi on Y, including effects shared")
    print("    due to its correlation with other input variables.")
    print("  - A high S1_full suggests Xi is important by itself, considering the system's correlations.")
    print("  - Sum of S1_full is not necessarily 1 and can be > 1 or < sum of standard S1.")
    print(f"  - Sum of S1_full in this case: {np.sum(Si_correlated['S1_full']):.3f}")
    print("  - Note: S1_full can sometimes be negative due to estimator variance or complex cancellation")
    print("    of effects in highly correlated, non-linear models, though this is less common for these estimators.")

    print("\nST_full (Full Total-Order Index):")
    print("  - Measures the total effect of Xi on Y, including its main effect, all interactions")
    print("    it's involved in, and all effects shared due to its correlations.")
    print("  - A high ST_full indicates Xi is influential overall in the correlated system.")
    print("  - If ST_full_i is close to zero, Xi (and its correlations/interactions) has little impact.")

    print("\nS2_full (Full Second-Order Index):")
    print("  - Measures the interaction effect between Xi and Xj, inclusive of correlation impacts.")
    print("  - A large S2_full_ij suggests a strong joint effect of Xi and Xj beyond their")
    print("    individual full first-order effects.")
    print("  - Like S1_full, S2_full can also sometimes be negative.")

    print("\nObservations for this Ishigami example with correlations:")
    print("  - Compare the S1_full values to the standard S1 values. How have they changed?")
    print("    For example, x1 is correlated with x2 (rho=0.6). If x1 and x2 both positively affect Y,")
    print("    their S1_full might increase compared to standard S1 as they 'share' some effect.")
    print("    If one's effect is positive and other negative, correlation might reduce S1_full.")
    print("  - x3 has S1=0 in the independent case. With correlations (e.g., x2-x3 rho=-0.4),")
    print("    does S1_full(x3) pick up any effect? It might if x3's correlation with an influential")
    print("    variable (like x2) allows it to 'explain' some variance that was previously purely x2's.")
    print("  - Look at ST_full(x3). In the independent case, ST(x3) is non-zero due to the x1*x3 interaction.")
    print("    How does correlation with x1 and x2 affect ST_full(x3)?")
    print("  - The S2_full(x1,x3) term is expected to be significant. How do correlations with x2")
    print("    affect this apparent interaction strength compared to an independent S2(x1,x3)?")

    # 7. Comparison/Discussion (Optional - to be added as comments)
    print("\n--- Comparison & Discussion Notes ---")
    print("Key differences from standard Sobol' (independent inputs):")
    print("1. Indices account for correlation: Standard Sobol' assumes independence. `S_full` indices try to attribute variance in a system where inputs move together.")
    print("2. No simple sum-to-one: Unlike standard S1 + S2 + ... = 1 (for sum of all orders), the sum of `S1_full` or other combinations do not have such a straightforward interpretation.")
    print("3. Interpretation focus: `S_full` indices are about the overall impact of a variable *within the defined correlated system*, not its 'independent' structural contribution to variance (which is hard to define when inputs are not independent).")
    print("4. Use Case: Helpful when input correlations are inherent and cannot be ignored or transformed away, and one wants to understand importance in that specific correlated context.")


    # 8. Conclusion & Caveats
    print("\n--- Conclusion & Caveats ---")
    print("This example demonstrates the experimental `sobol_correlated` method.")
    print("Key takeaways:")
    print("  - It provides 'full' sensitivity indices that account for input correlations.")
    print("  - Interpretation requires careful consideration of the correlation structure")
    print("    and reference to relevant academic literature.")
    print("  - This feature is EXPERIMENTAL. Use with caution and critical thinking.")
    print("-" * 60)

if __name__ == "__main__":
    main()
