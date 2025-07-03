# -*- coding: utf-8 -*-
"""
Example script demonstrating the usage of new interactive plotting functions
in SALib, specifically `bar_plot_interactive` and `raincloud_interactive`.

These plots require the Plotly library to be installed.
If Plotly is not installed, this script will print messages indicating that
the plots cannot be generated.

To run this example:
1. Ensure SALib is installed.
2. Install Plotly: `pip install plotly>=5.0`
   Alternatively, install SALib with the interactive plotting extra:
   `pip install salib[plotting_interactive]`
3. Execute this script: `python interactive_plots_demo.py`
   Interactive plots will typically open in your web browser or be displayed
   if run in an environment like a Jupyter notebook.
"""

import numpy as np
import pandas as pd

# SALib imports
from SALib.sample import saltelli
from SALib.analyze import sobol
from SALib.test_functions import Ishigami
from SALib.util import ResultDict

# Import the new interactive plotting functions
try:
    from SALib.plotting.interactive import bar_plot_interactive, raincloud_interactive
    PLOTLY_INSTALLED = True
except ImportError:
    PLOTLY_INSTALLED = False
    print("Plotly library not found. Interactive plots will not be generated.")
    print("Please install it: pip install plotly>=5.0 or pip install salib[plotting_interactive]")


def run_ishigami_sobol_analysis(N=512, seed=None):
    """Helper function to run Sobol analysis on Ishigami."""
    problem = {
        'num_vars': 3,
        'names': ['x1', 'x2', 'x3'],
        'bounds': [[-np.pi, np.pi]] * 3
    }
    param_values = saltelli.sample(problem, N, calc_second_order=True, seed=seed)
    Y = Ishigami.evaluate(param_values)
    Si = sobol.analyze(problem, Y, calc_second_order=True, num_resamples=100, # Low resamples for example speed
                       conf_level=0.95, seed=seed, print_to_console=False)
    return Si, problem


def main():
    print("--- SALib Interactive Plotting Demo ---")

    if not PLOTLY_INSTALLED:
        print("\nSkipping interactive plot generation as Plotly is not installed.")
        return

    # --- Generate Sample Sobol Analysis Results ---
    print("\n1. Generating sample Sobol analysis results for Ishigami function...")
    Si_sobol, problem_ishigami = run_ishigami_sobol_analysis(N=256, seed=101) # Smaller N for faster example
    print("   Sobol analysis complete.")

    # --- Demonstrate bar_plot_interactive ---
    print("\n2. Demonstrating interactive bar plot for S1 indices...")
    try:
        fig_bar_s1 = bar_plot_interactive(
            Si_sobol,
            index_key='S1',
            conf_key='S1_conf',
            title='Interactive First-Order Sobol Indices (S1) - Ishigami',
            sort_by_value='descending',
            layout_kwargs={'width': 800, 'height': 500}
        )
        print("   Showing S1 bar plot... (may open in browser or display in notebook)")
        fig_bar_s1.show()
        # In a script, .show() usually opens a browser tab.
        # fig_bar_s1.write_html("interactive_s1_bar.html") # Option to save
    except Exception as e:
        print(f"   Could not generate S1 bar plot: {e}")

    print("\n3. Demonstrating interactive bar plot for ST indices (horizontal)...")
    try:
        fig_bar_st = bar_plot_interactive(
            Si_sobol,
            index_key='ST',
            conf_key='ST_conf',
            title='Interactive Total-Order Sobol Indices (ST) - Ishigami (Horizontal)',
            orientation='h',
            bar_kwargs={'text_auto': '.2f'}, # Show values on bars
            layout_kwargs={'width': 700, 'height': 400}
        )
        print("   Showing ST bar plot (horizontal)...")
        fig_bar_st.show()
        # fig_bar_st.write_html("interactive_st_bar_horizontal.html")
    except Exception as e:
        print(f"   Could not generate ST bar plot: {e}")

    # --- Demonstrate raincloud_interactive ---
    print("\n4. Demonstrating interactive raincloud plot...")
    # For raincloud, we ideally want distributions (e.g., from bootstrap).
    # `sobol.analyze` with `keep_resamples=True` stores S1_conf_all, ST_conf_all, etc.
    # which are the resampled CI bounds, not the raw index values from each bootstrap.
    # For this demo, let's create some dummy bootstrap-like data for S1.
    # Each column is a parameter, each row is a bootstrap sample's S1 value for that param.

    # Re-run Sobol with keep_resamples=True to get S1_conf_all (though it's CI values, not raw S1s)
    # This is not ideal for raincloud. Raincloud is best for raw resampled values.
    # Let's simulate raw resampled S1 values for the demo.
    num_params = problem_ishigami['num_vars']
    num_bootstrap_samples = 200

    # Create dummy data: array of shape (num_bootstrap_samples, num_params)
    # Centered around the calculated S1 values, with some spread
    simulated_s1_bootstrap_data = np.random.normal(
        loc=Si_sobol['S1'],
        scale=Si_sobol['S1_conf'] * 2 + 0.02, # Scale CIs to get a reasonable spread
        size=(num_bootstrap_samples, num_params)
    )
    # Ensure non-negativity for S1-like data for plausibility
    simulated_s1_bootstrap_data = np.maximum(0, simulated_s1_bootstrap_data)


    print("   Using simulated bootstrap data for S1 index distributions.")
    try:
        fig_rain_s1 = raincloud_interactive(
            simulated_s1_bootstrap_data,
            param_names=problem_ishigami['names'],
            title='Interactive Raincloud Plot: Simulated S1 Index Distributions - Ishigami',
            orientation='v',
            show_box=True,
            show_violin=True,
            show_points='all',
            point_jitter=0.4,
            point_position=-0.8, # Points to the left of violin
            colors=['cornflowerblue', 'orange', 'lightgreen'],
            layout_kwargs={'width': 800, 'height': 600}
        )
        print("   Showing S1 raincloud plot...")
        fig_rain_s1.show()
        # fig_rain_s1.write_html("interactive_s1_raincloud.html")
    except Exception as e:
        print(f"   Could not generate S1 raincloud plot: {e}")

    print("\n--- Demo Complete ---")
    print("If plots did not appear, ensure you are in an environment that can display")
    print("Plotly figures (like Jupyter Notebook/Lab) or check your browser.")

if __name__ == "__main__":
    main()
